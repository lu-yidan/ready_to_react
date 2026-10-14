"""Generate a selected raw DuoBox segment using the released reactive checkpoint."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args()
request_path = Path(args.request)
request = json.loads(request_path.read_text())
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.argv = [sys.argv[0], '-t', 'test', '-c', 'configs/exps/reactive_model.yaml',
            'runner_cfg.evaluator_cfg.apd_compute=False']
import torch
from easyvolcap.engine import cfg
from reactmotion.utils.engine_utils import discover_modules
discover_modules()
from reactmotion.utils.net_utils import load_other_model, load_other_network
from reactmotion.dataloaders.datasets.motion_base_dataset import MotionBaseDataset
from reactmotion.utils.motion_repr_transform import MoReprTrans, U, downsample, motoken_cfg
from easyvolcap.utils.base_utils import dotdict

torch.set_num_threads(request.get('torch_threads', 1))
torch.manual_seed(request['seed'])
np.random.seed(request['seed'])
cfg.motoken_net = load_other_network(motoken_cfg)
model = load_other_model(cfg)
files = sorted((ROOT / 'data/DuoBox/fbx_export_joints').glob('*.npy'))
raw = np.load(files[request['clip']], allow_pickle=True).item()
names = sorted(k[:-9] for k in raw if k.endswith('_position'))
# Intervention starts after a common history; recompute derived velocities afterwards.
mode=request.get('opponent_mode','normal')
if mode!='normal':
    who=names[request['fixed']]
    cut=int(request.get('intervention_s',2)*120)+4
    for suffix in ('_position','_rotation'):
        values=raw[who+suffix];ix=np.arange(len(values))
        if mode=='freeze': ix[cut:]=min(cut,len(values)-1)
        elif mode=='delay': ix[cut:]=np.maximum(cut,ix[cut:]-int(request.get('opponent_delay_ms',500)*.12))
        else: raise ValueError('Unknown opponent intervention')
        raw[who+suffix]=values[ix]
dataset = object.__new__(MotionBaseDataset)
motions = {}
for name in names:
    motions[name] = dataset.get_motion_reactive(
        torch.tensor(raw[name + '_position'], dtype=torch.float32)[None],
        torch.tensor(raw[name + '_rotation'], dtype=torch.float32)[None])
for name, other in zip(names, names[::-1]):
    dataset.get_motion2oppo_reactive(motions[name], motions[other])
for motion in motions.values():
    for key in ('pose_pos', 'pose_rot', 'root_mat', 'root_rot', 'root_pos'):
        motion[key] = motion[key][:, downsample:]
start = request['start']
available = min(m.pose_vel.shape[1] for m in motions.values())
frames = list(range(start * downsample, min(available, (start + request['frames']) * downsample), downsample))
frames = frames[:len(frames) // U * U]
if len(frames) < 2 * U:
    raise ValueError('Segment must contain at least 8 model frames')
fixed = names[request['fixed']]
reactor = names[1 - request['fixed']]
batch = dotdict()
for i, name in enumerate((reactor, fixed), 1):
    motion = MoReprTrans.get_actor_motion(motions[name], frames, squeeze=False)
    batch[f'agent{i}'] = dotdict({k: v.cuda() for k, v in motion.items()})
    batch[f'cond_agent{i}'] = dotdict({k: v[:, :U].clone() for k, v in batch[f'agent{i}'].items()})
reference = batch.agent1.pose_pos[0].cpu().numpy().copy()
fixed_reference = batch.agent2.pose_pos[0].cpu().numpy().copy()
# Causal chunk producer: only revealed opponent history reaches pred_next.
import time
from types import SimpleNamespace
from reactmotion.dataloaders.datasets.motion_base_dataset import T_z2y
rotation = np.linalg.inv(T_z2y.detach().cpu().numpy().reshape(4,4))[:3,:3]
job = request_path.parent
connection = None
epoch = None
if request.get('socket'):
    from multiprocessing.connection import Client
    connection = Client(request['socket'], family='AF_UNIX')

def emit(end, prediction_ms, observed_until):
    begin = max(0,end-U) if connection is not None else 0
    positions = np.stack([agent.pose_pos[0,begin:end].cpu().numpy(), batch.agent2.pose_pos[0,begin:end].cpu().numpy()], 1)
    rotations = np.stack([agent.pose_rot[0,begin:end].cpu().numpy(), batch.agent2.pose_rot[0,begin:end].cpu().numpy()], 1)
    positions = np.einsum('ij,takj->taki', rotation, positions)
    rotations = np.einsum('ij,takjl->takil', rotation, rotations)
    if connection is not None:
        connection.send(dict(end=end, positions=positions, rotations=rotations,
                             actors=[reactor,fixed], observed_until=observed_until,
                             prediction_ms=prediction_ms, sent_at=time.monotonic(), total_frames=len(reference)))
        return
    temp = job / 'human.tmp.npz'
    np.savez_compressed(temp, positions=positions, rotations=rotations, fps=30,
                        actors=np.array([reactor, fixed]), observed_until=observed_until,
                        prediction_ms=prediction_ms)
    temp.replace(job / ('human_%04d.npz' % end))
    print('RELEASE', end, 'observed through', observed_until, flush=True)
    if request.get('benchmark', False):
        return
    deadline = time.monotonic() + 600
    while not (job / ('ack_%04d' % end)).exists():
        if (job / 'abort').exists(): raise RuntimeError('Pipeline aborted')
        if time.monotonic() > deadline: raise TimeoutError('Tracker acknowledgement timeout')
        time.sleep(.05)

with torch.inference_mode():
    agent, _ = model.prepare_agent(batch, 1)
    _, opponent = model.prepare_agent(batch, 2)
    emit(U, 0, U-1)
    if connection is not None:
        epoch = connection.recv()['epoch']
    for ts in range(U, len(reference), U):
        if connection is not None:
            # Source frames become available on the wall clock, independently of tracking.
            while time.monotonic() < epoch + ts / 30:
                time.sleep(min(.005, max(0, epoch + ts / 30 - time.monotonic())))
            if ts == request.get('stall_at', -1):
                time.sleep(request.get('stall_ms', 0) / 1000)
        # Tensor slicing physically removes all unrevealed opponent frames.
        fields = ('pose_pos','pose_rot','pose_vel','root_mat')
        history = {k:getattr(opponent,k)[:,:ts] for k in fields}
        history['root_rot'] = opponent.root_rot[:,:ts]
        history['root_pos'] = opponent.root_pos[:,:ts]
        visible = SimpleNamespace(**history)
        torch.cuda.synchronize(); started = time.perf_counter()
        future = model.pred_next(agent, visible, ts)
        # Decoder bookkeeping may request ts+i-1; use held last observation.
        held = {k:torch.cat([v, v[:,-1:].expand(-1,U,*v.shape[2:])],1) for k,v in history.items()}
        for i in range(U):
            model.update_agent(agent, future, ts, i, agent2=SimpleNamespace(**held))
        torch.cuda.synchronize()
        emit(ts+U, (time.perf_counter()-started)*1000, ts-1)
if connection is not None:
    connection.send({'done': True})
    connection.close()
(job / 'producer.done').touch()
