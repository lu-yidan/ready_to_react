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

torch.set_num_threads(4)
torch.manual_seed(request['seed'])
np.random.seed(request['seed'])
cfg.motoken_net = load_other_network(motoken_cfg)
model = load_other_model(cfg)
files = sorted((ROOT / 'data/DuoBox/fbx_export_joints').glob('*.npy'))
raw = np.load(files[request['clip']], allow_pickle=True).item()
names = sorted(k[:-9] for k in raw if k.endswith('_position'))
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
with torch.no_grad():
    output = model.inference_reactive(batch)
generated = output.agent1.pose_pos[0].cpu().numpy()
opponent = output.agent2.pose_pos[0].cpu().numpy()
assert np.isfinite(generated).all()
np.testing.assert_allclose(opponent, fixed_reference, atol=0, rtol=0)
np.testing.assert_allclose(generated[:U], reference[:U], atol=1e-6)
assert len(generated) == len(reference)
np.savez_compressed(request_path.parent / 'motion.npz', generated=generated, fixed=opponent,
                    reference=reference, fps=30, source_frames=np.array(frames)+downsample)
result = dict(file=files[request['clip']].name, fps=30,
              parents=[-1,0,1,2,3,2,5,6,7,2,9,10,11,0,13,14,15,0,17,18,19],
              actors=[dict(name=reactor+' · 模型响应', positions=generated.round(5).tolist()),
                      dict(name=fixed+' · 固定原始轨迹', positions=opponent.round(5).tolist()),
                      dict(name=reactor+' · 原始对照', positions=reference.round(5).tolist())],
              seed=request['seed'], warmup=U, source_start_frame=int(frames[0]+downsample),
              note='前4帧为原始热启动；此后蓝色由模型生成；灰色为原始对照；非物理仿真')
(request_path.parent / 'result.json').write_text(json.dumps(result))
print('DONE', len(generated), 'frames', flush=True)
