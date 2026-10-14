#!/usr/bin/env python3
"""Local, read-only DuoBox skeleton browser; run with the project Python."""
import argparse
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / 'data/DuoBox/fbx_export_joints').glob('*.npy'))
PARENTS = [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]


@lru_cache(maxsize=3)
def motion(index):
    data = np.load(FILES[index], allow_pickle=True).item()
    actors = []
    for key in sorted(k for k in data if k.endswith('_position')):
        raw = data[key][::4]  # Native 120 Hz -> display 30 Hz.
        # Z-up -> Y-up. A common transform preserves the two-person geometry.
        pos = np.stack((raw[..., 0], raw[..., 2], -raw[..., 1]), axis=-1)
        actors.append({'name': key[:-9], 'positions': pos.round(4).tolist()})
    return {'file': FILES[index].name, 'fps': 30, 'parents': PARENTS, 'actors': actors}


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>DuoBox 原始动作浏览器</title><style>
body{margin:0;background:#111827;color:#e5e7eb;font:15px system-ui}header{padding:18px 24px;background:#1f2937}h2{margin:0 0 8px}select{max-width:65vw}select,button,input{font:inherit}select,button{padding:7px;border-radius:6px;background:#334155;color:white;border:1px solid #64748b}#controls{display:flex;gap:12px;align-items:center;padding:12px 24px;flex-wrap:wrap}canvas{display:block;width:100%;height:65vh;touch-action:none}#seek{flex:1;min-width:200px}small{color:#94a3b8}#status{padding:0 24px 16px}#legend{margin-left:12px}</style>
<header><h2>DuoBox · 原始双人动作</h2><div>117 段录制数据 · 30 FPS 回放 · 双方均为数据集动作，不是模型生成或物理仿真</div></header>
<div id="controls"><label>动作 <select id="clip" aria-label="动作序列"></select></label><button id="play">暂停</button><label>速度 <select id="speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><button id="reset">重置视角</button><span id="legend"></span></div>
<canvas id="view" aria-label="双人骨架动作画面"></canvas><div id="controls"><input id="seek" aria-label="播放进度" type="range" min="0" value="0" step="1"><span id="time"></span></div><div id="status">加载中…</div><div style="padding:0 24px"><small>拖动旋转视角 · 滚轮缩放 · 拖动进度条逐帧查看 · 自动循环播放</small></div>
<script>
const $=id=>document.getElementById(id),canvas=$('view'),ctx=canvas.getContext('2d');
let data=null,playing=true,frame=0,last=0,yaw=.4,pitch=.15,zoom=1,center=[0,0,0],radius=2,loadId=0;
const colors=['#38bdf8','#fb923c'];
async function load(){const id=++loadId;playing=false;$('status').textContent='正在读取动作…';try{const r=await fetch('/motion/'+$('clip').value);if(!r.ok)throw Error('读取失败');const d=await r.json();if(id!==loadId)return;data=d;frame=0;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const a of d.actors)for(const f of a.positions)for(const p of f)for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],p[k]);hi[k]=Math.max(hi[k],p[k]);}center=lo.map((v,k)=>(v+hi[k])/2);radius=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],.1);$('seek').max=Math.min(...d.actors.map(a=>a.positions.length))-1;$('legend').replaceChildren();d.actors.forEach((a,i)=>{const s=document.createElement('span');s.style.color=colors[i%2];s.textContent=' ● '+a.name+' ';$('legend').append(s);});$('status').textContent=d.file+' · '+(Number($('seek').max)+1)+' 帧（30 FPS）';playing=true;$('play').textContent='暂停';}catch(e){$('status').textContent=String(e)}}
function project(p,w,h){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2];let xx=x*Math.cos(yaw)+z*Math.sin(yaw),zz=-x*Math.sin(yaw)+z*Math.cos(yaw);let yy=y*Math.cos(pitch)-zz*Math.sin(pitch);let scale=Math.min(w,h)*.78/radius*zoom;return [w/2+xx*scale,h/2-yy*scale];}
function render(t){const dt=last?Math.min((t-last)/1000,.1):0;last=t;let w=canvas.clientWidth,h=canvas.clientHeight,dpr=devicePixelRatio;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#111827';ctx.fillRect(0,0,w,h);if(data){const n=Number($('seek').max)+1;if(playing)frame=(frame+dt*30*Number($('speed').value))%n;let f=Math.floor(frame);$('seek').value=f;$('time').textContent=(f/30).toFixed(2)+' / '+(n/30).toFixed(2)+' s';for(let k=0;k<data.actors.length;k++){let points=data.actors[k].positions[f].map(p=>project(p,w,h));ctx.strokeStyle=colors[k%2];ctx.fillStyle=colors[k%2];ctx.lineWidth=4;ctx.lineCap='round';for(let j=1;j<points.length;j++){ctx.beginPath();ctx.moveTo(...points[data.parents[j]]);ctx.lineTo(...points[j]);ctx.stroke();}for(const p of points){ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.fill();}}}requestAnimationFrame(render);}
$('clip').onchange=load;$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放'};$('seek').oninput=()=>{frame=Number($('seek').value);playing=false;$('play').textContent='播放'};$('reset').onclick=()=>{yaw=.4;pitch=.15;zoom=1};let drag=null;canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>drag=null;canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-drag[0])*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-drag[1])*.008));drag=[e.clientX,e.clientY]}};canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(8,zoom*Math.exp(-e.deltaY*.001)))};
fetch('/catalog').then(r=>r.json()).then(rows=>{for(const row of rows){let o=document.createElement('option');o.value=row.id;o.textContent=row.file+' · '+row.seconds.toFixed(1)+'s';$('clip').append(o)}load()}).catch(e=>$('status').textContent=String(e));requestAnimationFrame(render);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/':
                body, mime = HTML.encode(), 'text/html; charset=utf-8'
            elif path == '/catalog':
                rows = []
                for i, file in enumerate(FILES):
                    d = np.load(file, allow_pickle=True).item()
                    frames = min(len(v) for k, v in d.items() if k.endswith('_position'))
                    rows.append({'id': i, 'file': file.name, 'seconds': frames / 120})
                body, mime = json.dumps(rows).encode(), 'application/json'
            elif path.startswith('/motion/') and path[8:].isdigit() and int(path[8:]) < len(FILES):
                body, mime = json.dumps(motion(int(path[8:])), separators=(',', ':')).encode(), 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(500, str(exc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=19091)
    args = parser.parse_args()
    print(f'DuoBox: {len(FILES)} clips at http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
