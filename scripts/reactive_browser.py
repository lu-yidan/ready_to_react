"""Local raw DuoBox selection and checkpoint-backed reaction generation UI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from browse_duobox import Handler as RawHandler, FILES, ROOT

CACHE = ROOT / 'data/reactive_browser'
POOL = ThreadPoolExecutor(max_workers=1)
LOCK = threading.Lock()
JOBS = {}


def generate(key, request):
    directory = CACHE / key
    try:
        directory.mkdir(parents=True)
        (directory / 'request.json').write_text(json.dumps(request))
        env = dict(os.environ, OMP_NUM_THREADS='4', WANDB_MODE='disabled')
        with (directory / 'log.txt').open('w') as log:
            subprocess.run([sys.executable, str(ROOT / 'scripts/reactive_browser_worker.py'),
                            '--request', str(directory / 'request.json')], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        with LOCK:
            JOBS[key] = 'ready'
    except Exception:
        with LOCK:
            JOBS[key] = 'error'


class Handler(RawHandler):
    def send(self, value, status=200, mime='application/json'):
        body = value if isinstance(value, bytes) else json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self.send((ROOT / 'scripts/reactive_browser.html').read_bytes(),
                             mime='text/html; charset=utf-8')
        if path.startswith('/job/'):
            parts = path.split('/')
            key = parts[2]
            if len(key) != 32 or any(c not in '0123456789abcdef' for c in key):
                return self.send({'error': 'Invalid job'}, 400)
            directory = CACHE / key
            if len(parts) == 3:
                with LOCK:
                    status = JOBS.get(key, 'ready' if (directory / 'result.json').exists() else 'missing')
                return self.send({'status': status, 'id': key})
            if parts[3] in ('result', 'download'):
                file = directory / ('result.json' if parts[3] == 'result' else 'motion.npz')
                if file.exists():
                    return self.send(file.read_bytes(), mime='application/json' if parts[3] == 'result' else 'application/octet-stream')
            return self.send({'error': 'Not ready'}, 404)
        return super().do_GET()

    def do_POST(self):
        if self.path != '/generate':
            return self.send({'error': 'Not found'}, 404)
        if self.headers.get('Origin') not in (None, 'http://' + self.headers.get('Host', '')):
            return self.send({'error': 'Origin rejected'}, 403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 2048:
                raise ValueError()
            request = json.loads(self.rfile.read(size))
            fields = ('clip', 'fixed', 'start', 'frames', 'seed')
            if set(request) != set(fields) or any(type(request[k]) is not int for k in fields):
                raise ValueError()
            if not (0 <= request['clip'] < len(FILES) and request['fixed'] in (0, 1)
                    and 0 <= request['start'] < 2000 and 8 <= request['frames'] <= 1920
                    and 0 <= request['seed'] < 2**31):
                raise ValueError()
            with LOCK:
                if 'running' in JOBS.values():
                    return self.send({'error': '已有生成任务，请等它完成后再提交。'}, 409)
                key = uuid.uuid4().hex
                JOBS[key] = 'running'
                POOL.submit(generate, key, request)
            return self.send({'id': key})
        except (ValueError, KeyError, TypeError):
            return self.send({'error': '参数无效；片段至少需要 8 帧。'}, 400)


if __name__ == '__main__':
    CACHE.mkdir(parents=True, exist_ok=True)
    print('Reactive browser http://127.0.0.1:19094/', flush=True)
    ThreadingHTTPServer(('127.0.0.1', 19094), Handler).serve_forever()
