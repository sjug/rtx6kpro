"""Enable reviewed timing for one startup matrix and retain logs on all ranks."""
import datetime
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--cuda-module-loading', choices=('LAZY', 'EAGER'))
args = parser.parse_args()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'receipts' / ('engram-host-timing-' + stamp)
out.mkdir(exist_ok=False)
nodes = ('dusty', 'toby', 'rusty', 'kirby')
cache = '/home/jugs/.cache/vllm-jj-ds41-tp4'
flag = cache + '/ds41-engram-timing.enable'
enabled = []
build = json.loads((root / 'receipts/engram-timing-build-receipt.json').read_text())
info = json.loads((root / 'receipts' / Path(build['directory']).name / 'image-inspect.json').read_text())[0]
cache_paths = {}
for item in info['Config']['Env']:
    key, _, value = item.partition('=')
    if key in {'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'B12X_COMPILE_CACHE_DIR',
               'TRITON_CACHE_DIR', 'XDG_CACHE_HOME'} and value.startswith('/cache/'):
        cache_paths[key] = cache + value[len('/cache'):]
try:
    for node in nodes:
        subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
            f'test -d {cache} && test ! -e {flag} && test ! -e {cache}/ds41-engram-timing'], check=True)
        collision = subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
            'podman container exists ds41-flash-karmic-main-tp4'])
        if collision.returncode != 1:
            raise RuntimeError(node + ': container-name check failed or name occupied')
        script = ('import json,os; paths=json.loads(' + repr(json.dumps(cache_paths)) + '); '
                  'print(json.dumps({k:{"path":v,"exists":os.path.exists(v)} for k,v in paths.items()}))')
        state = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node,
            shlex.join(['python3', '-c', script])], text=True)
        (out / f'{node}-cache-before.json').write_text(state)
        subprocess.run(['scp', str(root / 'engram-timing.enable'), node + ':' + flag], check=True)
        enabled.append(node)
    command = [sys.executable, '-u', str(root / 'start_moe_repaired.py'),
               '--arm', 'engram-timing', '--lengths', '128,385,16384']
    if args.cuda_module_loading:
        command += ['--cuda-module-loading', args.cuda_module_loading]
    (out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (out / 'run.log').open('x') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    (out / 'exit-code').write_text(str(code) + '\n')
finally:
    errors = []
    for node in enabled:
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
            f'mv {flag} {flag}-{stamp}.disabled'], text=True, capture_output=True)
        (out / f'{node}-disable.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            errors.append(node + ': disable failed')
        result = subprocess.run(['scp', '-r', node + ':' + cache + '/ds41-engram-timing',
                                 str(out / node)], text=True, capture_output=True)
        (out / f'{node}-collect.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            errors.append(node + ': log collection failed')
    if errors:
        raise RuntimeError('; '.join(errors))
print('ENGRAM-HOST-TIMING-CAPTURED', out, 'probe_exit', code, flush=True)
sys.exit(code)
