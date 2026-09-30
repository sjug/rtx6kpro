"""Run the production epoch kernel gate on an idle build GPU with receipts."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    root = Path(__file__).resolve().parent
    remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
    image = 'e7b273407022ae74726d6c7b7465aa567b2c5a0594d5e0b46002a742a11f5146'
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / 'receipts' / ('engram-epoch-gate-' + stamp)
    out.mkdir(exist_ok=False)

    def ssh(command):
        return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', 'dusty', command], text=True)

    if ssh('podman ps -q').strip() or ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader').strip():
        raise RuntimeError('dusty must be idle')
    inspection = json.loads(ssh('podman image inspect ' + image))
    if inspection[0]['Id'].removeprefix('sha256:') != image:
        raise RuntimeError('Wrong image')
    (out / 'image.json').write_text(json.dumps(inspection, indent=2))
    files = ['test_engram_epoch_gpu.py', 'engram-progress-engram.py']
    (out / 'inputs.json').write_text(json.dumps({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                               for name in files}, indent=2))
    for name in files:
        subprocess.run(['scp', str(root / name), 'dusty:' + remote + '/' + name], check=True)
    command = ['podman', 'run', '--rm', '--pull=never', '--network=none',
               '--device', 'nvidia.com/gpu=all', '-e', 'PYTHONUNBUFFERED=1',
               '-v', remote + ':/gate:ro', '--entrypoint', '/opt/venv/bin/python',
               image, '/gate/test_engram_epoch_gpu.py']
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    with (out / 'gpu.log').open('w') as log:
        proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    (out / 'exit').write_text(str(rc))
    print('EPOCH-GATE-RECEIPTS', out, flush=True)
    if rc:
        raise SystemExit(rc)
    if 'ENGRAM-EPOCH-GPU-PASS 8' not in (out / 'gpu.log').read_text():
        raise RuntimeError('Missing completion marker')


if __name__ == '__main__':
    main()
