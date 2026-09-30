"""Use the existing unchanged-archive fabric distributor and capture its output."""
import json
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
build = json.loads((root / 'receipts/engram-repair-build-receipt.json').read_text())
receipt = root / 'receipts' / Path(build['directory']).name
subprocess.run(['scp', str(root / 'distribute_determinism.py'), 'dusty:' + remote + '/'], check=True)
subprocess.run(['scp', str(receipt / 'BUILD-OK'), 'dusty:' + remote + '/engram-repair-build.ok'], check=True)
command = ['python3', '-u', remote + '/distribute_determinism.py', '--receipt', 'engram-repair-build.ok',
           '--prefix', 'engram-repair']
with (receipt / 'distribution.log').open('x') as log:
    proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        # Preserve full progress in the receipt without flooding the UI.
        if any(word in line for word in ('ARCHIVE-SAVED', 'IMAGE-VERIFIED', 'DISTRIBUTED', 'Error', 'Traceback')):
            print(line, end='', flush=True)
    code = proc.wait()
(receipt / 'distribution.exit').write_text(str(code))
if code:
    raise SystemExit(code)
for name in ['engram-repair-archive.json', *['engram-repair-' + n + '-image.json' for n in ('toby', 'rusty', 'kirby')]]:
    subprocess.run(['scp', 'dusty:' + remote + '/receipts/' + name, str(receipt / name)], check=True)
print('ENGRAM-REPAIR-DISTRIBUTION-COMPLETE', build['image_id'], flush=True)
