"""Transfer the diagnostic unchanged over the switched fabric, using the existing distributor."""
import json
import argparse
from pathlib import Path
import shlex
import subprocess

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
modes = parser.add_mutually_exclusive_group()
modes.add_argument('--moe-seams', action='store_true')
modes.add_argument('--moe-capture', action='store_true')
modes.add_argument('--router-fence', action='store_true')
args = parser.parse_args()
prefix = 'moe-seams' if args.moe_seams else 'activation-trace'
if args.moe_capture:
    prefix = 'moe-capture'
if args.router_fence:
    prefix = 'router-release'
remote = '/home/jugs/git/ds41-r38/karmic-main-20260924'
build = json.loads((root / ('receipts/' + prefix + '-build-receipt.json')).read_text())
receipt = root / 'receipts' / Path(build['directory']).name
subprocess.run(['scp', str(root / 'distribute_determinism.py'), 'dusty:' + remote + '/'], check=True)
subprocess.run(['scp', str(receipt / 'BUILD-OK'), 'dusty:' + remote + '/' + prefix + '-build.ok'], check=True)
command = ['python3', '-u', remote + '/distribute_determinism.py',
           '--receipt', prefix + '-build.ok', '--prefix', prefix,
           '--tag', 'localhost/voipmonitor/build-components:ds41-' + prefix]
with (receipt / 'distribution.log').open('x') as log:
    proc = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'dusty', shlex.join(command)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        log.write(line)
        log.flush()
        if any(word in line for word in ('ARCHIVE-SAVED', 'IMAGE-VERIFIED', 'DISTRIBUTED', 'Error', 'Traceback')):
            print(line, end='', flush=True)
    code = proc.wait()
(receipt / 'distribution.exit').write_text(str(code))
if code:
    raise SystemExit(code)
for name in [prefix + '-archive.json', *[prefix + '-' + n + '-image.json' for n in ('toby', 'rusty', 'kirby')]]:
    subprocess.run(['scp', 'dusty:' + remote + '/receipts/' + name, str(receipt / name)], check=True)
print(prefix.upper() + '-DISTRIBUTION-COMPLETE', build['image_id'], flush=True)
