"""Run a local qualification command with owned remote telemetry and final logs."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A qualification command is required')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    since = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    remote = '/home/jugs/git/ds41-r38/karmic-main-20260924/observe.sh'
    nodes = ('dusty', 'toby', 'rusty', 'kirby')
    observers = []
    try:
        for node in nodes:
            stream = (args.out / f'{node}-telemetry.log').open('x')
            process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', node,
                f'echo OBSERVER_PID=$$; exec bash {remote}'], stdout=stream, stderr=subprocess.STDOUT)
            observers.append((node, process, stream))
        with (args.out / 'run.log').open('x') as stream:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end='', flush=True)
            code = process.wait()
        for node, observer, _ in observers:
            if observer.poll() is not None:
                raise RuntimeError('Telemetry exited before qualification completed: ' + node)
        (args.out / 'exit-code').write_text(str(code) + '\n')
        return code
    finally:
        for node, observer, stream in observers:
            stream.flush()
            lines = (args.out / f'{node}-telemetry.log').read_text().splitlines()
            first = lines[0] if lines else ''
            pid = first.removeprefix('OBSERVER_PID=')
            if first.startswith('OBSERVER_PID=') and pid.isdigit():
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node,
                    f'if ps -p {pid} -o args= | grep -Fq "{remote}"; then kill -TERM {pid}; fi'],
                    text=True, capture_output=True, timeout=30)
                (args.out / f'{node}-observer-cleanup.log').write_text(result.stdout + result.stderr)
            try:
                observer.wait(timeout=30)
            except subprocess.TimeoutExpired:
                observer.terminate()
                observer.wait(timeout=30)
            stream.close()
        for node in nodes:
            for label, cmd in (
                ('container', 'podman logs ds41-flash-karmic-main-tp4'),
                ('kernel', f'journalctl -k --since "{since}" --no-pager'),
            ):
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, cmd],
                    text=True, capture_output=True, timeout=60)
                (args.out / f'{node}-{label}.log').write_text(result.stdout + result.stderr)
                (args.out / f'{node}-{label}.exit-code').write_text(str(result.returncode) + '\n')


if __name__ == '__main__':
    sys.exit(main())
