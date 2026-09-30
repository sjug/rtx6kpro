"""Prepare, but do not install, the explicit fixed-capacity router diagnostic.

79000 blocks is below the smallest historical profiled capacity (79435).
The normal launcher stays unchanged unless the diagnostic flag is supplied.
This is a comparison profile, not a proposed serving limit.
"""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTROL = 'DS41_ROUTER_CONTROL_BLOCKS'
BLOCKS = '79000'


def transform_contract(source):
    anchor = "    rank = NODES[node][0]\n"
    replacement = '''    # Diagnostic only: identical cache geometry for the router-fence A/B.
    # Explicitly retained in container environment and the rendered command.
    control_blocks = os.environ.get('DS41_ROUTER_CONTROL_BLOCKS')
    if control_blocks is not None:
        if control_blocks != '79000':
            raise ValueError('DS41_ROUTER_CONTROL_BLOCKS must be exactly 79000')
        env['DS41_ROUTER_CONTROL_BLOCKS'] = control_blocks
        model += ['--num-gpu-blocks-override', control_blocks]
    rank = NODES[node][0]
'''
    if source.count(anchor) != 1 or CONTROL in source:
        raise RuntimeError('Unexpected launch contract input')
    return source.replace(anchor, replacement)


def transform_driver(source):
    replacements = [
        ("out = ROOT / 'receipts' / f'{args.arm}-{stamp}'\n",
         "profile_label = args.arm + ('-fixed79000' if args.fixed_router_blocks else '')\nout = ROOT / 'receipts' / f'{profile_label}-{stamp}'\n"),
        ("args = parser.parse_args()\n", "parser.add_argument('--fixed-router-blocks', action='store_true', help='Diagnostic only: freeze KV geometry at 79000 blocks in both router arms')\nargs = parser.parse_args()\nif args.fixed_router_blocks and args.arm not in ('engram-repair', 'router-fence'):\n    parser.error('Fixed router blocks require the clean parent or router candidate')\n"),
        ("    if args.cuda_module_loading:\n", "    if args.fixed_router_blocks:\n        overrides += ' DS41_ROUTER_CONTROL_BLOCKS=79000'\n    if args.cuda_module_loading:\n"),
        ("    return f'cd {REMOTE} && {clean} {overrides} python3 run_node.py --node {node}'\n",
         "    clean += ' -u DS41_ROUTER_CONTROL_BLOCKS'\n    return f'cd {REMOTE} && {clean} {overrides} python3 run_node.py --node {node}'\n"),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError('Unexpected diagnostic driver anchor')
        source = source.replace(old, new)
    return source


def main():
    manifest = json.loads((ROOT / 'runtime-files.json').read_text())
    result = {'scope': 'prepared only, not installed; matched diagnostic, not serving default',
              'blocks': int(BLOCKS), 'files': {}}
    for name, transform in (('launch_contract.py', transform_contract), ('start_moe_repaired.py', transform_driver)):
        raw = (ROOT / name).read_bytes()
        before = hashlib.sha256(raw).hexdigest()
        if name in manifest and before != manifest[name]:
            raise RuntimeError('Current runtime manifest differs: ' + name)
        output = transform(raw.decode())
        compile(output, name, 'exec')
        patch = ''.join(difflib.unified_diff(raw.decode().splitlines(True), output.splitlines(True),
                                           fromfile='a/' + name, tofile='b/' + name))
        patch_name = 'router-fixed-' + name + '.patch'
        (ROOT / patch_name).write_text(patch)
        result['files'][name] = {'input_sha256': before,
            'output_sha256': hashlib.sha256(output.encode()).hexdigest(),
            'patch': patch_name, 'patch_sha256': hashlib.sha256(patch.encode()).hexdigest()}
    (ROOT / 'router-fixed-control.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ROUTER-FIXED-CONTROL-PREPARED-NOT-INSTALLED', flush=True)


if __name__ == '__main__':
    main()
