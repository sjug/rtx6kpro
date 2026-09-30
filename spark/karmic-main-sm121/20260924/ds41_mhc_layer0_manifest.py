"""Inputs manifest and prompt-token capture for the standalone layer-0 mHC operator comparison.

build     (local, no nodes): write ds41-mhc-layer0-inputs.json from receipts only: the frozen needle
          identity; both native configurations of the layer-0 mhc.pre 8192-row plan and both tf32
          configurations of mhc.pre.expanded 8192 (passing arm 233036Z vs release snapshot, identical on
          all ranks); their rebuilt selection keys; the retained layer-0 window capture used as the
          faithfulness gate (dusty rank 0 of 233036Z, path and sha256 from window-captures.json).
tokenize  (needs a serving DS4.1 head; run only when you choose): POST the gate's exact messages and
          template settings to /tokenize and save the 524288 token ids with their sha256. The server
          applies the same chat template and default kwargs as the original-needle requests.

Nothing here decides a numerical policy; it only fixes what the replay compares.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MANIFEST = ROOT / 'ds41-mhc-layer0-inputs.json'
PASSING = 'receipts/decision-row-matched8192-nccl-standard-upstream-capture-20260926T233036Z'
RELEASE = 'receipts/ratio1-namespace-snapshot-20260927T010531Z'
NEEDLE = ROOT.parents[1] / 'ds41/r38/receipts/20260916/admission-k7-1m-u80/needle-524288-input.json'
NEEDLE_SHA256 = '1b89c7f4d1e1047f6cf7ebe6000d6aff242981cbdbe8d8ce2a742ff61a7d6db6'
PROMPT_TOKENS = 524288
CHUNK = 8192
NODES = ('dusty', 'toby', 'rusty', 'kirby')
RELEASE_LOCK = 'ds41-precision-release.lock.json'
RELEASE_LOCK_SHA256 = '4b1afffe455d634cbadc82353a64beb3c96bbaf89b595c948c00dbfa3b1f1c32'
RELEASE_PIN_RECORD = 'ratio1-restore-20260927T015500Z-amendment.json'
IMAGE_LOCK = '/opt/ds41-precision-release/ds41-precision-release.lock.json'
SOURCE_ROOT = '/opt/jovian-judgement'
PYTHONPATH = '/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x'  # serving order (launch_contract.render)
# Files whose semantics the harness encodes, each cross-checked locally against the source it was read from.
B12X_PIN, VLLM_PIN = 'a7d7d29b', '1794dcf1'
VLLM_FROM_PIN = ('vllm/models/deepseek_v4_1/b12x_layers.py', 'vllm/models/deepseek_v4_1/attention.py',
                 'vllm/models/deepseek_v4_1/nvidia/b12x_attention.py')
# The release model differs from 1794dcf1 in engram progress only; its bytes are this kit file.
VLLM_FROM_KIT = {'vllm/models/deepseek_v4_1/nvidia/model.py': 'engram-progress-model.py'}
B12X_PREFIXES = ('b12x/norm/mhc/', 'b12x/preparation/', 'b12x/testing/mhc.py', 'tests/norm/test_mhc_lagged.py')
HELPER = 'claude_compare_selections.py'
# All from B12X a7d7d29b tests/norm/test_mhc_lagged.py; none is chosen here.
TOLERANCES = {
    # Against the reference: lagged tests (fp32 outputs, bf16 y) and the long-K test, which asserts the
    # tf32 projection keeps post within 1e-6 of fp64 at hidden 5120 (test_lagged_tf32_retains_fp32_mixing).
    'reference': {'post': {'atol': 1e-6, 'rtol': 1e-6}, 'comb': {'atol': 4e-5, 'rtol': 2e-5},
                  'pre_out': {'atol': 4e-5, 'rtol': 2e-5}, 'y': {'atol': 0.008, 'rtol': 2e-5}},
    # Between two configurations of the same operator: test_lagged_prefill_scalar_parity_graph requires
    # bf16 outputs bit-equal and fp32 outputs within 2e-6 across projection backends.
    'config_parity': {'fp32': {'atol': 2e-6, 'rtol': 2e-6}, 'bf16': {'atol': 0.0, 'rtol': 0.0}},
    'source': 'b12x a7d7d29b tests/norm/test_mhc_lagged.py',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def keys():
    from claude_compare_selections import mhc_key
    return {'pre': mhc_key('pre', CHUNK), 'pre_expanded': mhc_key('pre', CHUNK, True)}


def configs(root=ROOT):
    out, files = {}, {}
    for side, pattern in (('passing', PASSING + '/{}-selection.json'), ('release', RELEASE + '/{}.json')):
        per_rank = {}
        for node in NODES:
            raw = (root / pattern.format(node)).read_bytes()
            files[f'{side}/{node}'] = {'path': pattern.format(node), 'sha256': sha(raw)}
            records = json.loads(raw)['records']
            per_rank[node] = {name: records[key]['config'] for name, key in keys().items()}
        if any(per_rank[n] != per_rank['dusty'] for n in NODES):
            raise RuntimeError(f'{side} ranks disagree on the mHC configurations')
        out[side] = per_rank['dusty']
    for name in keys():
        if out['passing'][name] == out['release'][name]:
            raise RuntimeError(f'{name}: passing and release configurations are equal; nothing to compare')
    return out, files


def capture(root=ROOT):
    entry = json.loads((root / PASSING / 'window-captures.json').read_text())['dusty']
    if entry['problems']:
        raise RuntimeError('Retained layer-0 window capture reported problems')
    summary = json.loads((root / PASSING / 'summary.json').read_text())
    return {'receipt_remote_dir': summary['remote_dir'], 'host_file': entry['host_file'],
            'gathered_file': summary['remote_dir'] + '/captures/' + Path(entry['container_file']).name,
            'sha256': entry['sha256'], 'rank': 0, 'layer': 0,
            'positions': [PROMPT_TOKENS - 128, PROMPT_TOKENS - 1]}


def build(root=ROOT, needle=NEEDLE):
    if sha(Path(needle).read_bytes()) != NEEDLE_SHA256:
        raise RuntimeError('Frozen needle input changed')
    cfg, files = configs(root)
    manifest = {'schema': 'ds41-mhc-layer0-inputs-v1', 'needle_sha256': NEEDLE_SHA256,
                'prompt_tokens': PROMPT_TOKENS, 'chunk_rows': CHUNK, 'keys': keys(), 'configs': cfg,
                'selection_files': files, 'capture': capture(root), 'source': source_identity(root),
                'tolerances': TOLERANCES}
    return manifest


def git_bytes(repo, rev, path):
    import subprocess
    return subprocess.run(['git', '-C', str(repo), 'show', f'{rev}:{path}'], capture_output=True, check=True).stdout


def source_identity(root=ROOT, b12x_repo=Path.home() / 'git/b12x', vllm_repo=Path.home() / 'git/vllm'):
    """Release lock identity and the required file hashes, each proven equal to the source it was read from."""
    raw = (root / RELEASE_LOCK).read_bytes()
    if sha(raw) != RELEASE_LOCK_SHA256:
        raise RuntimeError('Release lock changed')
    # The release pin as last restored, not the live candidate.json (swapped for diagnostic boots).
    pin = json.loads((root / RELEASE_PIN_RECORD).read_text())['candidate']['diagnostic']
    lock = json.loads(raw)
    if pin['lock_sha256'] != RELEASE_LOCK_SHA256 or {'vllm': pin['vllm_tree'], 'b12x': pin['b12x_tree']} != lock['trees']:
        raise RuntimeError('Candidate pin and release lock disagree')
    after, required = lock['after'], {'b12x': {}, 'vllm': {}}
    for path, entry in after['b12x'].items():
        if path.startswith(B12X_PREFIXES) and path.endswith('.py'):
            if sha(git_bytes(b12x_repo, B12X_PIN, path)) != entry['sha256']:
                raise RuntimeError(f'Release B12X {path} is not the {B12X_PIN} source the harness encodes')
            required['b12x'][path] = entry['sha256']
    for path in VLLM_FROM_PIN:
        if sha(git_bytes(vllm_repo, VLLM_PIN, path)) != after['vllm'][path]['sha256']:
            raise RuntimeError(f'Release vLLM {path} is not the {VLLM_PIN} source the harness encodes')
        required['vllm'][path] = after['vllm'][path]['sha256']
    for path, kit in VLLM_FROM_KIT.items():
        if sha((root / kit).read_bytes()) != after['vllm'][path]['sha256']:
            raise RuntimeError(f'Release vLLM {path} is not {kit}')
        required['vllm'][path] = after['vllm'][path]['sha256']
    if not any(p.startswith('b12x/norm/mhc/') for p in required['b12x']):
        raise RuntimeError('No mHC sources in the release lock')
    return {'image_lock': IMAGE_LOCK, 'lock_sha256': RELEASE_LOCK_SHA256, 'trees': lock['trees'],
            'root': SOURCE_ROOT, 'pythonpath': PYTHONPATH, 'required': required,
            'helper': {'path': HELPER, 'sha256': sha((root / HELPER).read_bytes())},
            'pins': {'b12x': B12X_PIN, 'vllm': VLLM_PIN, 'vllm_kit': VLLM_FROM_KIT}}


def tokenize_body(needle=NEEDLE):
    spec = json.loads(Path(needle).read_text())
    return {'model': 'DeepSeek-V4.1-Flash', 'messages': spec['messages'],
            'chat_template_kwargs': spec['chat_template_kwargs'], 'add_generation_prompt': True}


def check_tokens(payload):
    tokens = payload.get('tokens')
    if not isinstance(tokens, list) or len(tokens) != PROMPT_TOKENS or payload.get('count') != PROMPT_TOKENS:
        raise RuntimeError('Tokenize response is not exactly the 524288-token frozen prompt')
    if not all(isinstance(t, int) and t >= 0 for t in tokens):
        raise RuntimeError('Token ids are not nonnegative integers')
    return tokens


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('build')
    t = sub.add_parser('tokenize')
    t.add_argument('--base-url', required=True)
    t.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'build':
        MANIFEST.write_text(json.dumps(build(), indent=2, sort_keys=True) + '\n')
        print('MHC-LAYER0-MANIFEST', sha(MANIFEST.read_bytes()))
        return
    import urllib.request
    request = urllib.request.Request(a.base_url + '/tokenize', json.dumps(tokenize_body()).encode(),
                                     {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
    tokens = check_tokens(payload)
    data = json.dumps(tokens).encode()
    with a.out.open('xb') as stream:
        stream.write(data)
    print('MHC-LAYER0-TOKENS', len(tokens), sha(data))


if __name__ == '__main__':
    main()
