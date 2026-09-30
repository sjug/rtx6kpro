"""Identity rules for the precision release candidate (host side; no node contact here).

The candidate is the router stage-release parent plus the worker-wide BF16 reduced-precision
disable (ds41-precision-release.lock.json). It is selected only over the exact router parent
pin and restored only to that exact pin. Serving it uses the normal profile: no fixed KV,
chunking, or NCCL diagnostic overrides; deterministic MoE on, dense split-K turbo off; and one
precision marker per worker rank, checked with the capture driver's reviewed rule.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KIND = 'precision-release-candidate'
PARENT_KIND = 'router-stage-release-candidate'
STEM = 'ds41-precision-release'
RECEIPT = 'receipts/precision-release-build-receipt.json'
NODES = ('dusty', 'toby', 'rusty', 'kirby')
# Every diagnostic control any earlier arm could leave in a container environment.
DIAGNOSTIC_ENV = ('DS41_DECISION_ROW_BLOCKS', 'DS41_PREFILL_THRESHOLD', 'DS41_CHUNKING_BLOCKS',
                  'DS41_NCCL_GEOMETRY', 'DS41_NCCL_ARM', 'DS41_ROUTER_CONTROL_BLOCKS',
                  'NCCL_ALGO', 'NCCL_PROTO', 'NCCL_MIN_NCHANNELS', 'NCCL_MAX_NCHANNELS',
                  'VLLM_DS41_L2_PREFETCH')
# Both the pinned router parent and gated release inherit LAZY in image ENV.
# Reject EAGER or a missing value, rather than rejecting the normal image default.
REQUIRED_ENV = {'B12X_DYNAMIC_DETERMINISTIC_OUTPUT': '1', 'B12X_DENSE_SPLITK_TURBO': '0',
                'CUDA_MODULE_LOADING': 'LAZY'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_lock(root=ROOT):
    raw = (root / (STEM + '.lock.json')).read_bytes()
    return json.loads(raw), sha(raw)


def router_pin_matches(pin, root=ROOT):
    """The exact router parent pin: its kind, lock, trees and gated build image."""
    raw = (root / 'router-release.lock.json').read_bytes()
    router = json.loads(raw)
    build = json.loads((root / 'receipts/router-release-build-receipt.json').read_text())
    diagnostic = pin.get('diagnostic', {})
    return (pin.get('image_id') == build['image_id'] and build['lock_sha256'] == sha(raw)
            and diagnostic.get('kind') == PARENT_KIND and diagnostic.get('lock_sha256') == sha(raw)
            and diagnostic.get('vllm_tree') == router['trees']['vllm']
            and diagnostic.get('b12x_tree') == router['trees']['b12x'] and 'diagnostic_overlay' not in pin)


def validate_build(pin, root=ROOT):
    """The pinned image is the gated release build of the current lock."""
    lock, digest = load_lock(root)
    build = json.loads((root / RECEIPT).read_text())
    directory = root / 'receipts' / Path(build['directory']).name
    if (build['lock_sha256'] != digest or pin.get('image_id') != build['image_id']
            or (directory / 'BUILD-OK').read_text().strip() != build['image_id']):
        raise RuntimeError('Release pin differs from the gated build of the current lock')
    diagnostic = pin.get('diagnostic', {})
    if (diagnostic.get('kind') != KIND or diagnostic.get('lock_sha256') != digest
            or diagnostic.get('vllm_tree') != lock['trees']['vllm']
            or diagnostic.get('b12x_tree') != lock['trees']['b12x'] or 'diagnostic_overlay' in pin):
        raise RuntimeError('Release pin identity differs from the lock')
    return lock, digest, build


def candidate(parent, root=ROOT):
    """Release pin derived from the exact router parent pin; every other field unchanged."""
    if not router_pin_matches(parent, root):
        raise RuntimeError('Release selection must start from the exact router parent pin')
    lock, digest = load_lock(root)
    build = json.loads((root / RECEIPT).read_text())
    if lock['base_image_id'] != parent['image_id'] or build['lock_sha256'] != digest:
        raise RuntimeError('Release lock or build is not based on the current router parent')
    result = dict(parent, image_id=build['image_id'])
    result['diagnostic'] = {'kind': KIND, 'lock_sha256': digest,
                            'vllm_tree': lock['trees']['vllm'], 'b12x_tree': lock['trees']['b12x']}
    validate_build(result, root)
    return result


def restored(amendment, current, manifest, root=ROOT):
    """The exact router parent recorded when the release was selected, and nothing else."""
    if current != amendment['candidate'] or manifest != amendment['manifest']:
        raise RuntimeError('Current runtime differs from the release amendment')
    if current.get('diagnostic', {}).get('kind') != KIND:
        raise RuntimeError('Current pin is not the release candidate')
    parent, prior = amendment['prior_candidate'], amendment['prior_manifest']
    if not router_pin_matches(parent, root):
        raise RuntimeError('Restoration target is not the exact router parent')
    if sha((json.dumps(parent, sort_keys=True, indent=2) + '\n').encode()) != prior['candidate.json']:
        raise RuntimeError('Prior candidate digest differs')
    if {k: v for k, v in prior.items() if k != 'candidate.json'} != {
            k: v for k, v in manifest.items() if k != 'candidate.json'}:
        raise RuntimeError('Restoration would change other runtime files')
    skip = {'image_id', 'diagnostic'}
    if {k: v for k, v in parent.items() if k not in skip} != {k: v for k, v in current.items() if k not in skip}:
        raise RuntimeError('Restoration would change more than image and source identity')
    return parent, prior


def expected_labels(pin, lock):
    return {'local-inference.ds41.diagnostic.kind': KIND,
            'local-inference.ds41.diagnostic.lock.sha256': pin['diagnostic']['lock_sha256'],
            'local-inference.ds41.release.base-image': lock['base_image_id'],
            'local-inference.ds41.release.base-lock.sha256': lock['base_lock_sha256'],
            'local-inference.ds41.release.precision-lock.sha256': lock['precision_lock_sha256'],
            'local-inference.cache.fingerprint': lock['cache_fingerprint'],
            'local-inference.status': lock['status'],
            'vllm.source-tree': pin['diagnostic']['vllm_tree'], 'b12x.source-tree': pin['diagnostic']['b12x_tree']}


def environment_problems(env):
    """Normal serving profile: no diagnostic control present, the two repaired settings exact."""
    problems = [f'diagnostic control present: {k}' for k in DIAGNOSTIC_ENV if k in env]
    problems += [f'{k} is not {v}' for k, v in REQUIRED_ENV.items() if env.get(k) != v]
    return problems


def marker_problems(log, node):
    from claude_run_decision_capture import check_precision_marker
    return check_precision_marker(log, NODES.index(node))
