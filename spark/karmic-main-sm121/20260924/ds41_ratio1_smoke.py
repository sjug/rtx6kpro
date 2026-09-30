"""Container smoke for the ratio-1 diagnostic image (run from /gate by the builder; no model load).

Gates: the imported attention module is the locked file; its pin constant equals the lock's pin;
the pin validates against the exact ratio-1 extend query this boot declares at the pinned KV
capacity (rebuilt by claude_decode_sparse_mla, which matched live cache keys); the worker
precision hook and the release lock are untouched.
"""
import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path('/opt/jovian-judgement')
lock = json.loads(Path('/opt/ds41-ratio1/ds41-ratio1.lock.json').read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


(target, entry), = lock['targets'].items()
attention = importlib.import_module(target.removesuffix('.py').replace('/', '.'))
if (Path(attention.__file__).resolve() != (ROOT / 'vllm' / target).resolve()
        or digest(attention.__file__) != entry['output_sha256']):
    raise RuntimeError('Wrong imported attention module')
pin = attention._DS41_RATIO1_EXTEND_PIN
if dataclasses.asdict(pin) != lock['pin']:
    raise RuntimeError('Pin constant differs from the lock')
import sys
sys.path.insert(0, '/opt/ds41-ratio1')
from claude_decode_sparse_mla import query_and_invocation  # noqa: E402
from b12x.attention.compressed_sparse_mla._tuning import TUNING, SparseMlaQuery  # noqa: E402

query_payload, _ = query_and_invocation('ratio1', 'extend', lock['kv_blocks'])
query = SparseMlaQuery(**{k: tuple(v) if isinstance(v, list) else v for k, v in query_payload.items()})
TUNING.validate_query(query, None)
TUNING.validate_config(query, pin, None)
default = TUNING.default_config(query, None)
if {k: v for k, v in dataclasses.asdict(default).items() if k != 'v41_compute_mode'} != {
        k: v for k, v in lock['pin'].items() if k != 'v41_compute_mode'}:
    raise RuntimeError('Pin differs from the default ratio-1 extend config beyond compute mode')
if digest('/opt/ds41-precision-release/' + lock['base_lock']) != lock['base_lock_sha256']:
    raise RuntimeError('Release lock changed')
print('RATIO1-PIN', json.dumps(lock['pin'], sort_keys=True), 'kv_blocks', lock['kv_blocks'], flush=True)
print(lock['smoke_pass'], flush=True)
