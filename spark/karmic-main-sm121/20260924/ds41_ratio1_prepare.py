"""Freeze the DIAGNOSTIC ratio-1 extend BF16 pin over the exact precision release image 1989e16d.

Review artifact only: no build, node or launch. One variable: the main-model ratio-1 (compress
ratio 1) extend attention plan is declared with B12X's supported complete configuration pin,
    SparseMlaConfig(max_chunks_per_row=1, split_chunk_size=1, single_pass=True,
                    v41_compute_mode="bf16", v41_heads_per_block=16),
which is the release's own tuned record with only v41_compute_mode changed. A pinned plan's
selection source is "override"; its compile jobs, scratch memory and materialized programs all
derive from the pinned config, and its selection key includes the pin, so no tuned record is
read, reused or overwritten for it. Every other plan (draft, SWA, ratio 2, all decode plans,
every GEMM/MoE/indexer family) resolves from the release selections copied byte for byte into
a new diagnostic cache namespace (seed_ratio1_namespace.py).

Source delta (AST-verified): one module-level constant, and in DeepseekV4Attention.bind_kv_cache
one assignment, one declaration-time marker print, and one keyword on the existing mla.plan call.
Not a serving candidate and not a precision policy decision.

Usage (local): python3 ds41_ratio1_prepare.py
Outputs: ds41-ratio1-attention.py, ds41-ratio1.patch, ds41-ratio1.lock.json, ds41-ratio1.ignore,
Dockerfile.ds41-ratio1.
"""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ds41_precision_release_prepare as release_prep  # noqa: E402

STEM = 'ds41-ratio1'
KIND = 'ratio1-bf16-diagnostic'
STATUS = 'diagnostic-only-not-qualified'
INSTALL_DIR = '/opt/ds41-ratio1'
INSTALL_PASS = 'RATIO1-INSTALL-PASS'
SMOKE_PASS = 'RATIO1-SMOKE-PASS'
MARKER = 'DS41-RATIO1-EXTEND-PIN'
TARGET = 'vllm/models/deepseek_v4_1/attention.py'
OUTPUT = 'ds41-ratio1-attention.py'
RELEASE_LOCK = 'ds41-precision-release.lock.json'
RELEASE_RECEIPT = 'receipts/precision-release-build-receipt.json'
PIN = {'max_chunks_per_row': 1, 'split_chunk_size': 1, 'single_pass': True,
       'v41_compute_mode': 'bf16', 'v41_heads_per_block': 16}
KV_BLOCKS = 80022
PACKAGED = ('ds41_ratio1_prepare.py', 'ds41_ratio1_install.py', 'ds41_ratio1_smoke.py', 'build_ratio1.py',
            'claude_decode_sparse_mla.py', 'test_ds41_ratio1.py', 'Dockerfile.' + STEM, OUTPUT)

CONSTANT_ANCHOR = 'from vllm.v1.worker.workspace import ('
CONSTANT = '''# DIAGNOSTIC ONLY (ds41-ratio1): pin the main-model ratio-1 extend attention plan to BF16
# compute; every other plan keeps its tuned selection. Not a serving default.
_DS41_RATIO1_EXTEND_PIN = mla.SparseMlaConfig(
    max_chunks_per_row=1, split_chunk_size=1, single_pass=True,
    v41_compute_mode="bf16", v41_heads_per_block=16,
)
'''
DECLARE_ANCHOR = '''            declaration = mla.plan(
                caps,
                invocation=mla.invocation_from_descriptors('''
DECLARE_REPLACEMENT = '''            ds41_pin = (
                _DS41_RATIO1_EXTEND_PIN
                if mode == "extend" and self.compress_ratio == 1 and not self.is_draft
                else None
            )
            if ds41_pin is not None:
                print(f"DS41-RATIO1-EXTEND-PIN layer={self.prefix} mode={mode} rows={capacity} "
                      f"v41_compute_mode={ds41_pin.v41_compute_mode}", flush=True)
            declaration = mla.plan(
                caps,
                override=ds41_pin,
                invocation=mla.invocation_from_descriptors('''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _module_insert_point(text):
    """The constant goes after the complete import block that starts at CONSTANT_ANCHOR."""
    start = text.index(CONSTANT_ANCHOR)
    end = text.index(')\n', start) + 2
    return end


def compose(base):
    if base.count(CONSTANT_ANCHOR) != 1 or base.count(DECLARE_ANCHOR) != 1:
        raise ValueError('Ratio-1 pin seam is ambiguous or missing')
    point = _module_insert_point(base)
    text = base[:point] + '\n' + CONSTANT + base[point:]
    text = text.replace(DECLARE_ANCHOR, DECLARE_REPLACEMENT)
    verify_delta(base, text)
    compile(text, TARGET, 'exec')
    return text


def verify_delta(base, composed):
    """Removing exactly the added constant, assignment, marker and keyword recovers the base AST."""
    before, after = ast.parse(base), ast.parse(composed)
    added_constant = [n for n in after.body if isinstance(n, ast.Assign)
                      and any(getattr(t, 'id', None) == '_DS41_RATIO1_EXTEND_PIN' for t in n.targets)]
    if len(added_constant) != 1:
        raise ValueError('Expected exactly one pin constant')
    call = added_constant[0].value
    if not (isinstance(call, ast.Call) and ast.unparse(call.func) == 'mla.SparseMlaConfig'
            and not call.args and {k.arg: ast.literal_eval(k.value) for k in call.keywords} == PIN):
        raise ValueError('Pin constant differs from the declared diagnostic config')

    class Strip(ast.NodeTransformer):
        removed = {'constant': 0, 'assign': 0, 'marker': 0, 'keyword': 0}

        def visit_Module(self, node):
            node.body = [n for n in node.body if n is not added_constant[0]]
            self.removed['constant'] += 1
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if [getattr(t, 'id', None) for t in node.targets] == ['ds41_pin']:
                expected = ('_DS41_RATIO1_EXTEND_PIN if mode == \'extend\' and self.compress_ratio == 1 '
                            'and (not self.is_draft) else None')
                if ast.unparse(node.value) != expected:
                    raise ValueError('Pin selection condition differs')
                self.removed['assign'] += 1
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == 'ds41_pin is not None':
                if (len(node.body) != 1 or node.orelse
                        or not ast.unparse(node.body[0]).startswith("print(f'DS41-RATIO1-EXTEND-PIN ")):
                    raise ValueError('Marker block differs')
                self.removed['marker'] += 1
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            if ast.unparse(node.func) == 'mla.plan':
                kept = [k for k in node.keywords if not (k.arg == 'override' and ast.unparse(k.value) == 'ds41_pin')]
                self.removed['keyword'] += len(node.keywords) - len(kept)
                node.keywords = kept
            return self.generic_visit(node)

    stripper = Strip()
    cleaned = stripper.visit(after)
    if stripper.removed != {'constant': 1, 'assign': 1, 'marker': 1, 'keyword': 1}:
        raise ValueError(f'Unexpected pin structure: {stripper.removed}')
    if ast.dump(cleaned) != ast.dump(before):
        raise ValueError('Changes extend beyond the ratio-1 extend pin')


def release_identity():
    raw = (ROOT / RELEASE_LOCK).read_bytes()
    lock = json.loads(raw)
    receipt = json.loads((ROOT / RELEASE_RECEIPT).read_text())
    build = ROOT / 'receipts' / Path(receipt['directory']).name
    if receipt['lock_sha256'] != sha(raw) or (build / 'BUILD-OK').read_text().strip() != receipt['image_id']:
        raise RuntimeError('Release build receipt, lock and gate result disagree')
    labels = json.loads((build / 'image-inspect.json').read_text())[0]['Config']['Labels']
    if (labels.get('local-inference.ds41.diagnostic.kind') != release_prep.KIND
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != sha(raw)):
        raise RuntimeError('Release image labels differ from its lock')
    return lock, sha(raw), receipt['image_id']


def release_attention(lock):
    base = release_prep.router_vllm_files()[TARGET][1]
    if sha(base) != lock['after']['vllm'][TARGET]['sha256']:
        raise RuntimeError('Release attention source differs from its lock')
    return base.decode()


DOCKERFILE = """# DIAGNOSTIC ONLY, not a serving candidate: precision release {short} with the main-model
# ratio-1 extend attention plan pinned to BF16 compute. Generated by ds41_ratio1_prepare.py.
FROM {base}
ARG RATIO1_LOCK
ARG RATIO1_CACHE
COPY {copy} {install}/
RUN /opt/venv/bin/python {install}/ds41_ratio1_install.py
ENV XDG_CACHE_HOME=/cache/jit/${{RATIO1_CACHE}} \\
    VLLM_CACHE_ROOT=/cache/jit/${{RATIO1_CACHE}}/vllm \\
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${{RATIO1_CACHE}}/inductor \\
    TRITON_CACHE_DIR=/cache/jit/${{RATIO1_CACHE}}/triton \\
    B12X_COMPILE_CACHE_DIR=/cache/jit/${{RATIO1_CACHE}}/b12x \\
    B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/${{RATIO1_CACHE}}/b12x/cute \\
    SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/${{RATIO1_CACHE}}/b12x/compile \\
    CUTE_DSL_CACHE_DIR=/cache/jit/${{RATIO1_CACHE}}/cute-dsl
LABEL local-inference.ds41.diagnostic.kind="{kind}" \\
    local-inference.ds41.diagnostic.lock.sha256="${{RATIO1_LOCK}}" \\
    local-inference.ds41.diagnostic.base-image="{base}" \\
    local-inference.ds41.diagnostic.base-kind="{base_kind}" \\
    local-inference.ds41.diagnostic.base-lock.sha256="{base_lock}" \\
    local-inference.cache.fingerprint="${{RATIO1_CACHE}}" \\
    vllm.source-tree="{vllm}" \\
    b12x.source-tree="{b12x}" \\
    local-inference.status="{status}" \\
    org.opencontainers.image.title="DS4.1 precision release with ratio-1 extend BF16 pin (diagnostic)"
"""


def main():
    from contracts import git_tree, manifest
    lock_release, release_sha, image = release_identity()
    base = release_attention(lock_release)
    composed = compose(base)
    (ROOT / OUTPUT).write_text(composed)
    (ROOT / (STEM + '.patch')).write_text(''.join(difflib.unified_diff(
        base.splitlines(True), composed.splitlines(True), 'a/' + TARGET, 'b/' + TARGET)))
    before_files = release_prep.router_vllm_files()
    _, after_release = release_prep.release(before_files)
    after = dict(after_release)
    after[TARGET] = (after[TARGET][0], composed.encode())
    if manifest(after_release) != lock_release['after']['vllm']:
        raise RuntimeError('Release manifest reconstruction differs')
    trees = {'vllm': git_tree(after), 'b12x': lock_release['trees']['b12x']}
    copy = ' '.join(sorted(PACKAGED) + [STEM + '.lock.json'])
    (ROOT / ('Dockerfile.' + STEM)).write_text(DOCKERFILE.format(
        short=image[:12], base=image, copy=copy, install=INSTALL_DIR, kind=KIND, base_kind=release_prep.KIND,
        base_lock=release_sha, vllm=trees['vllm'], b12x=trees['b12x'], status=STATUS))
    inputs = sorted(PACKAGED)
    lock = {'schema': 'ds41-ratio1-v1', 'kind': KIND, 'status': STATUS,
            'base_image_id': image, 'base_kind': release_prep.KIND, 'base_lock': RELEASE_LOCK,
            'base_lock_sha256': release_sha, 'base_install_dir': release_prep.INSTALL_DIR,
            'base_trees': lock_release['trees'], 'base_cache_fingerprint': lock_release['cache_fingerprint'],
            'before': lock_release['after'], 'after': {'vllm': manifest(after), 'b12x': lock_release['after']['b12x']},
            'trees': trees,
            'targets': {TARGET: {'source': OUTPUT, 'input_sha256': sha(base.encode()),
                                 'output_sha256': sha(composed.encode())}},
            'pin': PIN, 'pin_role': 'ratio1.extend', 'kv_blocks': KV_BLOCKS, 'marker': MARKER,
            'install_pass': INSTALL_PASS, 'smoke_pass': SMOKE_PASS,
            'inputs': {name: sha((ROOT / name).read_bytes()) for name in inputs}}
    lock['cache_fingerprint'] = 'ds41-ratio1-diag-' + sha(json.dumps(trees, sort_keys=True).encode())[:20]
    raw = json.dumps(lock, sort_keys=True, indent=2) + '\n'
    (ROOT / (STEM + '.lock.json')).write_text(raw)
    (ROOT / (STEM + '.ignore')).write_text('**\n' + ''.join('!' + n + '\n' for n in inputs + [STEM + '.lock.json']))
    print('RATIO1-PREPARED', sha(raw.encode()), json.dumps(trees, sort_keys=True), lock['cache_fingerprint'], flush=True)


if __name__ == '__main__':
    main()
