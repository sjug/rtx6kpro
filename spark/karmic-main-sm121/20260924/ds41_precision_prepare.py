"""Compose the DIAGNOSTIC worker-level BF16 reduced-precision-off derivative of the EXACT indexer image 25c92dde.

Review artifact only: no build, node, GPU or launch. Derives from the indexer-capture
image (kind indexer-capture, lock 20ea0e6e). The preimage is that image's unchanged
vllm/v1/worker/gpu_worker.py (router candidate bytes, sha e41fa1f4, reconstructed from
the router lock and kept beside this script as ds41-precision-gpu_worker-base.py). The
change is one import plus one call at the top of Worker.init_device, plus the new helper
vllm/v1/worker/ds41_precision.py, which sets
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False once per GPU
worker process before distributed init, model load, B12X preparation and graph capture.

Attention, the three capture helpers and B12X bytes are unchanged and stay installed, so the
same triggers, receipts and comparators apply. Inherited capture provenance is metadata only:
the installer preserves the indexer-era stubs and the indexer lock under explicit names and
writes stubs at all three helper read paths naming this image's trees (ds41_precision_install.py).

Source AST gate: removing exactly the new import and the call statement from the composed
worker source must recover the base AST node for node, and the call must be the first
statement of init_device, so no other worker behaviour can change.

Usage (local, no nodes): python3 ds41_precision_prepare.py
Outputs: ds41-precision-gpu_worker.py, ds41-precision.patch, ds41-precision.lock.json,
ds41-precision.ignore, Dockerfile.ds41-precision. Builder ARGs DECISION_LOCK and DECISION_CACHE
(generic builder, kind precision), install dir /opt/ds41-precision, install marker
PRECISION-INSTALL-PASS, image kind label precision-capture.
"""
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import textwrap

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
WORKER = 'vllm/vllm/v1/worker/gpu_worker.py'
PRECISION_HELPER = 'vllm/vllm/v1/worker/ds41_precision.py'
HELPER_SOURCE = 'ds41_precision.py'
BASE_WORKER = 'ds41-precision-gpu_worker-base.py'
OUTPUT = 'ds41-precision-gpu_worker.py'
INDEXER_LOCK = 'ds41-indexer.lock.json'
INDEXER_RECEIPT = 'receipts/indexer-build-receipt.json'
STEM = 'ds41-precision'
KIND = 'precision-capture'
INSTALL_DIR = '/opt/ds41-precision'
INSTALL_PASS = 'PRECISION-INSTALL-PASS'
PACKAGED = ['ds41_precision_prepare.py', 'ds41_precision_install.py', HELPER_SOURCE, BASE_WORKER,
            'Dockerfile.' + STEM, 'test_ds41_precision.py']

IMPORT_ANCHOR = 'import vllm.envs as envs\n'
NEW_IMPORT = 'import vllm.v1.worker.ds41_precision as _ds41_precision\n'
METHOD_ANCHOR = '    def init_device(self):\n        if self.device_config.device_type == "cuda":\n'
HOOK = '''        # DIAGNOSTIC ONLY: BF16 reduced-precision reduction off, once per worker process (ds41_precision.py).
        _ds41_precision.apply(rank=self.rank)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def indexer_kit():
    spec = importlib.util.spec_from_file_location('ds41_indexer_prep', root / 'ds41_indexer_prepare.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_only_hook_changed(original, instrumented):
    """Removing exactly the new import and the call must recover the base AST; the call leads init_device."""
    before, after = ast.parse(original), ast.parse(instrumented)
    hook_ast = ast.dump(ast.parse(textwrap.dedent(HOOK)).body[0])
    import_ast = ast.dump(ast.parse(NEW_IMPORT).body[0])
    leading = [node for node in ast.walk(after) if isinstance(node, ast.FunctionDef) and node.name == 'init_device'
               and node.body and ast.dump(node.body[0]) == hook_ast]
    if len(leading) != 1:
        raise ValueError('The precision call must lead exactly one init_device body')

    class RemoveHook(ast.NodeTransformer):
        imports = calls = 0

        def visit_Import(self, node):
            if ast.dump(node) == import_ast:
                self.imports += 1
                return None
            return node

        def visit_Expr(self, node):
            if ast.dump(node) == hook_ast:
                self.calls += 1
                return None
            return self.generic_visit(node)

    remover = RemoveHook()
    cleaned = remover.visit(after)
    if (remover.imports, remover.calls) != (1, 1) or ast.dump(before) != ast.dump(cleaned):
        raise ValueError('Changes extend beyond the single init_device precision call')


def compose(worker_text, expected_sha=None):
    if expected_sha is not None and sha(worker_text.encode()) != expected_sha:
        raise ValueError('Expected the exact router candidate worker source')
    if worker_text.count(IMPORT_ANCHOR) != 1 or worker_text.count(METHOD_ANCHOR) != 1:
        raise ValueError('Precision seam is ambiguous or missing')
    result = worker_text.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + NEW_IMPORT).replace(
        METHOD_ANCHOR, METHOD_ANCHOR.replace('        if self.device_config', HOOK + '        if self.device_config'))
    verify_only_hook_changed(worker_text, result)
    compile(result, WORKER, 'exec')
    return result


def base_identity():
    """The indexer-capture image, from its lock, build receipt and retained image inspection only."""
    raw = (root / INDEXER_LOCK).read_bytes()
    indexer = json.loads(raw)
    lock_sha = sha(raw)
    receipt = json.loads((root / INDEXER_RECEIPT).read_text())
    if receipt['lock_sha256'] != lock_sha:
        raise RuntimeError('Indexer build receipt names a different indexer lock')
    build = root / 'receipts' / Path(receipt['directory']).name
    image = receipt['image_id']
    if (build / 'BUILD-OK').read_text().strip() != image:
        raise RuntimeError('Indexer build gates did not pass for this image')
    inspect = json.loads((build / 'image-inspect.json').read_text())[0]
    labels = inspect['Config']['Labels']
    if (inspect['Id'].removeprefix('sha256:') != image
            or labels.get('local-inference.ds41.diagnostic.kind') != 'indexer-capture'
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock_sha
            or labels.get('vllm.source-tree') != indexer['trees']['vllm']
            or labels.get('b12x.source-tree') != indexer['trees']['b12x']):
        raise RuntimeError('Indexer image labels do not match its lock')
    preserved = {**indexer['preserved'], **{t: e['output_sha256'] for t, e in indexer['targets'].items()}}
    return {'base_image_id': image, 'base_kind': 'indexer-capture', 'base_lock': INDEXER_LOCK, 'base_lock_sha256': lock_sha,
            'base_parent_image_id': indexer['base_image_id'], 'base_trees': indexer['trees'],
            'base_router_target': indexer['base_router_target'], 'router_lock_sha256': indexer['router_lock_sha256'],
            'window_lock_sha256': indexer['base_lock_sha256'],
            'decision_row_lock_sha256': indexer['decision_row_lock_sha256'],
            'indexer_provenance': indexer['inherited_helper_provenance'],
            'preserved': preserved}


def router_worker_source():
    """The router candidate's worker bytes, reconstructed exactly as the decision-row kit proves its tree."""
    import prepare as runtime_prepare
    from contracts import git_tree, manifest
    router = json.loads((root / 'router-release.lock.json').read_text())
    ring = json.loads((root / 'ring-fix.lock.json').read_text())
    progress = json.loads((root / 'engram-progress-vllm.lock.json').read_text())
    files = runtime_prepare.patched_vllm(runtime_prepare.VLLM)
    for path, source, expected in [(ring['source_path'], 'ring-fix-sparse_mla.py', ring['output_sha256'])] + [
            (p, t['source'], t['output_sha256']) for p, t in progress['targets'].items()]:
        data = (root / source).read_bytes()
        if sha(data) != expected:
            raise RuntimeError('vLLM reconstruction input drift: ' + source)
        files[path] = (files[path][0], data)
    if manifest(files) != router['after']['vllm'] or git_tree(files) != router['trees']['vllm']:
        raise RuntimeError('Router candidate vLLM reconstruction differs')
    return files[WORKER.removeprefix('vllm/')][1]


def indexer_changes(indexer):
    """The indexer image's vLLM changes over the router candidate, from the indexer kit's own inputs."""
    changes = {}
    for target, source in ((indexer.ATTENTION, indexer.OUTPUT), (indexer.DECISION_HELPER, 'claude-decision-row-capture.py'),
                           (indexer.WINDOW_HELPER, 'claude-window-capture.py'), (indexer.INDEXER_HELPER, indexer.HELPER_SOURCE)):
        changes[target.removeprefix('vllm/')] = (root / source).read_bytes()
    return changes


def child_trees(base, new_worker, helper):
    indexer = indexer_kit()
    changes = indexer_changes(indexer)
    for target, data in changes.items():
        if sha(data) != base['preserved']['vllm/' + target]:
            raise RuntimeError('Indexer kit input differs from the indexer lock: ' + target)
    decision = indexer.decision_kit()
    if decision.vllm_tree_after(dict(changes)) != base['base_trees']['vllm']:
        raise RuntimeError('Indexer image vLLM tree does not reproduce from its kit inputs')
    changes[WORKER.removeprefix('vllm/')] = new_worker
    changes[PRECISION_HELPER.removeprefix('vllm/')] = helper
    return {'vllm': decision.vllm_tree_after(changes), 'b12x': base['base_trees']['b12x']}


DOCKERFILE = """# DIAGNOSTIC ONLY. Never a serving candidate: worker-level BF16 reduced-precision-reduction
# off (one torch global, set once per GPU worker in init_device) on top of the reviewed
# indexer-capture image (ds41_precision_prepare.py). Capture helpers and B12X unchanged.
# Base: indexer-capture diagnostic {short}. Generated by ds41_precision_prepare.py.
FROM {base_image_id}
ARG DECISION_LOCK
ARG DECISION_CACHE
COPY {copy} {install_dir}/
RUN /opt/venv/bin/python {install_dir}/ds41_precision_install.py
# B12X bytes are unchanged: the candidate's B12X caches and selection namespace are shared.
# vLLM and Inductor output goes to a separate directory.
ENV VLLM_CACHE_ROOT=/cache/jit/${{DECISION_CACHE}}/vllm \\
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${{DECISION_CACHE}}/inductor
LABEL local-inference.ds41.diagnostic.kind="{kind}" \\
    local-inference.ds41.diagnostic.lock.sha256="${{DECISION_LOCK}}" \\
    local-inference.ds41.diagnostic.base-image="{base_image_id}" \\
    local-inference.ds41.diagnostic.base-kind="{base_kind}" \\
    local-inference.ds41.diagnostic.base-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.indexer-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.window-lock.sha256="{window_lock_sha256}" \\
    local-inference.ds41.diagnostic.decision-row-lock.sha256="{decision_row_lock_sha256}" \\
    local-inference.ds41.diagnostic.vllm-cache="${{DECISION_CACHE}}" \\
    vllm.source-tree="{vllm_tree}" \\
    b12x.source-tree="{b12x_tree}" \\
    local-inference.status="diagnostic-only-not-qualified" \\
    org.opencontainers.image.title="DS4.1 indexer-capture diagnostic with worker-level BF16 reduced-precision reduction off"
"""


def render_dockerfile(lock):
    copy = ' '.join([STEM + '.lock.json', *PACKAGED, OUTPUT, STEM + '.patch'])
    return DOCKERFILE.format(short=lock['base_image_id'][:12], copy=copy, install_dir=INSTALL_DIR, kind=KIND,
                             vllm_tree=lock['trees']['vllm'], b12x_tree=lock['trees']['b12x'],
                             **{k: lock[k] for k in ('base_image_id', 'base_kind', 'base_lock_sha256',
                                                     'window_lock_sha256', 'decision_row_lock_sha256')})


def main():
    base = base_identity()
    old_bytes = (root / BASE_WORKER).read_bytes()
    if old_bytes != router_worker_source():
        raise RuntimeError('Local base worker source differs from the router candidate reconstruction')
    old = old_bytes.decode()
    new = compose(old, sha(old_bytes))
    helper = (root / HELPER_SOURCE).read_bytes()
    compile(helper, HELPER_SOURCE, 'exec')
    (root / OUTPUT).write_text(new)
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/' + WORKER, tofile='b/' + WORKER))
    patch += ''.join(difflib.unified_diff([], helper.decode().splitlines(True), fromfile='/dev/null',
                                          tofile='b/' + PRECISION_HELPER))
    (root / (STEM + '.patch')).write_text(patch)
    trees = child_trees(base, new.encode(), helper)
    provenance = base.pop('indexer_provenance')
    lock = {'kind': KIND + '-diagnostic', 'status': 'diagnostic-only-not-built-not-qualified',
            'scope': 'process-wide torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False, set once '
                     'per GPU worker at the top of init_device, on top of the indexer-capture image; one call, no '
                     'forward-path change, capture helpers and B12X unchanged; a diagnostic arm, not determinism, '
                     'batch-invariance, retrieval, performance or gate evidence',
            'install_dir': INSTALL_DIR, 'install_pass_marker': INSTALL_PASS, 'apply_marker': 'DS41-PRECISION-APPLIED',
            'build_args': ['DECISION_LOCK', 'DECISION_CACHE'], 'image_kind_label': KIND,
            'precision_schema': 'ds41-precision-v1', 'capture_schema': 'ds41-indexer-capture-v1',
            'window_schema': 'claude-window-v1', 'decision_row_schema': 'claude-decision-row-v3',
            **base, 'trees': trees,
            'targets': {WORKER: {'input_sha256': sha(old_bytes), 'output_sha256': sha(new.encode()), 'source': OUTPUT},
                        PRECISION_HELPER: {'input_sha256': None, 'output_sha256': sha(helper), 'source': HELPER_SOURCE}},
            'inherited_helper_provenance': {
                'disposition': 'metadata-only: attention and the decision-row, window and indexer helpers keep their '
                               'bytes; the three lock files they read for source_trees are replaced by provenance '
                               'stubs naming this image\'s trees and the precision lock, with the indexer-era files '
                               'preserved beside them and the window-era originals left in place',
                'window_read_path': provenance['window_read_path'],
                'decision_read_path': provenance['decision_read_path'],
                'indexer_read_path': '/opt/ds41-indexer/ds41-indexer.lock.json',
                'window_era_original_lock': provenance['original_window_lock'],
                'window_era_original_stub': provenance['original_decision_stub'],
                'indexer_era_window_stub': 'claude-window.lock.indexer-stub.json',
                'indexer_era_decision_stub': 'claude-decision-row.lock.indexer-stub.json',
                'indexer_era_lock': 'ds41-indexer.lock.original.json'}}
    (root / ('Dockerfile.' + STEM)).write_text(render_dockerfile(lock))
    lock['inputs'] = {name: sha((root / name).read_bytes()) for name in PACKAGED + [OUTPUT, STEM + '.patch']}
    (root / (STEM + '.lock.json')).write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + [STEM + '.lock.json']
    (root / (STEM + '.ignore')).write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('PRECISION-PREPARED', sha((root / (STEM + '.lock.json')).read_bytes()), trees)


if __name__ == '__main__':
    main()
