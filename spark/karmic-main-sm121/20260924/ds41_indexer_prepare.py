"""Compose the DIAGNOSTIC indexer-capture derivative of the reviewed window-capture image c4a51be4.

Review artifact only: no build, node, GPU or launch. Derives from the EXACT
window-capture image (kind window-capture, lock 59100837): the installed
attention.py of that image (sha 72bee917) is the preimage, and the change is
one import plus one guarded observation call after weight_scale.scale_index_weights
on layer 2, plus the new helper deepseek_v4_1/ds41_indexer_capture.py. The
decision-row and window helpers are untouched and stay installed. B12X bytes
are unchanged, so the router candidate's B12X caches and selection namespace
are reused.

Inherited helper provenance is metadata only: the installer preserves the
original window lock and the window-era decision-row stub under explicit
original names and writes stubs at both helper read paths naming this image's
trees and the indexer lock (see ds41_indexer_install.py). No helper source changes.

Source AST gate: removing exactly the new import and the guarded call from the
composed source must recover the window attention AST node for node, so no
projection, scaling, rotation or numerical policy can change (the draft
prepare_indexer_capture_hook.py logic, carried here for the installable kit).

Usage (local, no nodes): python3 ds41_indexer_prepare.py
Outputs: ds41-indexer-attention.py, ds41-indexer.patch, ds41-indexer.lock.json,
ds41-indexer.ignore, Dockerfile.ds41-indexer. Builder ARGs DECISION_LOCK and
DECISION_CACHE (generic builder), install dir /opt/ds41-indexer, install marker
INDEXER-INSTALL-PASS, image kind label indexer-capture.
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
ATTENTION = 'vllm/vllm/models/deepseek_v4_1/attention.py'
DECISION_HELPER = 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py'
WINDOW_HELPER = 'vllm/vllm/models/deepseek_v4_1/claude_window.py'
INDEXER_HELPER = 'vllm/vllm/models/deepseek_v4_1/ds41_indexer_capture.py'
HELPER_SOURCE = 'ds41_indexer_capture.py'
WINDOW_ATTENTION = 'claude-window-attention.py'
WINDOW_LOCK = 'claude-window.lock.json'
WINDOW_RECEIPT = 'receipts/window-build-receipt.json'
OUTPUT = 'ds41-indexer-attention.py'
STEM = 'ds41-indexer'
KIND = 'indexer-capture'
INSTALL_DIR = '/opt/ds41-indexer'
INSTALL_PASS = 'INDEXER-INSTALL-PASS'
PACKAGED = ['ds41_indexer_prepare.py', 'ds41_indexer_install.py', HELPER_SOURCE, 'Dockerfile.' + STEM,
            'test_ds41_indexer_capture.py']

IMPORT = 'import vllm.models.deepseek_v4_1.claude_window as _claude_window\n'
NEW_IMPORT = 'import vllm.models.deepseek_v4_1.ds41_indexer_capture as _indexer_capture\n'
ANCHOR = '''                weight_scale.scale_index_weights(
                    weights,
                    out=iw,
                    plan=self._helper_plan("index_weights"),
                )
                index_query = (iq_data, iq_scale, iw)
'''
HOOK = '''                # DIAGNOSTIC ONLY: layer-2 index-weight observation; no projection or scaling change.
                if self.layer_id == 2:
                    _indexer_capture.capture(
                        self, metadata=metadata, positions=positions,
                        hidden_input=hidden_states, kv_norm=kv, q_rotated=q,
                        index_query_rotated=iq, raw_weights=weights,
                        scaled_weights=iw, projection_weight=self.indexer.weights_proj.weight,
                    )
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decision_kit():
    spec = importlib.util.spec_from_file_location('claude_decision_prep', root / 'claude_prepare_decision_row.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_only_hook_changed(original, instrumented):
    """Removing exactly the new import and the guarded call must recover the original AST."""
    before, after = ast.parse(original), ast.parse(instrumented)
    hook_ast = ast.dump(ast.parse(textwrap.dedent(HOOK)).body[0])
    import_ast = ast.dump(ast.parse(NEW_IMPORT).body[0])

    class RemoveHook(ast.NodeTransformer):
        imports = calls = 0

        def visit_Import(self, node):
            if ast.dump(node) == import_ast:
                self.imports += 1
                return None
            return node

        def visit_If(self, node):
            if ast.dump(node) == hook_ast:
                self.calls += 1
                return None
            return self.generic_visit(node)

    remover = RemoveHook()
    cleaned = remover.visit(after)
    if (remover.imports, remover.calls) != (1, 1) or ast.dump(before) != ast.dump(cleaned):
        raise ValueError('Changes extend beyond the single guarded observation hook')


def compose(window_text, expected_sha=None):
    if expected_sha is not None and sha(window_text.encode()) != expected_sha:
        raise ValueError('Expected the exact reviewed window-capture attention source')
    if window_text.count(IMPORT) != 1 or window_text.count(ANCHOR) != 1:
        raise ValueError('Diagnostic hook seam is ambiguous or missing')
    result = window_text.replace(IMPORT, IMPORT + NEW_IMPORT).replace(
        ANCHOR, ANCHOR.replace('                index_query =', HOOK + '                index_query ='))
    verify_only_hook_changed(window_text, result)
    compile(result, ATTENTION, 'exec')
    return result


def base_identity():
    """The window-capture image, from its lock, build receipt and retained image inspection only."""
    raw = (root / WINDOW_LOCK).read_bytes()
    window = json.loads(raw)
    lock_sha = sha(raw)
    receipt = json.loads((root / WINDOW_RECEIPT).read_text())
    if receipt['lock_sha256'] != lock_sha:
        raise RuntimeError('Window build receipt names a different window lock')
    build = root / 'receipts' / Path(receipt['directory']).name
    image = receipt['image_id']
    if (build / 'BUILD-OK').read_text().strip() != image:
        raise RuntimeError('Window build gates did not pass for this image')
    inspect = json.loads((build / 'image-inspect.json').read_text())[0]
    labels = inspect['Config']['Labels']
    if (inspect['Id'].removeprefix('sha256:') != image
            or labels.get('local-inference.ds41.diagnostic.kind') != 'window-capture'
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock_sha
            or labels.get('vllm.source-tree') != window['trees']['vllm']
            or labels.get('b12x.source-tree') != window['trees']['b12x']):
        raise RuntimeError('Window image labels do not match its lock')
    attention = window['targets'][ATTENTION]['output_sha256']
    if sha((root / WINDOW_ATTENTION).read_bytes()) != attention:
        raise RuntimeError('Local window attention source differs from the window lock')
    return {'base_image_id': image, 'base_kind': 'window-capture', 'base_lock': WINDOW_LOCK, 'base_lock_sha256': lock_sha,
            'base_parent_image_id': window['base_image_id'], 'base_trees': window['trees'],
            'base_router_target': window['base_router_target'], 'router_lock_sha256': window['base_lock_sha256'],
            'attention_sha256': attention,
            'base_helpers': {DECISION_HELPER: window['targets'][DECISION_HELPER]['output_sha256'],
                             WINDOW_HELPER: window['targets'][WINDOW_HELPER]['output_sha256']},
            'decision_row_lock_sha256': window['decision_row_lock_sha256']}


DOCKERFILE = """# DIAGNOSTIC ONLY. Never a serving candidate: layer-2 index-weight observation on top of the
# reviewed window-capture image (ds41_indexer_prepare.py). One guarded call, copies only;
# equal responses show no observed output perturbation only.
# Base: window-capture diagnostic {short}. Generated by ds41_indexer_prepare.py.
FROM {base_image_id}
ARG DECISION_LOCK
ARG DECISION_CACHE
COPY {copy} {install_dir}/
RUN /opt/venv/bin/python {install_dir}/ds41_indexer_install.py
# B12X bytes are unchanged: the candidate's B12X caches and selection namespace are shared.
# vLLM and Inductor output goes to a separate directory.
ENV VLLM_CACHE_ROOT=/cache/jit/${{DECISION_CACHE}}/vllm \\
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${{DECISION_CACHE}}/inductor
LABEL local-inference.ds41.diagnostic.kind="{kind}" \\
    local-inference.ds41.diagnostic.lock.sha256="${{DECISION_LOCK}}" \\
    local-inference.ds41.diagnostic.base-image="{base_image_id}" \\
    local-inference.ds41.diagnostic.base-kind="{base_kind}" \\
    local-inference.ds41.diagnostic.base-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.window-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.decision-row-lock.sha256="{decision_row_lock_sha256}" \\
    local-inference.ds41.diagnostic.vllm-cache="${{DECISION_CACHE}}" \\
    vllm.source-tree="{vllm_tree}" \\
    b12x.source-tree="{b12x_tree}" \\
    local-inference.status="diagnostic-only-not-qualified" \\
    org.opencontainers.image.title="DS4.1 window-capture diagnostic with layer-2 index-weight observation"
"""


def render_dockerfile(lock):
    copy = ' '.join([STEM + '.lock.json', *PACKAGED, OUTPUT, STEM + '.patch'])
    return DOCKERFILE.format(short=lock['base_image_id'][:12], copy=copy, install_dir=INSTALL_DIR, kind=KIND,
                             vllm_tree=lock['trees']['vllm'], b12x_tree=lock['trees']['b12x'],
                             **{k: lock[k] for k in ('base_image_id', 'base_kind', 'base_lock_sha256',
                                                     'decision_row_lock_sha256')})


def main():
    decision = decision_kit()
    base = base_identity()
    old = (root / WINDOW_ATTENTION).read_text()
    new = compose(old, base['attention_sha256'])
    helper = (root / HELPER_SOURCE).read_bytes()
    compile(helper, HELPER_SOURCE, 'exec')
    decision_helper = (root / 'claude-decision-row-capture.py').read_bytes()
    window_helper = (root / 'claude-window-capture.py').read_bytes()
    for target, data in ((DECISION_HELPER, decision_helper), (WINDOW_HELPER, window_helper)):
        if sha(data) != base['base_helpers'][target]:
            raise RuntimeError('Base helper source differs from the window lock: ' + target)
    (root / OUTPUT).write_text(new)
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/' + ATTENTION, tofile='b/' + ATTENTION))
    patch += ''.join(difflib.unified_diff([], helper.decode().splitlines(True), fromfile='/dev/null',
                                          tofile='b/' + INDEXER_HELPER))
    (root / (STEM + '.patch')).write_text(patch)
    trees = {'vllm': decision.vllm_tree_after({ATTENTION.removeprefix('vllm/'): new.encode(),
                                               DECISION_HELPER.removeprefix('vllm/'): decision_helper,
                                               WINDOW_HELPER.removeprefix('vllm/'): window_helper,
                                               INDEXER_HELPER.removeprefix('vllm/'): helper}),
             'b12x': base['base_trees']['b12x']}
    lock = {'kind': KIND + '-diagnostic', 'status': 'diagnostic-only-not-built-not-qualified',
            'scope': 'layer-2 index-weight observation (hidden input, normalized KV, rotated q and index query, raw '
                     'and scaled index weights, projection weight, selected gemv plan) on top of the window-capture '
                     'image; one guarded call, copies only; not numerics, determinism, performance or gate evidence',
            'install_dir': INSTALL_DIR, 'install_pass_marker': INSTALL_PASS,
            'build_args': ['DECISION_LOCK', 'DECISION_CACHE'], 'image_kind_label': KIND,
            'capture_schema': 'ds41-indexer-capture-v1', 'window_schema': 'claude-window-v1',
            'decision_row_schema': 'claude-decision-row-v3',
            **{k: v for k, v in base.items() if k not in ('attention_sha256', 'base_helpers')},
            'trees': trees,
            'targets': {ATTENTION: {'input_sha256': base['attention_sha256'], 'output_sha256': sha(new.encode()),
                                    'source': OUTPUT},
                        INDEXER_HELPER: {'input_sha256': None, 'output_sha256': sha(helper), 'source': HELPER_SOURCE}},
            'preserved': base['base_helpers'],
            'inherited_helper_provenance': {
                'disposition': 'metadata-only: the unchanged decision-row and window helpers keep their bytes; the '
                               'lock files they read for source_trees are replaced by provenance stubs naming this '
                               'image\'s trees and the indexer lock, with the originals preserved beside them',
                'window_read_path': '/opt/ds41-window/claude-window.lock.json',
                'decision_read_path': '/opt/ds41-decision-row/claude-decision-row.lock.json',
                'original_window_lock': 'claude-window.lock.original.json',
                'original_decision_stub': 'claude-decision-row.lock.original.json'}}
    (root / ('Dockerfile.' + STEM)).write_text(render_dockerfile(lock))
    lock['inputs'] = {name: sha((root / name).read_bytes()) for name in PACKAGED + [OUTPUT, STEM + '.patch']}
    (root / (STEM + '.lock.json')).write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + [STEM + '.lock.json']
    (root / (STEM + '.ignore')).write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('INDEXER-PREPARED', sha((root / (STEM + '.lock.json')).read_bytes()), trees)


if __name__ == '__main__':
    main()
