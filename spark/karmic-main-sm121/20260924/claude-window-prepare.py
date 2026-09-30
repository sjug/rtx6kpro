"""Compose the DIAGNOSTIC window-capture derivative of the router fence candidate e06df11a.

Review artifact only: no build, node, GPU or launch. Derives from the SAME
router parent as the decision-row capture. Changes exactly three vLLM files:
deepseek_v4_1/attention.py (the four reviewed decision-row call sites plus five
guarded window call sites at existing eager boundaries), the UNCHANGED
decision-row helper deepseek_v4_1/claude_decision_row.py (bytes equal to the
decision-row lock's helper) and the new helper deepseek_v4_1/claude_window.py.
B12X bytes are unchanged, so the router candidate's B12X compile cache, native
build and selection namespace are reused.

Boundaries and why these five (MATCHED-CHUNK-CAPTURE-20260926.md addendum):
  _forward          custom-op body (vllm::dsv41_b12x_attention): incoming hidden rows and normalized KV,
                    after live SWA metadata is resolved and before _cache_context_kv
  insert_context_kv inside the eager_break _cache_context_kv: rotated KV and slots before mla.write_cache
  forward_mqa       eager_break: query/output rows and the records read, after the decision-row hook
  _o_proj           custom-op body: WO result before and after the TP all-reduce
Not instrumented: o-proj internals, mHC and FFN. They run inside compiled
regions of the model forward; a D2H copy there would add graph breaks and could
change Inductor fusion, i.e. the numerics under test. The reduced WO output is
the last eager tensor before those regions and the next layer's hidden_in is
the first eager tensor after them, so the compiled interval is bracketed, not
opened.

Usage (local, no nodes): python3 claude-window-prepare.py
Outputs: claude-window-attention.py, claude-window.patch, claude-window.lock.json,
claude-window.ignore, Dockerfile.claude-window (generic builder ARGs DECISION_LOCK/DECISION_CACHE). Build orchestration is main's.
"""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))
ATTENTION = 'vllm/vllm/models/deepseek_v4_1/attention.py'
DECISION_HELPER = 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py'
WINDOW_HELPER = 'vllm/vllm/models/deepseek_v4_1/claude_window.py'
DECISION_HELPER_SOURCE = 'claude-decision-row-capture.py'
DECISION_LOCK = 'claude-decision-row.lock.json'
HELPER_SOURCE = 'claude-window-capture.py'
OUTPUT = 'claude-window-attention.py'
STEM = 'claude-window'
KIND = 'window-capture'
INSTALL_DIR = '/opt/ds41-window'
INSTALL_PASS = 'WINDOW-INSTALL-PASS'
PACKAGED = ['claude-window-prepare.py', 'claude-window-install.py', HELPER_SOURCE, DECISION_HELPER_SOURCE,
            'Dockerfile.' + STEM, 'claude-window-tests.py', 'claude-window-compare.py']

EDITS = [
    ('import vllm.models.deepseek_v4_1.claude_decision_row as _claude_row\n',
     'import vllm.models.deepseek_v4_1.claude_decision_row as _claude_row\n'
     '# DIAGNOSTIC ONLY: final-window capture through layers 0 and 1 (claude-window-capture.py).\n'
     'import vllm.models.deepseek_v4_1.claude_window as _claude_window\n'),
    ('            swa = self._query_metadata(original_swa)\n'
     '            is_prefill = not swa.is_decode\n'
     '            self._cache_context_kv(kv, positions)\n',
     '            swa = self._query_metadata(original_swa)\n'
     '            is_prefill = not swa.is_decode\n'
     '            # DIAGNOSTIC ONLY: None unless armed, layer 0 or 1, and on the decision chunk.\n'
     '            _window = _claude_window.begin(self, metadata)\n'
     '            if _window is not None:\n'
     '                _window.inputs(self, hidden_states=hidden_states, kv=kv, positions=positions)\n'
     '            self._cache_context_kv(kv, positions)\n'),
    ('            plan=self._helper_plan("kv"),\n'
     '        )\n'
     '        mla.write_cache(\n'
     '            rotated,\n'
     '            self.swa_cache_layer.kv_cache,\n',
     '            plan=self._helper_plan("kv"),\n'
     '        )\n'
     '        _claude_window.rotated(self, rotated=rotated, slot_mapping=slot_mapping, positions=positions)\n'
     '        mla.write_cache(\n'
     '            rotated,\n'
     '            self.swa_cache_layer.kv_cache,\n'),
    ('                binding=binding, owner=owner,\n'
     '            )\n',
     '                binding=binding, owner=owner,\n'
     '            )\n'
     '        _claude_window.attention(\n'
     '            self, q=q, output=output, swa_indices=swa_indices[:rows], swa_lengths=swa_lengths[:rows],\n'
     '        )\n'),
    ('            raise TypeError("V4.1 WO projection must return BF16")\n'
     '        l2_prefetch.issue(self._l2pf_ffn, rows)\n',
     '            raise TypeError("V4.1 WO projection must return BF16")\n'
     '        _claude_window.wo_partial(self, local)\n'
     '        l2_prefetch.issue(self._l2pf_ffn, rows)\n'),
    ('            local = get_tp_group().all_reduce(local)\n'
     '        return local\n',
     '            local = get_tp_group().all_reduce(local)\n'
     '        _claude_window.wo_reduced(self, local)\n'
     '        return local\n'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decision_kit():
    spec = importlib.util.spec_from_file_location('claude_decision_prep', root / 'claude_prepare_decision_row.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compose(old_text, decision=None):
    """Upstream 1794dcf1 attention.py -> decision-row composition -> window composition."""
    decision = decision or decision_kit()
    text = decision.compose(old_text)
    for anchor, replacement in EDITS:
        if text.count(anchor) != 1:
            raise RuntimeError('anchor changed: ' + anchor.strip()[:60])
        text = text.replace(anchor, replacement)
    compile(text, ATTENTION, 'exec')
    return text


def decision_provenance():
    """The unchanged decision-row helper and its lock, as the window image carries them."""
    raw = (root / DECISION_LOCK).read_bytes()
    lock = json.loads(raw)
    helper = (root / DECISION_HELPER_SOURCE).read_bytes()
    recorded = lock['targets'][DECISION_HELPER]['output_sha256']
    if sha(helper) != recorded or lock['inputs'][DECISION_HELPER_SOURCE] != recorded:
        raise RuntimeError('Decision-row helper differs from its lock; the window kit must carry it unchanged')
    return {'decision_row_lock_sha256': sha(raw), 'decision_row_helper_sha256': recorded,
            'decision_row_attention_sha256': lock['targets'][ATTENTION]['output_sha256'],
            'decision_row_trees': lock['trees']}


DOCKERFILE = """# DIAGNOSTIC ONLY. Never a serving candidate: final-window capture through layers 0-1
# beside the unchanged decision-row capture (claude-window-prepare.py). Hooks add copies,
# not arithmetic; equal full responses show no observed output perturbation only.
# Base: router stage-release fence candidate {short}. Generated by claude-window-prepare.py.
FROM {base_image_id}
ARG DECISION_LOCK
ARG DECISION_CACHE
COPY {copy} {install_dir}/
RUN /opt/venv/bin/python {install_dir}/claude-window-install.py
# B12X bytes are unchanged: the candidate's B12X caches and selection namespace are shared.
# vLLM and Inductor output goes to a separate directory.
ENV VLLM_CACHE_ROOT=/cache/jit/${{DECISION_CACHE}}/vllm \\
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${{DECISION_CACHE}}/inductor
LABEL local-inference.ds41.diagnostic.kind="{kind}" \\
    local-inference.ds41.diagnostic.lock.sha256="${{DECISION_LOCK}}" \\
    local-inference.ds41.diagnostic.base-image="{base_image_id}" \\
    local-inference.ds41.diagnostic.base-kind="{base_kind}" \\
    local-inference.ds41.diagnostic.base-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.decision-row-lock.sha256="{decision_row_lock_sha256}" \\
    local-inference.ds41.diagnostic.vllm-cache="${{DECISION_CACHE}}" \\
    vllm.source-tree="{vllm_tree}" \\
    b12x.source-tree="{b12x_tree}" \\
    local-inference.status="diagnostic-only-not-qualified" \\
    org.opencontainers.image.title="DS4.1 router fence candidate with final-window and decision-row capture diagnostics"
"""


def render_dockerfile(lock):
    copy = ' '.join([STEM + '.lock.json', *PACKAGED, OUTPUT, STEM + '.patch'])
    return DOCKERFILE.format(short=lock['base_image_id'][:12], copy=copy, install_dir=INSTALL_DIR, kind=KIND,
                             vllm_tree=lock['trees']['vllm'], b12x_tree=lock['trees']['b12x'],
                             **{k: lock[k] for k in ('base_image_id', 'base_kind', 'base_lock_sha256',
                                                     'decision_row_lock_sha256')})


def main():
    decision = decision_kit()
    base = decision.base_identity()
    old = decision.upstream_attention()
    if sha(old) != base['attention_sha256']:
        raise RuntimeError('vLLM 1794dcf1 attention.py is not the router candidate installed file')
    provenance = decision_provenance()
    new = compose(old.decode(), decision)
    decision_helper = (root / DECISION_HELPER_SOURCE).read_bytes()
    helper = (root / HELPER_SOURCE).read_bytes()
    compile(helper, HELPER_SOURCE, 'exec')
    (root / OUTPUT).write_text(new)
    patch = ''.join(difflib.unified_diff(old.decode().splitlines(True), new.splitlines(True),
                                         fromfile='a/' + ATTENTION, tofile='b/' + ATTENTION))
    for target, data in ((DECISION_HELPER, decision_helper), (WINDOW_HELPER, helper)):
        patch += ''.join(difflib.unified_diff([], data.decode().splitlines(True), fromfile='/dev/null',
                                              tofile='b/' + target))
    (root / (STEM + '.patch')).write_text(patch)
    trees = {'vllm': decision.vllm_tree_after({ATTENTION.removeprefix('vllm/'): new.encode(),
                                               DECISION_HELPER.removeprefix('vllm/'): decision_helper,
                                               WINDOW_HELPER.removeprefix('vllm/'): helper}),
             'b12x': base['base_trees']['b12x']}
    lock = {'kind': KIND + '-diagnostic', 'status': 'diagnostic-only-not-built-not-qualified',
            'scope': 'final-window capture through layers 0-1 beside the unchanged decision-row capture; adds '
                     'copies at existing eager boundaries; not numerics, determinism, performance or router-fence '
                     'evidence',
            'install_dir': INSTALL_DIR, 'install_pass_marker': INSTALL_PASS,
            'build_args': ['DECISION_LOCK', 'DECISION_CACHE'], 'image_kind_label': KIND,
            'capture_schema': 'claude-window-v1', 'decision_row_schema': 'claude-decision-row-v3',
            **{k: v for k, v in base.items() if k != 'attention_sha256'}, **provenance, 'trees': trees,
            'targets': {ATTENTION: {'input_sha256': sha(old), 'output_sha256': sha(new.encode()), 'source': OUTPUT},
                        DECISION_HELPER: {'input_sha256': None, 'output_sha256': sha(decision_helper),
                                          'source': DECISION_HELPER_SOURCE},
                        WINDOW_HELPER: {'input_sha256': None, 'output_sha256': sha(helper), 'source': HELPER_SOURCE}}}
    (root / ('Dockerfile.' + STEM)).write_text(render_dockerfile(lock))
    lock['inputs'] = {name: sha((root / name).read_bytes()) for name in PACKAGED + [OUTPUT, STEM + '.patch']}
    (root / (STEM + '.lock.json')).write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + [STEM + '.lock.json']
    (root / (STEM + '.ignore')).write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('WINDOW-PREPARED', sha((root / (STEM + '.lock.json')).read_bytes()), trees)


if __name__ == '__main__':
    main()
