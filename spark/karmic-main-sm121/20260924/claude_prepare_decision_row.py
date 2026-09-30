"""Compose the DIAGNOSTIC decision-row capture derivative of the router fence candidate e06df11a.

Review artifact only: no build, node, GPU or launch. Changes exactly two vLLM
files: deepseek_v4_1/attention.py (four guarded call sites) and the new
helper deepseek_v4_1/claude_decision_row.py. B12X bytes are unchanged, so the
router candidate's B12X compile cache, native build and selection namespace
are reused (an armed boot at an unseen capacity adds capacity-keyed selection
records; declare the capacity).

Base identity, the router lock/receipt checks and the exact vLLM tree
reconstruction are shared with the Engram fault-injection kit.
"""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parent
ATTENTION = 'vllm/vllm/models/deepseek_v4_1/attention.py'
HELPER = 'vllm/vllm/models/deepseek_v4_1/claude_decision_row.py'
HELPER_SOURCE = 'claude-decision-row-capture.py'
OUTPUT = 'claude-decision-row-attention.py'
STEM = 'claude-decision-row'
PACKAGED = ['claude_prepare_decision_row.py', 'claude_install_decision_row.py', HELPER_SOURCE,
            'Dockerfile.' + STEM, 'claude_test_decision_row_capture.py']

EDITS = [
    ('from vllm.triton_utils import tl, triton\n',
     'from vllm.triton_utils import tl, triton\n'
     '# DIAGNOSTIC ONLY: decision-row capture (claude-decision-row-audit-DESIGN.md).\n'
     'import vllm.models.deepseek_v4_1.claude_decision_row as _claude_row\n'),
    ('        state = require_prepared(plan, "attention.compressed_sparse_mla")\n',
     '        state = require_prepared(plan, "attention.compressed_sparse_mla")\n'
     '        # DIAGNOSTIC ONLY: None unless armed and on the decision chunk.\n'
     '        _claude = _claude_row.begin(self, metadata)\n'),
    ('                dsa_indexer.select(binding)\n',
     '                dsa_indexer.select(binding)\n'
     '                if _claude is not None and end == rows:\n'
     '                    _claude.indexer(\n'
     '                        self, row_in_chunk=rows - 1 - offset, iq_data=iq_data,\n'
     '                        iq_scale=iq_scale, iw=iw, cache_lengths=im.cache_lengths,\n'
     '                        index_pages=index_pages,\n'
     '                    )\n'),
    ('            attn_sink=self.attn_sink,\n'
     '            out=output,\n'
     '            cache_format="deepseek_v41",\n'
     '        )\n',
     '            attn_sink=self.attn_sink,\n'
     '            out=output,\n'
     '            cache_format="deepseek_v41",\n'
     '        )\n'
     '        if _claude is not None:\n'
     '            _claude.attention(\n'
     '                self, state=state, q=q, output=output, swa_indices=swa_indices,\n'
     '                swa_lengths=swa_lengths, top_lengths=top_lengths,\n'
     '                page_table=buffers[3] if main is not None else None,\n'
     '                binding=binding, owner=owner,\n'
     '            )\n'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _fault_kit():
    spec = importlib.util.spec_from_file_location('claude_fault_prepare', root / 'claude_prepare_engram_fault_inject.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def base_identity():
    base = _fault_kit().base_identity()
    router = json.loads((root / base['base_lock']).read_text())
    attention = router['after']['vllm'][ATTENTION.removeprefix('vllm/')]['sha256']
    if HELPER.removeprefix('vllm/') in router['after']['vllm']:
        raise RuntimeError('Helper path already exists in the router candidate')
    return {**{k: v for k, v in base.items() if k != 'model_sha256'}, 'attention_sha256': attention}


def upstream_attention():
    import subprocess
    data = subprocess.run(['git', '-C', str(Path.home() / 'git/vllm'), 'show',
                           '1794dcf18454900263e0c66711af8ea4a1283ac1:vllm/models/deepseek_v4_1/attention.py'],
                          check=True, capture_output=True).stdout
    return data


def compose(old_text):
    text = old_text
    for anchor, replacement in EDITS:
        if text.count(anchor) != 1:
            raise RuntimeError('anchor changed: ' + anchor.strip()[:60])
        text = text.replace(anchor, replacement)
    compile(text, ATTENTION, 'exec')
    return text


def vllm_tree_after(changes):
    """Exact vLLM tree: router candidate sources (proven equal to the router lock) with `changes` applied."""
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
    for path, data in changes.items():
        files[path] = (files[path][0] if path in files else '100644', data)
    return git_tree(files)


DOCKERFILE = """# DIAGNOSTIC ONLY. Never a serving candidate: decision-row capture for the
# offline reference audit (claude-decision-row-audit-DESIGN.md). Hooks add copies,
# not arithmetic; equal full responses show no observed output perturbation only.
# Base: router stage-release fence candidate {short}. Generated by claude_prepare_decision_row.py.
FROM {base_image_id}
ARG DECISION_LOCK
ARG DECISION_CACHE
COPY {copy} /opt/ds41-decision-row/
RUN /opt/venv/bin/python /opt/ds41-decision-row/claude_install_decision_row.py
# B12X bytes are unchanged: the candidate's B12X caches and selection namespace are shared.
# vLLM and Inductor output goes to a separate directory.
ENV VLLM_CACHE_ROOT=/cache/jit/${{DECISION_CACHE}}/vllm \\
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${{DECISION_CACHE}}/inductor
LABEL local-inference.ds41.diagnostic.kind="decision-row-capture" \\
    local-inference.ds41.diagnostic.lock.sha256="${{DECISION_LOCK}}" \\
    local-inference.ds41.diagnostic.base-image="{base_image_id}" \\
    local-inference.ds41.diagnostic.base-kind="{base_kind}" \\
    local-inference.ds41.diagnostic.base-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.vllm-cache="${{DECISION_CACHE}}" \\
    vllm.source-tree="{vllm_tree}" \\
    b12x.source-tree="{b12x_tree}" \\
    local-inference.status="diagnostic-only-not-qualified" \\
    org.opencontainers.image.title="DS4.1 router fence candidate with decision-row capture diagnostic"
"""


def render_dockerfile(lock):
    copy = ' '.join([STEM + '.lock.json', *PACKAGED, OUTPUT, STEM + '.patch'])
    return DOCKERFILE.format(short=lock['base_image_id'][:12], copy=copy, vllm_tree=lock['trees']['vllm'],
                             b12x_tree=lock['trees']['b12x'],
                             **{k: lock[k] for k in ('base_image_id', 'base_kind', 'base_lock_sha256')})


def main():
    base = base_identity()
    old = upstream_attention()
    if sha(old) != base['attention_sha256']:
        raise RuntimeError('vLLM 1794dcf1 attention.py is not the router candidate installed file')
    new = compose(old.decode())
    helper = (root / HELPER_SOURCE).read_bytes()
    (root / OUTPUT).write_text(new)
    patch = ''.join(difflib.unified_diff(old.decode().splitlines(True), new.splitlines(True),
                                         fromfile='a/' + ATTENTION, tofile='b/' + ATTENTION))
    patch += ''.join(difflib.unified_diff([], helper.decode().splitlines(True), fromfile='/dev/null',
                                          tofile='b/' + HELPER))
    (root / (STEM + '.patch')).write_text(patch)
    trees = {'vllm': vllm_tree_after({ATTENTION.removeprefix('vllm/'): new.encode(),
                                      HELPER.removeprefix('vllm/'): helper}),
             'b12x': base['base_trees']['b12x']}
    lock = {'kind': 'decision-row-capture-diagnostic', 'status': 'diagnostic-only-not-built-not-qualified',
            'scope': 'decision-row capture for the offline reference audit; adds copies after producers; '
                     'not numerics, determinism, performance or router-fence evidence',
            **{k: v for k, v in base.items() if k != 'attention_sha256'}, 'trees': trees,
            'targets': {ATTENTION: {'input_sha256': sha(old), 'output_sha256': sha(new.encode()), 'source': OUTPUT},
                        HELPER: {'input_sha256': None, 'output_sha256': sha(helper), 'source': HELPER_SOURCE}}}
    (root / ('Dockerfile.' + STEM)).write_text(render_dockerfile(lock))
    lock['inputs'] = {name: sha((root / name).read_bytes()) for name in PACKAGED + [OUTPUT, STEM + '.patch']}
    (root / (STEM + '.lock.json')).write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + [STEM + '.lock.json']
    (root / (STEM + '.ignore')).write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('DECISION-ROW-PREPARED', sha((root / (STEM + '.lock.json')).read_bytes()), trees)


if __name__ == '__main__':
    main()
