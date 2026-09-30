"""Compose the DIAGNOSTIC one-shot publication-skip derivative of the router fence candidate.

Review artifact only: no build, no node, no GPU. Scope: the cross-rank
fail-closed Engram publication test. It does not test numerics,
determinism, performance or the router fence itself.

Base: the built router stage-release candidate e06df11a8ca1 (receipts/
router-release-build-receipt.json), whose build gates passed (BUILD-OK) and
whose recorded image labels name the router lock and source trees. The
router delta is B12X bf16_gemv/_prefill.py only; its vLLM tree equals the
Engram repair parent's, so the installed DS4.1 NVIDIA model.py is the Engram
repair output byte for byte (checked below through both locks). This
derivative changes exactly that one installed file and adds one helper
module; B12X bytes are unchanged, so the candidate's B12X compile cache and
native loader build are reused.
"""
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
MODEL_TARGET = 'vllm/vllm/models/deepseek_v4_1/nvidia/model.py'
HELPER_TARGET = 'vllm/vllm/models/deepseek_v4_1/claude_engram_fault_inject.py'
MODEL_SOURCE = 'engram-progress-model.py'
HELPER_SOURCE = 'claude_engram_fault_inject.py'
OUTPUT = 'claude-engram-fault-inject-model.py'

IMPORT_OLD = 'from b12x.sequence import engram as engram_native\n'
IMPORT_NEW = (IMPORT_OLD
              + '# DIAGNOSTIC ONLY: one-shot publication skip for the cross-rank fail-closed test.\n'
              + 'import vllm.models.deepseek_v4_1.claude_engram_fault_inject as _fault_inject\n')
JOB_OLD = '''        stream = self._engram_stream()
        # Every CUDA operation required by the producer is submitted here,
        # before the consumer forward. The stream's native I/O callback does
        # not call CUDA or Python and cannot wait on a later host CUDA launch.
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            self._engram_job = engram_native.enqueue_lookups(
                bindings,
                counts,
                status_host=self._engram_io_status.host_view,
                clear_tail=False,
            )
            _publish_engram_epoch[(1,)](
                self._engram_io_status.device_view,
                self._engram_epochs[0:1],
                epoch,
            )
'''
JOB_NEW = '''        stream = self._engram_stream()
        # DIAGNOSTIC ONLY: decided on the host before any CUDA work of this job
        # is queued; True at most once, after a durable receipt names `epoch`.
        skip_publish = _fault_inject.should_skip(epoch, counts[0] if counts else 0)
        # Every CUDA operation required by the producer is submitted here,
        # before the consumer forward. The stream's native I/O callback does
        # not call CUDA or Python and cannot wait on a later host CUDA launch.
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            self._engram_job = engram_native.enqueue_lookups(
                bindings,
                counts,
                status_host=self._engram_io_status.host_view,
                clear_tail=False,
            )
            if not skip_publish:
                _publish_engram_epoch[(1,)](
                    self._engram_io_status.device_view,
                    self._engram_epochs[0:1],
                    epoch,
                )
'''
PACKAGED = ['claude_prepare_engram_fault_inject.py', 'claude_engram_fault_inject.py',
            'claude_install_engram_fault_inject.py', 'Dockerfile.claude-engram-fault-inject',
            'claude_classify_fault_inject.py', 'claude_test_engram_fault_inject.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


ROUTER_LOCK = 'router-release.lock.json'
ROUTER_RECEIPT = 'receipts/router-release-build-receipt.json'
ROUTER_KIND = 'router-stage-release-candidate'


def base_identity():
    """The router candidate's identity, from its frozen lock and retained build receipts only."""
    lock_bytes = (root / ROUTER_LOCK).read_bytes()
    router = json.loads(lock_bytes)
    receipt = json.loads((root / ROUTER_RECEIPT).read_text())
    image, lock_sha = receipt['image_id'], sha(lock_bytes)
    if receipt['lock_sha256'] != lock_sha:
        raise RuntimeError('Router build receipt names a different router lock')
    build = root / 'receipts' / Path(receipt['directory']).name
    if (build / 'BUILD-OK').read_text().strip() != image:
        raise RuntimeError('Router build gates did not pass for this image')
    inspect = json.loads((build / 'image-inspect.json').read_text())[0]
    labels = inspect['Config']['Labels']
    if (inspect['Id'].removeprefix('sha256:') != image
            or labels.get('local-inference.ds41.diagnostic.kind') != ROUTER_KIND
            or labels.get('local-inference.ds41.diagnostic.lock.sha256') != lock_sha
            or labels.get('vllm.source-tree') != router['trees']['vllm']
            or labels.get('b12x.source-tree') != router['trees']['b12x']):
        raise RuntimeError('Router image labels do not match its lock')
    if router['before']['vllm'] != router['after']['vllm']:
        raise RuntimeError('Router candidate changed vLLM; the Engram model preimage is not established')
    repair = json.loads((root / 'engram-repair.lock.json').read_text())
    model = router['after']['vllm'][MODEL_TARGET.removeprefix('vllm/')]['sha256']
    if (model != repair['targets'][MODEL_TARGET]['output_sha256']
            or sha((root / 'engram-repair.lock.json').read_bytes()) != router['base_lock_sha256']):
        raise RuntimeError('Router candidate model.py is not the Engram repair output')
    return {'base_image_id': image, 'base_kind': ROUTER_KIND, 'base_lock': ROUTER_LOCK,
            'base_lock_sha256': lock_sha, 'base_trees': router['trees'],
            'base_parent_image_id': router['base_image_id'],
            'base_router_target': {'path': router['target'], 'sha256': router['output_sha256']},
            'model_sha256': model}


def vllm_tree_after(new_model, helper):
    """Exact vLLM source tree of the diagnostic image: the router candidate's vLLM sources (reconstructed
    from the pinned upstream plus the ring and Engram repair targets, proven equal to the router lock's
    vLLM manifest and tree) with model.py replaced and the helper added."""
    import prepare as runtime_prepare
    from contracts import git_tree, manifest
    router = json.loads((root / ROUTER_LOCK).read_text())
    ring = json.loads((root / 'ring-fix.lock.json').read_text())
    progress = json.loads((root / 'engram-progress-vllm.lock.json').read_text())
    files = runtime_prepare.patched_vllm(runtime_prepare.VLLM)
    changes = [(ring['source_path'], 'ring-fix-sparse_mla.py', ring['output_sha256'])]
    changes += [(path, target['source'], target['output_sha256']) for path, target in progress['targets'].items()]
    for path, source, expected in changes:
        data = (root / source).read_bytes()
        if sha(data) != expected:
            raise RuntimeError('vLLM reconstruction input drift: ' + source)
        files[path] = (files[path][0], data)
    if manifest(files) != router['after']['vllm'] or git_tree(files) != router['trees']['vllm']:
        raise RuntimeError('Router candidate vLLM reconstruction differs')
    model, helper_path = MODEL_TARGET.removeprefix('vllm/'), HELPER_TARGET.removeprefix('vllm/')
    if helper_path in files:
        raise RuntimeError('Helper path already exists in the candidate')
    files[model] = (files[model][0], new_model.encode())
    files[helper_path] = ('100644', helper.encode())
    return git_tree(files)


DOCKERFILE = """# DIAGNOSTIC ONLY. Never a serving candidate: one-shot Engram publication skip
# for the cross-rank fail-closed test (behavior, not numerics).
# Base: router stage-release fence candidate {short} (Engram repair lineage).
# Generated by claude_prepare_engram_fault_inject.py from the lock's base identity.
FROM {base_image_id}
ARG INJECT_LOCK
ARG INJECT_CACHE
COPY {copy} /opt/ds41-engram-fault-inject/
RUN /opt/venv/bin/python /opt/ds41-engram-fault-inject/claude_install_engram_fault_inject.py
# B12X bytes are unchanged, so the candidate's B12X compile cache, native loader
# build and preparation selection file are shared (inherited B12X_* and XDG
# paths). A boot at a new KV capacity adds capacity-keyed selection records to
# that shared file; snapshot it before and after. vLLM and Inductor output goes
# to a separate directory.
ENV VLLM_CACHE_ROOT=/cache/jit/${{INJECT_CACHE}}/vllm \\
    TORCHINDUCTOR_CACHE_DIR=/cache/jit/${{INJECT_CACHE}}/inductor
LABEL local-inference.ds41.diagnostic.kind="engram-fault-inject" \\
    local-inference.ds41.diagnostic.lock.sha256="${{INJECT_LOCK}}" \\
    local-inference.ds41.diagnostic.base-image="{base_image_id}" \\
    local-inference.ds41.diagnostic.base-kind="{base_kind}" \\
    local-inference.ds41.diagnostic.base-lock.sha256="{base_lock_sha256}" \\
    local-inference.ds41.diagnostic.vllm-cache="${{INJECT_CACHE}}" \\
    vllm.source-tree="{vllm_tree}" \\
    b12x.source-tree="{b12x_tree}" \\
    local-inference.status="diagnostic-only-not-qualified" \\
    org.opencontainers.image.title="DS4.1 router fence candidate with one-shot Engram publication-skip diagnostic"
"""


def render_dockerfile(lock):
    copy = ' '.join(['claude-engram-fault-inject.lock.json', *[n for n in PACKAGED if n != 'Dockerfile.claude-engram-fault-inject'],
                     'Dockerfile.claude-engram-fault-inject', OUTPUT, 'claude-engram-fault-inject.patch'])
    return DOCKERFILE.format(short=lock['base_image_id'][:12], copy=copy, vllm_tree=lock['trees']['vllm'],
                             b12x_tree=lock['trees']['b12x'], **{k: lock[k] for k in
                                                                ('base_image_id', 'base_kind', 'base_lock_sha256')})


def compose():
    base = base_identity()
    old = (root / MODEL_SOURCE).read_bytes()
    if sha(old) != base['model_sha256']:
        raise RuntimeError('engram-progress-model.py is not the installed candidate model.py')
    text = old.decode()
    for anchor, replacement in ((IMPORT_OLD, IMPORT_NEW), (JOB_OLD, JOB_NEW)):
        if text.count(anchor) != 1:
            raise RuntimeError('anchor changed: ' + anchor[:60])
        text = text.replace(anchor, replacement)
    compile(text, MODEL_TARGET, 'exec')
    return base, old.decode(), text


def main():
    base, old, new = compose()
    (root / OUTPUT).write_text(new)
    helper = (root / HELPER_SOURCE).read_text()
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/' + MODEL_TARGET, tofile='b/' + MODEL_TARGET))
    patch += ''.join(difflib.unified_diff([], helper.splitlines(True),
                                          fromfile='/dev/null', tofile='b/' + HELPER_TARGET))
    (root / 'claude-engram-fault-inject.patch').write_text(patch)
    trees = {'vllm': vllm_tree_after(new, helper), 'b12x': base['base_trees']['b12x']}
    lock = {
        'kind': 'engram-fault-inject-diagnostic',
        'status': 'diagnostic-only-not-built-not-qualified',
        'scope': 'cross-rank fail-closed Engram publication test only; not numerics, determinism, '
                 'performance or router-fence evidence',
        **{k: v for k, v in base.items() if k != 'model_sha256'},
        'trees': trees,
        'targets': {
            MODEL_TARGET: {'input_sha256': sha(old.encode()), 'output_sha256': sha(new.encode()),
                           'source': OUTPUT},
            HELPER_TARGET: {'input_sha256': None, 'output_sha256': sha(helper.encode()),
                            'source': HELPER_SOURCE},
        },
    }
    (root / 'Dockerfile.claude-engram-fault-inject').write_text(render_dockerfile(lock))
    lock['inputs'] = {name: sha((root / name).read_bytes())
                      for name in PACKAGED + [OUTPUT, 'claude-engram-fault-inject.patch']}
    (root / 'claude-engram-fault-inject.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + ['claude-engram-fault-inject.lock.json']
    (root / 'claude-engram-fault-inject.ignore').write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('ENGRAM-FAULT-INJECT-PREPARED', sha((root / 'claude-engram-fault-inject.lock.json').read_bytes()))


if __name__ == '__main__':
    main()
