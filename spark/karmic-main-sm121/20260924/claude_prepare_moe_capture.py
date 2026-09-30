"""Compose the DIAGNOSTIC MoE-capture layer over the MoE-seam image 14abc3e66a45.

v3 of the helper: v2 seams plus stash-only routed-call inputs (gate logits,
top-k weights and ids, input before/after) and a device-latched capture of
the first routed call that disagrees with its (rows, runner) reference. The
v2 kit (claude-moe-seams*) stays byte-identical so 14abc3e66a45 remains
reproducible; this layer replaces only the installed helper.

Original v2 description follows.


Review artifact only: no build, no node, no GPU. The v1 activation-digest kit
(claude_act_trace.py, claude-act-trace.lock.json, ...) stays byte-identical so
image 0b5c65a81b25 remains reproducible from its lock. This layer replaces
exactly one installed file, the helper module
vllm/models/deepseek_v4_1/claude_act_trace.py, with the v2 helper
(claude-moe-seams-act_trace.py). The model.py that imports it, every other
vLLM file and all B12X bytes are unchanged.

The seams shadow two callsites of MoERunner.forward on armed runner instances.
They are only meaningful for the exact pinned runner source, so this preparer
reads the pinned vLLM blobs (read-only, from the existing ~/git/vllm
checkout), requires their sha256 to equal the runtime lock's manifest for the
image's vLLM tree, checks the callsite structure by AST, and writes those
hashes into the lock; the installer refuses an image whose files differ.
"""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
VLLM_COMMIT = '1794dcf18454900263e0c66711af8ea4a1283ac1'
VLLM_CHECKOUT = Path(os.environ.get('CLAUDE_VLLM_CHECKOUT', Path.home() / 'git/vllm'))
BASE_AMENDMENT = 'moe-seams-14abc3e66a45-candidate-amendment.json'
V1_LOCK = 'claude-moe-seams.lock.json'
HELPER_TARGET = 'vllm/vllm/models/deepseek_v4_1/claude_act_trace.py'
HELPER_SOURCE = 'claude-moe-capture-act_trace.py'
RUNNER = 'vllm/model_executor/layers/fused_moe/runner/moe_runner.py'
PINNED = [RUNNER,
          'vllm/model_executor/layers/fused_moe/runner/shared_experts.py',
          'vllm/model_executor/layers/fused_moe/layer.py',
          'vllm/models/deepseek_v4/nvidia/model.py',
          'vllm/model_executor/layers/fused_moe/router/fused_moe_router.py']
LOCK = 'claude-moe-capture.lock.json'
PATCH = 'claude-moe-capture.patch'
PACKAGED = ['claude_prepare_moe_capture.py', HELPER_SOURCE, 'claude_install_moe_capture.py',
            'Dockerfile.claude-moe-capture', 'claude-moe-capture-act_analyze.py',
            'claude_test_moe_capture.py', 'claude_gate_moe_capture_cuda.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pinned_source(path):
    return subprocess.run(['git', '-C', str(VLLM_CHECKOUT), 'show', f'{VLLM_COMMIT}:{path}'],
                          check=True, capture_output=True).stdout


def _method(tree, cls, name):
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _self_calls(node, attr):
    return [c for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            and c.func.attr == attr and isinstance(c.func.value, ast.Name) and c.func.value.id == 'self']


def check_router(router_text):
    klass = _class(ast.parse(router_text), 'FusedMoERouter')
    select = next((n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'select_experts'), None)
    if select is None or any(ast.unparse(d) == 'abstractmethod' for d in select.decorator_list):
        raise RuntimeError('FusedMoERouter.select_experts is not a concrete method')
    if '__slots__' in router_text:
        raise RuntimeError('router objects may not accept instance attributes')


def check_callsites(runner_text, layer_text, model_text):
    """The seam contract, checked on the pinned sources; raises on any drift."""
    tree = ast.parse(runner_text)
    forward = _method(tree, 'MoERunner', 'forward')
    order = []
    for node in ast.walk(forward):
        if isinstance(node, ast.Call):
            func = node.func
            label = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            if label in ('_forward_entry', '_unpack', '_maybe_apply_routed_scale_to_output',
                         'apply_routed_output_transform', '_maybe_reduce_final_output'):
                order.append((node.lineno, label))
    labels = [label for _, label in sorted(order)]
    if labels != ['_forward_entry', '_unpack', '_maybe_apply_routed_scale_to_output',
                  'apply_routed_output_transform', '_maybe_reduce_final_output']:
        raise RuntimeError(f'MoERunner.forward callsite order drifted: {labels}')
    for attr in ('_forward_entry', '_maybe_reduce_final_output'):
        calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                 and c.func.attr == attr]
        if len(calls) != 1 or len(_self_calls(forward, attr)) != 1:
            raise RuntimeError(f'{attr} is not called exactly once, from MoERunner.forward')
    init = _method(tree, 'MoERunner', '__init__')
    if 'self._forward_entry = self._select_forward()' not in {ast.unparse(s) for s in init.body}:
        raise RuntimeError('MoERunner.__init__ no longer stores _forward_entry on the instance')
    if not any(isinstance(n, ast.FunctionDef) and n.name == '_maybe_reduce_final_output'
               for n in ast.walk(_class(tree, 'MoERunner'))):
        raise RuntimeError('_maybe_reduce_final_output is not a MoERunner method')
    factory = next(n for n in ast.parse(layer_text).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'FusedMoEFactory')
    runner_call = next(c for c in ast.walk(factory) if isinstance(c, ast.Call)
                       and ast.unparse(c.func) == 'runner_cls')
    scale = next(k for k in runner_call.keywords if k.arg == 'routed_scaling_factor')
    if ast.unparse(scale.value) != 'routed_scaling_factor if apply_routed_scale_to_output else 1.0':
        raise RuntimeError('runner routed_scaling_factor contract drifted')
    moe = _method(ast.parse(model_text), 'DeepseekV4MoE', '_init_fused_moe_experts')
    call = next(c for c in ast.walk(moe) if isinstance(c, ast.Call)
                and ast.unparse(c.func) == 'FusedMoEFactory')
    if any(k.arg == 'apply_routed_scale_to_output' for k in call.keywords):
        raise RuntimeError('DeepseekV4MoE now scales routed output in the runner')
    if any(k.arg == 'reduce_results' for k in call.keywords):
        raise RuntimeError('DeepseekV4MoE may defer its all-reduce out of the runner')
    impl = _method(tree, 'MoERunner', '_forward_impl')
    if [ast.unparse(s) for s in ast.walk(impl) if isinstance(s, ast.Assign)
            and ast.unparse(s.value) == 'self.gate(hidden_states)'] != ['router_logits, _ = self.gate(hidden_states)']:
        raise RuntimeError('MoERunner._forward_impl no longer calls its gate module exactly once')
    quant = _method(tree, 'MoERunner', '_apply_quant_method')
    if len([c for c in ast.walk(quant) if isinstance(c, ast.Call)
            and ast.unparse(c.func) == 'self.router.select_experts']) != 1:
        raise RuntimeError('MoERunner._apply_quant_method no longer calls router.select_experts once')
    if not any(k.arg == 'gate' and ast.unparse(k.value) == 'self.gate' for k in call.keywords):
        raise RuntimeError('DeepseekV4MoE no longer hands its gate to the runner')
    fused = _method(ast.parse(model_text), 'DeepseekV4MoE', '_forward_fused_moe')
    if not any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'self.experts' for c in ast.walk(fused)):
        raise RuntimeError('DeepseekV4MoE no longer calls its runner as a module')
    return labels


def _class(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def compose():
    amendment = json.loads((root / BASE_AMENDMENT).read_text())['candidate']
    v1_lock_bytes = (root / V1_LOCK).read_bytes()
    if sha(v1_lock_bytes) != amendment['diagnostic']['lock_sha256']:
        raise RuntimeError('v2 seam lock is not the one image 14abc3e66a45 was built from')
    v1 = json.loads(v1_lock_bytes)
    old_helper = (root / v1['targets'][HELPER_TARGET]['source']).read_bytes()
    if sha(old_helper) != v1['targets'][HELPER_TARGET]['output_sha256']:
        raise RuntimeError('v1 helper drifted from its lock')
    manifest = json.loads((root / 'runtime.lock.json').read_text())['sources']['vllm']['files']
    pins = {}
    texts = {}
    for path in PINNED:
        data = pinned_source(path)
        if sha(data) != manifest[path]['sha256']:
            raise RuntimeError('pinned blob differs from the image manifest: ' + path)
        pins['vllm/' + path] = sha(data)
        texts[path] = data.decode()
    check_callsites(texts[RUNNER], texts[PINNED[2]], texts[PINNED[3]])
    check_router(texts[PINNED[4]])
    new_helper = (root / HELPER_SOURCE).read_bytes()
    compile(new_helper, HELPER_TARGET, 'exec')
    return amendment['image_id'], sha(v1_lock_bytes), old_helper.decode(), new_helper.decode(), pins


def main():
    image, v1_lock, old, new, pins = compose()
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/' + HELPER_TARGET, tofile='b/' + HELPER_TARGET))
    (root / PATCH).write_text(patch)
    lock = {
        'kind': 'moe-capture-diagnostic',
        'status': 'diagnostic-only-not-built-not-qualified',
        'base_image_id': image,
        'base_diagnostic_lock_sha256': v1_lock,
        'vllm_commit': VLLM_COMMIT,
        'targets': {
            HELPER_TARGET: {'input_sha256': sha(old.encode()), 'output_sha256': sha(new.encode()),
                            'source': HELPER_SOURCE},
        },
        'verify_unchanged': pins,
        'inputs': {name: sha((root / name).read_bytes()) for name in PACKAGED + [PATCH]},
    }
    (root / LOCK).write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + [LOCK]
    (root / 'claude-moe-capture.ignore').write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('MOE-CAPTURE-PREPARED', sha((root / LOCK).read_bytes()))


if __name__ == '__main__':
    main()
