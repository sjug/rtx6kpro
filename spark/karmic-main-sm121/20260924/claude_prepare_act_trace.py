"""Compose the DIAGNOSTIC activation-digest derivative of the Engram repair.

Review artifact only: no build, no node, no GPU. The candidate's production
sources stay untouched; this derives a separate diagnostic image recipe
(Dockerfile.claude-act-trace) over the built candidate ee03505df7a3,
changing exactly one installed file (the DS4.1 NVIDIA model: one import and
one install call) and adding one helper module (claude_act_trace.py). B12X bytes are unchanged, so the candidate's
B12X compile cache and native loader build are reused.

Base identity: engram-progress-model.py must hash to the repair lock's
installed model.py output, and the base image is the candidate recorded in
receipts/engram-repair-build-receipt.json.
"""
import difflib
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
MODEL_TARGET = 'vllm/vllm/models/deepseek_v4_1/nvidia/model.py'
HELPER_TARGET = 'vllm/vllm/models/deepseek_v4_1/claude_act_trace.py'
MODEL_SOURCE = 'engram-progress-model.py'
HELPER_SOURCE = 'claude_act_trace.py'
OUTPUT = 'claude-act-trace-model.py'

IMPORT_OLD = 'from b12x.sequence import engram as engram_native\n'
IMPORT_NEW = (IMPORT_OLD
              + '# DIAGNOSTIC ONLY: activation digests of eager prefill steps.\n'
              + 'import vllm.models.deepseek_v4_1.claude_act_trace as _act_trace\n')
JOB_OLD = '''        else:
            self._mtp_hidden_buffer = None

    def _engram_stream(self) -> torch.cuda.Stream:
'''
JOB_NEW = '''        else:
            self._mtp_hidden_buffer = None
        # DIAGNOSTIC ONLY: two step hooks; module hooks exist only while the
        # /cache/ds41-act-trace.json trigger is present.
        _act_trace.install(self)

    def _engram_stream(self) -> torch.cuda.Stream:
'''
PACKAGED = ['claude_prepare_act_trace.py', 'claude_act_trace.py', 'claude_install_act_trace.py',
            'Dockerfile.claude-act-trace', 'claude_act_analyze.py', 'claude_test_act_trace.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def base_identity():
    repair = json.loads((root / 'engram-repair.lock.json').read_text())
    receipt = json.loads((root / 'receipts/engram-repair-build-receipt.json').read_text())
    expected = repair['targets'][MODEL_TARGET]['output_sha256']
    return receipt['image_id'], sha((root / 'engram-repair.lock.json').read_bytes()), expected


def compose():
    image, repair_lock, expected = base_identity()
    old = (root / MODEL_SOURCE).read_bytes()
    if sha(old) != expected:
        raise RuntimeError('engram-progress-model.py is not the installed candidate model.py')
    text = old.decode()
    for anchor, replacement in ((IMPORT_OLD, IMPORT_NEW), (JOB_OLD, JOB_NEW)):
        if text.count(anchor) != 1:
            raise RuntimeError('anchor changed: ' + anchor[:60])
        text = text.replace(anchor, replacement)
    compile(text, MODEL_TARGET, 'exec')
    return image, repair_lock, old.decode(), text


def main():
    image, repair_lock, old, new = compose()
    (root / OUTPUT).write_text(new)
    helper = (root / HELPER_SOURCE).read_text()
    patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                         fromfile='a/' + MODEL_TARGET, tofile='b/' + MODEL_TARGET))
    patch += ''.join(difflib.unified_diff([], helper.splitlines(True),
                                          fromfile='/dev/null', tofile='b/' + HELPER_TARGET))
    (root / 'claude-act-trace.patch').write_text(patch)
    lock = {
        'kind': 'activation-digest-diagnostic',
        'status': 'diagnostic-only-not-built-not-qualified',
        'base_image_id': image,
        'base_repair_lock_sha256': repair_lock,
        'targets': {
            MODEL_TARGET: {'input_sha256': sha(old.encode()), 'output_sha256': sha(new.encode()),
                           'source': OUTPUT},
            HELPER_TARGET: {'input_sha256': None, 'output_sha256': sha(helper.encode()),
                            'source': HELPER_SOURCE},
        },
        'inputs': {name: sha((root / name).read_bytes())
                   for name in PACKAGED + [OUTPUT, 'claude-act-trace.patch']},
    }
    (root / 'claude-act-trace.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + ['claude-act-trace.lock.json']
    (root / 'claude-act-trace.ignore').write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('ACT-TRACE-PREPARED', sha((root / 'claude-act-trace.lock.json').read_bytes()))


if __name__ == '__main__':
    main()
