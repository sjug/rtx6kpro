"""Compose the proposed main-thread Engram enqueue integration, not deploy it."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAILCLOSED_LOCK = '4693deea006f58687f62a83e7a8227fc83f0067c643941376a0ff5d1b76d3deb'
NATIVE_LOCK = '334eedf4692c8be65832bfef7277aa67a4416dac8d46142f0ea5783684e4779e'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Integration anchor count changed: ' + old[:80])
    return text.replace(old, new, 1)


PUBLISH = '''@triton.jit(do_not_specialize=["epoch"])
def _publish_engram_epoch(status, ready, epoch):
    """Publish only after the preceding CPU-only host read succeeded.

    The callback writes status before it returns. This kernel is queued after
    that callback on the same stream, before the model forward is submitted.
    A failed read deliberately leaves ready unchanged; the bounded consumer
    wait and cross-rank sticky fault check refuse that boot's outputs.
    """
    complete = tl.load(status, volatile=True) == 1
    tl.store(ready, epoch.to(tl.int64), mask=complete)


'''

START = '''    def _start_engram_job(self, bindings, counts) -> None:
        self._engram_epoch += 1
        epoch = self._engram_epoch
        self._engram_epochs[1:2].fill_(epoch)
        ready = torch.cuda.Event()
        ready.record()
        stream = self._engram_stream()
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


def compose():
    if digest((ROOT / 'claude-ple-batch.lock.json').read_bytes()) != NATIVE_LOCK:
        raise RuntimeError('Reviewed native API lock changed')
    lock_bytes = (ROOT / 'claude-engram-failclosed.lock.json').read_bytes()
    if digest(lock_bytes) != FAILCLOSED_LOCK:
        raise RuntimeError('Reviewed fail-closed lock changed')
    lock = json.loads(lock_bytes)
    outputs = []
    for path, target in lock['targets'].items():
        original = (ROOT / target['source']).read_bytes()
        if digest(original) != target['output_sha256']:
            raise RuntimeError('Changed fail-closed source: ' + path)
        text = original.decode()
        if path.endswith('/common/engram.py'):
            text = replace_once(text, 'class Engram(nn.Module):\n', PUBLISH + 'class Engram(nn.Module):\n')
        elif path.endswith('/nvidia/model.py'):
            text = replace_once(text, '            job.result()\n',
                '            # A skipped callback or hung disk read must fail instead\n'
                '            # of hanging the engine forever. Buffer ownership stays\n'
                '            # with the native pending job if this wait expires.\n'
                '            job.result(timeout=60.0)\n')
            text = replace_once(text, 'from concurrent.futures import Future, ThreadPoolExecutor\n', '')
            text = replace_once(text,
                'from ..common.engram import Engram, EngramLayout, NgramHashState\n',
                'from ..common.engram import Engram, EngramLayout, NgramHashState, _publish_engram_epoch\n')
            text = replace_once(text, '            self._engram_job: Future | None = None\n',
                                '            self._engram_job = None\n')
            text = replace_once(text,
                '            if self._engram_overlap:\n                self._engram_epochs = torch.zeros(\n',
                '            if self._engram_overlap:\n'
                '                from b12x.sequence._shared.disk_table import MappedHostAllocation\n\n'
                '                self._engram_io_status = MappedHostAllocation(\n'
                '                    (1,), torch.int64, caps.device\n'
                '                )\n'
                '                self._engram_io_status.host_view.zero_()\n'
                '                self._engram_epochs = torch.zeros(\n')
            begin = text.index('    def _start_engram_job(self, bindings, counts) -> None:\n')
            end = text.index('    def prepare_disk_engram(', begin)
            text = text[:begin] + START + text[end:]
        outputs.append((path, original, text.encode()))
    return digest(lock_bytes), outputs


def main():
    parent, outputs = compose()
    targets = {}
    patch = ''
    for path, old, new in outputs:
        name = 'engram-progress-' + Path(path).name
        (ROOT / name).write_bytes(new)
        targets[path] = {'source': name, 'failclosed_sha256': digest(old),
                         'output_sha256': digest(new)}
        if old != new:
            patch += ''.join(difflib.unified_diff(old.decode().splitlines(True),
                            new.decode().splitlines(True), fromfile='a/' + path,
                            tofile='b/' + path))
    (ROOT / 'engram-progress-vllm.patch').write_text(patch)
    record = {'kind': 'engram-prequeued-host-io-vllm-proposal',
              'status': 'unbuilt-unqualified-pending-native-api-review',
              'failclosed_lock_sha256': parent, 'targets': targets,
              'native_lock_sha256': NATIVE_LOCK,
              'patch_sha256': digest(patch.encode()),
              'preparer_sha256': digest(Path(__file__).read_bytes())}
    (ROOT / 'engram-progress-vllm.lock.json').write_text(json.dumps(record, indent=2) + '\n')
    print('ENGRAM-PROGRESS-VLLM-PREPARED', record['patch_sha256'])


if __name__ == '__main__':
    main()
