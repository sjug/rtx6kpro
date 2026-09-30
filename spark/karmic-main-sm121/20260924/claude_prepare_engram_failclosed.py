"""Compose the proposed Engram fail-closed repair from pinned vLLM blobs.

Review artifact only: nothing is installed, no image is built, no node is
contacted and ~/git/vllm is only read with `git show`. Rerunning reproduces
byte-identical outputs.

Mechanism (see CLAUDE-TUNING-WINNERS.md, "Engram fail-closed repair"):
  1. Each Engram layer's staged rows gain one leading row. The existing wait
     kernel always writes this rank's timeout state (1 or 0) into element 0
     of that row, then the EXISTING row all-reduce carries it, so every TP
     rank sees the sum of all ranks' flags. No new collective.
  2. After the all-reduce, plain torch ops fold a nonzero (or NaN) sum into a
     monotonic per-model fault epoch (the step's expected epoch). It is never
     reset: any fault since process start stays visible.
  3. After sampling, on the main stream, the runner asks the model state for
     a snapshot [fault_epoch, step_epoch]; AsyncOutput copies it on the
     existing output-copy stream and get_output raises after the existing
     copy_event.synchronize() whenever fault_epoch > 0, mirroring the EP
     all2all fault seam. Faults in steps without a checked output (warmup,
     dummy runs) therefore fail the next checked output instead of being
     lost.
  4. The staged-rows alias check is host-only and runs in the eager
     invalidate_disk_output() seam that every staging path calls, never in
     forward, so forward stays torch.compile(fullgraph=True) traceable.
Overlap, the 5000 ms timeout and B12X are unchanged. The only Triton edit
is the flag store in _wait_engram_rows.
"""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
VLLM, COMMIT = Path.home() / 'git/vllm', '1794dcf18454900263e0c66711af8ea4a1283ac1'

ENGRAM = 'vllm/models/deepseek_v4_1/common/engram.py'
MODEL = 'vllm/models/deepseek_v4_1/nvidia/model.py'
STATE = 'vllm/models/deepseek_v4_1/nvidia/model_state.py'
ASYNC = 'vllm/v1/worker/gpu/async_utils.py'
RUNNER = 'vllm/v1/worker/gpu/model_runner.py'

WAIT_OLD = '''@triton.jit
def _wait_engram_rows(ready, expected, failed, timeout_ms):
    """Spin until the side-stream lookup publishes this step's epoch."""
    target = tl.load(expected)
    timeout_ns = timeout_ms.to(tl.int64) * 1_000_000
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    waiting = (tl.load(ready, volatile=True) < target).to(tl.int32)
    while waiting != 0:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
        if now - start > timeout_ns:
            tl.store(failed, 1)
            waiting = 0
        else:
            waiting = (tl.load(ready, volatile=True) < target).to(tl.int32)
'''
WAIT_NEW = '''@triton.jit
def _wait_engram_rows(ready, expected, failed, timeout_ms, flag):
    """Spin until the side-stream lookup publishes this step's epoch.

    Always writes this rank's timeout state (1 or 0) to ``flag``, element 0
    of the row all-reduce input, so no earlier step's flag survives into
    this step's reduction.
    """
    target = tl.load(expected)
    timeout_ns = timeout_ms.to(tl.int64) * 1_000_000
    start = tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
    )
    waiting = (tl.load(ready, volatile=True) < target).to(tl.int32)
    timed_out = waiting * 0
    while waiting != 0:
        now = tl.inline_asm_elementwise(
            "mov.u64 $0, %globaltimer;", "=l", [], dtype=tl.int64, is_pure=False, pack=1
        )
        if now - start > timeout_ns:
            tl.store(failed, 1)
            timed_out = 1
            waiting = 0
        else:
            waiting = (tl.load(ready, volatile=True) < target).to(tl.int32)
    tl.store(flag, timed_out.to(tl.bfloat16))
'''

EDITS = {
    ENGRAM: [
        (WAIT_OLD, WAIT_NEW),
        ('''        # (ready, expected, failed) epochs when disk rows arrive asynchronously.
        self.overlap_epochs: tuple[torch.Tensor, ...] | None = None
''', '''        # (ready, expected, failed) epochs when disk rows arrive asynchronously.
        self.overlap_epochs: tuple[torch.Tensor, ...] | None = None
        # Model-wide monotonic epoch of the last step whose rows were consumed
        # unpublished on any TP rank; set together with overlap_epochs.
        self.overlap_fault_epoch: torch.Tensor | None = None
'''),
        ('''        self.register_buffer(
            "staged_rows",
            torch.empty(
                caps.max_tokens,
                6144,
                dtype=torch.bfloat16,
                device=caps.device,
            ),
            persistent=False,
        )
''', '''        self.register_buffer(
            "engram_io_rows",
            torch.empty(
                caps.max_tokens + 1,
                6144,
                dtype=torch.bfloat16,
                device=caps.device,
            ),
            persistent=False,
        )
        # Row 0 carries the per-step timeout flag through the row all-reduce.
        # Lookups bind only the token rows after it, so no lookup (including a
        # late one after a timeout) can write the flag.
        self.engram_io_rows[0].zero_()
        self.staged_rows = self.engram_io_rows[1:]
'''),
        ('''    def invalidate_disk_output(self, *, clear=False):
''', '''    def _check_fault_io_alias(self) -> None:
        """Host-only check that lookups still write inside engram_io_rows.

        Called from invalidate_disk_output(), which every staging path
        (disk, batched disk, dummy) runs eagerly before each forward.
        Never called from forward: data_ptr() is not traceable.
        """
        if self.overlap_epochs is None:
            return
        if self.overlap_fault_epoch is None:
            raise RuntimeError("Engram overlap requires a fault epoch buffer")
        if self.staged_rows.data_ptr() != self.engram_io_rows[1:].data_ptr():
            raise RuntimeError("Engram staged rows no longer alias the flag buffer")

    def invalidate_disk_output(self, *, clear=False):
        self._check_fault_io_alias()
'''),
        ('''        if self.overlap_epochs is not None:
            _wait_engram_rows[(1,)](*self.overlap_epochs, 5000)
        rows = tensor_model_parallel_all_reduce(self.staged_rows[: hash_ids.shape[0]])
''', '''        if self.overlap_epochs is not None:
            io = self.engram_io_rows
            fault = self.overlap_fault_epoch
            _wait_engram_rows[(1,)](*self.overlap_epochs, 5000, io)
            reduced = tensor_model_parallel_all_reduce(io[: hash_ids.shape[0] + 1])
            # Nonzero or NaN sum: some TP rank consumed unpublished rows.
            raised = reduced[0, :1] != 0
            fault.copy_(
                torch.where(raised, torch.maximum(fault, self.overlap_epochs[1]), fault)
            )
            rows = reduced[1:]
        else:
            rows = tensor_model_parallel_all_reduce(
                self.staged_rows[: hash_ids.shape[0]]
            )
'''),
    ],
    MODEL: [
        ('''                epochs = tuple(self._engram_epochs[i : i + 1] for i in range(3))
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    if getattr(layer, "engram", None) is not None:
                        layer.engram.overlap_epochs = epochs
''', '''                epochs = tuple(self._engram_epochs[i : i + 1] for i in range(3))
                self._engram_fault_epoch = torch.zeros(
                    1, dtype=torch.int64, device=caps.device
                )
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    if getattr(layer, "engram", None) is not None:
                        layer.engram.overlap_epochs = epochs
                        layer.engram.overlap_fault_epoch = self._engram_fault_epoch
'''),
        ('''    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
''', '''    def snapshot_engram_fault(self) -> torch.Tensor | None:
        """Snapshot the Engram fault state on the current stream.

        Returns a new int64 [fault_epoch, step_epoch]. fault_epoch is the
        latest epoch whose rows any TP rank consumed unpublished (0 = none,
        never reset); the runner calls this after sampling, so the snapshot
        follows this step's forward in stream order.
        """
        fault = getattr(self, "_engram_fault_epoch", None)
        if fault is None:
            return None
        return torch.cat((fault, self._engram_epochs[1:2]))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
'''),
    ],
    STATE: [
        ('''        for model in self.disk_engram_models:
            model.prepare_dummy_engram(num_tokens)
        return model_inputs
''', '''        for model in self.disk_engram_models:
            model.prepare_dummy_engram(num_tokens)
        return model_inputs

    def step_fault_snapshot(self) -> torch.Tensor | None:
        """Per-step Engram fault record for the async output path, or None."""
        snapshots = [
            snapshot
            for model in self.disk_engram_models
            if (snapshot := model.snapshot_engram_fault()) is not None
        ]
        if not snapshots:
            return None
        return snapshots[0] if len(snapshots) == 1 else torch.cat(snapshots)
'''),
    ],
    ASYNC: [
        ('''if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch
''', '''if TYPE_CHECKING:
    from vllm.v1.worker.gpu.input_batch import InputBatch


def _check_model_fault(values: list[int]) -> None:
    """Fail closed when a model reports any fault since process start.

    ``values`` holds [fault_epoch, step_epoch] pairs; fault_epoch is
    monotonic and never reset, so a fault in a step whose output was never
    checked (warmup, dummy run) fails the next checked output instead of
    being lost.
    """
    for i in range(0, len(values), 2):
        fault_epoch, step_epoch = values[i : i + 2]
        if fault_epoch > 0:
            where = "this step" if fault_epoch == step_epoch else "an earlier step"
            raise RuntimeError(
                f"Model reported a fault in {where} (fault epoch {fault_epoch}, "
                f"step epoch {step_epoch}): Engram rows were consumed before "
                "publication on at least one tensor-parallel rank. Refusing to "
                "emit outputs."
            )
'''),
        ('''        num_verified_draft_tokens: torch.Tensor | None = None,
    ):
''', '''        num_verified_draft_tokens: torch.Tensor | None = None,
        model_fault: torch.Tensor | None = None,
    ):
'''),
        ('''        self._has_fault: torch.Tensor | None = None
''', '''        self._has_fault: torch.Tensor | None = None
        # Keep the device snapshot alive until the copy stream has read it.
        self.model_fault = model_fault
        self._model_fault: torch.Tensor | None = None
'''),
        ('''            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)
''', '''            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            if model_fault is not None:
                self._model_fault = model_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)
'''),
        ('''    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk)''', '''    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()
        if self._model_fault is not None:
            _check_model_fault(self._model_fault.tolist())

        # NOTE(woosuk)'''),
    ],
    RUNNER: [
        ('''                check_ep_fault=self.check_ep_fault,
''', '''                check_ep_fault=self.check_ep_fault,
                model_fault=self._model_step_fault(),
''', 2),
        ('''    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
''', '''    def _model_step_fault(self) -> torch.Tensor | None:
        """Snapshot a model-reported step fault on the main stream, if any.

        Called after sampling and before the next step is prepared; the
        async output copies the snapshot with this step's results.
        """
        snapshot = getattr(self.model_state, "step_fault_snapshot", None)
        if snapshot is None:
            return None
        with torch.cuda.stream(self.main_stream):
            return snapshot()

    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
'''),
    ],
}
PACKAGED = ['claude_prepare_engram_failclosed.py', 'claude_test_engram_failclosed.py',
            'claude_test_engram_failclosed_torch.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def output_name(path):
    return 'claude-engram-failclosed-' + Path(path).name


def compose():
    lock_files = json.loads((root / 'runtime.lock.json').read_text())['sources']['vllm']['files']
    composed = []
    for path, edits in EDITS.items():
        old = subprocess.check_output(['git', '-C', str(VLLM), 'show', f'{COMMIT}:{path}']).decode()
        if lock_files[path]['sha256'] != sha(old.encode()):
            raise RuntimeError('Pinned blob differs from runtime.lock.json: ' + path)
        new = old
        for edit in edits:
            anchor, replacement, count = (*edit, 1)[:3]
            if new.count(anchor) != count:
                raise RuntimeError(f'{path}: anchor count {new.count(anchor)} != {count}: {anchor[:50]!r}')
            new = new.replace(anchor, replacement)
        compile(new, path, 'exec')
        composed.append((path, old, new))
    return composed


def main():
    composed = compose()
    patch = []
    for path, old, new in composed:
        (root / output_name(path)).write_text(new)
        patch.append(''.join(difflib.unified_diff(
            old.splitlines(True), new.splitlines(True), fromfile='a/' + path, tofile='b/' + path)))
    (root / 'claude-engram-failclosed.patch').write_text(''.join(patch))
    lock = {
        'kind': 'engram-fail-closed-proposal',
        'status': 'proposed-for-review-not-built-not-applied',
        'vllm_commit': COMMIT,
        'base': 'ring-fix candidate e7b273407022 (these five files equal the pinned blobs there)',
        'targets': {path: {'input_sha256': sha(old.encode()), 'output_sha256': sha(new.encode()),
                           'source': output_name(path)} for path, old, new in composed},
        'inputs': {name: sha((root / name).read_bytes())
                   for name in PACKAGED + ['claude-engram-failclosed.patch']
                   + [output_name(p) for p, _, _ in composed]},
    }
    (root / 'claude-engram-failclosed.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    print('ENGRAM-FAILCLOSED-PREPARED', sha((root / 'claude-engram-failclosed.lock.json').read_bytes()))


if __name__ == '__main__':
    main()
