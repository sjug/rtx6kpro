"""Bounded, isolated CUDA producer/consumer progress diagnostic.

Run in an idle GPU container, never beside serving. This is not a model test.
The consumer has a device timer so a stalled producer cannot spin indefinitely.
"""
import argparse
import ctypes
import json
import threading
import time

import torch
import triton
import triton.language as tl


@triton.jit
def consume(ready, failed, timeout_ns):
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                    dtype=tl.int64, is_pure=False, pack=1)
    waiting = tl.load(ready, volatile=True) == 0
    expired = tl.full((), 0, tl.int32)
    while waiting:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                      dtype=tl.int64, is_pure=False, pack=1)
        if now - start > timeout_ns:
            expired = 1
            waiting = False
        else:
            waiting = tl.load(ready, volatile=True) == 0
    tl.store(failed, expired)


@triton.jit
def publish(ready):
    tl.store(ready, 1)


@triton.jit
def publish_after(ready, delay_ns):
    start = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                    dtype=tl.int64, is_pure=False, pack=1)
    now = start
    while now - start < delay_ns:
        now = tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [],
                                      dtype=tl.int64, is_pure=False, pack=1)
    tl.store(ready, 1)


@triton.jit
def publish_mapped(ready, data, checksum):
    values = tl.load(data + tl.arange(0, 8192)).to(tl.int32)
    tl.store(checksum, tl.sum(values, 0))
    tl.store(ready, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=('no-mm', 'cold-mm', 'warm-mm', 'prequeued-cold-mm', 'host-callback-cold-mm'), required=True)
    a = p.parse_args()
    torch.cuda.set_device(0)
    ready = torch.ones(1, dtype=torch.int32, device='cuda')
    failed = torch.zeros_like(ready)
    x = torch.ones((385, 512), dtype=torch.bfloat16, device='cuda')
    weight = torch.ones((128, 512), dtype=torch.bfloat16, device='cuda')
    out = torch.empty((385, 128), dtype=torch.bfloat16, device='cuda')
    timeout_ns = 1_000_000_000
    # Compile and load only the tiny producer and consumer before the trial.
    consume[(1,)](ready, failed, timeout_ns)
    publish[(1,)](ready)
    if a.arm == 'prequeued-cold-mm':
        publish_after[(1,)](ready, 0)
    if a.arm == 'warm-mm':
        torch.mm(x, weight.T, out=out)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    callback_owner = None
    callback_state = None
    mapped_owner = None
    checksum = None
    if a.arm == 'host-callback-cold-mm':
        from cuda.bindings import runtime as cudart
        from b12x.sequence._shared.disk_table import MappedHostAllocation
        class CallbackState(ctypes.Structure):
            _fields_ = [('delay_ms', ctypes.c_uint32), ('completed', ctypes.c_uint32),
                        ('destination', ctypes.c_void_p), ('bytes', ctypes.c_uint64)]
        mapped_owner = MappedHostAllocation((8192,), torch.uint8, torch.device('cuda', 0))
        mapped_owner.host_view.zero_()
        checksum = torch.zeros(1, dtype=torch.int32, device='cuda')
        publish_mapped[(1,)](ready, mapped_owner.device_view, checksum)
        torch.cuda.synchronize()
        callback_state = CallbackState(100, 0, mapped_owner.host_view.data_ptr(), 8192)
        callback_owner = ctypes.CDLL('/progress_host_callback.so')
        callback_pointer = ctypes.cast(callback_owner.progress_host_callback, ctypes.c_void_p).value
    initialized = threading.Event()
    go = threading.Event()
    timings = {}
    errors = []

    def producer():
        try:
            torch.cuda.set_device(0)
            with torch.cuda.stream(stream):
                initialized.set()
                if not go.wait(30):
                    raise RuntimeError('No trial start')
                time.sleep(0.1)
                start = time.monotonic_ns()
                publish[(1,)](ready)
                timings['publish_host_ms'] = (time.monotonic_ns() - start) / 1e6
        except BaseException as exc:
            errors.append(repr(exc))
            initialized.set()

    worker = None
    if a.arm not in ('prequeued-cold-mm', 'host-callback-cold-mm'):
        worker = threading.Thread(target=producer, daemon=True)
        worker.start()
        if not initialized.wait(30) or errors:
            raise RuntimeError(str(errors) or 'Producer initialization stalled')
    ready.zero_()
    failed.zero_()
    torch.cuda.synchronize()
    start = time.monotonic_ns()
    if a.arm == 'prequeued-cold-mm':
        # This only models pre-submission of a delayed producer. It does not
        # implement the CPU disk-read handoff required by the real repair.
        with torch.cuda.stream(stream):
            publish_after[(1,)](ready, 100_000_000)
    elif a.arm == 'host-callback-cold-mm':
        # The CPU-only callback sleeps, writes the mapped payload and records
        # completion. Real I/O still needs ownership and error propagation.
        result = cudart.cudaLaunchHostFunc(stream.cuda_stream, callback_pointer,
                                         ctypes.addressof(callback_state))
        if result[0] != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError('cudaLaunchHostFunc failed: ' + str(result))
        with torch.cuda.stream(stream):
            publish_mapped[(1,)](ready, mapped_owner.device_view, checksum)
    consume[(1,)](ready, failed, timeout_ns)
    go.set()
    if a.arm != 'no-mm':
        before = time.monotonic_ns()
        torch.mm(x, weight.T, out=out)
        timings['mm_host_ms'] = (time.monotonic_ns() - before) / 1e6
    if worker is not None:
        worker.join(30)
    if (worker is not None and worker.is_alive()) or errors:
        raise RuntimeError(str(errors) or 'Producer did not return')
    torch.cuda.synchronize()
    timings['total_ms'] = (time.monotonic_ns() - start) / 1e6
    result = {'arm': a.arm, 'timed_out': bool(failed.item()),
              'published': int(ready.item()), 'timings': timings,
              'scope': 'isolated progress probe, not DS4.1 numerical qualification'}
    if a.arm != 'no-mm':
        result['mm_correct'] = bool(torch.all(out == 512).item())
    if callback_state is not None:
        result['cpu_callback_completed'] = callback_state.completed
        result['mapped_payload_checksum'] = int(checksum.item())
        result['mapped_payload_correct'] = result['mapped_payload_checksum'] == 32 * sum(range(256))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
