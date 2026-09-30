"""Diagnostic transport gate, not a regression test for model nondeterminism."""
import json
import threading
import time

import torch
from vllm.models.deepseek_v4_1 import claude_act_trace as trace


def main():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('Requires the GB10 GPU')
    torch.manual_seed(219)
    tensors = [torch.randn(m, 6144, device='cuda', dtype=torch.bfloat16)
               for m in (1, 128, 255, 256)]
    tensors += [tensors[-1].t()]
    expected = [trace.digest(x.cpu()).tolist() for x in tensors]
    records = []
    lock = threading.Lock()

    def sink(record):
        with lock:
            records.append(record)

    transport = trace.CudaTransport(sink, torch.device('cuda'), slots=2, capacity=8)
    for repeat in range(40):
        deadline = time.monotonic() + 30
        while True:
            slot = next((i for i, free in enumerate(transport.free) if free), None)
            if slot is not None:
                break
            if time.monotonic() > deadline:
                raise RuntimeError('Transport did not release a slot')
            time.sleep(.005)  # gate only; never on the model forward path
        transport.free[slot] = False
        transport.begin(slot)
        for i, tensor in enumerate(tensors):
            transport.acc[slot, i].copy_(trace.digest(tensor))
        transport.submit(slot, len(tensors), {'seq': repeat})
    deadline = time.monotonic() + 30
    while len(records) < 40 and time.monotonic() < deadline:
        time.sleep(.005)
    if len(records) != 40 or sorted(r['seq'] for r in records) != list(range(40)):
        raise RuntimeError('Missing or duplicate transport records')
    for record in records:
        if record['hashes'] != expected or record['gpu_ms'] < 0:
            raise RuntimeError('CUDA digest or asynchronous slot-reuse mismatch')
    print(json.dumps({'records': len(records), 'tensors_per_record': len(tensors)}), flush=True)
    print('ACTIVATION-TRACE-GPU-PASS', flush=True)


if __name__ == '__main__':
    main()
