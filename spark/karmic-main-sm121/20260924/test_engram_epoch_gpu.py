"""Execute the proposed publication/wait kernels on GB10, eager and captured.

This isolates actual generated kernel bodies without importing the model's
unrelated dependencies. It does not replace native-reader or TP4 model gates.
"""
import ast
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from b12x.sequence._shared.disk_table import MappedHostAllocation


def main():
    source = Path('/gate/engram-progress-engram.py')
    text = source.read_text()
    tree = ast.parse(text)
    names = {'_wait_engram_rows', '_publish_engram_epoch'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in names]
    if len(selected) != 2:
        raise RuntimeError('Expected exactly two production kernel definitions')
    namespace = {'triton': triton, 'tl': tl}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    wait = namespace['_wait_engram_rows']
    publish = namespace['_publish_engram_epoch']
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This gate requires GB10 SM121')
    torch.cuda.set_device(0)
    ready = torch.zeros(1, dtype=torch.int64, device='cuda')
    expected = torch.zeros_like(ready)
    failed = torch.zeros_like(ready)
    flag = torch.zeros(1, dtype=torch.bfloat16, device='cuda')
    status = MappedHostAllocation((1,), torch.int64, torch.device('cuda', 0))
    side = torch.cuda.Stream()
    status.host_view.fill_(1)
    publish[(1,)](status.device_view, ready, 1)
    wait[(1,)](ready, expected, failed, 2, flag)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wait[(1,)](ready, expected, failed, 2, flag)
    torch.cuda.synchronize()
    receipts = []
    # The last case proves the flag is overwritten on success even though the
    # compatibility failed word remains sticky from a preceding timeout.
    cases = [('success', 1, 0), ('io-error', -1, 1), ('not-ready', 0, 1),
             ('success-after-fault', 1, 0)]
    epoch = 1
    for captured in (False, True):
        for name, host_status, want_flag in cases:
            epoch += 1
            torch.cuda.synchronize()
            status.host_view.fill_(host_status)
            old_ready = int(ready.item())
            expected.fill_(epoch)
            flag.fill_(17)
            with torch.cuda.stream(side):
                publish[(1,)](status.device_view, ready, epoch)
            # The producer is already submitted. Do not replace the consumer
            # with a host synchronization or stream wait in this gate.
            if captured:
                graph.replay()
            else:
                wait[(1,)](ready, expected, failed, 2, flag)
            torch.cuda.synchronize()
            actual_flag = int(flag.item())
            actual_ready = int(ready.item())
            want_ready = epoch if host_status == 1 else old_ready
            if actual_flag != want_flag or actual_ready != want_ready:
                raise RuntimeError(f'{captured=} {name}: {actual_flag=} {actual_ready=}')
            receipts.append({'captured': captured, 'case': name, 'epoch': epoch,
                             'flag': actual_flag, 'ready': actual_ready,
                             'failed_word': int(failed.item())})
    print(json.dumps({'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'cases': receipts, 'count': len(receipts),
                      'scope': 'production epoch kernels only'}), flush=True)
    print('ENGRAM-EPOCH-GPU-PASS 8', flush=True)


if __name__ == '__main__':
    main()
