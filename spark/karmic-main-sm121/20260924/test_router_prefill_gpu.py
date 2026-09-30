"""Arithmetic and repeatability smoke for the fenced router, not the rare-fault gate."""
import hashlib
import json
from pathlib import Path

import torch
from b12x.gemm.bf16_gemv import _prefill


def main():
    if not __debug__ or torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('Expected GB10 and assertions enabled')
    lock = json.loads(Path('/opt/ds41-router-release/router-release.lock.json').read_text())
    if hashlib.sha256(Path(_prefill.__file__).read_bytes()).hexdigest() != lock['output_sha256']:
        raise RuntimeError('Wrong loaded router source')
    torch.manual_seed(20260925)
    weight = torch.randn(384, 5120, device='cuda', dtype=torch.bfloat16) * .02
    launch = _prefill.compile_prefill(0, 8192, 384, 5120, 'float32')
    cases = 0
    for rows in (128, 254, 256, 258):
        x = torch.randn(rows, 5120, device='cuda', dtype=torch.bfloat16) * .02
        output = torch.empty(rows, 384, device='cuda', dtype=torch.float32)
        launch(x, weight, output)
        first = output.clone()
        reference = (x.cpu().double() @ weight.cpu().double().T).float()
        torch.testing.assert_close(first.cpu(), reference, rtol=1e-5, atol=1e-6)
        for _ in range(100):
            output.fill_(float('nan'))
            launch(x, weight, output)
            if not torch.equal(first, output):
                raise RuntimeError('Router smoke changed output at rows=' + str(rows))
        cases += 1
    torch.cuda.synchronize()
    print('ROUTER-PREFILL-GPU-PASS cases=' + str(cases), flush=True)


if __name__ == '__main__':
    main()
