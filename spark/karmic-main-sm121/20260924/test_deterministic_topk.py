"""GPU regression: unchanged scores, deterministic index tie-break, graph replay.

Run with --stock to demonstrate the existing selector fails the contract.
Otherwise test the proposed repair after the actual production selector.
"""
import argparse
import json
import torch
from b12x.attention.dsa_indexer.tiled_topk import run_row_topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', action='store_true')
    args = parser.parse_args()
    if not args.stock:
        from deterministic_topk import repair_topk
    torch.manual_seed(391)
    cases = 0
    for width in (17, 511, 512, 513, 514, 4096, 32768, 524288):
        for kind in ('equal', 'bf16', 'overflow'):
            scores = torch.randn((3, width), device='cuda').bfloat16().float()
            if kind == 'equal':
                scores.fill_(0.5)
            elif kind == 'overflow':
                scores[:, :width // 2] = 3.0
            lengths = torch.tensor([width, max(0, width - 31), 0], device='cuda', dtype=torch.int32)
            masked = scores.masked_fill(torch.arange(width, device='cuda')[None] >= lengths[:, None], -torch.inf)
            positions = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :min(width, 512)]
            vals = masked.gather(1, positions)
            reference = torch.where(vals != -torch.inf, positions, -1)
            ref = torch.full((3, 512), -1, device='cuda', dtype=torch.int64)
            ref[:, :reference.shape[1]] = reference
            ref = ref.sort(dim=1).values
            values = torch.empty((3, 512), device='cuda')
            indices = torch.empty((3, 512), device='cuda', dtype=torch.int32)
            def call():
                run_row_topk(row_logits=scores, lengths=lengths, topk=512,
                             output_values=values, output_indices=indices)
                if not args.stock:
                    repair_topk(scores, lengths, values, indices)
            for repeat in range(10):
                call()
                actual = indices.long().sort(dim=1).values
                if not torch.equal(actual, ref):
                    raise RuntimeError(f'DETERMINISM-FAIL width={width} kind={kind} repeat={repeat}')
                valid = indices >= 0
                gathered = scores.gather(1, indices.long().clamp_min(0))
                if not torch.equal(values[valid], gathered[valid]):
                    raise RuntimeError('Selected score was modified')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call()
            for _ in range(3):
                indices.fill_(-99)
                values.fill_(999)
                graph.replay()
                if not torch.equal(indices.long().sort(dim=1).values, ref):
                    raise RuntimeError('Graph replay did not restore deterministic output')
            cases += 1
            print(json.dumps({'width': width, 'kind': kind, 'pass': True}), flush=True)
    if not args.stock:
        # The hierarchical candidate path emits mapped indices, not positions.
        width = 4096
        scores = torch.ones((33, width), device='cuda')
        lengths = torch.full((33,), width, device='cuda', dtype=torch.int32)
        gather = torch.arange(width - 1, -1, -1, device='cuda', dtype=torch.int32).repeat(33, 1)
        for k in (512, 1024):
            values = torch.empty((33, k), device='cuda')
            indices = torch.empty((33, k), device='cuda', dtype=torch.int32)
            run_row_topk(row_logits=scores, lengths=lengths, topk=k,
                         output_values=values, output_indices=indices, output_gather_table=gather)
            repair_topk(scores, lengths, values, indices, gather=gather)
            if not torch.equal(indices, gather[:, :k]):
                raise RuntimeError('Candidate position-to-index mapping failed')
            cases += 1
    print(f'DETERMINISTIC-TOPK-PASS cases={cases}', flush=True)


if __name__ == '__main__':
    main()
