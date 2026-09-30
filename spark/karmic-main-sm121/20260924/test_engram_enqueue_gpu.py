"""Real io_uring -> mapped rows -> lookup -> captured consumer integration gate.

Uses the upstream small-table oracle with the proposed enqueue API. The first
consumer is followed by a cold torch.mm, the isolated progress reproducer's
trigger. No CPU wait is inserted between submission and that forward.
"""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile

import torch
from b12x.preparation import PreparedCall, PreparationSession
from b12x.sequence import engram
from b12x.sequence._shared.disk_table import MappedHostAllocation
from vllm.models.deepseek_v4_1.common.engram import _publish_engram_epoch, _wait_engram_rows


@torch.inference_mode()
def run(resident_scales, folder):
    device = torch.device('cuda', 0)
    tokens = 385
    geometry = engram.build_geometry(base_table_size=101, compressed_vocab_size=32)
    plan = engram.plan(
        engram.Caps(device=device, max_tokens=tokens, max_seqs=3, max_requests=4,
                    vocab_size=32, layer_id=1, tp_rank=1, tp_size=3),
        token_map=list(range(32)), geometry=geometry,
        invocation={'operation': 'lookup', 'compact_rows': True,
                    'resident_scales': resident_scales})
    rows = plan.query.table_rows
    sources = []
    for table_number in range(2):
        weights = (torch.arange(rows * 256, dtype=torch.float32) + table_number * 5).remainder(15).sub(7)
        weights = weights.reshape(rows, 256).to(torch.float8_e4m3fn)
        scales = (torch.arange(rows * 8) + table_number).remainder(5).add(125).to(torch.uint8).reshape(rows, 8)
        paths = [folder / f'{resident_scales}-{table_number}-{kind}.bin' for kind in ('weights', 'scales')]
        for path, data in zip(paths, (weights, scales)):
            path.write_bytes(bytes(4093) + data.view(torch.uint8).numpy().tobytes())
        sources.append((weights, scales, paths))
    ids = [torch.full((tokens, 24), -1, dtype=torch.int64, device=device) for _ in range(2)]
    count = torch.tensor([tokens], dtype=torch.int32, device=device)
    outs = [torch.full((tokens, 6144), 73, dtype=torch.bfloat16, device=device) for _ in range(2)]
    consumed = [torch.empty_like(out) for out in outs]
    ready = torch.zeros(1, dtype=torch.int64, device=device)
    expected_epoch = torch.zeros_like(ready)
    failed = torch.zeros_like(ready)
    flag = torch.zeros(1, dtype=torch.bfloat16, device=device)
    status = MappedHostAllocation((1,), torch.int64, device)
    status.host_view.zero_()
    side = torch.cuda.Stream()
    x = torch.ones((385, 512), dtype=torch.bfloat16, device=device)
    weight = torch.ones((128, 512), dtype=torch.bfloat16, device=device)
    mmout = torch.empty((385, 128), dtype=torch.bfloat16, device=device)
    receipts = []
    states = []
    with ExitStack() as resources:
        resources.callback(status.close)
        session = resources.enter_context(PreparationSession(device=device, autotune=False, compile_workers=20))

        def prepare_call(state):
            states.append(state)
            return PreparedCall(run=lambda: None)

        resources.enter_context(session.prepare((plan.request(name='engram-prequeued-gate', prepare_call=prepare_call),)))
        tables = []
        for weights, scales, paths in sources:
            table = engram.DiskTable(states[-1], queue_depth=4, resident_scales=resident_scales)
            resources.callback(table.close)
            for scale, path in zip((False, True), paths):
                table.add_shard(0, str(path), 4093, scale=scale)
            tables.append(table)
        bindings = [engram.bind_lookup(plan, disk_table=table, hash_ids=table_ids,
                                      num_tokens=count, out=out)
                    for table, table_ids, out in zip(tables, ids, outs)]
        start = tables[0].state.shard_start
        end = min(tables[0].state.shard_end, rows)
        cases = torch.tensor([start, end - 1, start, start - 1, end, -1, start + 3, start + 17],
                             device=device).repeat(tokens * 3).reshape(tokens, 24)
        # Warm only publication and wait kernels, never the cold GEMM.
        _publish_engram_epoch[(1,)](status.device_view, ready, 0)
        _wait_engram_rows[(1,)](ready, expected_epoch, failed, 1000, flag)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        resources.callback(graph.reset)
        with session.capture(), torch.cuda.graph(graph):
            _wait_engram_rows[(1,)](ready, expected_epoch, failed, 1000, flag)
            for src, dst in zip(outs, consumed):
                torch.mul(src, 2, out=dst)
        torch.cuda.synchronize()
        for epoch, live in enumerate((385, 1, 9, 0, 384, 385, 2), 1):
            for k in range(2):
                ids[k].copy_(cases.roll(epoch + k, dims=1))
                outs[k].fill_(73)
            count.fill_(live)
            expected_epoch.fill_(epoch)
            inputs_ready = torch.cuda.Event()
            inputs_ready.record()
            with torch.cuda.stream(side):
                side.wait_event(inputs_ready)
                job = engram.enqueue_lookups(bindings, [live, live], status_host=status.host_view,
                                             clear_tail=True)
                _publish_engram_epoch[(1,)](status.device_view, ready, epoch)
            graph.replay()
            if epoch == 1:
                torch.mm(x, weight.T, out=mmout)
            # Host observation begins only after the whole consumer is queued.
            job.result(timeout=30)
            torch.cuda.synchronize()
            if int(flag.item()) or int(failed.item()) or int(ready.item()) != epoch:
                raise RuntimeError(f'Publication failed at {epoch=} {resident_scales=}')
            if epoch == 1 and not bool(torch.all(mmout == 512).item()):
                raise RuntimeError('GEMM result changed')
            for k, (weights, scales, _) in enumerate(sources):
                expected = torch.zeros((tokens, 24, 256), dtype=torch.bfloat16)
                cpu_ids = ids[k].cpu()
                valid = (cpu_ids[:live] >= start) & (cpu_ids[:live] < end)
                selected = cpu_ids[:live][valid]
                expected[:live][valid] = (weights.float()[selected] * scales[selected].view(
                    torch.float8_e8m0fnu).float().repeat_interleave(32, dim=1)).to(torch.bfloat16)
                torch.testing.assert_close(outs[k].cpu(), expected.flatten(1), rtol=0, atol=0)
                torch.testing.assert_close(consumed[k].cpu(), expected.flatten(1) * 2, rtol=0, atol=0)
            receipts.append({'resident_scales': resident_scales, 'epoch': epoch, 'live': live,
                             'captured_consumer': True, 'rows_exact': True})
        # A real native short-read must not publish ready or silently succeed.
        sources[0][2][0].write_bytes(b'')
        epoch += 1
        count.fill_(1)
        expected_epoch.fill_(epoch)
        inputs_ready = torch.cuda.Event()
        inputs_ready.record()
        with torch.cuda.stream(side):
            side.wait_event(inputs_ready)
            job = engram.enqueue_lookups(bindings, [1, 1], status_host=status.host_view, clear_tail=True)
            _publish_engram_epoch[(1,)](status.device_view, ready, epoch)
        graph.replay()
        try:
            job.result(timeout=30)
        except RuntimeError as error:
            if 'read' not in str(error).lower():
                raise
        else:
            raise RuntimeError('Injected short read was silently accepted')
        torch.cuda.synchronize()
        if int(flag.item()) != 1 or int(ready.item()) == epoch:
            raise RuntimeError('Failed read was published to the consumer')
        receipts.append({'resident_scales': resident_scales, 'injected_short_read': 'rejected'})
    return receipts


def main():
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This gate requires GB10')
    with tempfile.TemporaryDirectory(prefix='engram-enqueue-') as temp:
        results = run(False, Path(temp)) + run(True, Path(temp))
    print(json.dumps({'cases': results, 'count': len(results)}), flush=True)
    print('ENGRAM-ENQUEUE-GPU-PASS 16', flush=True)


if __name__ == '__main__':
    main()
