#!/usr/bin/env python3
"""DIAGNOSTIC gate for the v3 MoE capture on a real CUDA device (no model, no weights).

Exercises what the CPU tests cannot: the device-side latch, the asynchronous
CudaTransport (side-stream pinned copies, writer thread), the pinned capture
fetch and the saved file, plus capture generations across a re-arm while an
earlier generation's latch record is still queued.

Run inside the capture image on an idle GPU (or anywhere with torch+CUDA):
    /opt/venv/bin/python /opt/ds41-moe-capture/claude_gate_moe_capture_cuda.py
It prints CLAUDE-MOE-CAPTURE-CUDA-GATE-PASS and exits 0, or raises.
`--device cpu` runs the same assertions through SyncTransport (logic check only).

Phases:
 1. basic: generation 1 arms capture on layers.2; clean steps 1-3; step 4
    injects a fault into layers.2's routed output; the saved file must hold
    the exact bytes of that call's input, logits, top-k weights and ids and
    routed output, with epoch 4, and the receipts must show the latch word
    clear at epochs 1-3, set at epoch 4, and status lines fetching then saved.
 2. delayed writer and re-arm: the transport's records are held; generation
    2 latches on a fault; the trigger is rewritten (generation 3) before the
    held records are released. Generation 3 must not become ready from the
    generation-2 record; receipts must show generation 2 discarded and
    latched-discarded, and no generation-2 file. A fault under generation 3
    must then save its own file with the right epoch.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
INSTALLED = Path('/opt/jovian-judgement/vllm/vllm/models/deepseek_v4_1/claude_act_trace.py')
WIDTH, EXPERTS, TOPK, LAYERS = 64, 8, 2, 4


def load_helper(path):
    spec = importlib.util.spec_from_file_location('claude_capture_gate_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(WIDTH, EXPERTS, bias=False)

    def forward(self, x):
        return self.lin(x).float(), None


class Router:
    def select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        weights, ids = torch.topk(torch.softmax(router_logits, -1), TOPK, dim=-1)
        return weights, ids.to(torch.int32)


class Runner(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = Gate()
        self.router = Router()
        self.shared_lin = nn.Linear(WIDTH, WIDTH)
        self.experts = nn.Parameter(torch.randn(EXPERTS, WIDTH, WIDTH) * 0.05)
        self.fault = False
        self.seen = None
        self._forward_entry = self._entry

    def _entry(self, hidden_states, router_logits, shared_experts_input, input_ids, layer_name, dim, dtype):
        logits, _ = self.gate(hidden_states.float())
        weights, ids = self.router.select_experts(hidden_states=hidden_states, router_logits=logits)
        routed = torch.einsum('tk,tkij,tj->ti', weights, self.experts[ids.long()],
                              hidden_states.float()).to(torch.bfloat16)
        if self.fault:
            routed[3, 5] += 1.0
            self.fault = False
        self.seen = {'input': hidden_states, 'logits': logits, 'topk_weights': weights,
                     'topk_ids': ids, 'routed': routed}
        return self.shared_lin(shared_experts_input.float()).to(torch.bfloat16), routed

    def _maybe_reduce_final_output(self, states, trunc_size, output_is_reduced=None):
        return states * 2

    def forward(self, hidden_states, router_logits):
        shared, fused = self._forward_entry(hidden_states, router_logits, hidden_states, None, 'x', 0, None)
        return self._maybe_reduce_final_output(shared + fused, None, False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(WIDTH, WIDTH)
        self.ffn = nn.Module()
        self.ffn.experts = Runner()

    def forward(self, h):
        a = torch.tanh(self.attn(h.float())).to(torch.bfloat16)
        return (self.ffn.experts(hidden_states=a, router_logits=a) * 0.01 + h).to(torch.bfloat16), a


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.embed = nn.Embedding(1000, WIDTH)
        self.layers = nn.ModuleList(Block() for _ in range(LAYERS))

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        h = self.embed(input_ids).to(torch.bfloat16)
        for layer in self.layers:
            h, _ = layer(h)
        return h


class Held:
    """Holds step records between the transport and the tracer until released."""

    def __init__(self, tracer):
        self.tracer, self.hold, self.queue = tracer, False, []

    def __call__(self, meta):
        if self.hold:
            self.queue.append(meta)
        else:
            self.tracer._sink(meta)

    def release(self):
        self.hold = False
        queued, self.queue = self.queue, []
        for meta in queued:
            self.tracer._sink(meta)


def check(condition, message):
    if not condition:
        raise AssertionError('CLAUDE-MOE-CAPTURE-CUDA-GATE-FAIL: ' + message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--helper', type=Path,
                        default=INSTALLED if INSTALLED.exists() else HERE / 'claude-moe-capture-act_trace.py')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--rows', type=int, default=48)
    parser.add_argument('--timeout', type=float, default=60.0)
    args = parser.parse_args()
    helper = load_helper(args.helper)
    device = torch.device(args.device)
    if args.device == 'cuda':
        check(torch.cuda.is_available(), 'CUDA is not available')
    with tempfile.TemporaryDirectory(prefix='claude-capture-gate-') as tmp:
        trigger, out = Path(tmp) / 'trigger.json', Path(tmp) / 'out'
        model = Toy().to(device)
        helper._INSTALLED.clear()
        transport = helper.SyncTransport(None, slots=4, capacity=512) if args.device == 'cpu' else None
        tracer = helper.install(model, transport=transport, trigger=str(trigger), out_dir=str(out),
                                rank=1, node='gate')
        held = Held(tracer)
        if transport is not None:
            transport.sink = held
        state = {'epoch': 0}

        def step(tokens=None):
            state['epoch'] += 1
            model._engram_epoch = state['epoch']
            ids = (torch.arange(args.rows, device=device) * 7 + 3) % 1000 if tokens is None else tokens
            with torch.inference_mode():
                model(ids, torch.arange(args.rows, device=device), None)
            if tracer.transport is not None and args.device == 'cuda':
                tracer.transport.sink = held      # CudaTransport is created lazily at the first traced step

        def arm(token):
            trigger.write_text(json.dumps({
                'arm': token, 'min_tokens': 8, 'max_tokens': 512, 'post': [],
                'moe': ['layers.*.ffn.experts'],
                'capture': {'rows': [args.rows], 'moe': ['layers.2.ffn.experts']}}))
            os.utime(trigger, ns=(time.time_ns(), time.time_ns()))

        def drain():
            if args.device == 'cuda':
                torch.cuda.synchronize()
                deadline = time.monotonic() + args.timeout
                while not tracer.transport.pending.empty() and time.monotonic() < deadline:
                    time.sleep(0.05)
                time.sleep(0.2)                      # the writer finishes its current item

        def lines():
            records = []
            for file in sorted(out.glob('*.jsonl')):
                records += [json.loads(l) for l in file.read_text().splitlines() if l.strip()]
            return records

        def wait_for(path):
            deadline = time.monotonic() + args.timeout
            while not path.exists() and time.monotonic() < deadline:
                step()
                drain()
            check(path.exists(), f'{path.name} was not saved')

        runner = model.layers[2].ffn.experts
        pid = os.getpid()

        # Phase 1: basic latch, fetch and save.
        arm('gate-basic-0001')
        for _ in range(3):
            step()
        runner.fault = True
        step()                                       # epoch 4, faulty
        expected = {k: v.detach().to('cpu').clone() for k, v in runner.seen.items()}
        drain()
        first = out / f'gate-{pid}-g1-capture.pt'
        wait_for(first)
        saved = torch.load(first)
        check(saved['generation'] == 1 and saved['meta']['epoch'] == 4 and saved['meta']['rows'] == args.rows,
              f'phase 1 meta {saved["meta"]}')
        check(saved['runner'] == 'layers.2.ffn.experts', 'phase 1 runner')
        for key, tensor in expected.items():
            got = saved['tensors'][key]
            check(got.dtype == tensor.dtype and got.shape == tensor.shape, f'phase 1 {key} layout')
            check(torch.equal(got.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8)),
                  f'phase 1 {key} bytes')
        check(saved['meta']['digest0'] != saved['meta']['ref0'] or saved['meta']['digest1'] != saved['meta']['ref1'],
              'phase 1 digest equals reference')
        records = [r for r in lines() if r.get('arm') == 'gate-basic-0001']
        words = {r['epoch']: r['hashes'][r['names'].index('<capture>')]
                 for r in records if r.get('type') == 'step' and '<capture>' in r['names']}
        check([words[e][0] for e in (1, 2, 3, 4)] == [0, 0, 0, 1] and words[4][1] == 4, f'latch words {words}')
        states = [r['state'] for r in records if r.get('type') == 'capture' and r['generation'] == 1]
        check(states == ['fetching', 'saved'], f'phase 1 status lines {states}')

        # Phase 2: delayed writer, re-arm while a latch record is queued.
        arm('gate-rearm-0002')
        step()
        step()
        drain()
        held.hold = True
        runner.fault = True
        step()                                       # generation 2 latches on the device
        drain()
        check(len(held.queue) >= 1, 'no held record')
        arm('gate-rearm-0003')
        step()                                       # disarms generation 2, arms generation 3
        second = tracer.capture
        check(second is not None and second.generation == 3, 'generation 3 not armed')
        drain()
        held.release()
        drain()
        check(not second.ready and second.state in ('armed', 'allocated'),
              f'generation 3 marked from a stale record: {second.state}')
        check(not (out / f'gate-{pid}-g2-capture.pt').exists(), 'generation 2 was saved after disarm')
        status2 = [r['state'] for r in lines() if r.get('type') == 'capture' and r.get('generation') == 2]
        check('discarded' in status2 and 'latched-discarded' in status2, f'generation 2 status {status2}')
        step()
        runner.fault = True
        step()
        fault_epoch = state['epoch']
        expected = {k: v.detach().to('cpu').clone() for k, v in runner.seen.items()}
        drain()
        third = out / f'gate-{pid}-g3-capture.pt'
        wait_for(third)
        saved = torch.load(third)
        check(saved['generation'] == 3 and saved['meta']['epoch'] == fault_epoch, f'phase 2 meta {saved["meta"]}')
        check(torch.equal(saved['tensors']['routed'].view(torch.uint8), expected['routed'].view(torch.uint8)),
              'phase 2 routed bytes')
        print(json.dumps({'device': args.device, 'helper': str(args.helper), 'phase1_epoch': 4,
                          'phase2_epoch': fault_epoch, 'generation2_status': status2}), flush=True)
    print('CLAUDE-MOE-CAPTURE-CUDA-GATE-PASS', flush=True)


if __name__ == '__main__':
    main()
