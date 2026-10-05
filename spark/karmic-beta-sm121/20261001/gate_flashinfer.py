#!/usr/bin/env python3
"""Numerically qualify rebuilt FlashInfer AOT kernels on SM121, without JIT fallback."""
import json
import os
from pathlib import Path

os.environ['FLASHINFER_DISABLE_JIT'] = '1'


def main():
    import torch
    import flashinfer
    import flashinfer_jit_cache
    import cutlass
    from verify_compiler import validate_cutlass
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('FlashInfer gate requires SM121')
    validate_cutlass(cutlass)
    for module in (flashinfer, flashinfer_jit_cache):
        if not Path(module.__file__).resolve().is_relative_to('/opt/venv'):
            raise RuntimeError(f'Wrong FlashInfer import: {module.__file__}')
    if flashinfer.__git_commit__ != 'dbd6238c6655b98195fdf77f04bba6facf5a38a4':
        raise RuntimeError('Wrong rebuilt FlashInfer source')
    if flashinfer_jit_cache.__git_version__ != flashinfer.__git_commit__:
        raise RuntimeError('Wrong rebuilt JIT-cache source')
    from flashinfer.jit import core
    from aot_guard import enforce_aot
    cache = Path(flashinfer_jit_cache.__file__).resolve().parent
    used = enforce_aot(core, cache)
    torch.manual_seed(20261001)
    count = 0
    for dtype in (torch.float16, torch.bfloat16):
        k = torch.randn(64, 8, 128, device='cuda', dtype=dtype)
        v = torch.randn_like(k)
        q = torch.randn(8, 128, device='cuda', dtype=dtype)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float().unsqueeze(1), k.float().transpose(0, 1), v.float().transpose(0, 1)).squeeze(1)
        workspace = torch.empty(16 * 1024 * 1024, device='cuda', dtype=torch.uint8)
        decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout='NHD')
        decode.plan(torch.tensor([0, 4], dtype=torch.int32),
                    torch.arange(4, dtype=torch.int32),
                    torch.tensor([16], dtype=torch.int32),
                    8, 8, 128, 16, q_data_type=dtype, kv_data_type=dtype)
        actual = decode.run(q.unsqueeze(0), (k.reshape(4, 16, 8, 128),
                                            v.reshape(4, 16, 8, 128))).squeeze(0)
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
        q = torch.randn(16, 8, 128, device='cuda', dtype=dtype)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(0, 1), k.float().transpose(0, 1), v.float().transpose(0, 1)).transpose(0, 1)
        prefill = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout='NHD', backend='fa2')
        prefill.plan(torch.tensor([0, 16], dtype=torch.int32),
                     torch.tensor([0, 64], dtype=torch.int32),
                     8, 8, 128, causal=False, q_data_type=dtype, kv_data_type=dtype)
        actual = prefill.run(q, k, v)
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
        x = torch.randn(16, 256, device='cuda', dtype=dtype)
        weight = torch.randn(256, device='cuda', dtype=dtype)
        expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()
        actual = flashinfer.norm.rmsnorm(x, weight, eps=1e-6)
        torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
        torch.cuda.synchronize()
        count += 3
    # Exercise vLLM's actual unseeded CUDA sampler path used by these models.
    from vllm import envs
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
    if not envs.VLLM_USE_FLASHINFER_SAMPLER:
        raise RuntimeError('Production FlashInfer sampler is disabled')
    sampler = TopKTopPSampler()
    if sampler.forward.__func__ is not TopKTopPSampler.forward_cuda:
        raise RuntimeError('vLLM did not select its FlashInfer sampler')
    def deny_native(*args, **kwargs):
        raise RuntimeError('Sampler unexpectedly fell back to native Torch')
    sampler.forward_native = deny_native
    # 1024 fits one sampler pass; the served vocabularies (DS4 Vision, GLM, Qwen) need
    # the multi-pass loop, so their kept tokens sit in late blocks of the vocabulary.
    for batch, vocab, high, second, tolerance in ((8192, 1024, 0, 1, 0.025),
                                                  (4096, 129280, 129279, 43093, 0.03),
                                                  (4096, 154880, 154879, 51626, 0.03),
                                                  (4096, 248320, 248319, 82773, 0.03)):
        probabilities = torch.full((vocab,), 0.03 / (vocab - 2), device='cuda')
        probabilities[high], probabilities[second] = 0.65, 0.32
        logits = probabilities.log().expand(batch, -1).contiguous()
        top_k = torch.full((batch,), 2, device='cuda', dtype=torch.int32)
        top_p = torch.full((batch,), 0.95, device='cuda')
        for k, p in ((None, top_p), (top_k, None), (top_k, top_p)):
            samples, _ = sampler(logits.clone(), {}, k, p)
            if samples.shape != (batch,) or not ((samples == high) | (samples == second)).all().item():
                raise RuntimeError(f'Sampler violated the Torch-derived top-k/top-p support: vocab={vocab}')
            frequency = (samples == high).float().mean().item()
            if abs(frequency - 0.65 / 0.97) > tolerance:
                raise RuntimeError(f'Sampler distribution mismatch: vocab={vocab} frequency={frequency}')
            torch.cuda.synchronize()
            count += 1
        del probabilities, logits
        torch.cuda.empty_cache()
    if not any('sampling' in row['name'] for row in used):
        raise RuntimeError('No rebuilt sampling AOT kernel was exercised')
    # The attention kernels must have loaded from the newly installed wheel;
    # disabling JIT prevents recompilation from masking a broken packaged cache.
    cache = Path(flashinfer_jit_cache.__file__).resolve().parent
    mapped = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if '.so' in line and str(cache) in line})
    if not mapped:
        raise RuntimeError('No rebuilt FlashInfer JIT-cache libraries were loaded')
    print(json.dumps({'cases': count, 'loaded_cache_libraries': mapped, 'jit_fallback': False, 'aot_kernels': used}))
    for path in ('build', 'inputs', 'flashinfer-wheels'):
        directory = Path('/opt/karmic-beta-refresh') / path
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f'Build inputs leaked into the serving image: {directory}')
    print(f'FLASHINFER-KERNEL-GATE-PASS cases={count}', flush=True)


if __name__ == '__main__':
    if not __debug__:
        raise RuntimeError('FlashInfer gates require assertions')
    main()
