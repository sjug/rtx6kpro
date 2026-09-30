"""CPU unit tests for claude_dense_regression.py (no GPU, no b12x import).

The layout tests exercise the harness readers against a CPU port of the pinned
packing expressions; `PinnedSourceTests` then checks, through `git show` on the
existing ~/git/b12x checkout at pin a7d7d29b, that the pinned source still
contains exactly those expressions and the cpasync contract the harness relies
on. That is source verification, not execution; the GPU run verifies the real
consumed planes.
"""
import importlib.util
import os
import subprocess
import unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("claude_dense_regression", HERE / "claude_dense_regression.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

B12X_PIN = "a7d7d29b2ef8869086e0ceaa787321f17544e3c9"
B12X_CHECKOUT = Path(os.environ.get("B12X_CHECKOUT", Path.home() / "git" / "b12x"))


def pinned(path: str) -> str:
    return subprocess.check_output(["git", "-C", str(B12X_CHECKOUT), "show", f"{B12X_PIN}:{path}"], text=True)


class Obj:
    pass


def fake_case(m, k, n, seed=0):
    """Consumed-plane fakes in the pinned layouts (Kphys = align_up(k, 128))."""
    g = torch.Generator().manual_seed(seed)
    kp = mod.physical_k(k)
    xq = torch.zeros((m, kp), dtype=torch.float8_e4m3fn)
    xq[:, :k] = (torch.randn(m, k, generator=g) * 0.5).to(torch.float8_e4m3fn)
    sx = torch.randint(120, 130, (m, kp // 32), generator=g, dtype=torch.uint8)
    w = (torch.randn(n, k, generator=g) * 0.3).to(torch.float8_e4m3fn)
    s = torch.randint(118, 131, (n // 32, k // 32), generator=g, dtype=torch.uint8)
    wq = torch.zeros((n, kp), dtype=torch.float8_e4m3fn); wq[:, :k] = w
    s_phys = torch.full((n // 32, kp // 32), 127, dtype=torch.uint8); s_phys[:, : k // 32] = s
    sw = mod.mirror_expand_block_scales_cpu(s_phys, n, kp)[0]
    b = Obj(); b.x_q = mod.fake_rows(xq, sx)
    p = Obj(); p.weight = mod.fake_rows(wq, sw)
    return b, p, w, s, xq, sx, wq, sw


class HarnessTests(unittest.TestCase):
    def test_enumeration_matches_combined_receipt(self):
        identity, winners = mod.enumerate_winners(HERE / "receipts" / "combined-tuning.json", turbo=True)
        self.assertEqual(len(winners), 152)
        shapes = {(k, n) for k, n, _, _ in winners}
        self.assertEqual(shapes, {(576, 5120), (1280, 4096), (1280, 8192), (5120, 512), (5120, 1152), (5120, 1792), (6144, 25600), (15360, 5120)})
        caps = {c for _, _, c, _ in winners}
        self.assertTrue({1, 2, 3, 4, 5, 6, 7, 8, 384, 512}.issubset(caps))
        self.assertTrue(any(a["split_k_slices"] == 4 and c <= 8 for _, _, c, a in winners))
        self.assertTrue(any(a["split_k_slices"] == 2 for *_, a in winners))
        self.assertTrue(any(a["tile_k"] == 64 for *_, a in winners))
        self.assertTrue(all(a["load_path"] == "tma" for *_, a in winners))

    def test_turbo_off_changes_keys(self):
        with self.assertRaisesRegex(RuntimeError, "Dense receipt coverage mismatch"):
            mod.enumerate_winners(HERE / "receipts" / "combined-tuning.json", turbo=False)

    def test_default_compile_workers_policy(self):
        self.assertEqual(mod.COMPILE_WORKERS_DEFAULT, 20)

    def test_sampling_spans_edges_interior_and_boundaries(self):
        r = mod.sample_indices(8192, (64, 128, 128))
        self.assertEqual((r[0], r[-1]), (0, 8191)); self.assertIn(4096, r)
        self.assertTrue({63, 64}.issubset(r)); self.assertEqual(r, sorted(set(r)))
        self.assertLessEqual(len(r), 3 + 2 * mod.BOUNDARY_BUDGET)
        c = mod.sample_indices(25600, (64, 128, 128))
        self.assertTrue({0, 12800, 25599}.issubset(c)); self.assertTrue(all(0 <= i < 25600 for i in c))
        self.assertEqual(mod.sample_indices(1, (64, 128, 128)), [0])
        self.assertEqual(mod.sample_indices(129, (64, 128, 128)), [0, 63, 64, 127, 128])
        s = mod.sampling_for(129, 25600, {"tile_m": 64, "tile_n": 128})
        self.assertEqual(s["tiles"], [64, 128, 128]); self.assertEqual(s["rows"], [0, 63, 64, 127, 128])

    def test_scale_mma_reader_matches_pinned_packing_mirror(self):
        b, p, w, s, xq, sx, wq, sw = fake_case(300, 576, 256)
        self.assertEqual({nm: tuple(getattr(b.x_q, nm).shape) for nm in ("values", "scale_rows", "scale_mma")},
                         mod.expected_shapes(300, 640))
        self.assertEqual(tuple(b.x_q.scale_mma.shape), (32, 4, 3, 4, 5, 1))
        rows = mod.sample_indices(300, (64, 128, 128))
        self.assertTrue(torch.equal(mod.logical_scales_from_mma(b.x_q.scale_mma, rows), sx[rows]))
        self.assertTrue(torch.equal(mod.logical_scales_from_mma(b.x_q.scale_mma, list(range(300))), sx))
        # padding rows and chunks of the physical plane hold 127 (UE8M0 one)
        pad = mod.logical_scales_from_mma(b.x_q.scale_mma, list(range(300, 384)))
        self.assertTrue(bool((pad == 127).all()))

    def test_layout_verifiers_accept_pinned_layouts_and_reject_deviations(self):
        b, p, w, s, xq, sx, wq, sw = fake_case(37, 576, 256)
        rows, cols = mod.sample_indices(37, (16, 64, 128)), mod.sample_indices(256, (64, 128, 128))
        self.assertEqual(mod.verify_packed_weight(p, w, s, 576, 256, cols)["k_padding"], 64)
        self.assertTrue(mod.verify_binding(b, 37, 576, rows)["padding_finite"])
        bad = Obj(); bad.weight = mod.fake_rows(wq, sw)
        bad.weight.values = wq.clone(); bad.weight.values[cols[1], 600] = torch.tensor(1.0).to(torch.float8_e4m3fn)
        with self.assertRaisesRegex(RuntimeError, "padding is not zero"):
            mod.verify_packed_weight(bad, w, s, 576, 256, cols)
        bad2 = Obj(); bad2.weight = mod.fake_rows(wq, sw)
        bad2.weight.scale_rows = sw.reshape(1, 256, 20).view(torch.float8_e8m0fnu)[:, :, :18]
        with self.assertRaisesRegex(RuntimeError, "shapes"):
            mod.verify_packed_weight(bad2, w, s, 576, 256, cols)
        bad3 = Obj(); bad3.x_q = mod.fake_rows(xq, sx)
        mma = bad3.x_q.scale_mma.view(torch.uint8).clone(); mma[rows[0] % 32, 0, 0, 1, 0, 0] ^= 1
        bad3.x_q.scale_mma = mma.view(torch.float8_e8m0fnu)
        with self.assertRaisesRegex(RuntimeError, "scale_mma does not map"):
            mod.verify_binding(bad3, 37, 576, rows)

    def test_sampled_reference_equals_full_oracle_and_tolerance(self):
        b, p, w, s, xq, sx, wq, sw = fake_case(37, 576, 256)
        rows, cols = mod.sample_indices(37, (16, 64, 128)), mod.sample_indices(256, (64, 128, 128))
        full = mod.dequant_rows_f64(xq, sx) @ mod.dequant_rows_f64(wq, sw).T
        ref_rows, ref_cols = mod.sampled_reference(xq, sx, wq, sw, rows, cols, chunk=50)
        self.assertTrue(torch.equal(ref_rows, full[rows])); self.assertTrue(torch.equal(ref_cols, full[:, cols]))
        rounded = full.to(torch.float32).to(torch.bfloat16).to(torch.float64)
        ok, ratio = mod.accuracy_check(rounded[rows], ref_rows); self.assertTrue(ok); self.assertLessEqual(ratio, 1.0)
        bad = rounded.clone(); bad[rows[0], cols[0]] += 0.1 * abs(float(full[rows[0], cols[0]])) + 1.0
        self.assertFalse(mod.accuracy_check(bad[rows], ref_rows)[0])
        self.assertFalse(mod.accuracy_check(bad[:, cols], ref_cols)[0])

    def test_rows_for(self):
        self.assertEqual(mod.rows_for(256, 3), [256, 255, 128])
        self.assertEqual(mod.rows_for(8, 2), [8, 7])
        self.assertEqual(mod.rows_for(1, 3), [1])

    def test_sha_detects_single_bit(self):
        t = torch.randn(4, 4).to(torch.bfloat16)
        u = t.clone(); u.view(torch.int16)[0, 0] ^= 1
        self.assertNotEqual(mod._sha(t), mod._sha(u))


@unittest.skipUnless((B12X_CHECKOUT / ".git").exists(), "no ~/git/b12x checkout")
class PinnedSourceTests(unittest.TestCase):
    """The pinned b12x source still contains the exact expressions the harness relies on."""

    def test_block_fp8_shared_layouts(self):
        src = pinned("b12x/gemm/_shared/block_fp8.py")
        self.assertIn("return _align_up(logical_k, 128)", src)
        self.assertIn("shape=(int(tokens), _physical_mxfp8_k(in_features)),\n        dtype=torch.float8_e4m3fn,", src)
        self.assertIn("shape=(\n            1,\n            int(tokens),\n            _physical_mxfp8_k(in_features) // MXFP8_SCALE_VEC_SIZE,\n        ),", src)
        self.assertIn("x_scale_mma_u8.view(torch.float8_e8m0fnu).permute(\n        3,\n        4,\n        1,\n        5,\n        2,\n        0,\n    )", src)
        self.assertIn("padded_weight = torch.zeros(", src)
        self.assertIn("padded_weight[:, :in_features] = packed_weight", src)
        self.assertIn("neutral_scale = 127 if packed_scale.dtype == torch.uint8 else 1.0", src)
        self.assertIn("if output.shape != (tokens, packed_weight.out_features, 1):", src)

    def test_wo_mxfp8_scale_packing(self):
        src = pinned("b12x/gemm/_shared/wo_mxfp8.py")
        self.assertIn(".expand(num_groups, m_tiles, block_n, k_tiles, scales_per_block_k)", src)
        self.assertIn("[:, :m, : k // MXFP8_SCALE_VEC_SIZE]", src)
        self.assertIn("padded.view(num_groups, m_tiles, 4, 32, k_tiles, 4)\n        .permute(0, 1, 4, 3, 2, 5)", src)
        self.assertIn("return physical.permute(3, 4, 1, 5, 2, 0)", src)
        self.assertIn("if block_size == (32, 32) and not _scale_is_exact_ue8m0(scale):", src)
        self.assertIn("values = weight.contiguous()", src)

    def test_preparation_consumes_scale_mma_and_prime_pattern(self):
        src = pinned("b12x/gemm/block_fp8_linear/_preparation.py")
        self.assertIn("(x_q.values.view(source_2d.shape[0], physical_k, 1), x_q.scale_mma),", src)
        self.assertIn("(weight.values.view(self.query.out_features, physical_k, 1), weight.scale_mma),", src)
        self.assertIn("def run_binding(self, binding: BlockFP8LinearBinding, *, stream=None):", src)
        self.assertIn("def bind(self, *, plan=None, **kwargs):", src)
        types = pinned("b12x/preparation/types.py")
        self.assertIn("def _prime(call: PreparedCall):", types)
        self.assertIn("return call.invoke()", types)

    def test_cpasync_contract(self):
        tuning = pinned("b12x/gemm/_tuning.py")
        self.assertIn('load_path=("tma", "cpasync") if query.recipe == "nvfp4" else ("tma",),', tuning)
        self.assertIn('raise ValueError(f"dense GEMM {name} is outside its recipe/caller domain")', tuning)
        self.assertIn("load_path=config.load_path,\n        swap_ab=config.swap_ab,\n        block_fp8=query.recipe == \"block_fp8\",", tuning)
        dense = pinned("b12x/_lib/dense_gemm.py")
        self.assertIn('if load_path == "cpasync" and (\n            ab_dtype != cutlass.Float4E2M1FN or sf_vec_size != 16 or l != 1\n        ):\n            return False', dense)
        bfl = pinned("b12x/gemm/block_fp8_linear/_tuning.py")
        self.assertIn('recipe="mxfp8", entry_point="gemm.mm", weight_storage="native",', bfl)
        contract = pinned("b12x/preparation/tuning.py")
        self.assertIn("default = override if override is not None else self.default_config(query, device)\n        self.validate_config(query, default, device)", contract)
        self.assertEqual(mod.CPASYNC_DOMAIN_ERROR, "dense GEMM load_path is outside its recipe/caller domain")


if __name__ == "__main__":
    unittest.main()
