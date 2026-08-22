#!/usr/bin/env python3
"""Create the opt-in r18p A8/native per-batch experiment source.

This is intentionally an exact-source transformer rather than a loose patch:
every anchor must occur exactly once, so drift fails before a staging boot.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(old, new, 1)


def transform(source: str) -> str:
    source = replace_once(
        source,
        """def _moe_force_a16_enabled() -> bool:
    return _env_flag("B12X_MOE_FORCE_A16")


def _moe_activation_amax_enabled() -> bool:
""",
        """def _moe_force_a16_enabled() -> bool:
    return _env_flag("B12X_MOE_FORCE_A16")


def _moe_nvfp4_native_token_counts() -> frozenset[int]:
    raw = _env_first("VLLM_B12X_MOE_NVFP4_NATIVE_TOKEN_COUNTS")
    if raw is None:
        return frozenset()
    try:
        counts = frozenset(int(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            "VLLM_B12X_MOE_NVFP4_NATIVE_TOKEN_COUNTS must be a "
            "comma-separated list of positive integers"
        ) from exc
    if not counts or any(value <= 0 for value in counts):
        raise ValueError(
            "VLLM_B12X_MOE_NVFP4_NATIVE_TOKEN_COUNTS must contain "
            "positive integers"
        )
    return counts


def _moe_activation_amax_enabled() -> bool:
""",
        "environment parser",
    )

    source = replace_once(
        source,
        """        self._kquant_capture_prefix: str | None = None

    def _quant_mode(self) -> str:
""",
        """        self._kquant_capture_prefix: str | None = None
        self._nvfp4_native_token_counts = _moe_nvfp4_native_token_counts()

    def _quant_mode(self) -> str:
""",
        "per-layer policy cache",
    )

    source = replace_once(
        source,
        """        return "nvfp4" if self.quant_config.quant_dtype == "nvfp4" else "w4a16"

    def _source_format(self) -> str:
""",
        """        return "nvfp4" if self.quant_config.quant_dtype == "nvfp4" else "w4a16"

    def _quant_modes(self) -> tuple[str, ...]:
        quant_mode = self._quant_mode()
        if (
            not self._nvfp4_native_token_counts
            or quant_mode != "w4a8_nvfp4"
        ):
            return (quant_mode,)
        return (quant_mode, "nvfp4")

    def _quant_mode_for_tokens(self, tokens: int) -> str:
        quant_mode = self._quant_mode()
        if (
            quant_mode == "w4a8_nvfp4"
            and int(tokens) in self._nvfp4_native_token_counts
        ):
            return "nvfp4"
        return quant_mode

    def _source_format(self) -> str:
""",
        "runtime mode selection",
    )

    source = replace_once(
        source,
        """        quant_mode = self._quant_mode()
        prepared = self._prepared_experts
        if prepared is not None:
            requested_dtype = str(params_dtype).removeprefix("torch.")
            if (
                quant_mode in prepared.plan.quant_modes
                and requested_dtype == prepared.plan.io_dtype
""",
        """        quant_mode = self._quant_mode()
        requested_quant_modes = self._quant_modes()
        prepared = self._prepared_experts
        if prepared is not None:
            requested_dtype = str(params_dtype).removeprefix("torch.")
            if (
                frozenset(requested_quant_modes) <= prepared.plan.quant_modes
                and requested_dtype == prepared.plan.io_dtype
""",
        "prepared owner validation",
    )

    source = replace_once(
        source,
        """                f"quant_mode={quant_mode!r}, dtype={requested_dtype!r}, "
                f"activation={_b12x_activation_name(activation)!r}; "
""",
        """                f"quant_mode={quant_mode!r}, "
                f"requested_modes={sorted(requested_quant_modes)}, "
                f"dtype={requested_dtype!r}, "
                f"activation={_b12x_activation_name(activation)!r}; "
""",
        "prepared owner diagnostic",
    )

    source = replace_once(
        source,
        """        weight_plan = plan_b12x_fp4_moe_weights(
            quant_modes=quant_mode,
""",
        """        weight_plan = plan_b12x_fp4_moe_weights(
            quant_modes=requested_quant_modes,
""",
        "dual-mode weight plan",
    )

    source = replace_once(
        source,
        """    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        quant_mode = self._quant_mode()
        prepared = self._lookup_prepared_experts()
""",
        """    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        quant_mode = self._quant_mode_for_tokens(M)
        prepared = self._lookup_prepared_experts()
""",
        "workspace mode selection",
    )

    source = replace_once(
        source,
        """        )
        quant_mode = self._quant_mode()

        if expert_map is not None:
""",
        """        )
        quant_mode = self._quant_mode_for_tokens(int(hidden_states.shape[0]))

        if expert_map is not None:
""",
        "execution mode selection",
    )

    ast.parse(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = args.source.read_text()
    transformed = transform(original)
    args.output.write_text(transformed)
    print(f"patched {args.source} -> {args.output}")


if __name__ == "__main__":
    main()
