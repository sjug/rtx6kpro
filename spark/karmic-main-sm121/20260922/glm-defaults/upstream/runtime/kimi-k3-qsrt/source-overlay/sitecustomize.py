"""Run a source-locked vLLM tree with extensions from a binary image.

Set ``VLLM_SOURCE_OVERLAY_ACTIVE=1``, ``VLLM_SOURCE_OVERLAY_ROOT`` to the
repository root, and ``VLLM_BINARY_PACKAGE_DIR`` to the image's compiled
``vllm`` package. Activation only configures import resolution; it does not
import vLLM. The Kimi runtime image therefore enables the overlay globally so
custom vLLM commands and the bundled launchers execute the same source tree.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
from pathlib import Path


def _activate_vllm_source_overlay() -> None:
    source_root = os.environ.get("VLLM_SOURCE_OVERLAY_ROOT")
    binary_package_dir = os.environ.get("VLLM_BINARY_PACKAGE_DIR")
    if not source_root:
        raise RuntimeError("VLLM_SOURCE_OVERLAY_ROOT is not configured")

    source_root_path = Path(source_root).resolve()
    source_package_dir = source_root_path / "vllm"
    if not (source_package_dir / "__init__.py").is_file():
        raise RuntimeError(
            "VLLM_SOURCE_OVERLAY_ROOT does not contain a vLLM package: "
            f"{source_package_dir}"
        )

    source_root = str(source_root_path)
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)

    if binary_package_dir:
        binary_package_path = Path(binary_package_dir).resolve()
        if not binary_package_path.is_dir():
            raise RuntimeError(
                f"VLLM_BINARY_PACKAGE_DIR does not exist: {binary_package_path}"
            )
        binary_package_dir = str(binary_package_path)

        class _VLLMBinaryFallback(importlib.abc.MetaPathFinder):
            """Resolve compiled vLLM submodules from the binary package."""

            def find_spec(self, fullname, path=None, target=None):
                if not fullname.startswith("vllm."):
                    return None
                relative_parent = fullname.split(".")[1:-1]
                search_dir = str(Path(binary_package_dir, *relative_parent))
                spec = importlib.machinery.PathFinder.find_spec(fullname, [search_dir])
                if spec is None or spec.origin is None:
                    return None
                if fullname == "vllm.vllm_flash_attn.cute" or fullname.startswith(
                    "vllm.vllm_flash_attn.cute."
                ):
                    return spec
                if not any(
                    spec.origin.endswith(suffix)
                    for suffix in importlib.machinery.EXTENSION_SUFFIXES
                ):
                    return None
                return spec

        sys.meta_path.insert(0, _VLLMBinaryFallback())


if os.environ.get("VLLM_SOURCE_OVERLAY_ACTIVE") == "1":
    try:
        _activate_vllm_source_overlay()
    except Exception as exc:
        # site.execsitecustomize catches ordinary exceptions and lets Python
        # continue. SystemExit preserves the source-identity invariant by
        # terminating startup before an unintended vLLM package can execute.
        raise SystemExit(f"vLLM source overlay activation failed: {exc}") from exc
