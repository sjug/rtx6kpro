"""The selected upstream workspace fixture without unrelated model-runner imports.

Matches vLLM 8e1f1e58 tests/conftest.py workspace_init. The gate's confcutdir
excludes that large conftest and its unrelated test-only dependency closure.
"""
import pytest
import torch


@pytest.fixture
def workspace_init():
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    if torch.accelerator.is_available():
        init_workspace_manager(torch.device(0))
    yield
    reset_workspace_manager()
