#!/usr/bin/env python3
"""Run the frozen Qwen qualification asset from the dated repository layout."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("frozen_qwen_qualification", HERE / "qualify_qwen.py")
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)
qualification.ROOT = HERE.parents[2]
if __name__ == "__main__":
    qualification.main()
