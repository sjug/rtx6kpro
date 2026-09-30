#!/usr/bin/env python3
"""Run the unchanged semantic assertions with temperature-one sampling."""
import argparse
import hashlib
import importlib.util
from pathlib import Path

SOURCE_SHA256 = 'ef1a02e1112ccc6d394f4870466632c4a9476cf5fd388743cd4f56f831ba216a'


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--receipt-file', type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.receipt_file.exists():
        raise SystemExit('Refuse to overwrite a semantic receipt')
    source = Path(__file__).resolve().parents[3] / 'glm53/verify-semantic-admission.py'
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('Shared semantic probe changed; review before reuse')
    spec = importlib.util.spec_from_file_location('glm_semantic', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_payload, original_emit = module.payload, module.emit

    def sampled_payload(*args, **kwargs):
        return {**original_payload(*args, **kwargs), 'temperature': 1.0}

    def sampled_emit(receipt, output):
        original_emit({**receipt, 'temperature': 1.0, 'probe_source_sha256': SOURCE_SHA256}, output)

    module.payload = sampled_payload
    module.emit = sampled_emit
    module.main()


if __name__ == '__main__':
    main()
