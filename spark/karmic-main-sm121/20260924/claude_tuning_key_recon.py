#!/usr/bin/env python3
"""Rebuild B12X tuning-cache record keys offline and print the measured winners.

Reads a retained tuning receipt (a copy of the B12X selection cache, schema 6)
and reconstructs the record key that `b12x.preparation.session._choice_key`
computes for `gemm.block_fp8_linear` queries at pinned commit a7d7d29b:

    sha256(json({component, query_schema, config_schema, semantic_version,
                 candidate_contract_version, query, invocation, pin,
                 dependencies}, sort_keys, compact))

The block-FP8 query is `BlockFp8LinearQuery` (max_tokens, in/out features,
dtypes, output_mode, weight_block_size, exhaustive, codegen snapshot). The
codegen snapshot is {"split_k_atomic": B12X_DENSE_SPLITK_TURBO, "fused_fp6_quant":
B12X_DENSE_FUSED_QUANT}; the serving launcher sets turbo=1 and leaves fused
quant at its default (off), and B12X_AUTOTUNE_EXHAUSTIVE is unset.

Shapes are brute-forced over plausible (k, n) pairs so nothing about the model
is assumed beyond the capacity list. Read-only: it never touches a checkout,
host or cache; it only reads the receipt path given.

    python3 claude_tuning_key_recon.py receipts/moe-control-tuning.json
"""
import hashlib
import json
import sys
from pathlib import Path

COMPONENT = "gemm.block_fp8_linear"
QUERY_SCHEMA, CONFIG_SCHEMA = 6, 4          # block_fp8_linear/_tuning.py
SEMANTIC_VERSION = 1                          # TuningContract default
CANDIDATE_CONTRACT_VERSION = 4                # dense gemm/_tuning.py TUNING
CAPACITIES = (1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32, 128, 256, 384, 512, 8192)
KS = (512, 1280, 2048, 2560, 3072, 4096, 5120, 6144, 8192)
NS = sorted({*range(64, 16385, 64), 448, 1856, 25600, 30720})


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def record_key(max_tokens, k, n, *, weight_block=32, exhaustive=False, turbo=True, fused=False):
    query = {
        "max_tokens": max_tokens, "in_features": k, "out_features": n,
        "source_dtype": "bfloat16", "output_dtype": "bfloat16", "output_mode": "provided",
        "weight_block_size": weight_block, "exhaustive": exhaustive,
        "codegen": {"split_k_atomic": turbo, "fused_fp6_quant": fused},
    }
    payload = {
        "component": COMPONENT, "query_schema": QUERY_SCHEMA, "config_schema": CONFIG_SCHEMA,
        "semantic_version": SEMANTIC_VERSION,
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "query": query, "invocation": {}, "pin": None, "dependencies": [],
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def main(path):
    raw = Path(path).read_bytes()
    receipt = json.loads(raw)
    records = receipt["records"]
    print(f"receipt {path} sha256 {hashlib.sha256(raw).hexdigest()}")
    print(f"identity {_json(receipt['identity'])}")
    hits = {}
    for mt in CAPACITIES:
        for k in KS:
            for n in NS:
                for wb in (32, 128):
                    key = record_key(mt, k, n, weight_block=wb)
                    if key in records:
                        hits[(mt, k, n, wb)] = (key, records[key]["assignment"])
    print(f"matched records: {len(hits)}")
    for (k, n, wb) in sorted({(k, n, wb) for (_, k, n, wb) in hits}):
        print(f"\nshape in={k} out={n} weight_block={wb}")
        for mt in CAPACITIES:
            hit = hits.get((mt, k, n, wb))
            if hit is None:
                continue
            key, a = hit
            print(f"  m{mt:<5} tile=({a['tile_m']},{a['tile_n']}) tile_k={a['tile_k']} "
                  f"split_k={a['split_k_slices']} large_m_unroll={a['large_m_unroll']} "
                  f"swap_ab={a['swap_ab']} load={a['load_path']} key={key[:16]}")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "receipts/moe-control-tuning.json"))
