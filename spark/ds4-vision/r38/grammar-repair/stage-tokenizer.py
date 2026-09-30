#!/usr/bin/env python3
"""Stage only pinned GPT2 tokenizer assets in the gate's writable HF cache."""
from huggingface_hub import snapshot_download
from pathlib import Path

snapshot = Path(snapshot_download(
    repo_id='gpt2', revision='607a30d783dfa663caf39e06633721c8d4cfcd7e',
    allow_patterns=['config.json', 'tokenizer.json', 'tokenizer_config.json',
                    'vocab.json', 'merges.txt'],
))
# Upstream's unchanged fixture requests "gpt2" without a revision. Resolve its
# offline main alias to this pinned snapshot, not the mutable upstream tip.
refs = snapshot.parent.parent / 'refs'
refs.mkdir(exist_ok=True)
(refs / 'main').write_text(snapshot.name)
print(snapshot, flush=True)
config = Path(snapshot_download(
    repo_id='Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca',
    allow_patterns=['config.json', 'generation_config.json'],
))
refs = config.parent.parent / 'refs'
refs.mkdir(exist_ok=True)
(refs / 'main').write_text(config.name)
print(config, flush=True)
opt = Path(snapshot_download(
    repo_id='facebook/opt-125m', revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6',
    allow_patterns=['config.json', 'generation_config.json', 'tokenizer.json',
                    'tokenizer_config.json', 'vocab.json', 'merges.txt',
                    'special_tokens_map.json'],
))
refs = opt.parent.parent / 'refs'
refs.mkdir(exist_ok=True)
(refs / 'main').write_text(opt.name)
print(opt, flush=True)
