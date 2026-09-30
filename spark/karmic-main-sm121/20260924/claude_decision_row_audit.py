#!/usr/bin/env python3
"""Offline decision-row audit of DS4.1 sparse attention and the MXFP4 DSA indexer.

Input: exactly four per-rank capture files (schema claude-decision-row-v2, see
claude-decision-row-audit-DESIGN.md) for the final prompt row of the
524288-token request; the expected identity of the B2 decision and the
observed identity of the capture run (image, kit, source trees, decoded
sparse-MLA regime, full response signature). The regime and response must be
identical, the B12X tree equal. CPU only; weights_only loads.

Per rank and target layer it compares production operator outputs with
references computed from the exact bytes the kernels read:
  * attention: pinned B12X compressed_sparse_mla_reference (a7d7d29b), fed
    FP32 q holding the captured BF16 values, so the reference output is FP32;
  * indexer (8 index sources): MXFP4 score with the production kernel's
    rounding stages (dsa_indexer/mxfp4.py _PagedScore), structural checks of
    the selection, the exact deterministic top-k (lowest logical index wins
    exact ties, determinism-tiled-topk), and, separately, an approximate
    one-BF16-ulp boundary band for residual summation-order differences;
  * layer-20 candidate production (8-entry blocks, block max, the block of
    the last visible entry forced, top-2048 blocks expanded);
  * the native indexed-page mapper against an independent page-table
    resolution;
  * cross-rank: replicated indexer inputs (q, scales, weights, key pages,
    candidates) are hashed per rank; selections must agree when they agree.

Limits, stated in every report:
  * the references share the cache formats and published rounding with
    production: agreement means the kernels compute what the format defines
    at this row, not that the model is right, and a shared systematic bias is
    invisible to this audit;
  * the attention envelope is a conformance/sensitivity measure from
    production-plan runs, not a certification;
  * equal response signatures show no observed output perturbation by the
    hooks, not that hidden numerics were untouched.

  python3 claude_decision_row_audit.py R0.pt R1.pt R2.pt R3.pt --expected b2.json --observed run.json \
      [--envelope E.json] [--spans S.json] --json out.json
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
SCHEMA = 'claude-decision-row-v3'
PINNED_REFERENCE = HERE / 'claude-pinned-b12x-compressed_reference.py'
PINNED_REFERENCE_SHA256 = '123afd7522d9969eedab2b049702d70dd70e993d4e74a31ef2cecb46b8f95959'
CONFIG = HERE / 'claude-ds41-attention-config.json'
RANKS = 4
PROMPT_TOKENS, CED_WINDOW = 524288, 128
TOPK, CANDIDATE_BLOCKS, CANDIDATE_BLOCK = 512, 2048, 8
SWA_RECORD, INDEXED_RECORD, KEY_DATA, KEY_SCALES, SWA_PAGE = 528, 288, 64, 4, 128
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PROVISIONAL_ENVELOPE = {'bf16': 1.5e-2, 'fp8': 1.0e-1}
LIMITS = ('references share cache formats and published rounding with production; not an independent '
          'model oracle; a shared systematic bias is not detectable here',
          'attention envelope is a conformance/sensitivity measure, not a certification',
          'equal response signatures against the same-boot unarmed control show no observed output perturbation, '
          'not untouched hidden numerics; a historical B2 full-response mismatch is reported separately',
          'KV and index-key contents written by earlier prefill chunks are inputs here, not validated',
          'in matched mode (--regime-reference) validity is equality with the pinned 81389-block regime; B2 '
          'kernel-plan equivalence is reported per plan, not claimed, and a different chunk grid is a different '
          'prefill schedule, not a numerical change of any operator')
# Known inside the serving process (capture meta):
META_KEYS = ('rank', 'node', 'kit_sha256', 'source_trees', 'generation', 'problems',
             'prompt_tokens', 'row_position', 'chunk_rows', 'row_index', 'batch_requests')
# Known only to the external driver (observed identity of the capture run):
IDENTITY_KEYS = ('image_id', 'kit_sha256', 'source_trees', 'regime', 'response_signature')


def sha(t):
    """sha256 of a CPU tensor's contiguous bytes (no numpy dependency)."""
    t = t.detach().to('cpu').contiguous().view(torch.uint8)
    return hashlib.sha256(ctypes.string_at(t.data_ptr(), t.numel()) if t.numel() else b'').hexdigest()


def load_reference():
    data = PINNED_REFERENCE.read_bytes()
    if hashlib.sha256(data).hexdigest() != PINNED_REFERENCE_SHA256:
        raise SystemExit('pinned B12X reference differs from a7d7d29b')
    spec = importlib.util.spec_from_file_location('claude_pinned_compressed_reference', PINNED_REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layer_table(config=None):
    """Target-layer roles from the pinned model config subset (draft/nextn layers excluded)."""
    text = (config or json.loads(CONFIG.read_text()))['text_config']
    layers = text['num_hidden_layers']
    ratios = text['compress_ratios'][:layers]
    kv, index = text['kv_source_layer_ids'], text['index_source_layer_ids']
    source = text['candidate_source_layer_id']
    # ced.py ced_decoder_start: the first full-resolution KV source after a compressed encoder.
    boundary = next((s for s in sorted(kv) if 0 < s < layers and ratios[s] == 1 and any(r > 1 for r in ratios[:s])), None)
    table = {}
    for layer in range(layers):
        ratio = ratios[layer]
        role = {'ratio': ratio, 'page': 256 // max(ratio, 1), 'ced_decoder': boundary is not None and layer >= boundary,
                'kv_owner': max(s for s in kv if s <= layer) if ratio else None,
                'index_owner': max(s for s in index if s <= layer) if ratio else None,
                'index_source': bool(ratio) and layer in index, 'candidate_role': None}
        if role['index_source']:
            role['candidate_role'] = 'produce' if layer == source else 'consume' if layer > source else 'full'
        table[layer] = role
    return table


# ---- decoding and scoring -------------------------------------------------------------------

def ue8m0(scales):
    return torch.pow(2.0, scales.to(torch.float32) - 127.0)


def decode_mxfp4(data, scales):
    """[..., 64] packed E2M1 (low nibble even dim, bit 3 sign) and [..., 4] UE8M0 per 32 dims -> BF16 (exact)."""
    codes = torch.stack((data & 15, data >> 4), dim=-1).reshape(*data.shape[:-1], 128)
    values = torch.tensor(E2M1)[(codes & 7).long()]
    values = torch.where((codes & 8) != 0, -values, values)
    return (values * ue8m0(scales).repeat_interleave(32, dim=-1)).to(torch.bfloat16)


def key_matrix(pages, page_size, count):
    """Logical keys [count, 128] from pages laid out [page_size,64] data then [page_size,4] UE8M0."""
    if pages.ndim != 2 or pages.shape[1] != page_size * (KEY_DATA + KEY_SCALES):
        raise ValueError('index key page width differs from the MXFP4 layout')
    data = pages[:, :page_size * KEY_DATA].reshape(-1, KEY_DATA)
    scales = pages[:, page_size * KEY_DATA:].reshape(-1, KEY_SCALES)
    if data.shape[0] < count:
        raise ValueError('captured key pages do not cover cache_length')
    return decode_mxfp4(data[:count], scales[:count])


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def mxfp4_scores(q_data, q_scales, weights, keys):
    """(published BF16-staged score, exact FP32 score) per key."""
    q = decode_mxfp4(q_data, q_scales).float()
    dots = keys.float() @ q.t()
    w = weights.float()
    per_head = bf16(torch.clamp(bf16(dots), min=0.0) * w)
    return bf16(per_head.sum(dim=1)), (torch.clamp(dots, min=0.0) * w).sum(dim=1)


def bf16_ulp(x):
    x = abs(float(x))
    if x == 0.0 or x == float('inf'):
        return float(torch.finfo(torch.bfloat16).tiny)
    return float(2.0 ** (torch.floor(torch.log2(torch.tensor(x))) - 7))


def deterministic_topk(values, positions, k):
    """Top-k by (-value, ascending position): the image's deterministic exact-tie rule."""
    order = sorted(range(len(positions)), key=lambda i: (-float(values[i]), int(positions[i])))
    return [int(positions[i]) for i in order[:k]]


def opposite_topk(values, positions, k):
    """Top-k by (-value, descending position): the mirror of the image's tie rule, equally valid on exact ties."""
    order = sorted(range(len(positions)), key=lambda i: (-float(values[i]), -int(positions[i])))
    return [int(positions[i]) for i in order[:k]]


def tie_exposure(values, positions, selected, k, *, bound):
    """How much of a k-selection over `positions` (scores `values`) is decided by the exact-tie rule.

    threshold: the k-th reference score; tie_population: pool entries scoring exactly the threshold;
    strictly_above: entries above it (always admitted); tie_admitted: tied entries in the production
    selection; opposite_rule_changes: entries that the mirror rule (highest index first) would swap;
    opposite_selection: that mirror selection, for needle coverage under both rules."""
    positions = [int(p) for p in positions]
    reference = deterministic_topk(values, positions, k)
    if not reference:
        return {'threshold': None, 'tie_population': 0, 'strictly_above': 0, 'tie_admitted': 0,
                'opposite_rule_changes': 0, 'tie_admitted_position_quartiles': [0, 0, 0, 0], 'opposite_selection': []}
    scores = {p: float(v) for p, v in zip(positions, values.tolist())}
    threshold = min(scores[p] for p in reference)
    tied = [p for p in positions if scores[p] == threshold]
    chosen = set(int(s) for s in selected)
    admitted = [p for p in tied if p in chosen]
    opposite = opposite_topk(values, positions, k)
    quartiles = [0, 0, 0, 0]
    for p in admitted:
        quartiles[min(3, 4 * p // max(int(bound), 1))] += 1
    return {'threshold': threshold, 'tie_population': len(tied),
            'strictly_above': sum(1 for p in positions if scores[p] > threshold),
            'tie_admitted': len(admitted), 'opposite_rule_changes': len(set(reference) ^ set(opposite)) // 2,
            'tie_admitted_position_quartiles': quartiles, 'opposite_selection': sorted(opposite)}


# ---- structural checks --------------------------------------------------------------------

def selection_structure(indices, *, expected, bound, pool=None, name='selection'):
    """Exactly `expected` valid ascending unique entries first, then -1; all in range and in pool."""
    values = [int(v) for v in indices.tolist()]
    valid = [v for v in values if v >= 0]
    problems = []
    if values[:len(valid)] != valid or any(v != -1 for v in values[len(valid):]):
        problems.append(f'{name}: -1 padding is not a pure tail')
    if len(valid) != expected:
        problems.append(f'{name}: {len(valid)} entries, expected {expected}')
    if len(set(valid)) != len(valid):
        problems.append(f'{name}: duplicate entries')
    if valid != sorted(valid):
        problems.append(f'{name}: not ascending')
    if any(v >= bound for v in valid):
        problems.append(f'{name}: entries beyond cache_length')
    if any(v < -1 for v in values):
        problems.append(f'{name}: invalid negative entries')
    if pool is not None and not set(valid) <= pool:
        problems.append(f'{name}: entries outside the candidate pool')
    return valid, problems


# ---- audits ---------------------------------------------------------------------------------

def audit_attention(entry, reference, *, compute_mode, envelope):
    heads = entry['q'].shape[0]
    swa_len = int(entry['swa_len'])
    kwargs = {}
    if entry.get('indexed_len') is not None:
        n = int(entry['indexed_len'])
        kwargs = dict(extra_k_cache=entry['indexed_records'][:n].contiguous(),
                      extra_indices=torch.arange(n, dtype=torch.int32).reshape(1, n),
                      extra_topk_lengths=torch.tensor([n], dtype=torch.int32), extra_page_size=1)
    q32 = entry['q'].to(torch.float32).reshape(1, heads, -1)          # exact BF16 values, FP32 storage
    expected = reference.compressed_sparse_mla_reference(
        q32, entry['swa_records'][:swa_len].contiguous(),
        torch.arange(swa_len, dtype=torch.int32).reshape(1, swa_len), torch.tensor([swa_len], dtype=torch.int32),
        sm_scale=512 ** -0.5, attn_sink=entry['attn_sink'].float(), swa_page_size=1,
        cache_format='deepseek_v41', **kwargs)
    if expected.dtype != torch.float32:
        raise RuntimeError('reference output is not FP32')
    ref = expected.reshape(heads, -1)
    got = entry['out'].float().reshape(heads, -1)
    diff = got - ref
    rel = diff.norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=1)
    bound = envelope[compute_mode]
    return {'swa_len': swa_len, 'indexed_len': entry.get('indexed_len'), 'compute_mode': compute_mode,
            'rel_l2_max': float(rel.max()), 'rel_l2_median': float(rel.median()),
            'cosine_min': float(cos.min()), 'max_abs': float(diff.abs().max()), 'bound': bound,
            'heads_over_bound': [int(h) for h in torch.nonzero(rel > bound).flatten()],
            'within': bool((rel <= bound).all())}


def compare_selection(production, reference_exact, published, threshold_value):
    """Exact deterministic equality, then an approximate boundary band for residual order differences."""
    band = 1.01 * bf16_ulp(threshold_value)
    prod, ref = set(production), set(reference_exact)
    near = lambda i: abs(float(published[i]) - float(threshold_value)) <= band
    difference = prod ^ ref
    outside = sorted(i for i in difference if not near(i))
    return {'exact_equal': prod == ref, 'symmetric_difference': len(difference),
            'within_band_only': not outside, 'outside_band': outside[:32], 'band': band}


def audit_indexer(entry, role, *, candidates=None):
    ix = entry['indexer']
    count = int(ix['cache_length'])
    keys = key_matrix(ix['key_pages'], int(ix['page_size']), count)
    published, exact = mxfp4_scores(ix['q_data'], ix['q_scales'], ix['weights'], keys)
    pool = list(range(count)) if candidates is None else [int(c) for c in candidates]
    k = min(TOPK, len(pool))
    valid, problems = selection_structure(ix['topk'], expected=k, bound=count,
                                          pool=None if candidates is None else set(pool), name='topk')
    pool_scores = published[torch.tensor(pool, dtype=torch.long)] if pool else published[:0]
    reference = deterministic_topk(pool_scores, pool, k)
    threshold = float(published[reference[-1]]) if reference else float('-inf')
    row = {'cache_length': count, 'pool': len(pool), 'structure_problems': problems,
           'threshold': threshold,
           'boundary_count': int(((pool_scores - threshold).abs() <= 1.01 * bf16_ulp(threshold)).sum()),
           'exact_vs_published_max': float((exact - published).abs().max()) if count else 0.0,
           # out-of-range entries are already structural failures; compare only addressable ones
           'selection': compare_selection([v for v in valid if v < count], reference, published, threshold),
           'selected': valid}
    ties = tie_exposure(pool_scores, pool, [v for v in valid if v < count], k, bound=count)
    row['opposite_selected'] = ties.pop('opposite_selection')
    row['ties'] = ties
    if role['candidate_role'] == 'produce':
        row['candidates'] = audit_candidates(ix, published, count)
    return row


def audit_candidates(ix, published, count):
    """Layer 20: blocks of 8, block logit = max valid score, the block of the last visible entry forced,
    top-2048 blocks by (-logit, block), expanded to ascending positions."""
    blocks = (count + CANDIDATE_BLOCK - 1) // CANDIDATE_BLOCK
    logits = torch.full((blocks,), float('-inf'))
    for b in range(blocks):
        logits[b] = published[b * CANDIDATE_BLOCK:min((b + 1) * CANDIDATE_BLOCK, count)].max()
    if count:
        logits[(count - 1) // CANDIDATE_BLOCK] = float('inf')
    chosen = deterministic_topk(logits, list(range(blocks)), min(CANDIDATE_BLOCKS, blocks))
    expected = sorted(p for b in chosen for p in range(b * CANDIDATE_BLOCK, min((b + 1) * CANDIDATE_BLOCK, count)))
    length = int(ix['candidate_len'])
    valid, problems = selection_structure(ix['candidates'][:max(length, 0)], expected=len(expected),
                                          bound=count, name='candidates')
    finite = [float(logits[b]) for b in chosen if logits[b] != float('inf')]
    threshold = min(finite) if finite else float('-inf')
    band = 1.01 * bf16_ulp(threshold)
    prod_blocks = {p // CANDIDATE_BLOCK for p in valid if p < count}
    difference = prod_blocks ^ set(chosen)
    outside = sorted(b for b in difference if abs(float(logits[b]) - threshold) > band)
    ties = tie_exposure(logits, list(range(blocks)), prod_blocks, min(CANDIDATE_BLOCKS, blocks), bound=blocks)
    ties.pop('opposite_selection')                                    # block-level rule exposure only
    return {'length': length, 'structure_problems': problems, 'exact_equal': valid == expected,
            'block_symmetric_difference': len(difference), 'outside_band_blocks': outside[:32],
            'within_band_only': not outside, 'ties': ties}


def check_mapper(entry, role):
    """Independent logical -> physical resolution against the native mapper output."""
    n = int(entry['indexed_len'])
    logical = entry['indexed_logical'][:n].long()
    table = entry['indexed_page_table_row'].long()
    page = role['page']
    resolved = table[logical // page] * page + logical % page
    resolved = torch.where((logical >= 0) & (table[logical.clamp_min(0) // page] > 0), resolved, -1)
    return bool(torch.equal(resolved, entry['indexed_slots'][:n].long()))


def needle_coverage(selected, tokens_per_entry, spans):
    covered = {}
    for name, (start, end) in spans.items():
        entries = set(range(start // tokens_per_entry, (end - 1) // tokens_per_entry + 1))
        covered[name] = sorted(entries & set(selected))
    return covered


# ---- validation -----------------------------------------------------------------------------

def validate(capture, table, rank, observed):
    problems = []
    meta = capture.get('meta') if isinstance(capture, dict) else None
    if not isinstance(meta, dict) or capture.get('schema') != SCHEMA:
        return ['schema']
    missing = [k for k in META_KEYS if meta.get(k) is None]
    if missing:
        return [f'missing provenance {missing}']
    if meta['rank'] != rank:
        problems.append('rank label')
    if meta['problems']:
        problems.append(f'capture reported problems: {meta["problems"][:4]}')
    for key in ('kit_sha256', 'source_trees'):
        if meta[key] != observed.get(key):
            problems.append(f'{key} differs from the observed run identity')
    chunk_rows = meta['chunk_rows']
    if type(chunk_rows) is not int or chunk_rows not in (8192, 4096):
        return problems + ['unsupported capture chunk_rows']
    if observed.get('chunk_rows') is not None and observed.get('chunk_rows') != chunk_rows:
        problems.append(f'capture chunk grid {chunk_rows} differs from the observed run identity {observed.get("chunk_rows")}')
    if (meta['prompt_tokens'], meta['row_position'], meta['row_index'],
            meta['batch_requests']) != (PROMPT_TOKENS, PROMPT_TOKENS - 1, chunk_rows - 1, 1):
        problems.append(f'not the final row of the single-request final {chunk_rows}-row chunk of the 524288-token prompt')
    layers = capture.get('layers', {})
    if sorted(layers) != sorted(table):
        return problems + ['layer set differs from the 40 target layers']
    for layer, role in table.items():
        entry = layers[layer]
        base = {'q', 'out', 'attn_sink', 'swa_len', 'swa_slots', 'swa_records', 'plan', 'ced_decoder', 'query_rows',
                'row', 'position'}
        compressed = {'indexed_len', 'indexed_logical', 'indexed_page_table_row', 'indexed_slots', 'indexed_records'}
        need = base | (compressed if role['ratio'] else set())
        if not need <= set(entry):
            problems.append(f'layer {layer}: missing {sorted(need - set(entry))}')
            continue
        rows = CED_WINDOW if role['ced_decoder'] else chunk_rows
        if (entry['ced_decoder'], entry['query_rows'], entry['row']) != (role['ced_decoder'], rows, rows - 1):
            problems.append(f'layer {layer}: row geometry {entry["query_rows"]}/{entry["row"]} is not the '
                            f'{"CED decoder" if role["ced_decoder"] else "full-row"} decision row')
        if entry['position'] != PROMPT_TOKENS - 1:
            problems.append(f'layer {layer}: captured position {entry["position"]} is not the decision row')
        if not role['ratio'] and entry.get('indexed_len') is not None:
            problems.append(f'layer {layer}: indexed stream on an SWA-only layer')
        if ('indexer' in entry) != role['index_source']:
            problems.append(f'layer {layer}: indexer capture differs from the index-source role')
        if tuple(entry['q'].shape) != (16, 512) or tuple(entry['out'].shape) != (16, 512):
            problems.append(f'layer {layer}: local head geometry')
        if entry['q'].dtype != torch.bfloat16 or entry['out'].dtype != torch.bfloat16:
            problems.append(f'layer {layer}: q/out dtype')
        if entry['swa_records'].shape[-1] != SWA_RECORD or not 0 < int(entry['swa_len']) <= SWA_PAGE:
            problems.append(f'layer {layer}: SWA records')
        if role['ratio'] and (entry['indexed_records'].shape[-1] != INDEXED_RECORD
                              or not 0 <= int(entry['indexed_len']) <= TOPK):
            problems.append(f'layer {layer}: indexed records')
        if entry['plan'].get('config', {}).get('v41_compute_mode') not in ('fp8', 'bf16'):
            problems.append(f'layer {layer}: plan config')
        if role['index_source']:
            ix = entry['indexer']
            need_ix = {'q_data', 'q_scales', 'weights', 'cache_length', 'page_size', 'topk', 'key_pages_sha256'}
            need_ix |= {'key_pages'} if rank == 0 else set()
            need_ix |= {'candidates', 'candidate_len'} if role['candidate_role'] == 'produce' else set()
            if not need_ix <= set(ix):
                problems.append(f'layer {layer}: indexer missing {sorted(need_ix - set(ix))}')
            elif int(ix['page_size']) != role['page']:
                problems.append(f'layer {layer}: index page size')
            elif rank == 0 and sha(ix['key_pages']) != ix['key_pages_sha256']:
                problems.append(f'layer {layer}: key page bytes differ from their recorded hash')
    return problems


def replicated_inputs(entry, role):
    ix = entry['indexer']
    fields = {name: sha(ix[name]) for name in ('q_data', 'q_scales', 'weights')}
    fields['key_pages'] = ix['key_pages_sha256']
    fields['cache_length'] = int(ix['cache_length'])
    if role['candidate_role'] == 'produce':
        fields['candidates_out'] = sha(ix['candidates'])
    return fields


ACCURACY = ('the original 524288-token dual-needle retrieval remains a failed gate; this audit never assesses '
            'or changes it')


def compare_identities(expected, observed, control=None, matched=None):
    """Explicit, separate comparisons; returns (comparisons, validity problems).

    B2 (expected): the historical decision. The B12X tree and the eight-plan regime must match for the
    capture to speak about B2's kernels; the full-response signature is compared and reported. When a
    same-boot control is supplied, a B2 response mismatch is reported as an observation across every
    difference between the two runs (KV capacity, boot, image, hooks absent in B2) and is not attributed
    to any one of them; it is not a validity failure. Without a control the B2 response must match.
    Control: two unarmed cold requests on the very same boot with identical signatures (driver-verified).
    The armed capture must equal it in image, kit, full source trees, regime and response. Equality is an
    observation on this boot only (no output difference was observed with the hooks armed); it does not
    establish that the hooks are inert, and inequality does not by itself attribute the difference.
    Matched (optional, explicit): a pinned eight-plan regime record (driver-produced from the reviewed
    045954Z snapshot). When given, validity requires the observed regime to equal it, and the B2 regime
    comparison is reported with its per-plan differences instead of being required; B2 kernel-plan
    equivalence is then explicitly not claimed. Without it the historical B2 requirement stands."""
    problems = []
    for key in IDENTITY_KEYS:
        if not expected.get(key) or not observed.get(key):
            problems.append(f'expected or observed identity lacks {key}')
    b2 = {'regime_equal': expected.get('regime') == observed.get('regime'),
          'response_signature_equal': expected.get('response_signature') == observed.get('response_signature'),
          'b12x_tree_equal': ((expected.get('source_trees') or {}).get('b12x')
                              == (observed.get('source_trees') or {}).get('b12x')),
          'image_id_equal': expected.get('image_id') == observed.get('image_id'),
          'accuracy': ACCURACY}
    if not b2['b12x_tree_equal']:
        problems.append('B12X source tree differs from the B2 image')
    regime_reference = {'mode': 'historical-b2', 'validity_regime': 'b2', 'observed_equals_validity_regime': b2['regime_equal']}
    if matched is None:
        if not b2['regime_equal']:
            problems.append('observed regime differs from the B2 decision (plans not equivalent)')
    else:
        for key in ('regime', 'sha256', 'source', 'blocks', 'chunk_rows'):
            if not matched.get(key):
                problems.append(f'matched regime reference lacks {key}')
        matched_equal = bool(matched.get('regime')) and matched.get('regime') == observed.get('regime')
        if not matched_equal:
            problems.append('observed regime differs from the pinned matched regime (plans not the pinned family)')
        if observed.get('chunk_rows') is not None and observed.get('chunk_rows') != matched.get('chunk_rows'):
            problems.append('observed chunk grid differs from the matched regime record')
        b2['regime_difference_from_pinned'] = {name: {'b2': (expected.get('regime') or {}).get(name), 'observed': value}
                                               for name, value in (observed.get('regime') or {}).items()
                                               if (expected.get('regime') or {}).get(name) != value}
        b2['regime_note'] = ('B2 REGIME NOT REQUIRED IN MATCHED MODE: validity is equality with the pinned '
                             'matched regime; B2 kernel-plan equivalence is reported, not claimed'
                             if not b2['regime_equal'] else 'observed regime also equals the B2 regime')
        regime_reference = {'mode': 'matched-pinned-regime', 'validity_regime': 'pinned', 'source': matched.get('source'),
                            'sha256': matched.get('sha256'), 'blocks': matched.get('blocks'),
                            'chunk_rows': matched.get('chunk_rows'), 'observed_equals_validity_regime': matched_equal,
                            'observed_equals_b2_regime': b2['regime_equal']}
    comparisons = {'b2': b2, 'control': None, 'regime_reference': regime_reference}
    if control is None:
        if not b2['response_signature_equal']:
            problems.append('observed response_signature differs from the B2 decision (no observed-equality; '
                            'supply a same-boot unarmed control to separate capacity from hooks)')
        b2['response_note'] = ('full response equals the historical B2 decision' if b2['response_signature_equal']
                               else 'full response differs from the historical B2 decision')
        return comparisons, problems
    ctl = {}
    for key in IDENTITY_KEYS:
        if not control.get(key):
            problems.append(f'control identity lacks {key}')
        ctl[key + '_equal'] = bool(control.get(key)) and control.get(key) == observed.get(key)
        if control.get(key) and not ctl[key + '_equal']:
            problems.append(f'observed {key} differs from the same-boot unarmed control')
    ctl['note'] = ('armed capture equals the unarmed control on this boot: no observed output perturbation '
                   'with the hooks armed (an observation on this boot; hidden numerics are not proven untouched)'
                   if all(v for k, v in ctl.items() if k.endswith('_equal'))
                   else 'armed capture differs from the same-boot unarmed control (difference not attributed)')
    comparisons['control'] = ctl
    b2['response_note'] = ('full response equals the historical B2 decision' if b2['response_signature_equal'] else
                           'HISTORICAL B2 FULL-RESPONSE MISMATCH: this boot produced a different full response than '
                           'B2 with an equal B12X tree and eight-plan regime; the mismatch is observed across the '
                           'differences between the runs (KV capacity, boot, image) and is not attributed to any of '
                           'them here; the audit concerns this boot\'s decision row, not B2\'s')
    return comparisons, problems


def audit(captures, expected, observed, *, envelope=None, spans=None, control=None, matched=None):
    """expected: identity of the B2 decision; observed: identity of the capture run (driver-produced);
    control: identity of the same-boot unarmed cold requests (driver-produced, optional);
    matched: pinned regime record for matched-grid captures (driver-produced, optional, explicit)."""
    table = layer_table()
    calibrated = envelope is not None
    envelope = envelope or PROVISIONAL_ENVELOPE
    comparisons, problems = compare_identities(expected, observed, control, matched)
    report = {'limits': LIMITS, 'envelope': envelope, 'envelope_calibrated': calibrated,
              'mode': comparisons['regime_reference']['mode'],
              'comparisons': comparisons, 'ranks': {}, 'cross_rank': {}, 'problems': problems}
    if len(captures) != RANKS:
        report['problems'].append(f'exactly {RANKS} rank captures required, got {len(captures)}')
    for rank, capture in enumerate(captures):
        report['problems'] += [f'rank {rank}: {p}' for p in validate(capture, table, rank, observed)]
    if report['problems']:
        report['verdict'] = 'invalid-capture'
        return report
    reference = load_reference()
    for rank, capture in enumerate(captures):
        layers, rows = capture['layers'], {}
        for layer, role in table.items():
            entry = layers[layer]
            row = {'role': role, 'attention': audit_attention(
                entry, reference, compute_mode=entry['plan']['config']['v41_compute_mode'], envelope=envelope)}
            if role['ratio']:
                row['mapper_equal'] = check_mapper(entry, role)
            if role['index_source']:
                row['replicated_inputs'] = replicated_inputs(entry, role)
                if rank == 0:
                    candidates = None
                    if role['candidate_role'] == 'consume':
                        source = layers[20]['indexer']
                        candidates = source['candidates'][:int(source['candidate_len'])].tolist()
                    row['indexer'] = audit_indexer(entry, role, candidates=candidates)
                    if spans:
                        tokens = PROMPT_TOKENS // int(entry['indexer']['cache_length'])
                        row['indexer']['needles'] = {
                            'lowest_index_rule': needle_coverage(row['indexer']['selected'], tokens, spans),
                            'highest_index_rule': needle_coverage(row['indexer']['opposite_selected'], tokens, spans)}
                row['topk'] = [int(v) for v in entry['indexer']['topk'].tolist()]
            rows[layer] = row
        report['ranks'][rank] = rows
    for layer, role in table.items():
        if not role['index_source']:
            continue
        inputs = {json.dumps(report['ranks'][r][layer]['replicated_inputs'], sort_keys=True) for r in range(RANKS)}
        selections = {tuple(report['ranks'][r][layer]['topk']) for r in range(RANKS)}
        report['cross_rank'][layer] = {'inputs_equal': len(inputs) == 1, 'selections_equal': len(selections) == 1}
    s = {
        'attention_outside_envelope': [(r, l) for r, rows in report['ranks'].items() for l, row in rows.items()
                                       if not row['attention']['within']],
        'mapper_mismatch': [(r, l) for r, rows in report['ranks'].items() for l, row in rows.items()
                            if row.get('mapper_equal') is False],
        'indexer_structure': [l for l, row in report['ranks'][0].items()
                              if row.get('indexer', {}).get('structure_problems')
                              or row.get('indexer', {}).get('candidates', {}).get('structure_problems')],
        'indexer_outside_band': [l for l, row in report['ranks'][0].items() if 'indexer' in row and
                                 (not row['indexer']['selection']['within_band_only']
                                  or not row['indexer'].get('candidates', {}).get('within_band_only', True))],
        'indexer_exact_order_differences': [l for l, row in report['ranks'][0].items() if 'indexer' in row and
                                            (not row['indexer']['selection']['exact_equal']
                                             or not row['indexer'].get('candidates', {}).get('exact_equal', True))],
        'replicated_input_divergence': [l for l, v in report['cross_rank'].items() if not v['inputs_equal']],
        'replicated_selection_mismatch': [l for l, v in report['cross_rank'].items()
                                          if v['inputs_equal'] and not v['selections_equal']],
        # attribution context, never part of the verdict: where the exact-tie rule decided entries
        'tie_exposed_layers': {l: {k: row['indexer']['ties'][k] for k in ('tie_population', 'tie_admitted',
                                                                            'strictly_above', 'opposite_rule_changes')}
                               for l, row in report['ranks'][0].items()
                               if row.get('indexer', {}).get('ties', {}).get('tie_admitted')},
        'needle_coverage_rule_dependent': [l for l, row in report['ranks'][0].items() if 'needles' in row.get('indexer', {})
                                           and row['indexer']['needles']['lowest_index_rule']
                                           != row['indexer']['needles']['highest_index_rule']],
    }
    report['summary'] = s
    deviations = [k for k in s if k not in ('indexer_exact_order_differences', 'tie_exposed_layers',
                                            'needle_coverage_rule_dependent') and s[k]]
    if deviations:
        report['verdict'] = 'deviation-found'
    elif s['indexer_exact_order_differences']:
        report['verdict'] = 'unresolved-boundary-order'
    else:
        report['verdict'] = 'within-conformance-envelope' + ('' if calibrated else ' (provisional envelope)')
    for rows in report['ranks'].values():
        for row in rows.values():
            row.pop('topk', None)
            row.get('indexer', {}).pop('selected', None)
            row.get('indexer', {}).pop('opposite_selected', None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('captures', nargs='+', type=Path, help='rank0 .. rank3, in order')
    parser.add_argument('--expected', type=Path, required=True,
                        help='identity of the B2 decision: image_id, kit_sha256, source_trees, regime, response_signature')
    parser.add_argument('--observed', type=Path, required=True,
                        help='identity of the capture run (driver-produced, same keys)')
    parser.add_argument('--control', type=Path,
                        help='identity of the same-boot unarmed cold requests (same keys); when given, the capture '
                             'must equal it and the B2 full response is reported, not required')
    parser.add_argument('--regime-reference', type=Path,
                        help='matched-grid captures only: the driver\'s pinned regime record; validity then means '
                             'equality with that regime and the B2 regime is reported, not required')
    parser.add_argument('--envelope', type=Path, help='conformance envelope {"bf16": x, "fp8": y}')
    parser.add_argument('--spans', type=Path, help='{"identity": [start, end], ...} token spans')
    parser.add_argument('--json', type=Path)
    a = parser.parse_args(argv)
    captures = [torch.load(p, map_location='cpu', weights_only=True) for p in a.captures]
    expected = json.loads(a.expected.read_text())
    observed = json.loads(a.observed.read_text())
    envelope = json.loads(a.envelope.read_text()) if a.envelope else None
    spans = {k: tuple(v) for k, v in json.loads(a.spans.read_text()).items()} if a.spans else None
    control = json.loads(a.control.read_text()) if a.control else None
    matched = json.loads(a.regime_reference.read_text()) if a.regime_reference else None
    report = audit(captures, expected, observed, envelope=envelope, spans=spans, control=control, matched=matched)
    report['captures'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in a.captures}
    text = json.dumps(report, indent=2, default=str)
    if a.json:
        a.json.write_text(text + '\n')
    print(json.dumps({k: report.get(k) for k in ('verdict', 'mode', 'comparisons', 'summary', 'problems')}, indent=2, default=str))
    return {'within-conformance-envelope': 0, 'deviation-found': 1}.get(report['verdict'].split(' (')[0], 3)


if __name__ == '__main__':
    sys.exit(main())
