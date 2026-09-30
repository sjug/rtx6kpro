"""Token spans of the needle markers in the frozen historical 524288-token input (CPU, in-image).

The serving contract runs vLLM with ``--tokenizer-mode deepseek_v41`` (upstream-launch.json). That
mode has no Jinja chat template: ``vllm.tokenizers.deepseek_v41`` wraps the HF fast tokenizer and
``apply_chat_template`` renders the DeepSeek V4.1 prompt text with the pinned encoder, then encodes it
with ``add_special_tokens=False``. The served prompt ids are that list (renderer ``deepseek_v4``,
``tokenize`` left at its default). ``--default-chat-template-kwargs`` supplies thinking=true and
reasoning_effort=high; the request's ``chat_template_kwargs`` override per key (renderers/params.py
merge_kwargs), so the frozen request renders in chat mode with no reasoning-effort prefix.

This helper imports that tokenizer from the image's pinned tree (the launch contract's PYTHONPATH
entry), renders the frozen messages the same way, and maps each marker's characters to token
positions with the fast tokenizer's offsets. Nothing is approximated: it fails closed unless
  * the ids from the served path equal the ids of the offsets pass,
  * the prompt has exactly the historical token count,
  * every marker occurs once and maps to a contiguous token span,
  * rendering the input's text up to the late marker (the historical ``before_late`` string that
    qualify.py posted to /tokenize) reproduces the recorded prefix count exactly, and
  * the served ids share that prefix's ids up to the late marker, give or take the one token that
    may absorb the newline in front of the marker.
The spans feed claude_decision_row_audit.py --spans (needle coverage of the selected index entries).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

MARKERS = {'identity': '510c94b1bb4f42d2998093bd6b9c3d99', 'early_code': '739184',
           'late_marker': 'The late retrieval code is', 'late_code': '482617'}
# launch_contract.py: PYTHONPATH=/opt/jovian-judgement/vllm:/opt/jovian-judgement/b12x
VLLM_ROOT = '/opt/jovian-judgement/vllm'
# upstream-launch.json: --default-chat-template-kwargs.thinking=true and .reasoning_effort=high
DEFAULT_CHAT_TEMPLATE_KWARGS = {'thinking': True, 'reasoning_effort': 'high'}
ENCODER_FILES = ('vllm/tokenizers/deepseek_v41.py', 'vllm/tokenizers/deepseek_v41_encoding.py')


def chat_template_kwargs(spec, defaults=DEFAULT_CHAT_TEMPLATE_KWARGS):
    """Serving-side merge: request keys override the launch defaults (None/'auto' count as unset)."""
    request = spec.get('chat_template_kwargs') or {}
    return dict(defaults) | {k: v for k, v in request.items() if v not in (None, 'auto')}


def load_tokenizer(model_dir, vllm_root=VLLM_ROOT):
    """The pinned serving tokenizer class from the image's vLLM tree, plus its identity."""
    if vllm_root not in sys.path:
        sys.path.insert(0, vllm_root)
    import vllm
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer
    if not Path(vllm.__file__).resolve().is_relative_to(Path(vllm_root) / 'vllm'):
        raise RuntimeError(f'vllm resolved outside the pinned tree: {vllm.__file__}')
    identity = {'vllm_root': vllm_root, 'model_dir': model_dir, 'tokenizer_mode': 'deepseek_v41',
                'encoder_sha256': {name: hashlib.sha256((Path(vllm_root) / name).read_bytes()).hexdigest()
                                   for name in ENCODER_FILES}}
    return DeepseekV41Tokenizer.from_pretrained(model_dir), identity


def render(tokenizer, messages, kwargs):
    """(served prompt ids, rendered text) exactly as the deepseek_v41 chat path produces them."""
    ids = tokenizer.apply_chat_template(messages, **kwargs)
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    return list(ids), text


def encode_with_offsets(tokenizer, text):
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoding['input_ids']), [tuple(o) for o in encoding['offset_mapping']]


def prefix_messages(spec, marker=MARKERS['late_marker']):
    """The historical ``before_late`` message: the input text up to the newline before the late marker."""
    messages = spec['messages']
    if len(messages) != 1 or messages[0].get('role') != 'user' or not isinstance(messages[0].get('content'), str):
        raise ValueError('historical input must be a single user text message')
    content = messages[0]['content']
    first = content.find(marker)
    if first < 1 or content.find(marker, first + 1) >= 0 or content[first - 1] != '\n':
        raise ValueError('late marker must occur once, preceded by the newline the historical builder inserted')
    return [dict(messages[0], content=content[:first - 1])]


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def spans(text, offsets, spec, *, served_ids, offset_ids, prefix_ids, markers=MARKERS):
    """{marker: [first_token, last_token + 1]} with the checks listed in the module docstring."""
    if served_ids != offset_ids:
        raise ValueError('served prompt ids differ from the offsets pass')
    if len(offsets) != spec['prompt_tokens']:
        raise ValueError(f'rendered prompt has {len(offsets)} tokens, historical {spec["prompt_tokens"]}')
    result = {}
    for name, needle in markers.items():
        first = text.find(needle)
        if first < 0 or text.find(needle, first + 1) >= 0:
            raise ValueError(f'marker {name} must occur exactly once')
        last = first + len(needle)
        tokens = [i for i, (s, e) in enumerate(offsets) if e > first and s < last]
        if not tokens or tokens != list(range(tokens[0], tokens[-1] + 1)):
            raise ValueError(f'marker {name} does not map to a contiguous token span')
        result[name] = [tokens[0], tokens[-1] + 1]
    recorded = spec['late_marker_prefix_tokens_including_template']
    if len(prefix_ids) != recorded:
        raise ValueError(f'rendered prefix has {len(prefix_ids)} tokens, input recorded {recorded}')
    shared = common_prefix(prefix_ids, served_ids)
    start = result['late_marker'][0]
    if shared not in (start - 1, start):
        raise ValueError(f'prefix ids agree with the prompt for {shared} tokens, late marker starts at {start}')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='frozen historical input json')
    parser.add_argument('--model-dir', required=True, help='HF snapshot directory holding the tokenizer')
    parser.add_argument('--vllm-root', default=VLLM_ROOT, help='pinned vLLM tree inside the image')
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    spec = json.loads(a.input.read_text())
    kwargs = chat_template_kwargs(spec)
    tokenizer, identity = load_tokenizer(a.model_dir, a.vllm_root)
    served_ids, text = render(tokenizer, spec['messages'], kwargs)
    offset_ids, offsets = encode_with_offsets(tokenizer, text)
    prefix_ids, _ = render(tokenizer, prefix_messages(spec), kwargs)
    result = spans(text, offsets, spec, served_ids=served_ids, offset_ids=offset_ids, prefix_ids=prefix_ids)
    identity.update(chat_template_kwargs=kwargs, prompt_tokens=len(served_ids), prefix_tokens=len(prefix_ids),
                    prefix_shared_tokens=common_prefix(prefix_ids, served_ids),
                    prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    tokenizer_class=type(tokenizer).__name__)
    a.out.write_text(json.dumps(result, indent=2) + '\n')
    print('NEEDLE-SPANS-IDENTITY', json.dumps(identity, sort_keys=True), flush=True)
    print('NEEDLE-SPANS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
