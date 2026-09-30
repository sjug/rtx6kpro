"""Build the capture comparison identity from retained B2 receipts, offline only.

This does not approve a KV override, select an image, arm a capture or contact
any node. The original accuracy failure remains a failure.
"""
import argparse
import hashlib
import json
from pathlib import Path

from claude_decode_sparse_mla import decode, regimes

ROOT = Path(__file__).resolve().parent
NODES = ('dusty', 'toby', 'rusty', 'kirby')
B2_IMAGE = 'e06df11a8ca18fa514d9f28f67cc691aef296da2eeb22b113a734519853bccd7'
PROMPT_SHA = 'b410c6b19d492e83ecb775805266d91eed15ded5e3afd0cf264c25aae41bc7e4'


def signature(response):
    choice = response['choices'][0]
    if not (choice.get('logprobs') or {}).get('content'):
        raise ValueError('Missing full response logprobs')
    payload = {key: choice[key] for key in ('message', 'finish_reason', 'logprobs')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def exact_regime(payload, blocks):
    records = payload['records']
    found, _ = decode(records, [blocks], None)
    result = regimes(records, found).get(blocks, {})
    wanted = {f'{role}.{mode}' for role in ('swa', 'draft', 'ratio1', 'ratio2')
              for mode in ('extend', 'decode')}
    if set(result) != wanted:
        raise ValueError('Incomplete eight-plan regime')
    return result


def build_reference(root=ROOT):
    sources = {}

    def read(relative):
        raw = (root / relative).read_bytes()
        sources[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    before = 'receipts/stop-router-b2-completed-20260926T021208Z'
    identities, selected = [], []
    for node in NODES:
        info = read(f'{before}/{node}-before.json')
        labels = info['Config']['Labels']
        identity = dict(image_id=info['Image'].removeprefix('sha256:'),
                        kit_sha256=labels['local-inference.ds41.kit.sha256'],
                        source_trees={'vllm': labels['vllm.source-tree'],
                                      'b12x': labels['b12x.source-tree']})
        if identity['image_id'] != B2_IMAGE:
            raise ValueError('Not the B2 candidate image')
        identities.append(identity)
        b1 = exact_regime(read(f'receipts/router-b1-tuning-20260925/{node}.json'), 80927)
        b2 = exact_regime(read(f'receipts/router-b2-tuning-20260926/{node}.json'), 81592)
        if b1 != b2:
            raise ValueError('Proposed diagnostic capacity changes the B2 regime')
        selected.append(b2)
    if any(value != identities[0] for value in identities) or any(value != selected[0] for value in selected):
        raise ValueError('B2 identities or regimes differ across ranks')
    needle = 'receipts/router-b2-full-qualification-20260926/gates/correctness/original-524k'
    report = read(needle + '/report.json')
    if (len(report) != 3 or [r['repeat'] for r in report] != [0, 1, 2]
            or any(r['cached_tokens'] != 0 or not r['identical'] or r['correct'] for r in report)):
        raise ValueError('B2 historical failure is not three cold identical wrong trials')
    signatures = []
    salts = []
    for index in range(3):
        row = read(f'{needle}/{index}.json')
        request, response = row['request'], row['response']
        if (hashlib.sha256(request['messages'][0]['content'].encode()).hexdigest() != PROMPT_SHA
                or response['usage']['prompt_tokens'] != 524288):
            raise ValueError('Frozen historical prompt or token accounting changed')
        if (request['temperature'] != 0 or request['max_tokens'] != 64
                or request['top_logprobs'] != 20 or not request['logprobs']
                or request['chat_template_kwargs'] != {'thinking': False}):
            raise ValueError('Historical request settings differ')
        salts.append(request['cache_salt'])
        signatures.append(signature(response))
    if len(set(salts)) != 3 or len(set(signatures)) != 1:
        raise ValueError('B2 cold trial salts or complete signatures differ')
    return dict(**identities[0], regime=selected[0], response_signature=signatures[0],
                source_receipts=sources, proposed_capture_blocks=80927,
                scope='offline B2 comparison identity only; no capture or launch authorization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build_reference()
    with args.out.open('x') as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print('DECISION-REFERENCE-PREPARED', result['response_signature'])


if __name__ == '__main__':
    main()
