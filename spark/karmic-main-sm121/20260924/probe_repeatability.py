"""Frozen-prompt, cold-cache full-response and top-logprob repeatability gate."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid
from cache_metrics import idle_snapshot, finish


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', default='http://dusty:8000')
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lengths', default='256,513,514,1024,16384')
    p.add_argument('--repeats', type=int, default=6)
    p.add_argument('--max-tokens', type=int, default=8)
    p.add_argument('--filler', default=' filler')
    p.add_argument('--retrieval-code', default='739184')
    p.add_argument('--prompt-logprobs', type=int, default=None,
                   help='Diagnostic request override; omitted in the baseline')
    a = p.parse_args()
    if not (a.retrieval_code.isascii() and a.retrieval_code.isdigit()):
        p.error('--retrieval-code must contain ASCII digits only')
    a.out.mkdir(parents=True, exist_ok=False)
    def post(path, body):
        request = urllib.request.Request(a.base_url + path, json.dumps(body).encode(),
                                        {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.load(response)
    model = 'DeepSeek-V4.1-Flash'
    corpus = json.loads(a.corpus.read_text()) if a.corpus.exists() else {}
    report = []
    for length in map(int, a.lengths.split(',')):
        key = str(length)
        if key not in corpus:
            n = max(1, length - 100)
            for _ in range(12):
                messages = [{'role': 'user', 'content':
                    f'Archive identity 510c94b1bb4f42d2998093bd6b9c3d99. Memorize this unique retrieval code: {a.retrieval_code}.\nArchive:\n'
                    + a.filler * n + '\nArchive ends. Reply with only the retrieval code stated before the archive.'}]
                tokenized = post('/tokenize', {'model': model, 'messages': messages,
                    'add_generation_prompt': True, 'chat_template_kwargs': {'thinking': False}})
                if tokenized['count'] == length:
                    break
                n += length - tokenized['count']
            else:
                raise RuntimeError('Could not create exact-length prompt')
            corpus[key] = {'messages': messages, 'tokenize': tokenized, 'filler': a.filler,
                           'retrieval_code': a.retrieval_code}
            a.corpus.write_text(json.dumps(corpus, indent=1) + '\n')
        if corpus[key].get('filler', ' filler') != a.filler:
            raise RuntimeError('Filler differs from frozen corpus; use a separate corpus')
        if corpus[key].get('retrieval_code', '739184') != a.retrieval_code:
            raise RuntimeError('Retrieval code differs from frozen corpus; use a separate corpus')
        signatures = []
        for repeat in range(a.repeats):
            body = {'model': model, 'messages': corpus[key]['messages'], 'temperature': 0,
                    'max_tokens': a.max_tokens, 'logprobs': True, 'top_logprobs': 20,
                    'chat_template_kwargs': {'thinking': False}, 'cache_salt': uuid.uuid4().hex}
            if a.prompt_logprobs is not None:
                body['prompt_logprobs'] = a.prompt_logprobs
            before = idle_snapshot(a.base_url)
            start = time.monotonic()
            result = post('/v1/chat/completions', body)
            receipt = {'request': body, 'response': result, 'elapsed_s': time.monotonic() - start}
            name = f'{length}-{repeat}'
            (a.out / (name + '.json')).write_text(json.dumps(receipt, indent=1) + '\n')
            if result['usage']['prompt_tokens'] != length:
                raise RuntimeError('Token count drift')
            if finish(a.base_url, before, length, a.out / (name + '-cache.json'),
                      expected_queries=0 if a.prompt_logprobs is not None else None) != 0:
                raise RuntimeError('Unexpected cache hit')
            choice = result['choices'][0]
            if not choice.get('logprobs', {}).get('content'):
                raise RuntimeError('Missing logprobs')
            signature = {'message': choice['message'], 'finish_reason': choice['finish_reason'],
                         'logprobs': choice['logprobs']}
            signatures.append(signature)
            print(json.dumps({'length': length, 'repeat': repeat, 'elapsed_s': receipt['elapsed_s'],
                              'same_as_first': signature == signatures[0]}), flush=True)
        row = {'length': length, 'repeats': a.repeats,
               'identical': all(s == signatures[0] for s in signatures),
               'correct': all(s['message'].get('content', '').strip() == a.retrieval_code
                              and s['finish_reason'] == 'stop' for s in signatures)}
        report.append(row)
        (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if not all(r['identical'] and r['correct'] for r in report):
        raise SystemExit('REPEATABILITY-FAIL')
    print('REPEATABILITY-PASS', flush=True)


if __name__ == '__main__':
    main()
