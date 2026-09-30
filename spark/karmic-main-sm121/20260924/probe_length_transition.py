"""Replay frozen cold prompts around a request-length transition without restarts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--prime-lengths', default='385,384,400')
    parser.add_argument('--prime-corpus', type=Path)
    parser.add_argument('--prime-filler', default=' filler')
    parser.add_argument('--prime-code', default='739184')
    parser.add_argument('--target-prompt-logprobs', type=int, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name('probe_repeatability.py')
    results = []
    # Prime twice, then compare three cold 385-token requests. Repeated 385 is
    # the same-length control; 384 and 400 straddle it without long I/O stalls.
    for cycle in range(args.cycles):
        for prior in map(int, args.prime_lengths.split(',')):
            for label, length, repeats in (('prime', prior, 2), ('target', 385, 3)):
                dest = args.out / f'{cycle}-{prior}-{label}'
                corpus = args.prime_corpus if label == 'prime' and args.prime_corpus else args.corpus
                command = [sys.executable, '-u', str(script), '--corpus', str(corpus),
                           '--out', str(dest), '--lengths', str(length),
                           '--repeats', str(repeats)]
                if label == 'prime':
                    command += ['--filler', args.prime_filler, '--retrieval-code', args.prime_code]
                elif args.target_prompt_logprobs is not None:
                    command += ['--prompt-logprobs', str(args.target_prompt_logprobs)]
                proc = subprocess.run(command, text=True, capture_output=True)
                print(proc.stdout, end='', flush=True)
                # A repeatability verdict is evidence here, not authorization
                # to stop collecting the matched transition controls.
                if proc.returncode and proc.stderr.strip() != 'REPEATABILITY-FAIL':
                    raise RuntimeError(proc.stderr)
                choices = [json.loads((dest / f'{length}-{i}.json').read_text())
                           ['response']['choices'][0] for i in range(repeats)]
                results.append({'cycle': cycle, 'prior': prior, 'label': label,
                    'command': command, 'returncode': proc.returncode,
                    'stderr': proc.stderr,
                    'answers': [c['message']['content'] for c in choices],
                    'logprob_sha256': [hashlib.sha256(json.dumps(c['logprobs'],
                        sort_keys=True).encode()).hexdigest() for c in choices]})
                (args.out / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print('TRANSITION-DIAGNOSTIC-COMPLETE; not a qualification verdict', flush=True)


if __name__ == '__main__':
    main()
