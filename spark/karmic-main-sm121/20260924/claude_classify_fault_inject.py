"""Classify one cross-rank fail-closed injection run from its evidence files.

Inputs:
  --receipt   the targeted node's ds41-engram-fault-inject-<token>.consumed (may be absent)
  --rank0-log the output rank's container log from the injected request onward
              (podman logs --since <request start>), so startup lines cannot
              count as confounders
  --response  JSON {"status": int | null, "elapsed_s": float, "body": <parsed JSON or text>,
              "error": <transport error text when no status was received>}
              for a NON-streaming request
  --prompt-tokens  the injected request's prompt token count (optional, recommended)

Verdicts (exit code):
  PASS 0          receipt fired for epoch E; the API rejected the request
                  (HTTP 4xx/5xx or a recorded transport error); the FIRST model
                  fault rank 0 logged is "Model reported a fault in this step
                  (fault epoch E, step epoch E)", and no confounder precedes it.
  FAIL 1          receipt fired but the request was served (any content,
                  reasoning, tool call or text in a 2xx response).
  CONFOUNDED 2    another failure (RoCE/NCCL timeout, disk read timeout or
                  failure, CUDA error, a fault for a different epoch) came first
                  or instead.
  NOT-FIRED 3     no receipt: the injection never happened (inconclusive).
  MISTARGETED 4   the skipped job is clearly larger than the injected prompt,
                  so it belonged to another request.
  INCONCLUSIVE 5  fired, but rejection is unproven (2xx without output, other
                  status) or rank 0 logged no fault message.
  CONFOUNDED also covers an earlier model fault for another epoch.
"""
import argparse
import json
from pathlib import Path
import re
import sys

FAULT = re.compile(r'Model reported a fault in (this step|an earlier step) '
                   r'\(fault epoch (\d+), step epoch (\d+)\)')
CONFOUNDERS = [
    ('roce-timeout', re.compile(r'RoCE collective on rank \d+ timed out')),
    ('roce-proxy', re.compile(r'RoCE (proxy|queue-pair)')),
    ('nccl', re.compile(r'NCCL error|ncclInternalError|ncclSystemError|ncclRemoteError|'
                        r'Watchdog caught collective operation timeout')),
    ('engram-read-timeout', re.compile(r'disk batch read has not completed')),
    ('engram-read-failure', re.compile(r'short PLE read|PLE batch launch failed|'
                                      r'io_uring (READ_FIXED|submit|completion wait)')),
    ('cuda', re.compile(r'CUDA error|cudaError|illegal memory access')),
]
PADDING_TOLERANCE = 64
CODES = {'PASS': 0, 'FAIL': 1, 'CONFOUNDED': 2, 'NOT-FIRED': 3, 'MISTARGETED': 4, 'INCONCLUSIVE': 5}


GENERATION_FIELDS = ('content', 'reasoning_content', 'reasoning', 'tool_calls', 'function_call', 'text')


def _nonempty(value):
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def has_generation(body):
    """True if any choice carries generated output of any kind."""
    if not isinstance(body, dict):
        return False
    for choice in body.get('choices') or []:
        if not isinstance(choice, dict):
            continue
        for part in (choice, choice.get('message') or {}, choice.get('delta') or {}):
            if isinstance(part, dict) and any(_nonempty(part.get(field)) for field in GENERATION_FIELDS):
                return True
    return False


def response_outcome(response):
    """served, rejected, or unproven.

    Only an HTTP 4xx/5xx status, or a transport error recorded with no
    status (connection reset by a dying engine), proves the API rejected
    the request. A 2xx response is served if it carries any generated
    output and unproven otherwise: an empty or non-JSON 2xx body does not
    prove rejection. Non-streaming requests are assumed.
    """
    status = response.get('status')
    if isinstance(status, int) and 200 <= status < 300:
        return 'served' if has_generation(response.get('body')) else 'unproven'
    if isinstance(status, int) and status >= 400:
        return 'rejected'
    if status is None and _nonempty(response.get('error')):
        return 'rejected'
    return 'unproven'


def classify(receipt, log, response, prompt_tokens=None):
    reasons = []
    first_confounder = None
    for name, pattern in CONFOUNDERS:
        match = pattern.search(log)
        if match and (first_confounder is None or match.start() < first_confounder[1]):
            first_confounder = (name, match.start())
    faults = [(m.start(), m.group(1), int(m.group(2)), int(m.group(3))) for m in FAULT.finditer(log)]
    outcome = response_outcome(response)
    if receipt is None:
        return 'NOT-FIRED', [f'no receipt; response {outcome}']
    epoch = int(receipt['epoch'])
    if prompt_tokens is not None:
        rows, prompt = int(receipt['tokens']), int(prompt_tokens)
        # A chunked prefill makes the job smaller and padding slightly larger;
        # a job clearly larger than the request belongs to another request.
        if rows > prompt + PADDING_TOLERANCE:
            return 'MISTARGETED', [f'skipped job had {rows} rows, larger than the {prompt}-token prompt']
        if rows != prompt:
            reasons.append(f'skipped job had {rows} rows for a {prompt}-token prompt (chunked or padded)')
    if outcome == 'served':
        return 'FAIL', [f'epoch {epoch} was skipped on rank {receipt["rank"]} but output was served '
                        f'after {response.get("elapsed_s")} s']
    if outcome == 'unproven':
        return 'INCONCLUSIVE', [f'status {response.get("status")!r} without output does not prove '
                                f'API rejection']
    if faults:
        first = faults[0]
        if not (first[1] == 'this step' and first[2] == epoch and first[3] == epoch):
            return 'CONFOUNDED', [f'first model fault was {first[1]!r} epoch {first[2]}/{first[3]}, '
                                  f'expected this step epoch {epoch}']
        if first_confounder and first_confounder[1] < first[0]:
            return 'CONFOUNDED', [f'{first_confounder[0]} preceded the expected fault message']
        if first_confounder:
            reasons.append(f'{first_confounder[0]} logged after the fault message (teardown)')
        return 'PASS', reasons + [f'rank 0 rejected epoch {epoch}; receipt from rank {receipt["rank"]} '
                                  f'on {receipt["node"]}']
    if first_confounder:
        return 'CONFOUNDED', [f'{first_confounder[0]} without the expected fault message']
    return 'INCONCLUSIVE', ['request rejected but rank 0 logged no model fault']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--rank0-log', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    parser.add_argument('--prompt-tokens', type=int)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text()) if args.receipt.is_file() else None
    verdict, reasons = classify(receipt, args.rank0_log.read_text(errors='replace'),
                                json.loads(args.response.read_text()), args.prompt_tokens)
    print(json.dumps({'verdict': verdict, 'reasons': reasons, 'receipt': receipt}, indent=2))
    sys.exit(CODES[verdict])


if __name__ == '__main__':
    main()
