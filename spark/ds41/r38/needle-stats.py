#!/usr/bin/env python3
"""Summarize first-token margins across needle diagnostic series.

Provenance: added 2026-09-16. For each run record it reports the answer, cache
state, salt use and the logprob gap between the retrieval code token "739" and
the strongest identity-token alternative on the first output position. Positive
gap means the code led. Output is JSON on stdout.
"""
import json
from pathlib import Path
import statistics
import sys

CODE = "739"


def first_gap(content):
    if not content:
        return None
    top = {a["token"]: a["logprob"] for a in content[0]["top_logprobs"]}
    chosen = content[0]["token"]
    code = top.get(CODE)
    if code is None:
        code = content[0]["logprob"] if chosen == CODE else None
    others = [lp for tok, lp in top.items() if tok != CODE]
    if code is None or not others:
        return None
    return round(code - max(others), 4)


def load(series):
    rows = []
    for path in sorted(Path(series).glob("*.json")):
        if path.name in ("manifest.json", "summary.json") or path.name.endswith("-input.json") or "http-error" in path.name:
            continue
        rec = json.loads(path.read_text())
        if not isinstance(rec, dict) or "response" not in rec:
            continue  # analysis side files, not run records
        resp = rec["response"]
        if resp is None:
            continue
        choice = resp["choices"][0]
        body = rec["request_without_messages"]
        content = (choice.get("logprobs") or {}).get("content") or []
        rows.append({"label": rec["label"], "series": Path(series).name,
                     "prompt_tokens": resp["usage"]["prompt_tokens"],
                     "cached": resp["usage"]["prompt_tokens_details"]["cached_tokens"],
                     "salted": "cache_salt" in body, "condition": rec.get("condition"),
                     "source": rec.get("source"), "change": rec.get("change_from_source"),
                     "answer": choice["message"]["content"], "expected": rec["expected"],
                     "exact": choice["message"]["content"].strip() == rec["expected"],
                     "first_gap": first_gap(content), "first_token": content[0]["token"] if content else None})
    return rows


def group(rows):
    out = {}
    for row in rows:
        key = (row["series"], row["prompt_tokens"], row["condition"] or "original", "cold" if row["cached"] == 0 else "warm")
        out.setdefault(key, []).append(row)
    summary = []
    for key, items in sorted(out.items()):
        gaps = [r["first_gap"] for r in items if r["first_gap"] is not None]
        summary.append({"series": key[0], "prompt_tokens": key[1], "condition": key[2], "cache": key[3],
                        "n": len(items), "exact": sum(r["exact"] for r in items),
                        "gap_min": min(gaps) if gaps else None, "gap_max": max(gaps) if gaps else None,
                        "gap_mean": round(statistics.mean(gaps), 3) if gaps else None,
                        "gap_stdev": round(statistics.stdev(gaps), 3) if len(gaps) > 1 else None,
                        "labels": [r["label"] for r in items]})
    return summary


if __name__ == "__main__":
    rows = [row for series in sys.argv[1:] for row in load(series)]
    print(json.dumps({"groups": group(rows), "rows": rows}, indent=1))
