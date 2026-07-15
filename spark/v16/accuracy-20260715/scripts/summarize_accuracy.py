#!/usr/bin/env python3
"""One-line summary of a completion-stats results JSON (accuracy + request errors)."""
import json
import sys


def walk(o):
    if isinstance(o, dict):
        if isinstance(o.get("correct"), bool):
            yield o
        for v in o.values():
            yield from walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from walk(v)


def main(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    s = data.get("all_summary")
    if isinstance(s, dict) and "attempted" in s:
        toks = s.get("completion_tokens") or {}
        rate = s.get("correct_rate")
        acc = f"{100 * rate:.1f}%" if rate is not None else "n/a"
        print(
            f"correct={s.get('correct')} wrong={s.get('wrong')} acc={acc} "
            f"errors={s.get('errors')}/{s.get('attempted')} "
            f"p50_tokens={int(toks.get('p50') or 0)} max_tokens={int(toks.get('max') or 0)}"
        )
        return
    items = list(walk(data))
    if not items:
        print("no-summary no-correct-field")
        return
    right = sum(1 for it in items if it["correct"])
    print(f"correct={right}/{len(items)} ({100.0 * right / len(items):.1f}%) errors=0/{len(items)}")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as exc:  # tolerant: summary must never fail the sweep
        print(f"summary-error: {exc}")
