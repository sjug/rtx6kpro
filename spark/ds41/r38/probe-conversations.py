#!/usr/bin/env python3
"""Exact-answer prefix reuse and concurrent multi-turn DS4.1 admission."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import re
import urllib.error
import urllib.request
import uuid

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base-url", default="http://dusty:8000")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--mixed-needle-input", type=Path,
               help="Replay this needle with a fresh archive ID while four conversations run")
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
MODEL = "DeepSeek-V4.1-Flash"


def chat(label, messages, expected):
    body = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 64,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(a.base_url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        (a.out / f"{label}-http-error.json").write_text(json.dumps({
            "status": error.code, "body": error.read().decode(errors="replace"),
            "request": body, "started_unix": start,
        }, indent=2) + "\n")
        raise
    receipt = {"label": label, "started_unix": start, "elapsed_s": time.time() - start,
               "expected": expected, "request": body, "response": result}
    (a.out / f"{label}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    choice = result["choices"][0]
    answer = choice["message"].get("content") or ""
    if answer.strip() != expected or choice["finish_reason"] != "stop":
        raise RuntimeError(f"Wrong or incomplete {label}: {choice}")
    details = result["usage"].get("prompt_tokens_details", {})
    cached = details.get("cached_tokens")
    if cached is None:
        raise RuntimeError(f"Missing cached-token accounting for {label}")
    print(json.dumps({"label": label, "answer": answer, "elapsed_s": receipt["elapsed_s"],
                      "prompt_tokens": result["usage"]["prompt_tokens"], "cached_tokens": cached}), flush=True)
    return choice["message"], cached, result["usage"]["prompt_tokens"]


def user(text):
    return {"role": "user", "content": text}


identity = uuid.uuid4().hex
archive = (f"Archive {identity}. Remember access code 739184.\n"
           + " archived material" * 16384 + "\nReply only with the access code.")
base = [user(archive)]
first, cached, total = chat("prefix-cold", base, "739184")
if cached:
    raise RuntimeError("Unique prefix was not cold")
repeat, cached, total = chat("prefix-repeat", base, "739184")
if cached / total < 0.95:
    raise RuntimeError("Exact repeat did not reuse prefix")
extension = base + [first, user("Repeat the access code alone.")]
_, cached, total = chat("prefix-extension", extension, "739184")
if cached / total < 0.95:
    raise RuntimeError("Conversation extension did not reuse prefix")
_, cached, total = chat("prefix-divergent", [user(archive + "\nUse digits only.")], "739184")
if cached / total < 0.95:
    raise RuntimeError("Shared-prefix divergent suffix did not reuse prefix")


def conversation(index):
    code = str(630140 + index)
    messages = [user(f"Conversation {uuid.uuid4().hex}. Remember code {code}. Reply only with that code.")]
    first, _, _ = chat(f"concurrent-{index}-turn1", messages, code)
    messages += [first, user("What code did I give you? Reply only with the code.")]
    second, _, _ = chat(f"concurrent-{index}-turn2", messages, code)
    messages += [second, user("Add one to that code and reply only with the integer.")]
    chat(f"concurrent-{index}-turn3", messages, str(int(code) + 1))


def metric_total(text, name):
    """Sum one gauge over its label sets; an absent gauge is an error, not zero."""
    values = [float(match.group(1)) for match in re.finditer(
        rf"^{re.escape(name)}(?:\{{[^}}\n]*\}})?\s+([0-9.eE+-]+)$", text, re.M)]
    if not values:
        raise RuntimeError(f"Metric {name} is missing from /metrics")
    return sum(values)


def scrape():
    with urllib.request.urlopen(a.base_url + "/metrics", timeout=10) as response:
        text = response.read().decode()
    return {"unix": time.time(), "text": text,
            "running": metric_total(text, "vllm:num_requests_running"),
            "waiting": metric_total(text, "vllm:num_requests_waiting")}


def sample_engine(stop, long_run, samples, errors):
    # long_done is client-side: whether the long response had returned yet.
    while not stop.is_set():
        try:
            sample = scrape()
        except Exception as error:  # Incomplete evidence is reported, never ignored.
            errors.append(repr(error))
            return
        samples.append({"unix": sample["unix"], "running": sample["running"],
                        "waiting": sample["waiting"], "long_done": long_run.done()})
        stop.wait(0.25)


def mixed_evidence(baseline, submitted, started, samples, errors):
    long = json.loads((a.out / "mixed-long-needle.json").read_text())
    long_end = long["started_unix"] + long["elapsed_s"]
    turns = []
    for path in sorted(a.out.glob("concurrent-*.json")):
        if re.fullmatch(r"concurrent-\d+-turn\d\.json", path.name):
            turn = json.loads(path.read_text())
            turns.append({"label": turn["label"], "started_unix": turn["started_unix"],
                          "ended_unix": turn["started_unix"] + turn["elapsed_s"]})
    active = [sample for sample in samples if not sample["long_done"]]
    shared = [sample for sample in active if sample["running"] >= 2]
    before_end = [turn["label"] for turn in turns if turn["ended_unix"] < long_end]
    return {
        "baseline": {key: baseline[key] for key in ("unix", "running", "waiting")},
        "long_submitted_unix": submitted,
        "long_running_observed": {key: started[key] for key in ("unix", "running", "waiting")},
        "long_ended_unix": long_end,
        "turns": turns,
        "turns_completed_before_long_end": before_end,
        "sampler_errors": errors,
        "samples": len(samples),
        "samples_while_long_active": len(active),
        "samples_running_ge_2_while_long_active": len(shared),
        "max_running_while_long_active": max((s["running"] for s in active), default=None),
        "max_waiting_while_long_active": max((s["waiting"] for s in active), default=None),
        "queue_observed": any(s["waiting"] >= 1 for s in active),
        # Submission alone never counts: require an idle baseline, the long request
        # observed running, a concurrent engine sample and a turn finishing first.
        "overlap_observed": not errors and bool(shared) and bool(before_end),
    }


with ThreadPoolExecutor(max_workers=5) as pool:
    long_run = sampler = baseline = started = submitted = None
    stop = threading.Event()
    samples, sampler_errors = [], []
    if a.mixed_needle_input:
        needle = json.loads(a.mixed_needle_input.read_text())
        text, replacements = re.subn(r"Archive identity [0-9a-f]{32}",
                                    f"Archive identity {uuid.uuid4().hex}", needle["messages"][0]["content"])
        if replacements != 1:
            raise RuntimeError("Cannot give the mixed-load needle a unique prefix")
        # Attribute later engine activity to this probe only from an idle engine.
        deadline = time.monotonic() + 30
        while True:
            baseline = scrape()
            if baseline["running"] == 0 and baseline["waiting"] == 0:
                break
            if time.monotonic() >= deadline:
                (a.out / "mixed-baseline.prom").write_text(baseline["text"])
                raise RuntimeError("Engine was not idle before the mixed-load needle: "
                                   f"running={baseline['running']} waiting={baseline['waiting']}")
            time.sleep(0.5)
        (a.out / "mixed-baseline.prom").write_text(baseline["text"])
        submitted = time.time()
        long_run = pool.submit(chat, "mixed-long-needle", [user(text)], needle["expected"])
        # Observe actual engine activity before injecting the short requests.
        deadline = time.monotonic() + 30
        while True:
            if long_run.done():
                long_run.result()
                raise RuntimeError("Long request finished before overlap was established")
            started = scrape()
            if started["running"] >= 1:
                (a.out / "mixed-overlap-start.prom").write_text(started["text"])
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("No running long request observed")
            time.sleep(0.2)
        sampler = threading.Thread(target=sample_engine,
                                   args=(stop, long_run, samples, sampler_errors), daemon=True)
        sampler.start()
    try:
        list(pool.map(conversation, range(4)))
    finally:
        if sampler is not None:
            stop.set()
            sampler.join()
            (a.out / "mixed-samples.json").write_text(json.dumps(
                {"samples": samples, "sampler_errors": sampler_errors}, indent=2) + "\n")
    if long_run is not None and baseline and started and submitted is not None:
        _, cached, _ = long_run.result()
        if cached != 0:
            raise RuntimeError("Mixed-load needle was not cold")
        evidence = mixed_evidence(baseline, submitted, started, samples, sampler_errors)
        (a.out / "mixed-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps({"mixed_evidence": {key: value for key, value in evidence.items()
                                             if key != "turns"}}), flush=True)
        if not evidence["overlap_observed"]:
            raise RuntimeError("Mixed-load overlap was not observed; see mixed-evidence.json")
print("DS41-CONVERSATIONS-PASS", flush=True)
