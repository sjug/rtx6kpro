#!/usr/bin/env python3
"""Structured-output probes for II r14 #320 (speculative structured output
validated before commit) + #294/#295 grammar fixes, under DSpark K5.

Three sections, all against a live pair (PROBE_URL, default dusty:8000):
  1. json_schema response_format arithmetic at c1: N requests must return
     schema-valid JSON with the CORRECT sum (mirrors the upstream r15
     receipt's "expected arithmetic JSON").
  2. tool_choice=required strict tool calls at c1 and c8: every response
     must finish with tool_calls whose arguments parse and conform to the
     tool schema (keys + types). Values are model-chosen; structure is the
     gate (DSpark probabilistic outputs are nondeterministic at temp 0).
  3. One 225k-context strict-tools request (r14 receipt pattern at 2-GPU
     scale): long filler prefill, then a required tool call, same
     conformance gate plus needle recall inside the tool arguments.
"""

import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("PROBE_URL", "http://dusty:8000")
MODEL = os.environ.get("PROBE_MODEL", "DeepSeek-V4-Flash-0731")
N_SCHEMA = int(os.environ.get("N_SCHEMA", "24"))
N_TOOLS_C1 = int(os.environ.get("N_TOOLS_C1", "24"))
N_TOOLS_C8 = int(os.environ.get("N_TOOLS_C8", "64"))

failures = []


def post(obj, timeout=2400):
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        json.dumps(obj).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# 1. json_schema arithmetic, c1.
schema = {
    "type": "object",
    "properties": {"sum": {"type": "integer"}},
    "required": ["sum"],
    "additionalProperties": False,
}
ok = 0
for i in range(N_SCHEMA):
    a, b = 1000 + 37 * i, 4000 + 91 * i
    d = post(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Compute {a}+{b}. Answer only via the JSON schema.",
                }
            ],
            "max_tokens": 600,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "arith", "schema": schema, "strict": True},
            },
        }
    )
    txt = d["choices"][0]["message"]["content"]
    try:
        obj = json.loads(txt)
        assert set(obj) == {"sum"} and isinstance(obj["sum"], int)
        if obj["sum"] == a + b:
            ok += 1
        else:
            failures.append(f"schema[{i}]: wrong sum {obj['sum']} != {a + b}")
    except Exception as exc:
        failures.append(f"schema[{i}]: invalid JSON/schema: {exc}: {txt[:120]!r}")
print(f"json_schema arithmetic c1: {ok}/{N_SCHEMA} correct-sum, schema-valid")

# 2. strict tool calls.
TOOL = {
    "type": "function",
    "function": {
        "name": "record_reading",
        "description": "Record a sensor reading",
        "parameters": {
            "type": "object",
            "properties": {
                "sensor": {"type": "string"},
                "value": {"type": "number"},
                "unit": {"type": "string", "enum": ["C", "F", "K"]},
            },
            "required": ["sensor", "value", "unit"],
            "additionalProperties": False,
        },
    },
}


def tool_call(i, extra_ctx=""):
    d = post(
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": extra_ctx
                    + f"Sensor probe {i} reads 21.{i % 10} degrees Celsius. "
                    "Record it with the tool.",
                }
            ],
            "max_tokens": 600,
            "temperature": 0,
            "tools": [TOOL],
            "tool_choice": "required",
        }
    )
    c = d["choices"][0]
    if c["finish_reason"] != "tool_calls":
        return f"[{i}] finish_reason={c['finish_reason']}"
    calls = c["message"].get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "record_reading":
        return f"[{i}] bad tool selection: {calls!r:.120}"
    try:
        args = json.loads(calls[0]["function"]["arguments"])
        assert set(args) == {"sensor", "value", "unit"}
        assert isinstance(args["value"], (int, float))
        assert args["unit"] in ("C", "F", "K")
    except Exception as exc:
        return f"[{i}] non-conformant arguments: {exc}"
    return None


bad = [r for r in (tool_call(i) for i in range(N_TOOLS_C1)) if r]
print(f"strict tools c1: {N_TOOLS_C1 - len(bad)}/{N_TOOLS_C1} conformant")
failures += [f"tools-c1{b}" for b in bad]

with cf.ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(tool_call, range(1000, 1000 + N_TOOLS_C8)))
bad = [r for r in res if r]
print(f"strict tools c8: {N_TOOLS_C8 - len(bad)}/{N_TOOLS_C8} conformant")
failures += [f"tools-c8{b}" for b in bad]

# 3. 225k-context strict tool call with needle recall in the arguments.
unit = (
    "Telemetry frame %d: coolant nominal, fan curve steady, link margin "
    "within spec, no parity faults observed during the interval. "
)
needle = "CRITICAL: sensor of record is 'aft-plenum-7' at 63.4 degrees Celsius. "
n_units = int(os.environ.get("LONG_UNITS", "8600"))  # ~225k tokens
i_needle = int(n_units * 0.05)
body = "".join((needle if i == i_needle else "") + (unit % i) for i in range(n_units))
t0 = time.time()
d = post(
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": body
                + "\n\nRecord the sensor named in the CRITICAL line far above, "
                "with its stated value and unit, using the tool.",
            }
        ],
        "max_tokens": 600,
        "temperature": 0,
        "tools": [TOOL],
        "tool_choice": "required",
    }
)
dt = time.time() - t0
c = d["choices"][0]
ptoks = d.get("usage", {}).get("prompt_tokens", 0)
label = f"long-ctx ({ptoks} prompt tokens, {dt:.0f}s)"
if c["finish_reason"] != "tool_calls":
    failures.append(f"{label}: finish_reason={c['finish_reason']}")
else:
    try:
        args = json.loads(c["message"]["tool_calls"][0]["function"]["arguments"])
        assert set(args) == {"sensor", "value", "unit"}
        recall = args["sensor"] == "aft-plenum-7" and abs(args["value"] - 63.4) < 0.05
        print(f"{label}: conformant, needle-recall={'EXACT' if recall else 'MISS'}"
              f" args={args}")
        if not recall:
            failures.append(f"{label}: needle miss: {args}")
    except Exception as exc:
        failures.append(f"{label}: non-conformant: {exc}")

if failures:
    print("STRICT-TOOLS-FAIL:", "; ".join(failures[:12]))
    sys.exit(1)
print("STRICT-TOOLS-OK")
