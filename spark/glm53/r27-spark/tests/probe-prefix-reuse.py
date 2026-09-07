#!/usr/bin/env python3
"""Quantify prefix reuse for one long prompt: fresh, multi-turn extension, and
verbatim repeat. Reports TTFT and the server's prefix-cache counter deltas
(queries/hits in blocks) per request, plus reported cached_tokens if present."""
import argparse, json, re, time, urllib.request
p=argparse.ArgumentParser(); p.add_argument("--base-url", default="http://dusty:8000"); p.add_argument("--model", default="Qwen3.8-Flash-Next-NVFP4-4p89")
p.add_argument("--filler-tokens", type=int, default=32000); p.add_argument("--label", required=True); p.add_argument("--receipt-file", required=True)
a=p.parse_args()
def metrics():
    raw=urllib.request.urlopen(a.base_url+"/metrics",timeout=30).read().decode(); v={}
    for line in raw.splitlines():
        m=re.match(r"^vllm:(prefix_cache_queries_total|prefix_cache_hits_total)(\{[^}]*\})? (\S+)$", line)
        if m: v[m.group(1)]=v.get(m.group(1),0.0)+float(m.group(3))
    return v
def chat(messages, max_tokens=48):
    body=json.dumps({"model":a.model,"messages":messages,"max_tokens":max_tokens,"temperature":0,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(a.base_url+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.monotonic(); ttft=None; usage=None; text=""
    with urllib.request.urlopen(req,timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data:") or b"[DONE]" in line: continue
            d=json.loads(line[5:]); ch=d.get("choices") or []
            if ch and ch[0].get("delta",{}).get("content"):
                text+=ch[0]["delta"]["content"]
                if ttft is None: ttft=time.monotonic()-t0
            if d.get("usage"): usage=d["usage"]
    return ttft, time.monotonic()-t0, usage, text
base=[{"role":"user","content":"Remember the code 739184. Archive follows:"+" filler"*a.filler_tokens+"\nWhat was the code? Reply with the code only."}]
steps=[("fresh", base)]
out=open(a.receipt_file,"a"); results=[]
for name,msgs in steps+[]:
    pass
def run(name,msgs):
    m0=metrics(); ttft,total,usage,text=chat(msgs); m1=metrics()
    rec={"label":a.label,"step":name,"ttft_s":round(ttft or -1,3),"total_s":round(total,3),"prompt_tokens":(usage or {}).get("prompt_tokens"),
         "cached_tokens":((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
         "prefix_queries_delta":m1.get("prefix_cache_queries_total",0)-m0.get("prefix_cache_queries_total",0),
         "prefix_hits_delta":m1.get("prefix_cache_hits_total",0)-m0.get("prefix_cache_hits_total",0),"answer":text.strip()[:40]}
    print(json.dumps(rec)); out.write(json.dumps(rec)+"\n"); return text
r1=run("fresh", base)
ext=base+[{"role":"assistant","content":r1},{"role":"user","content":"Now state the code again followed by the word done."}]
run("extension", ext)
run("verbatim_repeat", base)
run("extension_repeat", ext)
