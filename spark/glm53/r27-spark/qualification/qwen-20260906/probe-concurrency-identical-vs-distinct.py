import json, sys, time, urllib.request, concurrent.futures
base="http://dusty:8000"; model="Qwen3.8-Flash-Next-NVFP4-4p89"
def chat(prompt, max_tokens=200):
    body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(base+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=600) as r: p=json.load(r)
    return time.monotonic()-t, p["usage"]["completion_tokens"]
def run(label, prompts):
    t=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as ex: res=list(ex.map(chat,prompts))
    wall=time.monotonic()-t; toks=sum(c for _,c in res)
    print(f"{label}: n={len(prompts)} wall={wall:.1f}s agg_tok/s={toks/wall:.1f} per_req_s={[round(d,1) for d,_ in res]}", flush=True)
distinct=[f"Write a long essay number {i} about the history of railway signalling in a different country for each number." for i in range(4)]
same=["Write a long essay about the history of railway signalling in Switzerland."]*4
run("warm c1", distinct[:1])
run("c4 distinct", distinct)
run("c4 identical", same)
run("c4 distinct again", distinct)
run("c1", distinct[:1])
