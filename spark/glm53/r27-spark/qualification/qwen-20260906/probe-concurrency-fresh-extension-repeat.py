import json, time, urllib.request, concurrent.futures, uuid
base="http://dusty:8000"; model="Qwen3.8-Flash-Next-NVFP4-4p89"
def chat(messages, max_tokens=200):
    body=json.dumps({"model":model,"messages":messages,"max_tokens":max_tokens,"temperature":0,"ignore_eos":True,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(base+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=600) as r: p=json.load(r)
    return time.monotonic()-t, p["usage"]["completion_tokens"], p["choices"][0]["message"]["content"]
def run(label, msgs_list):
    t=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(len(msgs_list)) as ex: res=list(ex.map(chat,msgs_list))
    wall=time.monotonic()-t; toks=sum(c for _,c,_ in res)
    print(f"{label}: n={len(msgs_list)} wall={wall:.1f}s agg_tok/s={toks/wall:.1f} per_req_s={[round(d,1) for d,_,_ in res]}", flush=True)
    return res
fresh=lambda: [[{"role":"user","content":f"Essay {uuid.uuid4().hex}: describe the geology of a volcanic island in detail."}] for _ in range(4)]
a=fresh(); r=run("A: 4 fresh distinct", a)
# multi-turn extensions of A: each appends the assistant reply and a new user turn (prefix = a previous full prompt+response)
ext=[m+[{"role":"assistant","content":c},{"role":"user","content":"Continue with the island's climate."}] for m,(_,_,c) in zip(a,r)]
run("B: 4 extensions of A (multi-turn)", ext)
run("C: 4 fresh distinct after B", fresh())
run("D: repeat A verbatim", a)
run("E: 4 fresh distinct after D", fresh())
