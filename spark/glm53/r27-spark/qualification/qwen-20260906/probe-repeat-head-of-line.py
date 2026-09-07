import json, time, urllib.request, threading, uuid
base="http://dusty:8000"; model="Qwen3.8-Flash-Next-NVFP4-4p89"
def chat(label, prompt, max_tokens, out):
    body=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0,"ignore_eos":True,"stream":True,"chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(base+"/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
    t0=time.monotonic(); ttft=None
    with urllib.request.urlopen(req,timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b'"content"' in line and ttft is None: ttft=time.monotonic()-t0
    out[label]=(ttft, time.monotonic()-t0)
seed=f"Essay {uuid.uuid4().hex}: describe the geology of a volcanic island in detail."
o={}; chat("seed", seed, 60, o); print("seed done", {k:tuple(round(x,1) for x in v) for k,v in o.items()})
res={}; th=[]
def start(label, prompt, n, delay):
    time.sleep(delay); t=threading.Thread(target=chat,args=(label,prompt,n,res)); t.start(); th.append(t)
T0=time.monotonic()
for i in range(2): start(f"long{i}", f"Essay {uuid.uuid4().hex}: history of canals.", 600, 0)
start("repeat", seed, 60, 2.0)          # verbatim repeat of a completed prompt, submitted while two long decodes run
for i in range(2): start(f"fresh{i}", f"Essay {uuid.uuid4().hex}: history of bridges.", 60, 3.0)
time.sleep(4); [t.join() for t in th]
for k,(ttft,tot) in res.items(): print(f"{k:8s} ttft={ttft:6.1f}s total={tot:6.1f}s")
