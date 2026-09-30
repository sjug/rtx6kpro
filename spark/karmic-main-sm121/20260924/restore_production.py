"""Restore the exact retained pre-investigation containers, without replacing them."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import time
import urllib.request

HERE=Path(__file__).resolve().parent
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=HERE/'receipts'/('production-restore-'+STAMP)
OUT.mkdir()
spec={
 'dusty':('qwen38-flash-next-nvfp4-karmic-main-hcbase-qsa865-tp2','082691f6285ceb9c215cb51a7a08e8ca45f5f57e672362b4750c2a5cba242dc6','a25bedd43581dffecca2f9eefb40e60b170af6ae19124fef1c292850b2a0483e'),
 'kirby':('qwen38-flash-next-nvfp4-karmic-main-hcbase-qsa865-tp2','280538ed0dbc8e80ef02c49ada1aef665fc7cb34284d3c235811f3b4a7a89e75','a25bedd43581dffecca2f9eefb40e60b170af6ae19124fef1c292850b2a0483e'),
}
for h in ('rusty','toby'):
 c=json.loads((HERE/'receipts'/('previous-'+h+'.json')).read_text())[0]
 spec[h]=(c['Name'],c['Id'],c['Image'])
def ssh(h,cmd):
 p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',h,cmd],text=True,capture_output=True,timeout=90)
 if p.returncode: raise RuntimeError(h+': '+p.stderr)
 return p.stdout
def preflight(h):
 name,cid,image=spec[h]
 if ssh(h,'podman ps -q').strip(): raise RuntimeError(h+' is not idle')
 c=json.loads(ssh(h,'podman inspect '+name))[0]
 if (c['Id'],c['Image'],c['State']['ExitCode'],c['State']['OOMKilled']) != (cid,image,0,False): raise RuntimeError(h+' identity/state mismatch')
 ssh(h,'podman image exists '+image)
 paths=[m['Source'] for m in c.get('Mounts',[]) if m['Type']=='bind']
 import shlex
 ssh(h,' && '.join('test -e '+shlex.quote(p) for p in paths) or 'true')
 result={'host':h,'Id':cid,'Image':image,'Name':name,'Mounts':paths,'State':c['State']}
 (OUT/(h+'-before.json')).write_text(json.dumps(result,indent=2)+'\n')
 print('PREFLIGHT PASS',h,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(preflight,spec))
for h in ('kirby','dusty','toby','rusty'):
 print('START',h,ssh(h,'podman start '+spec[h][0]).strip(),flush=True)
ready=set()
deadline=time.monotonic()+1200
while time.monotonic()<deadline and len(ready)<2:
 for h in ('dusty','rusty'):
  if h in ready: continue
  try:
   with urllib.request.urlopen('http://'+h+':8000/v1/models',timeout=5) as r: data=json.load(r)
   (OUT/(h+'-models.json')).write_text(json.dumps(data,indent=2)+'\n')
   ready.add(h); print('API READY',h,flush=True)
  except (OSError,ValueError) as e: print('WAIT',h,type(e).__name__,flush=True)
 for h,(name,cid,image) in spec.items():
  status=ssh(h,"podman inspect --format '{{.State.Status}}' "+name).strip()
  if status!='running': raise RuntimeError(h+' container '+status)
 if len(ready)<2: time.sleep(20)
if len(ready)<2: raise RuntimeError('Readiness deadline exceeded')
for h,model in [('dusty','Qwen3.8-Flash-Next'),('rusty','DeepSeek-V4-Flash-Vision-Exp')]:
 payload={'model':model,'messages':[{'role':'user','content':'What is 3 + 4? Reply with only the number.'}],'temperature':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}}
 req=urllib.request.Request('http://'+h+':8000/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r: data=json.load(r)
 (OUT/(h+'-completion.json')).write_text(json.dumps(data,indent=2)+'\n')
 choice=data['choices'][0]
 print('COMPLETION',h,json.dumps(choice),flush=True)
 if choice['message']['content'].strip()!='7' or choice['finish_reason']!='stop': raise RuntimeError('Unexpected completion '+h)
for h,(name,cid,image) in spec.items():
 logs=ssh(h,'podman logs --since '+STAMP[:4]+'-'+STAMP[4:6]+'-'+STAMP[6:8]+'T'+STAMP[9:11]+':'+STAMP[11:13]+':'+STAMP[13:15]+'Z '+name+' 2>&1')
 (OUT/(h+'-startup.log')).write_text(logs)
print('RESTORATION PASS',OUT,flush=True)
