"""Read-only final health receipts for the September 28 production restoration."""
import concurrent.futures
import json
from pathlib import Path
import subprocess

out=Path(__file__).resolve().parent/'receipts/production-restore-20260928T024357Z'
def check(h):
    cmd="podman ps --format '{{.Names}} {{.Image}} {{.Status}}'; free -b; journalctl -k --since '2026-09-28 02:43:57 UTC' --no-pager -o short-iso"
    p=subprocess.run(['ssh','-o','BatchMode=yes',h,cmd],text=True,capture_output=True,timeout=60)
    (out/(h+'-final-health.log')).write_text(p.stdout+p.stderr)
    if p.returncode: raise RuntimeError(h+p.stderr)
    lines=p.stdout.splitlines()
    return {'host':h,'container':lines[0],'NV_ERR_NO_MEMORY':sum('NV_ERR_NO_MEMORY' in s for s in lines),
            'other_faults':[s for s in lines if any(x in s for x in ('Xid (','Killed process','oom-kill'))]}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results=list(pool.map(check,('dusty','kirby','rusty','toby')))
(out/'final-health.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
