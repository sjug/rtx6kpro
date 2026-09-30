"""Read-only remote inventory. Writes one local receipt; performs no cleanup."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import json, os, subprocess
def run(argv, parse=False):
    p = subprocess.run(argv, text=True, capture_output=True, timeout=240)
    value = json.loads(p.stdout) if parse and p.returncode == 0 else p.stdout
    return {'exit': p.returncode, 'data': value, 'stderr': p.stderr}
result = {}
result['images'] = run(['podman','images','--no-trunc','--format','json'], True)
ids = sorted({i['Id'] for i in result['images']['data']}) if result['images']['exit'] == 0 else []
if ids:
    inspected = run(['podman','image','inspect',*ids], True)
    if inspected['exit'] == 0:
        inspected['data'] = [{k:i.get(k) for k in ('Id','RepoTags','Size','Created','Parent','Labels','RootFS')} for i in inspected['data']]
    result['image_details'] = inspected
ps = run(['podman','ps','-aq'])
if ps['exit'] == 0 and ps['data'].strip():
    inspected = run(['podman','inspect',*ps['data'].split()], True)
    if inspected['exit'] == 0:
        inspected['data'] = [{'Id':i['Id'],'Name':i['Name'],'Image':i['Image'],'ImageName':i.get('ImageName'),
          'State':{k:i['State'].get(k) for k in ('Status','Running','ExitCode','OOMKilled')},
          'Mounts':[{k:m.get(k) for k in ('Source','Destination','Type')} for m in i.get('Mounts',[])]} for i in inspected['data']]
    result['containers'] = inspected
else:
    result['containers'] = {'exit':ps['exit'],'data':[],'stderr':ps['stderr']}
result['storage'] = run(['podman','system','df','--format','json'],True)
result['filesystem'] = run(['df','-B1','/home/jugs'])
for key, path, depth in [('task','/home/jugs/git/ds41-r38',2),('git','/home/jugs/git',1),('cache','/home/jugs/.cache',1)]:
    result[key] = run(['du','-x','-B1','--max-depth='+str(depth),path])
root='/home/jugs/git/ds41-r38'
result['large_task_files']=run(['find',root,'-xdev','-type','f','-size','+512M','-printf','%s\t%p\n'])
result['temporary']=run(['find','/tmp','-maxdepth','2','-user',str(os.getuid()),'-iname','*ds41*','-printf','%y\t%s\t%p\n'])
units=run(['systemctl','--user','list-units','--all','--no-pager','--plain'])
if units['exit']==0:
    units['data']='\n'.join(line for line in units['data'].splitlines() if any(s in line.lower() for s in ('ds41','ds4.1','mhc','determinism')))
result['units']=units
print(json.dumps(result))
'''

def host(node):
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',node,'python3 -'],
                            input=REMOTE,text=True,capture_output=True,timeout=900)
    if result.returncode:
        return node, {'error':result.stderr,'exit':result.returncode}
    return node, json.loads(result.stdout)

if __name__ == '__main__':
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(__file__).resolve().parent / 'receipts' / f'cleanup-inventory-{stamp}.json'
    result = {'timestamp':stamp,'hosts':{}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for node, data in pool.map(host, ('dusty','toby','rusty','kirby')):
            result['hosts'][node]=data
            out.write_text(json.dumps(result,indent=2)+'\n')
            print('INVENTORIED',node,flush=True)
    print('RECEIPT',out,flush=True)
