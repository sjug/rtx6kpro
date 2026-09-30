"""Execute the host-specific cleanup approved after the September 28 inventory."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
INVENTORY = json.loads((HERE / 'receipts/cleanup-inventory-20260928T021125Z.json').read_text())
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OLD = {
    'dusty': ['bld-jj-r38-spark/jj-r38-spark-sm121.docker.tar', 'bld-jj-r32-spark/jj-r32-spark-sm121.docker.tar', 'bld-jj-r28-spark/jj-r28-spark-sm121.docker.tar', 'bld-jj-r29-spark/jj-r29-spark-sm121.docker.tar', 'bld-jj-r29-spark/jj-r29-0b15723cb87646bb4628c5cd67aa2a7c879ac5a27105109ea5db06c30113dc0b.docker.tar'],
    'kirby': ['bld-jj-r29-spark/jj-r29-spark-sm121.docker.tar', 'bld-jj-r38-spark/jj-r38-spark-sm121.docker.tar', 'bld-jj-r32-spark/jj-r32-spark-sm121.docker.tar', 'bld-jj-r28-spark/jj-r28-spark-sm121.docker.tar'],
    'toby': ['bld-jj-r38-spark/jj-r38-spark-sm121.docker.tar', 'ds4-vision/jj-r32-spark-sm121.docker.tar', 'ds4-vision-r38/grammar-repair/jj-r38p-spark-sm121.docker-archive'],
    'rusty': ['bld-jj-r38-spark/jj-r38-spark-sm121.docker.tar', 'ds4-vision/jj-r32-spark-sm121.docker.tar', 'ds4-vision-r38/grammar-repair/jj-r38p-spark-sm121.docker-archive', 'ds4-vision-r38/grammar-repair/jj-r38p-oci-rejected.docker-archive'],
}

REMOTE = r'''
import json, pathlib, subprocess, shutil, os
p = PAYLOAD
def run(args):
    r = subprocess.run(args, text=True, capture_output=True, timeout=300)
    if r.returncode: raise RuntimeError(str(args)+': '+r.stderr)
    return r.stdout
def containers():
    ids=run(['podman','ps','-aq']).split()
    return json.loads(run(['podman','inspect',*ids])) if ids else []
cs=containers()
if any(c['State']['Running'] for c in cs): raise RuntimeError('A container is running; refusing cleanup')
byid={c['Id']:c for c in cs}
targets=[]
for expected in p['containers']:
    c=byid[expected['Id']]
    if c['Name'] != expected['Name'] or c['Image'] != expected['Image']: raise RuntimeError('Container identity changed')
    targets.append(c)
targetids={c['Id'] for c in targets}
keep=[c for c in cs if c['Id'] not in targetids]
protected={c['Image'] for c in keep}
images=sorted({c['Image'] for c in targets}-protected)
for path in p['archives']:
    q=pathlib.Path(path)
    if not q.is_file() or q.is_symlink() or str(q.resolve()) != path: raise RuntimeError('Archive target not a plain resolved file: '+path)
for path in p['caches']:
    q=pathlib.Path(path)
    if not q.is_dir() or q.is_symlink() or str(q.resolve()) != path: raise RuntimeError('Cache target changed: '+path)
    for c in keep:
        for m in c.get('Mounts',[]):
            src=m.get('Source','').rstrip('/')
            if src and (path==src or path.startswith(src+'/') or src.startswith(path+'/')):
                raise RuntimeError('Preserved container mounts cache: '+path)
root=pathlib.Path('/home/jugs/git/ds41-r38/karmic-main-20260924/receipts') / ('cleanup-'+p['stamp'])
root.mkdir(parents=True)
before=shutil.disk_usage('/home/jugs').free
report={'before_free':before,'removed_archives':[], 'removed_containers':[], 'removed_images':[], 'image_errors':[], 'removed_caches':[], 'preserved_containers':[{'Id':c['Id'],'Name':c['Name'],'Image':c['Image']} for c in keep]}
def save(): (root/'cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
# Preserve logs and metadata before any destructive operation, omitting environments.
for c in targets:
    with (root/(c['Name']+'.log')).open('w') as f:
        r=subprocess.run(['podman','logs',c['Id']],stdout=f,stderr=subprocess.STDOUT,timeout=300)
    if r.returncode: raise RuntimeError('Cannot preserve logs for '+c['Name'])
(root/'containers.json').write_text(json.dumps(p['containers'],indent=2)+'\n')
inspected=json.loads(run(['podman','image','inspect',*images]))
(root/'images.json').write_text(json.dumps([{k:i.get(k) for k in ('Id','RepoTags','Size','Labels','RootFS')} for i in inspected],indent=2)+'\n')
save()
for path in p['archives']:
    size=os.stat(path).st_size
    os.unlink(path)
    report['removed_archives'].append({'path':path,'bytes':size}); save()
for c in targets:
    run(['podman','rm',c['Id']])
    report['removed_containers'].append(c['Name']); save()
for i in inspected:
    try:
        # Do not prune parents or force deletion of referenced images.
        for tag in i.get('RepoTags') or []: run(['podman','untag',i['Id'],tag])
        run(['podman','rmi','--no-prune',i['Id']])
        report['removed_images'].append(i['Id'])
    except RuntimeError as e: report['image_errors'].append(str(e))
    save()
for path in p['caches']:
    shutil.rmtree(path)
    report['removed_caches'].append(path); save()
after=containers()
if {(c['Id'],c['Image']) for c in after} != {(c['Id'],c['Image']) for c in keep}: raise RuntimeError('Preserved-container verification failed')
for image in protected: run(['podman','image','exists',image])
report['after_free']=shutil.disk_usage('/home/jugs').free
report['receipt']=str(root)
report['preserved_verified']=True
save()
print(json.dumps(report))
'''

def cleanup(host):
    v = INVENTORY['hosts'][host]
    archives = [line.split('\t',1)[1] for line in v['large_task_files']['data'].splitlines()]
    if not all(path.startswith('/home/jugs/git/ds41-r38/karmic-main-20260924/') and path.endswith('.tar') for path in archives):
        raise RuntimeError('Unexpected archive inventory')
    caches = ['/home/jugs/.cache/vllm-jj-ds41-tp4']
    for line in v['task']['data'].splitlines():
        path = line.split('\t',1)[1]
        if Path(path).name.startswith(('selection-transplant-cache-', 'selection-replay-cache-', 'mhc-layer0-cache-', 'mhc-expanded-operator-cache-', 'router-compile-')):
            caches.append(path)
    payload = {'stamp':STAMP,'containers':[c for c in v['containers']['data'] if c['Name'].startswith('ds41')],
               'archives':archives+['/home/jugs/git/'+s for s in OLD[host]], 'caches':caches}
    code = REMOTE.replace('PAYLOAD', repr(payload), 1)
    r = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,'python3 -'],input=code,text=True,capture_output=True,timeout=1800)
    out = HERE/'receipts'/('cleanup-executed-'+STAMP+'-'+host+'.json')
    if r.returncode:
        out.write_text(json.dumps({'error':r.stderr,'stdout':r.stdout},indent=2)+'\n')
        return host, {'error':r.stderr}
    out.write_text(r.stdout)
    report=json.loads(r.stdout)
    return host, {k:report[k] for k in ('before_free','after_free','image_errors','preserved_verified','receipt')} | {k:len(report[k]) for k in ('removed_archives','removed_containers','removed_images','removed_caches')}

if __name__ == '__main__':
    hosts = sys.argv[1:] or ['dusty','kirby','rusty','toby']
    if any(h not in OLD for h in hosts):
        raise SystemExit('Unapproved host')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for host,result in pool.map(cleanup, hosts):
            print(host,json.dumps(result),flush=True)
