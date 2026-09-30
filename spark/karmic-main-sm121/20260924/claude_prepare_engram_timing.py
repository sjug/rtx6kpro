"""Compose the diagnostic-only Engram host-path timing layer from pinned blobs.

Reads the three target files with `git show` from the pinned commits (their
bytes equal the image's files per runtime.lock.json "files"), applies exact
one-match text insertions that only call claude_engram_timing, and writes the
patched files, one unified diff and a lock. Nothing in ~/git is modified and
no node is contacted. Rerunning reproduces byte-identical outputs.

Delivery differs by tree. The vLLM model.py and the helper are written into
the image. The two B12X files are NOT written: every B12X disk-cache key
includes a hash of every file in the b12x package, so any byte change there
would cold-compile all CuTe kernels on first boot and change the first-use
conditions this diagnostic measures. Instead the helper compiles the four
edited functions from the patched copies shipped in /opt/ds41-engram-timing
and binds them over the live ones at model import, after checking that the
live B12X files are the pinned inputs.
"""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
VLLM, VLLM_COMMIT = Path.home() / 'git/vllm', '1794dcf18454900263e0c66711af8ea4a1283ac1'
B12X, B12X_COMMIT = Path.home() / 'git/b12x', 'a7d7d29b'
BASE_IMAGE = 'e7b273407022ae74726d6c7b7465aa567b2c5a0594d5e0b46002a742a11f5146'
RING_FIX_SPARSE_MLA_SHA256 = '8a8dffa8417f0fd8b9a96723d36ba5a842949b6dd674751a19e55ec9031a4c34'
HELPER_TARGET = 'vllm/vllm/models/deepseek_v4_1/claude_engram_timing.py'
IMPORT = ('import vllm.models.deepseek_v4_1.claude_engram_timing as _cet\n'
          '\n'
          '_cet.install_b12x()\n')

# (image path relative to /opt/jovian-judgement, repo, commit, repo path, wrapper lines, output name, edits)
# Wrapper lines are bare statements the edit adds around original code (the
# try/finally that guarantees job_end); they are ignored by the equality check.
TARGETS = [
    ('vllm/vllm/models/deepseek_v4_1/nvidia/model.py', VLLM, VLLM_COMMIT,
     'vllm/models/deepseek_v4_1/nvidia/model.py',
     {'try:', 'finally:'},
     'claude-engram-timing-model.py', [
        ('from b12x.sequence import engram as engram_native\n',
         'from b12x.sequence import engram as engram_native\n' + IMPORT),
        ('        if job is not None:\n'
         '            job.result()\n',
         '        if job is not None:\n'
         '            _cet.event("finish-wait-begin")\n'
         '            job.result()\n'
         '            _cet.event("finish-wait-end")\n'),
        ('        def lookup():\n'
         '            torch.accelerator.set_device_index(device)\n'
         '            with torch.inference_mode(), torch.cuda.stream(stream):\n'
         '                stream.wait_event(ready)\n'
         '                engram_native.run_lookups(bindings, counts, clear_tail=False)\n'
         '                self._engram_epochs[0:1].fill_(epoch)\n',
         '        def lookup():\n'
         '            _cet.event("lookup-begin")\n'
         '            try:\n'
         '                torch.accelerator.set_device_index(device)\n'
         '                _cet.event("lookup-device-set")\n'
         '                with torch.inference_mode(), torch.cuda.stream(stream):\n'
         '                    stream.wait_event(ready)\n'
         '                    _cet.event("lookup-ready-wait-enqueued")\n'
         '                    engram_native.run_lookups(bindings, counts, clear_tail=False)\n'
         '                    _cet.event("lookup-run-lookups-returned")\n'
         '                    self._engram_epochs[0:1].fill_(epoch)\n'
         '                    _cet.event("lookup-epoch-fill-enqueued", epoch=epoch)\n'
         '            finally:\n'
         '                _cet.job_end()\n'),
        ('        self._engram_job = pool.submit(lookup)\n',
         '        _cet.job_begin(epoch)\n'
         '        self._engram_job = pool.submit(lookup)\n'),
        ('            )\n'
         '        self._finish_engram_job()\n'
         '        engrams = tuple(\n',
         '            )\n'
         '        _cet.event("prepare-begin", tokens=input_ids.shape[0], last_epoch=self._engram_epoch)\n'
         '        self._finish_engram_job()\n'
         '        _cet.event("prepare-prev-job-done")\n'
         '        engrams = tuple(\n'),
        ('                hashes,\n'
         '            )\n'
         '            # One batch lets the per-layer disk reads overlap on the host.\n',
         '                hashes,\n'
         '            )\n'
         '            _cet.event("prepare-hash-launched")\n'
         '            # One batch lets the per-layer disk reads overlap on the host.\n'),
        ('            counts = [hashes.shape[0]] * len(bindings)\n',
         '            counts = [hashes.shape[0]] * len(bindings)\n'
         '            _cet.event("prepare-staged", tables=len(bindings), rows=hashes.shape[0])\n'),
        ('            for engram in engrams:\n'
         '                engram.finish_disk(hashes.shape[0])\n',
         '            _cet.event("prepare-dispatched", overlap=int(bool(self._engram_overlap)))\n'
         '            for engram in engrams:\n'
         '                engram.finish_disk(hashes.shape[0])\n'
         '            _cet.event("prepare-end")\n'),
    ]),
    ('b12x/b12x/sequence/engram/_impl.py', B12X, B12X_COMMIT, 'b12x/sequence/engram/_impl.py',
     set(),
     'claude-engram-timing-_impl.py', [
        ('  if prepared: self.programs[0][(prepared,24,1)](binding.weight,binding.scale_bytes,binding.hash_ids,binding.num_tokens,binding.out,prepared)\n',
         '  if prepared:\n'
         '   _cet.event("kernel-launch-begin",prepared=prepared)\n'
         '   self.programs[0][(prepared,24,1)](binding.weight,binding.scale_bytes,binding.hash_ids,binding.num_tokens,binding.out,prepared)\n'
         '   _cet.event("kernel-launch-end")\n'),
        (' if len(disk)<2: return tuple(run_lookup(b,n,clear_tail=clear_tail) for b,n in zip(bindings,counts))\n',
         ' _cet.event("run-lookups-begin",disk=len(disk),counts=",".join(map(str,counts)))\n'
         ' if len(disk)<2: return tuple(run_lookup(b,n,clear_tail=clear_tail) for b,n in zip(bindings,counts))\n'),
        ('   caches[i]=stack.enter_context(bindings[i].disk_table._cache.transaction())\n'
         '   caches[i]._stage_ids(bindings[i].hash_ids,counts[i]*24)\n',
         '   _cet.event("txn-enter-begin",table=i)\n'
         '   caches[i]=stack.enter_context(bindings[i].disk_table._cache.transaction())\n'
         '   _cet.event("txn-entered",table=i,cache=f"{id(caches[i]):x}")\n'
         '   caches[i]._stage_ids(bindings[i].hash_ids,counts[i]*24)\n'),
        ('  finally: errors=[f.exception() for f in futures]\n',
         '  finally:\n'
         '   _cet.event("reads-primary-returned")\n'
         '   errors=[f.exception() for f in futures]\n'
         '   _cet.event("reads-joined")\n'),
        ('  for b,n in zip(bindings,counts):\n'
         '   lookup_op(b.plan.handle,b.weight,b.scale_bytes,b.hash_ids,b.num_tokens,b.out,n,clear_tail)\n'
         ' return tuple(b.out for b in bindings)\n',
         '  for b,n in zip(bindings,counts):\n'
         '   _cet.event("lookup-op-begin",rows=n)\n'
         '   lookup_op(b.plan.handle,b.weight,b.scale_bytes,b.hash_ids,b.num_tokens,b.out,n,clear_tail)\n'
         '   _cet.event("lookup-op-end")\n'
         ' _cet.event("run-lookups-end")\n'
         ' return tuple(b.out for b in bindings)\n'),
    ]),
    ('b12x/b12x/sequence/_shared/disk_table.py', B12X, B12X_COMMIT, 'b12x/sequence/_shared/disk_table.py',
     set(),
     'claude-engram-timing-disk_table.py', [
        ('        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)\n'
         '        self._ids_ready.record(self._transaction_stream)\n',
         '        _cet.event("stage-copy-begin", cache=f"{id(self):x}", count=count)\n'
         '        self.ids_host[:count].copy_(ids.view(-1)[:count], non_blocking=True)\n'
         '        self._ids_ready.record(self._transaction_stream)\n'
         '        _cet.event("stage-recorded", cache=f"{id(self):x}")\n'),
        ('            self._ids_ready.synchronize()\n'
         '            if self._gds is not None:\n'
         '                self._gds.read(self._ids_buffer, count, self._transaction_stream)\n'
         '                return\n'
         '            self._native.ple_reader_run(\n'
         '                self._reader,\n'
         '                self._ids_buffer,\n'
         '                self._weight_buffer,\n'
         '                self._scale_buffer,\n'
         '                count,\n'
         '            )\n',
         '            _cet.event("ids-sync-begin", cache=f"{id(self):x}", count=count)\n'
         '            self._ids_ready.synchronize()\n'
         '            _cet.event("ids-sync-end", cache=f"{id(self):x}")\n'
         '            if self._gds is not None:\n'
         '                self._gds.read(self._ids_buffer, count, self._transaction_stream)\n'
         '                _cet.event("gds-read-returned", cache=f"{id(self):x}")\n'
         '                return\n'
         '            self._native.ple_reader_run(\n'
         '                self._reader,\n'
         '                self._ids_buffer,\n'
         '                self._weight_buffer,\n'
         '                self._scale_buffer,\n'
         '                count,\n'
         '            )\n'
         '            _cet.event("read-returned", cache=f"{id(self):x}")\n'),
    ]),
]
PACKAGED = ['claude_engram_timing.py', 'claude_prepare_engram_timing.py',
            'claude_install_engram_timing.py', 'Dockerfile.claude-engram-timing',
            'claude_test_engram_timing.py']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def compose():
    """Return [(image_path, old, new, output_name)] with every anchor matched exactly once."""
    runtime = json.loads((root / 'runtime.lock.json').read_text())['sources']
    lock_files = {**runtime['vllm']['files'], **runtime['b12x']['files']}
    out = []
    for image_path, repo, commit, repo_path, wrappers, name, edits in TARGETS:
        old = subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{repo_path}']).decode()
        if lock_files[repo_path]['sha256'] != sha(old.encode()):
            raise RuntimeError('Pinned blob differs from runtime.lock.json files: ' + repo_path)
        new = old
        for anchor, replacement in edits:
            if new.count(anchor) != 1:
                raise RuntimeError(f'Anchor count {new.count(anchor)} in {repo_path}: {anchor[:60]!r}')
            new = new.replace(anchor, replacement)
        compile(new, repo_path, 'exec')
        if original_code(new, wrappers) != original_code(old, wrappers):
            raise RuntimeError('Edit changed original code in ' + repo_path)
        out.append((image_path, old, new, name))
    return out


def original_code(text, wrappers):
    """Source with timing lines removed and all whitespace dropped.

    Equality with the pinned source proves the edits only add `_cet` calls,
    the import and the named wrapper statements, and re-indent nothing else.
    """
    kept = [line for line in text.splitlines()
            if '_cet' not in line and line.strip() not in wrappers]
    return ''.join(''.join(kept).split())


def main():
    composed = compose()
    patch = []
    for image_path, old, new, name in composed:
        (root / name).write_text(new)
        patch.append(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                  fromfile='a/' + image_path, tofile='b/' + image_path)))
    helper = (root / 'claude_engram_timing.py').read_text()
    patch.append(''.join(difflib.unified_diff([], helper.splitlines(True),
                                              fromfile='/dev/null', tofile='b/' + HELPER_TARGET)))
    (root / 'claude-engram-timing.patch').write_text(''.join(patch))
    lock = {
        'kind': 'engram-host-timing',
        'status': 'diagnostic-only-not-applied',
        'base_image_id': BASE_IMAGE,
        'base_ring_fix_sparse_mla_sha256': RING_FIX_SPARSE_MLA_SHA256,
        'vllm_commit': VLLM_COMMIT, 'b12x_commit': B12X_COMMIT,
        'targets': {image_path: {'input_sha256': sha(old.encode()), 'output_sha256': sha(new.encode()),
                                 'source': name,
                                 'delivery': 'file' if image_path.startswith('vllm/') else 'runtime-bound'}
                    for image_path, old, new, name in composed},
        'helper': {'target': HELPER_TARGET, 'source': 'claude_engram_timing.py',
                   'sha256': sha(helper.encode())},
        'inputs': {name: sha((root / name).read_bytes()) for name in PACKAGED
                   + [name for *_, name in composed] + ['claude-engram-timing.patch']},
    }
    (root / 'claude-engram-timing.lock.json').write_text(json.dumps(lock, indent=2, sort_keys=True) + '\n')
    names = sorted(lock['inputs']) + ['claude-engram-timing.lock.json']
    (root / 'claude-engram-timing.ignore').write_text('**\n' + ''.join('!' + n + '\n' for n in names))
    print('ENGRAM-TIMING-PREPARED', sha((root / 'claude-engram-timing.lock.json').read_bytes()))


if __name__ == '__main__':
    main()
