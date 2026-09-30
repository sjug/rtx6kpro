"""Diagnostic-only timestamps for the DS4.1 disk-Engram host path.

Installed as `vllm/models/deepseek_v4_1/claude_engram_timing.py` by
`claude_install_engram_timing.py`. The patched DS4.1 model.py imports it and
calls `install_b12x()` once; that binds timed copies of four B12X functions
(`run_lookups`, `_State.run_lookup`, `DiskRowCache._stage_ids`,
`DiskRowCache._read_staged`) compiled from the reviewed patched files in
/opt/ds41-engram-timing. B12X files on disk stay byte-identical, so the B12X
package fingerprint, which keys every CuTe disk-cache entry, is unchanged
and the inherited compile cache still hits. The timed functions make the
same calls in the same order as the pinned ones, and no CUDA call,
synchronisation or wait is added; timing itself is not preserved (see
Perturbation). When disabled (the default) an event still evaluates its
arguments at the call site, reads the clock and checks a cached flag; the
enable file is opened and read at most once per second.

Enable:  touch /cache/ds41-engram-timing.enable
         (file content containing "stacks" also arms faulthandler per job)
Output:  /cache/ds41-engram-timing/<node>-<pid>.log     one line per event
         /cache/ds41-engram-timing/<node>-<pid>.stacks  all-thread dumps

Line format (space separated):
  <monotonic_ns> node=<DS41_NODE> pid=<pid> tid=<native tid> thread=<name>
  job=<epoch of the outstanding lookup job or -> <event> [key=value ...]

Stack dumps: while a lookup job is outstanding and "stacks" is enabled,
`faulthandler.dump_traceback_later(STACK_AFTER_S, repeat=True)` is armed by
the main thread just before the job is submitted (so a lookup thread that is
slow to start is covered) and cancelled in the lookup's `finally`, which runs
before the job's future completes. Dumps therefore appear only for jobs
outstanding longer than STACK_AFTER_S. faulthandler's watchdog is a C thread that writes
Python frames of every thread without taking the GIL; it shows the last
Python frame of each thread, not native frames, so a thread blocked inside a
native call and a thread waiting for the GIL at that line look alike. The
event timestamps do not separate them either: an event after a native call
runs only once the thread holds the GIL again, so each segment is native
time plus GIL reacquisition. They bound the segment, not its cause.

Perturbation: every enabled event runs Python (string formatting) and one
os.write on the calling thread, about 25 events per engine step across the
main thread, the lookup thread and the read pool. The first enabled event in
a process also creates the output directory and opens the log, and the first
armed job opens the stacks file, so the measurement adds its own first-use
filesystem work on exactly the path under test. None of this cost has been
measured. Any added work changes thread interleaving and GIL hand-off, and
the forward waits on this job with a fixed 5000 ms timeout, so enabling
timing can move a stall across that threshold in either direction: a
timeout may disappear or appear, not only shift. A run where the residual
does not reproduce with timing enabled is not evidence that the stall is
gone; compare with the same image and the enable file absent.
"""
from __future__ import annotations

import ast
import faulthandler
import hashlib
import json
import os
import threading
import time

KIT = '/opt/ds41-engram-timing'
ENABLE = '/cache/ds41-engram-timing.enable'
OUT_DIR = '/cache/ds41-engram-timing'
STACK_AFTER_S = 3.0
RECHECK_NS = 1_000_000_000

_state = {'checked': -RECHECK_NS - 1, 'enabled': False, 'stacks': False,
          'fd': None, 'stack_fd': None, 'job': '-', 'armed': False}
_open_lock = threading.Lock()


def _refresh(now: int) -> bool:
    if now - _state['checked'] >= RECHECK_NS:
        _state['checked'] = now
        try:
            with open(ENABLE, 'rb') as handle:
                content = handle.read(256)
            _state['enabled'], _state['stacks'] = True, b'stacks' in content
        except OSError:
            _state['enabled'] = _state['stacks'] = False
    return _state['enabled']


def _open(key: str, suffix: str):
    if _state[key] is None:
        with _open_lock:
            if _state[key] is None:
                os.makedirs(OUT_DIR, exist_ok=True)
                name = f"{os.environ.get('DS41_NODE', 'unknown')}-{os.getpid()}{suffix}"
                _state[key] = os.open(os.path.join(OUT_DIR, name),
                                      os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    return _state[key]


def event(name: str, **fields) -> None:
    """Append one timestamped line; never raises into the caller."""
    now = time.monotonic_ns()
    try:
        if not _refresh(now):
            return
        thread = threading.current_thread()
        extra = ''.join(f' {key}={value}' for key, value in fields.items())
        line = (f"{now} node={os.environ.get('DS41_NODE', 'unknown')} pid={os.getpid()} "
                f"tid={threading.get_native_id()} thread={thread.name} job={_state['job']} "
                f"{name}{extra}\n")
        os.write(_open('fd', '.log'), line.encode())
    except Exception:  # diagnostics must not change serving behaviour
        pass


def job_begin(epoch: int) -> None:
    """Mark the lookup job for `epoch` as outstanding; arm stack dumps if requested."""
    _state['job'] = epoch
    event('job-begin')
    try:
        if _state['enabled'] and _state['stacks'] and not _state['armed']:
            faulthandler.dump_traceback_later(STACK_AFTER_S, repeat=True,
                                              file=_open('stack_fd', '.stacks'), exit=False)
            _state['armed'] = True
    except Exception:
        pass


def job_end() -> None:
    """Cancel stack dumps armed for this job; record the end."""
    try:
        if _state['armed']:
            faulthandler.cancel_dump_traceback_later()
            _state['armed'] = False
    except Exception:
        pass
    event('job-end')
    _state['job'] = '-'


# (lock target, module, class or None, function names) bound by install_b12x.
B12X_BINDINGS = (
    ('b12x/b12x/sequence/engram/_impl.py', 'b12x.sequence.engram._impl', None, ('run_lookups',)),
    ('b12x/b12x/sequence/engram/_impl.py', 'b12x.sequence.engram._impl', '_State', ('run_lookup',)),
    ('b12x/b12x/sequence/_shared/disk_table.py', 'b12x.sequence._shared.disk_table', 'DiskRowCache',
     ('_stage_ids', '_read_staged')),
)
_installed = []


def _read(path: str) -> bytes:
    with open(path, 'rb') as handle:
        return handle.read()


def _functions(source: str, path: str, owner, names):
    """Compile only the named functions from the patched file, keeping its line numbers."""
    tree = ast.parse(source, path)
    scope = tree.body
    if owner is not None:
        (cls,) = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner]
        scope = cls.body
    found = [node for node in scope if isinstance(node, ast.FunctionDef) and node.name in names]
    if sorted(node.name for node in found) != sorted(names):
        raise RuntimeError(f'timing overlay: missing {names} in {path}')
    return compile(ast.Module(body=found, type_ignores=[]), path, 'exec')


def install_b12x(kit: str = KIT) -> None:
    """Bind the timed B12X functions once per process; fail closed on any mismatch."""
    if _installed:
        return
    import importlib
    import sys
    lock = json.loads(_read(os.path.join(kit, 'claude-engram-timing.lock.json')))
    plan = []
    for target, module_name, owner, names in B12X_BINDINGS:
        entry = lock['targets'][target]
        module = importlib.import_module(module_name)
        live = _read(module.__file__)
        if hashlib.sha256(live).hexdigest() != entry['input_sha256']:
            raise RuntimeError('timing overlay: live B12X module is not the pinned input: ' + target)
        path = os.path.join(kit, entry['source'])
        patched = _read(path)
        if hashlib.sha256(patched).hexdigest() != entry['output_sha256']:
            raise RuntimeError('timing overlay: patched source digest mismatch: ' + target)
        code = _functions(patched.decode(), path, owner, names)
        namespace = dict(vars(module))       # compile against the module's own globals
        exec(code, namespace)
        plan.append((module, owner, names, namespace))
    for module, owner, names, namespace in plan:
        module._cet = sys.modules[__name__]
        for name in names:
            function = namespace[name]
            # Functions must see the live module globals (for example the
            # `global _READ_POOL` in run_lookups), not the scratch namespace.
            function = type(function)(function.__code__, vars(module), function.__name__,
                                      function.__defaults__, function.__closure__)
            function.__kwdefaults__ = namespace[name].__kwdefaults__
            function.__qualname__ = (owner + '.' + name) if owner else name
            function.__module__ = module.__name__
            if owner is None:
                setattr(module, name, function)
            else:
                setattr(getattr(module, owner), name, function)
    # run_lookups is re-exported: api imports it by value, and the package's
    # lazy __getattr__ caches whatever api holds on first access.
    impl = importlib.import_module('b12x.sequence.engram._impl')
    api = importlib.import_module('b12x.sequence.engram.api')
    package = importlib.import_module('b12x.sequence.engram')
    api.run_lookups = impl.run_lookups
    package.run_lookups = impl.run_lookups
    _installed.append(True)
