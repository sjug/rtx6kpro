"""b12x package initializer for the kit's B12X root (/opt/jovian-judgement/b12x).

FlashInfer 0.7.1 ships the package at flashinfer/experimental/b12x and installs a `b12x` alias
package (its _compat initializer) that executes the embedded _api.py with the embedded directory
as the package path and installs a finder aliasing `flashinfer.experimental.b12x.*` to `b12x.*`.
The kit's root is byte-identical to the embedded package apart from this file; it performs the
same bootstrap rooted here, so every module loads once from the root, first on PYTHONPATH, and
both import paths share module identity (the wheel's experimental/b12x/__init__.py aliases to
`b12x`, which resolves to this root).
"""

import importlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, canonical_name):
        self.canonical_name = canonical_name

    def create_module(self, spec):
        module = importlib.import_module(self.canonical_name)
        self.canonical_spec = module.__spec__
        return module

    def exec_module(self, module):
        # Import machinery sets __spec__ even when create_module reuses a module.
        module.__spec__ = self.canonical_spec


class _AliasFinder(importlib.abc.MetaPathFinder):
    prefix = "flashinfer.experimental.b12x."

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix):
            return None
        canonical = "b12x." + fullname[len(self.prefix) :]
        module = importlib.import_module(canonical)
        return importlib.util.spec_from_loader(
            fullname, _AliasLoader(canonical), is_package=hasattr(module, "__path__")
        )


_root = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    __name__, _root / "_api.py", submodule_search_locations=[str(_root)]
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
try:
    _spec.loader.exec_module(_module)
except BaseException:
    sys.modules.pop(__name__, None)
    raise
if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())
