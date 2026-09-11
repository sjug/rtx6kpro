"""Package Python-only LMCache refreshes with exact ARM native payloads.

Wheel metadata algorithm adapted from upstream pack_lmcache_native_reuse.py
whose R32 input digest is d98c184998a343adef3c49e8904ed8509e418b5acc5d6005996a40dfa1880bfe
(upstream-r32.source.lock), with an ARM contract and no native builds.
"""

if not __debug__:
    raise RuntimeError("R32 verification requires Python assertions enabled")

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

TAG = "cp312-cp312-linux_aarch64"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_manifest():
    vllm = Path("/opt/jovian-judgement/vllm/vllm")
    lmcache = Path(importlib.metadata.distribution("lmcache").locate_file("lmcache"))
    roots = [vllm, lmcache, Path("/opt/jovian-judgement/b12x-roce")]
    files = {str(p): digest(p) for root in roots for p in root.rglob("*.so")}
    assert len(list(vllm.rglob("*.abi3.so"))) == 10
    assert len(list(lmcache.rglob("*.so"))) >= 4
    for p in (vllm / "vllm-rs", Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so")):
        files[str(p)] = digest(p)
    assert any("roce_proxy-" in path for path in files)
    return files


def verify_manifest(expected, actual):
    if expected != actual:
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        raise RuntimeError(f"Native reuse mismatch: {changed}")


def classify_installed_files(paths, source_files, native_files, generated_files, metadata_root):
    """Classify an inventory, rejecting anything the refresh cannot account for."""
    result = {}
    unknown = []
    for path in sorted(paths):
        if path in native_files:
            category = "preserved-native"
        elif path in source_files:
            category = "tracked-source"
        elif path in generated_files:
            category = "generated"
        elif path.is_relative_to(metadata_root):
            category = "distribution-metadata"
        elif path.parent.name == "__pycache__" and path.suffix == ".pyc":
            category = "generated-bytecode"
        else:
            unknown.append(str(path))
            continue
        result[str(path)] = category
    if unknown:
        raise RuntimeError(f"Unclassified LMCache installed files before reinstall: {unknown}")
    return result


def verify_wheel_native_payload(native_files, site_packages, wheel_files):
    """A known native object also needs an exact replacement in the new wheel."""
    for path, expected in native_files.items():
        try:
            name = path.relative_to(site_packages).as_posix()
        except ValueError as exc:
            raise RuntimeError(f"Native wheel relocation is not qualified: {path}") from exc
        if name not in wheel_files or hashlib.sha256(wheel_files[name]).hexdigest() != expected:
            raise RuntimeError(f"Wheel does not preserve installed native payload: {path}")


def lmcache_preinstall_inventory(source_root, base_tree, target_tree, native_files, wheel):
    """Inventory before uv can remove the old RECORD's files, including siblings.

    Unknown payloads fail closed here, not after uninstall. Generated version,
    console scripts, distribution metadata and bytecode are intentionally not
    mistaken for compiled extensions. Source files removed by R32 are classified
    using the base tree as well as the replacement tree.
    """
    dist = importlib.metadata.distribution("lmcache")
    if not dist.files:
        raise RuntimeError("LMCache has no installed RECORD to inventory")

    def absolute(path):
        # Normalize RECORD's ../../../bin entries without dereferencing source
        # documentation symlinks; it is the installed filename we classify.
        return Path(os.path.abspath(path))

    package = absolute(dist.locate_file("lmcache"))
    owned = {absolute(dist.locate_file(p)) for p in dist.files}
    metadata = {p.parent for p in owned if p.name == "WHEEL" and p.parent.name.endswith(".dist-info")}
    if len(metadata) != 1:
        raise RuntimeError("Expected exactly one LMCache wheel metadata directory")
    paths = owned | {p for p in package.rglob("*") if p.is_file() or p.is_symlink()}
    source_files = set()
    for tree in (base_tree, target_tree):
        names = subprocess.check_output([
            "git", "-C", str(source_root), "ls-tree", "-rz", "--name-only", tree, "--", "lmcache",
        ]).decode().split("\0")
        source_files.update(package.parent / p for p in names if p)
    generated = {package / "_version.py"}
    generated.update(
        absolute(Path(sysconfig.get_path("scripts")) / ep.name)
        for ep in dist.entry_points if ep.group == "console_scripts"
    )
    native = {Path(p): sha for p, sha in native_files.items()}
    classified = classify_installed_files(paths, source_files, native, generated, metadata.pop())
    missing = [str(p) for p in owned if not (p.is_file() or p.is_symlink())
               and classified[str(p)] != "generated-bytecode"]
    if missing:
        raise RuntimeError(f"LMCache RECORD names missing installed files: {missing}")
    preserved = {p: native[p] for p in paths if classified[str(p)] == "preserved-native"}
    with zipfile.ZipFile(wheel) as archive:
        # Only native payloads need byte identity; Python and metadata change.
        payloads = {n: archive.read(n) for n in archive.namelist() if n.endswith(".so")}
    verify_wheel_native_payload(preserved, package.parent, payloads)
    return {
        "version_before": dist.version,
        "base_tree": base_tree,
        "replacement_tree": target_tree,
        "files": classified,
        "preserved_native_sha256": {str(p): sha for p, sha in preserved.items()},
    }


def platform_wheel(files, native, tag):
    if tag != TAG or not native or any(not n.startswith("lmcache/") or not n.endswith(".so") for n in native):
        raise ValueError("Expected CPython 3.12 Linux aarch64 LMCache native payload")
    wheels = [n for n in files if n.endswith(".dist-info/WHEEL")]
    if len(wheels) != 1 or any(n.endswith(".so") for n in files):
        raise ValueError("Expected exactly one pure-Python wheel without native objects")
    wheel_path = wheels[0]
    result = {**files, **native}
    lines = [line for line in files[wheel_path].decode().splitlines()
             if line and not line.startswith(("Tag:", "Root-Is-Purelib:"))]
    result[wheel_path] = ("\n".join(lines + ["Root-Is-Purelib: false", f"Tag: {tag}"]) + "\n").encode()
    record = wheel_path.removesuffix("WHEEL") + "RECORD"
    rows = io.StringIO(newline="")
    writer = csv.writer(rows, lineterminator="\n")
    for name, data in sorted(result.items()):
        if name != record:
            sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            writer.writerow([name, f"sha256={sha}", len(data)])
    writer.writerow([record, "", ""])
    result[record] = rows.getvalue().encode()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pack", "verify"))
    args = parser.parse_args()
    expected = json.loads(Path("/opt/local-inference/r32-native-manifest.json").read_text())
    verify_manifest(expected, native_manifest())
    if args.mode == "verify":
        print("R32 all-native byte reuse: PASS")
        return
    assert platform.machine() == "aarch64" and sys.version_info[:2] == (3, 12)
    dist = importlib.metadata.distribution("lmcache")
    tags = [line.removeprefix("Tag: ") for line in dist.read_text("WHEEL").splitlines() if line.startswith("Tag: ")]
    assert tags == [TAG], tags
    package = Path(dist.locate_file("lmcache"))
    native = {"lmcache/" + p.relative_to(package).as_posix(): p.read_bytes() for p in package.rglob("*.so")}
    wheels = list(Path("/artifacts").glob("*.whl"))
    assert len(wheels) == 1 and wheels[0].name.endswith("-py3-none-any.whl")
    pure = wheels[0]
    output = pure.with_name(pure.name.removesuffix("-py3-none-any.whl") + f"-{TAG}.whl")
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(pure) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in platform_wheel(files, native, TAG).items():
            archive.writestr(name, data)
    # Keep the pure artifact as evidence; installer selects only the ARM wheel.
    print(f"R32 preserved {len(native)} native payloads: {output.name}")


if __name__ == "__main__":
    main()
