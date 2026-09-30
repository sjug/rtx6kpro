#!/usr/bin/env python3
"""Read pinned objects from existing repositories; emit a narrow Spark overlay.

No checkout, index, ref or source repository is modified. Stdout is the generated
patch. Source trees and the PR756 snapshot implementation are checked first.
"""
import argparse
import difflib
import json
from pathlib import Path
import subprocess


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one source match: {before!r}")
    return text.replace(before, after, 1)


def adapt_arches(text):
    for arches in (
        "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.7;11.0;12.0",
        "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0",
    ):
        text = replace_once(text, f'"{arches}"', f'"{arches};12.1"')
    return text


def adapt_head(text):
    text = replace_once(text, "if (major, minor) != (12, 0):",
                        "if (major, minor) not in ((12, 0), (12, 1)):")
    return replace_once(text, 'f"12.0; got {major}.{minor}"',
                        'f"12.0 or 12.1; got {major}.{minor}"')


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos", type=Path, default=Path.home() / "git")
    args = parser.parse_args()
    selection = json.loads(Path(__file__).with_name("source-selection.json").read_text())
    for name, pin in selection["sources"].items():
        actual = git(args.repos / name, "rev-parse", pin["commit"] + "^{tree}").strip()
        if actual != pin["tree"]:
            raise ValueError(f"{name}: source tree mismatch {actual}")
    repo = args.repos / "vllm"
    commit = selection["sources"]["vllm"]["commit"]
    # The existing checkout is shallow; absence of an ancestry path is not
    # evidence that the fix is missing. Inspect the frozen runtime source.
    async_source = git(repo, "show", f"{commit}:vllm/v1/worker/gpu/async_utils.py")
    snapshot = ("self.num_verified_draft_tokens = (\n"
                "            num_verified_draft_tokens.clone()\n"
                "            if num_verified_draft_tokens is not None\n"
                "            else None\n"
                "        )")
    if async_source.count(snapshot) != 1:
        raise ValueError("PR756 snapshot implementation must be reviewed")
    result = []
    for path, transform in (
        ("CMakeLists.txt", adapt_arches),
        ("vllm/models/glm5next/nvidia/mtp_draft_head.py", adapt_head),
    ):
        before = git(repo, "show", f"{commit}:{path}")
        after = transform(before)
        result.append(f"diff --git a/{path} b/{path}\n")
        result.extend(difflib.unified_diff(before.splitlines(keepends=True),
                      after.splitlines(keepends=True), f"a/{path}", f"b/{path}"))
    print("".join(result), end="")


if __name__ == "__main__":
    main()
