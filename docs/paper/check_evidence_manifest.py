# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Verify every sha256 in docs/paper/EVIDENCE_MANIFEST.json against the working tree.

    python docs/paper/check_evidence_manifest.py            # repository root

Exit status 1 on any mismatch or on a missing tracked file. Entries marked
"tracked": false live on a branch or in an operator working tree; they are
verified when present and reported when absent, without failing.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/paper/EVIDENCE_MANIFEST.json"


def file_entries(node):
    """Yield every object that carries both 'path' and 'sha256'."""
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            yield node
        for value in node.values():
            yield from file_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from file_entries(value)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    seen, failures, absent_untracked = {}, [], []
    for entry in file_entries(manifest):
        path, expected = entry["path"], entry["sha256"]
        if seen.get(path) == expected:
            continue
        if path in seen:
            failures.append(f"{path}: two different hashes in the manifest")
            continue
        seen[path] = expected
        target = ROOT / path
        if not target.is_file():
            if entry.get("tracked") is False:
                absent_untracked.append(path)
            else:
                failures.append(f"{path}: missing")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{path}: sha256 {actual} != manifest {expected}")
    verified = len(seen) - len(failures) - len(absent_untracked)
    print(f"{verified} files verified, {len(absent_untracked)} untracked files absent, {len(failures)} failures")
    for path in absent_untracked:
        print(f"  absent (untracked): {path}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
