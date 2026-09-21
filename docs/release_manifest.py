# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Write or check the release manifest.

The manifest records the source revision, the Python environment, and the
SHA-256 of every released input and output file. It binds a release tag to the
exact bytes of the specification, the frozen inputs, the experiment results and
the paper PDFs. It does not establish that any measurement is correct.

    python docs/release_manifest.py            # write docs/release_manifest.json
    python docs/release_manifest.py --check    # fail if any hashed file changed

The manifest is not committed; ``.gitignore`` lists it. The release procedure
writes it after the release commit, checks it, tags that commit and attaches
the file to the GitHub release, so ``git.commit`` names the tagged commit.
``git.branch`` and ``git.dirty`` are recorded as found; at release they must
read ``main`` and ``false``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "release_manifest.json"
AUDIT = "docs/REPOSITORY_AUDIT.md"
HASHED_GLOBS = ("spec/**/*", "data/**/*", "experiments/*/results/**/*", "docs/paper/latex/*.pdf")


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashed_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for pattern in HASHED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file() and ".gitignore" not in path.name:
                files[path.relative_to(ROOT).as_posix()] = sha256(path)
    return files


def environment() -> dict:
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                      if d.metadata["Name"])
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": Path(sys.executable).name,
        "pip_freeze": packages,
    }


def build_manifest() -> dict:
    files = hashed_files()
    if not (ROOT / AUDIT).is_file():
        raise FileNotFoundError(f"repository audit missing: {AUDIT}")
    return {
        "manifest_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "describe": git("describe", "--tags", "--always", "--dirty"),
            "tags_at_commit": [t for t in git("tag", "--points-at", "HEAD").split("\n") if t],
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        },
        "environment": environment(),
        "repository_audit": AUDIT,
        "hashed_globs": list(HASHED_GLOBS),
        "file_count": len(files),
        "files": files,
    }


def check(manifest_path: Path) -> list[str]:
    if not manifest_path.is_file():
        # The manifest is written at release time, not committed, so absence is
        # the repository's normal state rather than an error in the tree.
        return [f"no manifest at {manifest_path}; write one first with "
                f"`python docs/release_manifest.py --out {manifest_path}`"]
    recorded = json.loads(manifest_path.read_text())["files"]
    current = hashed_files()
    problems = [f"changed: {p}" for p in recorded if p in current and current[p] != recorded[p]]
    problems += [f"missing: {p}" for p in recorded if p not in current]
    problems += [f"unlisted: {p}" for p in current if p not in recorded]
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true",
                        help="compare current file hashes with the manifest at --out")
    args = parser.parse_args()
    if args.check:
        problems = check(args.out)
        for line in problems:
            print(line)
        print(f"{'FAIL' if problems else 'OK'}: {len(problems)} difference(s) against {args.out}")
        return 1 if problems else 0
    manifest = build_manifest()
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out} ({manifest['file_count']} files, commit {manifest['git']['commit'][:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
