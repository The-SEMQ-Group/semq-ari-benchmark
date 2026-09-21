# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Check docs/paper/EVIDENCE_MANIFEST.json against the working tree.

    python docs/paper/check_evidence_manifest.py            # repository root
    python docs/paper/check_evidence_manifest.py --render   # rewrite the table in EVIDENCE.md

Entries with "sha256" (result files, attestation sidecars, archived raw
artifacts) are hashed and compared. Entries with "git" and no "sha256"
(specification files, generator scripts, generated tables and figures,
ari.tex) are recorded by the commit that last changed them on main and are
not hashed, so unrelated edits to those files do not fail this check.
Entries marked "tracked": false live on a branch or in an operator working
tree; they are verified when present and reported when absent, without
failing. Exit status 1 on any hash mismatch, on a missing tracked file, or
when two entries disagree about one path.

--render regenerates the "Tables and figures" table in EVIDENCE.md from the
manifest. tests/test_evidence_manifest.py asserts the committed table equals
the rendered one.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "docs/paper/EVIDENCE_MANIFEST.json"
EVIDENCE = ROOT / "docs/paper/EVIDENCE.md"

COLUMNS = ("Label", "Item", "Experiment", "Summary files (in repository)",
           "Generator and command", "Environment record", "Raw artifacts")


def file_entries(node):
    """Yield every object that carries 'path'."""
    if isinstance(node, dict):
        if "path" in node:
            yield node
        for value in node.values():
            yield from file_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from file_entries(value)


def check(manifest) -> tuple[int, int, list[str], list[str]]:
    """Return (verified, by_commit, absent_untracked, failures)."""
    seen: dict[str, dict] = {}
    failures, by_commit = [], set()
    for entry in file_entries(manifest):
        path = entry["path"]
        if "sha256" not in entry:
            by_commit.add(path)
            continue
        record = {"sha256": entry["sha256"], "tracked": entry.get("tracked", True)}
        previous = seen.setdefault(path, record)
        if previous["sha256"] != record["sha256"]:
            failures.append(f"{path}: two different hashes in the manifest")
        elif previous["tracked"] != record["tracked"]:
            failures.append(f"{path}: entries disagree about 'tracked'")
    conflicting = {f.split(":")[0] for f in failures}
    verified, absent_untracked = 0, []
    for path, record in seen.items():
        if path in conflicting:
            continue
        target = ROOT / path
        if not target.is_file():
            if record["tracked"]:
                failures.append(f"{path}: missing")
            else:
                absent_untracked.append(path)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual == record["sha256"]:
            verified += 1
        else:
            failures.append(f"{path}: sha256 {actual} != manifest {record['sha256']}")
    return verified, len(by_commit), absent_untracked, failures


def _code(paths) -> str:
    return ", ".join(f"`{p}`" for p in paths)


def _dirs(value) -> str:
    if value is None:
        return "none"
    return _code(value if isinstance(value, list) else [value])


def _relative(item, entries):
    """Strip the item's experiment directory from each path, as the Experiment column names it."""
    dirs = item["experiment_dir"] or []
    dirs = dirs if isinstance(dirs, list) else [dirs]
    for entry in entries:
        path = entry["path"]
        for directory in dirs:
            if path.startswith(directory + "/"):
                path = path[len(directory) + 1:]
                break
        yield path


def _summary_files(item) -> str:
    parts = [_code(_relative(item, item["result_files"]))]
    sidecars = item.get("sidecars_in_repo") or []
    if sidecars:
        parts.append(f"(+ {len(sidecars)} attestation sidecars)")
    related = item.get("related_files") or []
    if related:
        parts.append(f"(related: {_code(_relative(item, related))})")
    return " ".join(parts)


def _generator(item) -> str:
    gen = item["generator"]
    if not gen.get("script"):
        return gen.get("note", "none")
    return f"`{gen['script']['path']}` -> `{gen['output']['path']}`; `{gen['command']}`"


def _text(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_text(sub)}" for key, sub in value.items())
    return str(value)


def _raw(node, name=None) -> list[str]:
    parts = []
    if "status" in node:
        parts.append(f"{name}: {node['status']}" if name else node["status"])
    for key, value in node.items():
        if not isinstance(value, dict):
            continue
        if "prefix" in value:
            parts.append(f"`{value['prefix']}` ({value.get('verified') or value.get('state')})")
        else:
            parts.extend(_raw(value, key))
    return parts


def render_table(manifest) -> str:
    """Return the "Tables and figures" markdown table for EVIDENCE.md."""
    lines = ["| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for item in manifest["items"]:
        number = item["number"] or "Text"
        cells = (
            f"`{item['label']}`",
            f"{number}, {item['title']}",
            _dirs(item["experiment_dir"]),
            _summary_files(item),
            _generator(item),
            _text(item.get("environment_record")),
            "; ".join(_raw(item["raw_artifacts"])),
        )
        lines.append("| " + " | ".join(c.replace("|", "\\|").replace("\n", " ") for c in cells) + " |")
    return "\n".join(lines)


def table_span(text: str) -> tuple[int, int]:
    """Return the [start, end) character offsets of the rendered table in EVIDENCE.md."""
    start = text.index("| " + COLUMNS[0] + " |")
    end = start
    for line in text[start:].split("\n"):
        if not line.startswith("|"):
            break
        end += len(line) + 1
    return start, end - 1


def render(manifest) -> None:
    text = EVIDENCE.read_text()
    start, end = table_span(text)
    EVIDENCE.write_text(text[:start] + render_table(manifest) + text[end:])


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    manifest = json.loads(MANIFEST.read_text())
    if argv == ["--render"]:
        render(manifest)
        return 0
    verified, by_commit, absent_untracked, failures = check(manifest)
    print(f"{verified} files verified, {by_commit} recorded by commit, "
          f"{len(absent_untracked)} untracked files absent, {len(failures)} failures")
    for path in absent_untracked:
        print(f"  absent (untracked): {path}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
