# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The release manifest names the audit, hashes the released files, and detects a changed file."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_manifest", ROOT / "docs" / "release_manifest.py")
release_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_manifest)


def test_manifest_covers_spec_data_results_and_audit():
    manifest = release_manifest.build_manifest()
    files = manifest["files"]
    assert manifest["repository_audit"] == "docs/REPOSITORY_AUDIT.md"
    assert (ROOT / manifest["repository_audit"]).is_file()
    assert "spec/ari-canonical-v0.1.md" in files
    assert "data/ari-bench-v0.1.jsonl" in files
    assert "experiments/regime-discrimination/results/regime_matrix.json" in files
    assert any(p.startswith("docs/paper/latex/") and p.endswith(".pdf") for p in files)
    assert all(len(h) == 64 for h in files.values())
    assert manifest["file_count"] == len(files)
    assert manifest["environment"]["python_version"]
    assert any(line.startswith("numpy==") for line in manifest["environment"]["pip_freeze"])
    assert "/" not in manifest["environment"]["executable"]


def test_check_passes_on_fresh_manifest_and_fails_on_tamper(tmp_path):
    path = tmp_path / "m.json"
    manifest = release_manifest.build_manifest()
    path.write_text(json.dumps(manifest))
    assert release_manifest.check(path) == []
    first = next(iter(manifest["files"]))
    manifest["files"][first] = "0" * 64
    manifest["files"]["spec/does-not-exist.md"] = "1" * 64
    path.write_text(json.dumps(manifest))
    problems = release_manifest.check(path)
    assert f"changed: {first}" in problems
    assert "missing: spec/does-not-exist.md" in problems

