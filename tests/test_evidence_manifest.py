# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The evidence manifest must match the working tree, name every paper label, and render EVIDENCE.md's table."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper"
BY_COMMIT_PREFIXES = ("spec/", "docs/paper/latex/")


def load_checker():
    spec = importlib.util.spec_from_file_location("check_evidence_manifest", PAPER / "check_evidence_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return load_checker()


@pytest.fixture(scope="module")
def manifest():
    return json.loads((PAPER / "EVIDENCE_MANIFEST.json").read_text())


def test_manifest_hashes_match_working_tree(checker, capsys):
    assert checker.main([]) == 0, capsys.readouterr().out


def test_spec_and_generator_entries_are_recorded_by_commit(checker, manifest):
    for entry in checker.file_entries(manifest):
        if entry["path"].startswith(BY_COMMIT_PREFIXES):
            assert "sha256" not in entry and "git" in entry, entry["path"]
        else:
            assert "sha256" in entry, entry["path"]


def test_check_counts_hashed_files_and_ignores_commit_entries(checker, tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    (tmp_path / "a.json").write_bytes(b"a")
    (tmp_path / "b.json").write_bytes(b"b")
    sha_a, sha_b = (hashlib.sha256(b).hexdigest() for b in (b"a", b"b"))
    manifest = {"items": [
        {"path": "a.json", "sha256": sha_a},
        {"path": "a.json", "sha256": sha_a},
        {"path": "b.json", "sha256": sha_b, "tracked": False},
        {"path": "b.json", "sha256": "0" * 64},
        {"path": "c.json", "sha256": sha_a, "tracked": False},
        {"path": "d.json", "sha256": sha_a, "tracked": True},
        {"path": "d.json", "sha256": sha_a, "tracked": False},
        {"path": "spec/x.md", "git": "0" * 40},
    ]}
    verified, by_commit, absent, failures = checker.check(manifest)
    assert verified == 1
    assert by_commit == 1
    assert absent == ["c.json"]
    assert failures == ["b.json: two different hashes in the manifest", "d.json: entries disagree about 'tracked'"]


def test_evidence_table_matches_rendered_manifest(checker, manifest):
    text = (PAPER / "EVIDENCE.md").read_text()
    start, end = checker.table_span(text)
    assert text[start:end] == checker.render_table(manifest)


def test_manifest_covers_every_table_and_figure(manifest):
    labels = {item["label"] for item in manifest["items"]}
    expected = {"tab:panel", "fig:arid", "fig:overview", "tab:decoding", "tab:conditions", "tab:arid",
                "fig:batch", "tab:scifact", "tab:normalized", "tab:replication", "tab:decfull"}
    assert expected <= labels
    for item in manifest["items"]:
        assert "raw_artifacts" in item and "s3_upload_needed" in item, item["label"]


def test_manifest_names_no_bucket_or_account():
    text = (PAPER / "EVIDENCE_MANIFEST.json").read_text() + (PAPER / "EVIDENCE.md").read_text()
    assert "s3://semq-" not in text
    assert "dkr.ecr" not in text
