# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The evidence manifest must match the working tree and name every paper label."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/paper"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_evidence_manifest", PAPER / "check_evidence_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_hashes_match_working_tree(capsys):
    assert load_checker().main() == 0, capsys.readouterr().out


def test_manifest_covers_every_table_and_figure():
    manifest = json.loads((PAPER / "EVIDENCE_MANIFEST.json").read_text())
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
