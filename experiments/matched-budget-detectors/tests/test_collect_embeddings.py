"""Failure semantics for confirmatory episode publication."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "collect_embeddings.py"
SPEC = importlib.util.spec_from_file_location("collect_embeddings", MODULE_PATH)
collect_embeddings = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_embeddings)


def test_publish_files_uploads_every_artifact(monkeypatch, tmp_path):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(collect_embeddings.subprocess, "run", run)
    files = (tmp_path / "ep000.npz", tmp_path / "ep000.json")
    collect_embeddings.publish_files(files, "s3://bucket/condition", "us-east-2")

    assert [c[4] for c in calls] == [
        "s3://bucket/condition/ep000.npz",
        "s3://bucket/condition/ep000.json",
    ]


def test_publish_files_fails_closed_on_upload_error(monkeypatch, tmp_path):
    def run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="access denied")

    monkeypatch.setattr(collect_embeddings.subprocess, "run", run)
    files = (tmp_path / "ep000.npz", tmp_path / "ep000.json")
    with pytest.raises(RuntimeError, match="episode publication failed"):
        collect_embeddings.publish_files(files, "s3://bucket/condition", "us-east-2")
