"""Failure semantics for confirmatory episode publication."""

from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


def test_existing_episode_is_reused_only_after_validation(tmp_path):
    paths = collect_embeddings._episode_paths(tmp_path, "control", 3)
    embeddings = np.ones((2, 4), dtype=np.float32)
    np.savez_compressed(paths[0], embeddings=embeddings)
    manifest = {
        "condition": "control",
        "episode": 3,
        "model": "model",
        "n_inputs": 2,
        "inputs_sha256": "inputs",
        "embeddings_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
    }
    paths[1].write_text(json.dumps(manifest))

    assert collect_embeddings._reuse_existing_episode(
        paths, {k: manifest[k] for k in ("condition", "episode", "model", "n_inputs", "inputs_sha256")}
    )

    embeddings[0, 0] = 2.0
    np.savez_compressed(paths[0], embeddings=embeddings)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        collect_embeddings._reuse_existing_episode(paths, {
            k: manifest[k] for k in ("condition", "episode", "model", "n_inputs", "inputs_sha256")
        })


def test_partial_existing_episode_fails_closed(tmp_path):
    paths = collect_embeddings._episode_paths(tmp_path, "control", 4)
    paths[0].parent.mkdir(parents=True)
    paths[0].write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="incomplete episode"):
        collect_embeddings._reuse_existing_episode(paths, {})
