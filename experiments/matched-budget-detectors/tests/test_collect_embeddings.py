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
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.ones((2, 4), dtype=np.float32)
    np.savez_compressed(paths[0], embeddings=embeddings)
    manifest = {
        "condition": "control",
        "episode": 3,
        "model": "model",
        "n_inputs": 2,
        "inputs_sha256": "inputs",
        "requested": {"dtype": "fp32", "tf32": False, "batch": 32, "threads": None},
        "embeddings_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
    }
    paths[1].write_text(json.dumps(manifest))

    assert collect_embeddings._reuse_existing_episode(
        paths, {k: manifest[k] for k in ("condition", "episode", "model", "n_inputs", "inputs_sha256", "requested")}
    )

    embeddings[0, 0] = 2.0
    np.savez_compressed(paths[0], embeddings=embeddings)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        collect_embeddings._reuse_existing_episode(paths, {
            k: manifest[k] for k in ("condition", "episode", "model", "n_inputs", "inputs_sha256", "requested")
        })


def test_partial_existing_episode_fails_closed(tmp_path):
    paths = collect_embeddings._episode_paths(tmp_path, "control", 4)
    paths[0].parent.mkdir(parents=True)
    paths[0].write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="incomplete episode"):
        collect_embeddings._reuse_existing_episode(paths, {})


def test_corrupt_archive_is_reported_as_invalid_not_raised_raw(tmp_path):
    """A half-written npz is a zipfile.BadZipFile, which is not an OSError."""
    paths = collect_embeddings._episode_paths(tmp_path, "control", 5)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.ones((64, 32), dtype=np.float32)
    np.savez_compressed(paths[0], embeddings=embeddings)
    raw = paths[0].read_bytes()
    paths[0].write_bytes(raw[: len(raw) // 2])
    paths[1].write_text(json.dumps({"embeddings_sha256": "x"}))
    with pytest.raises(RuntimeError, match="invalid existing episode"):
        collect_embeddings._reuse_existing_episode(paths, {})


def test_reuse_publishes_so_a_retry_after_a_failed_upload_still_uploads(
    monkeypatch, tmp_path
):
    """Fail-closed publish is pointless if the retry path skips the upload."""
    paths = collect_embeddings._episode_paths(tmp_path, "control", 6)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    embeddings = np.ones((2, 4), dtype=np.float32)
    np.savez_compressed(paths[0], embeddings=embeddings)
    manifest = {
        "condition": "control",
        "episode": 6,
        "model": "m",
        "n_inputs": 2,
        "inputs_sha256": collect_embeddings._inputs_sha256(["a", "b"]),
        "requested": {"dtype": "fp32", "tf32": False, "batch": 32, "threads": None},
        "embeddings_sha256": hashlib.sha256(embeddings.tobytes()).hexdigest(),
    }
    paths[1].write_text(json.dumps(manifest))

    uploaded = []
    monkeypatch.setattr(collect_embeddings.subprocess, "run",
                        lambda cmd, **kw: (uploaded.append(cmd[4]),
                                           SimpleNamespace(returncode=0, stdout="",
                                                           stderr=""))[1])
    monkeypatch.setattr(
        collect_embeddings.sys, "argv",
        ["collect_embeddings.py", "--condition", "control", "--episode", "6",
         "--model", "m", "--inputs", str(tmp_path / "in.jsonl"), "--n", "2",
         "--out", str(tmp_path), "--publish-s3", "s3://bucket/run"],
    )
    (tmp_path / "in.jsonl").write_text('{"text": "a"}\n{"text": "b"}\n')

    assert collect_embeddings.main() == 0
    assert uploaded == ["s3://bucket/run/control/ep006.npz",
                        "s3://bucket/run/control/ep006.json"]
