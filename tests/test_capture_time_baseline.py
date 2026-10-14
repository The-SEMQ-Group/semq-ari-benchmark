# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""The `time` capture tool end to end, with a stub agent in place of a model.

Capture a baseline, move its timestamps 30 hours back, and compare: the standalone
cell must pass the evidence check and be written. A hardware change must stop the
comparison. Merging a cell into a report recomputes ARI and ARI-R.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("semq")

from ari.tools import capture_time_baseline as ctb  # noqa: E402

INPUTS = Path(__file__).resolve().parents[1] / "data" / "ari-bench-v0.1.jsonl"


class _Agent:
    agent_id = "org/stub"
    revision = "a" * 40

    def encode(self, texts):
        x = np.random.default_rng(len(texts)).standard_normal((len(texts), 64))
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _env(hardware="x86_64 Linux-host-a"):
    return {"model_revision": "a" * 40, "tokenizer_revision": "a" * 40, "precision": "fp32",
            "hardware": hardware, "sdk_version": "1.0.0", "source_commit": "b" * 40}


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.setattr(ctb, "build", lambda *a, **k: (_Agent(), "self_hosted"))
    monkeypatch.setattr(ctb.os.environ, "update", lambda *a, **k: None)
    state = {"env": _env()}
    monkeypatch.setattr(ctb, "environment_record", lambda agent, cls: dict(state["env"]))
    return state


def _run(mode, d):
    return ctb.main(["--mode", mode, "--agent", "st", "--model", "org/stub",
                     "--inputs", str(INPUTS), "--limit", "20", "--baseline-dir", str(d)])


def _age_baseline(d, hours=30):
    meta = json.loads((d / "meta.json").read_text())
    then = datetime.now(timezone.utc) - timedelta(hours=hours)
    meta["started_at"] = meta["finished_at"] = then.strftime(ctb.TS_FMT)
    meta["unix_ts"] = time.time() - hours * 3600
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def test_standalone_compare_passes_and_writes_the_cell(tool, tmp_path):
    assert _run("capture", tmp_path) == 0
    meta = _age_baseline(tmp_path)
    assert meta["probe_calibration"] == "ARI-Canonical-v0.2"
    assert _run("compare", tmp_path) == 0
    cell = json.loads((tmp_path / "time_cell.json").read_text())
    semq = cell["results_per_condition"]["time"]["detectors"]["semq"]
    assert semq["HER"] == 1.0 and semq["coordinate_change_rate"] == 0.0
    assert cell["results_per_condition"]["time"]["time_evidence"]["calibration_scale"] == meta["scale_s"]


def test_a_hardware_change_is_not_a_time_comparison(tool, tmp_path):
    assert _run("capture", tmp_path) == 0
    _age_baseline(tmp_path)
    tool["env"] = _env(hardware="x86_64 Linux-host-b")
    with pytest.raises(SystemExit, match="hardware differs"):
        _run("compare", tmp_path)
    assert not (tmp_path / "time_cell.json").exists()


def _cell(her, rho):
    return {"detectors": {"semq": {"HER": her, "coordinate_change_rate": rho}}}


def test_merge_recomputes_ari_and_ari_r():
    rep = {"results_per_condition": {"proc": _cell(0.5, 0.1), "conc": _cell(0.5, 0.1),
                                     "time": _cell(0.5, 0.1)},
           "ARI": 0.5, "ARI_R": 0.9, "audit_hashes": {}}
    merged = ctb.merge_time_cell(rep, _cell(0.2, 0.4), "d")
    assert merged["ARI"] == pytest.approx(0.4)
    assert merged["ARI_R"] == pytest.approx(0.8)
    no_rho = {"detectors": {"semq": {"HER": 0.2}}}
    assert "ARI_R" not in ctb.merge_time_cell(rep, no_rho, "d")
