# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The decoding-layer (ARI-D) `time` evidence, from capture to gate.

run_dry.py records what a `time` comparison needs in each manifest run (the
HTTP client is stubbed; nothing else is), build_arid_submission.py builds the
block from two `same` runs, and check_report accepts the result. A run
without the fields, or two runs that differ in a held-fixed field, is refused.
"""
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from ari import time_evidence as te  # noqa: E402
from ari.check_evidence import check_report  # noqa: E402
from ari.tools import build_arid_submission as bas  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DRY = REPO / "experiments" / "arid-dry-run"
sys.path.insert(0, str(DRY))
import run_dry  # noqa: E402


class _Caller:
    """Stands in for providers.Caller: no network."""
    calls = 0

    def __init__(self, subject, auth):
        self.subject = subject
        self.reconnects = 0

    def call(self, text):
        _Caller.calls += 1
        version = "gemini-3.6-flash" if self.subject.endpoint_kind == "gemini" else None
        return {"ok": True, "parsed": {"text": "x", "upstream_model": version},
                "raw": {"model": "gpt-4o-mini-2024-07-18"} if version is None
                else {"modelVersion": version}}

    def close(self):
        pass


@pytest.fixture
def capture(monkeypatch, tmp_path):
    monkeypatch.setattr(run_dry, "Caller", _Caller)
    monkeypatch.setattr(run_dry, "build_auth", lambda subject: None)
    monkeypatch.setattr(run_dry, "HERE", tmp_path)
    _Caller.calls = 0

    def run(subject="gemini", label=None):
        argv = ["--subject", subject, "--condition", "same", "--limit", "2"]
        assert run_dry.main(argv + (["--label", label] if label else [])) == 0
        return json.loads((tmp_path / "results" / "manifest.json").read_text())["runs"][-1]
    return run


def _age(run, hours=30):
    end = datetime.strptime(run["finished_at"], te.TS_FMT) - timedelta(hours=hours)
    return {**run, "started_at": (end - timedelta(minutes=1)).strftime(te.TS_FMT),
            "finished_at": end.strftime(te.TS_FMT)}


def test_capture_records_what_time_needs(capture):
    run = capture()
    assert len(run["source_commit"]) == 40
    assert run["protocol"] == "arid-bench-v0.1"
    assert run["model_versions"] == ["gemini-3.6-flash"]
    cfg = run["request"]["body"]["generationConfig"]
    assert (cfg["temperature"], cfg["topP"], cfg["maxOutputTokens"], cfg["seed"]) == (
        0, 1, 128, 20260831)
    assert run["request_sha256"] == te.request_digest(run["request"])
    assert run["started_at"] <= run["finished_at"] and "wall_s" in run


def test_capture_refuses_an_unresolved_commit_before_any_request(capture, monkeypatch):
    monkeypatch.setattr(te, "source_commit", lambda: "unknown")
    with pytest.raises(SystemExit, match="harness commit"):
        capture()
    assert _Caller.calls == 0


def _dry_copy(tmp_path, runs):
    """This branch's dry-run results, with the openai `same` pair replaced by `runs`."""
    d = tmp_path / "dry"
    d.mkdir()
    for name in ("analysis.json", "time_pair.json"):
        shutil.copy(DRY / "results" / name, d / name)
    manifest = json.loads((DRY / "results" / "manifest.json").read_text())
    by_name = {r["transcript"]: r for r in manifest["runs"]}
    for name, run in zip(("openai_same.jsonl.gz", "openai_same_t2.jsonl.gz"), runs):
        by_name[name].update({**run, "transcript": name, "sha256": by_name[name]["sha256"]})
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def _build(monkeypatch, tmp_path, dry):
    monkeypatch.setattr(bas, "DRY", dry)
    bas.main(["--subject", "openai", "--agent-id", "openai/gpt-4o-mini",
              "--measured-at", "2026-10-14", "--out", str(tmp_path)])
    return json.loads((tmp_path / "openai_gpt-4o-mini.decoding.json").read_text())


def test_decoding_evidence_passes_the_gate_end_to_end(capture, monkeypatch, tmp_path):
    base, comp = _age(capture("openai")), capture("openai", label="t2")
    rep = _build(monkeypatch, tmp_path, _dry_copy(tmp_path, [base, comp]))
    ev = rep["results_per_condition"]["time"]["time_evidence"]
    assert ev["gap_hours"] > 24 and ev["model_versions"] == ["gpt-4o-mini-2024-07-18"]
    assert check_report(rep) == []


def test_existing_captures_without_the_fields_are_refused(monkeypatch, tmp_path):
    """This branch's openai pair predates the fields; it is not backfilled."""
    monkeypatch.setattr(bas, "DRY", DRY / "results")
    with pytest.raises(SystemExit, match="does not record protocol, request_sha256, started_at"):
        bas.main(["--subject", "openai", "--agent-id", "openai/gpt-4o-mini",
                  "--measured-at", "2026-10-14", "--out", str(tmp_path)])


@pytest.mark.parametrize("field,value", [
    ("prompt_content_hash", "f" * 64), ("protocol", "arid-bench-v0.2"),
    ("model", "gemini-3.7-flash"), ("scope", "full")])
def test_a_held_fixed_field_that_differs_is_refused(capture, field, value):
    base, comp = _age(capture()), capture()
    with pytest.raises(te.TimeEvidenceError, match=field):
        te.decoding_time_evidence(base, {**comp, field: value}, agent_id="gemini/x")


def test_a_changed_request_param_is_refused(capture):
    base, comp = _age(capture()), capture()
    request = json.loads(json.dumps(comp["request"]))
    request["body"]["generationConfig"]["temperature"] = 0.7
    changed = {**comp, "request": request, "request_sha256": te.request_digest(request)}
    with pytest.raises(te.TimeEvidenceError, match="request_sha256"):
        te.decoding_time_evidence(base, changed, agent_id="gemini/x")
    with pytest.raises(te.TimeEvidenceError, match="does not match its request"):
        te.decoding_time_evidence(base, {**comp, "request": request}, agent_id="gemini/x")


def test_versions_and_commits_are_recorded_not_held_fixed(capture):
    base, comp = _age(capture()), capture()
    comp = {**comp, "model_versions": ["gemini-3.6-flash-002"], "source_commit": "c" * 40}
    ev = te.decoding_time_evidence(base, comp, agent_id="gemini/x")
    assert ev["baseline_model_versions"] == ["gemini-3.6-flash"]
    assert ev["model_versions"] == ["gemini-3.6-flash-002"]
    assert ev["source_commit"] == "c" * 40 != ev["baseline_source_commit"]


def test_a_short_gap_is_refused(capture):
    with pytest.raises(te.TimeEvidenceError, match="more than 24"):
        te.decoding_time_evidence(capture(), capture(), agent_id="gemini/x")


def test_the_layer_selects_the_form(capture, monkeypatch, tmp_path):
    """A decoding report needs the decoding fields; an embedding block in it fails,
    and the decoding block in an embedding report fails."""
    base, comp = _age(capture("openai")), capture("openai", label="t2")
    rep = _build(monkeypatch, tmp_path, _dry_copy(tmp_path, [base, comp]))
    ev = rep["results_per_condition"]["time"]["time_evidence"]
    del ev["protocol"]
    assert any(v.endswith("time_evidence.protocol: missing") for v in check_report(rep))
    ev["protocol"] = "arid-bench-v0.1"
    rep["layer"] = "representation"
    assert any(v.endswith("time_evidence.probe_calibration: missing") for v in check_report(rep))
