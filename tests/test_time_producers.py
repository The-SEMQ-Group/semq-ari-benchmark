# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""The tools that produce `time` cells, end to end, with stub encoders.

ari/time_evidence.py builds the records for real: the hosted and self-hosted
environment, the hardware id and the harness commit are not stubbed. Only the
API clients and the sentence-transformers model are. Each producer captures a
baseline, the test moves its timestamps 30 hours back, and the comparison must
carry evidence that passes check_report.
"""
import json
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("semq")
pytest.importorskip("jsonschema")

from ari import metrics, report  # noqa: E402
from ari import time_evidence as te  # noqa: E402
from ari.check_evidence import check_report  # noqa: E402
from ari.inputs import InputSet, load_ari_bench  # noqa: E402
from ari.tools import capture_time_baseline as ctb  # noqa: E402

INPUTS = Path(__file__).resolve().parents[1] / "data" / "ari-bench-v0.1.jsonl"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
DIM = 64


def _vectors(texts):
    x = np.random.default_rng(len(texts)).standard_normal((len(texts), DIM))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class _APIAgent:
    """Stands in for an ari.agents API agent: same attributes, no network."""
    snapshot = "pinned"     # what every real API agent sets

    def __init__(self, model_id="stub-embed", max_workers=1):
        self.agent_id = model_id
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return _vectors(texts)

    encode_fresh = encode

    def encode_batched(self, texts):
        raise NotImplementedError


def _age(meta_path, hours=30):
    meta = json.loads(meta_path.read_text())
    end = datetime.now(timezone.utc) - timedelta(hours=hours)
    meta["started_at"] = (end - timedelta(minutes=1)).strftime(te.TS_FMT)
    meta["finished_at"] = end.strftime(te.TS_FMT)
    meta_path.write_text(json.dumps(meta))
    return meta


def _inputs(n=20):
    full = load_ari_bench(INPUTS)
    return InputSet(full.name, full.ids[:n], full.texts[:n])


# --------------------------------------------------------------------- shared module
def test_hosted_record_names_the_model_not_pinned():
    """A3: the revision is the model id; the 'pinned' placeholder is dropped."""
    env = te.hosted_environment(_APIAgent())
    assert env["model_revision"] == "stub-embed"
    assert env["sdk_version"] == "1.0.0"
    assert len(env["source_commit"]) == 40
    assert {env[f] for f in ("tokenizer_revision", "precision", "hardware")} == {"provider-internal"}
    agent = _APIAgent()
    agent.snapshot = "2026-01-15"
    assert te.hosted_environment(agent)["model_revision"] == "stub-embed@2026-01-15"


class _Model:
    """A loaded sentence-transformers model, as far as the record reads it."""

    def __init__(self, dtype="torch.float32"):
        self.dtype = dtype

    def parameters(self):
        yield types.SimpleNamespace(dtype=self.dtype)

    def encode(self, texts, **kwargs):
        return _vectors(texts)


def test_self_hosted_record_reads_the_loaded_model():
    env = te.self_hosted_environment(_Model(), REVISION)
    assert env["model_revision"] == env["tokenizer_revision"] == REVISION
    assert env["precision"] == "fp32"
    assert env["hardware"] == te.hardware_id()
    assert te.self_hosted_environment(_Model("torch.bfloat16"), REVISION)["precision"] == "bf16"
    te.check_identities("org/model", "self_hosted", env)


@pytest.mark.parametrize("revision", ["v1.0", "abc1234", "main", "unknown"])
def test_a_bad_pin_is_refused_at_capture_time(revision):
    """A8: refused when the baseline is recorded, not after the 24-hour wait."""
    env = te.self_hosted_environment(_Model(), revision)
    with pytest.raises(te.TimeEvidenceError, match="model_revision"):
        te.baseline_record(agent_id="org/model", agent_class="self_hosted",
                           input_content_hash="a" * 64, dim=DIM, started_at="",
                           finished_at="", environment=env)


def test_an_unresolved_harness_commit_is_refused(monkeypatch):
    monkeypatch.setattr(te, "source_commit", lambda: "unknown")
    with pytest.raises(te.TimeEvidenceError, match="source_commit"):
        te.check_identities("openai/x", "api", te.hosted_environment(_APIAgent()))


def test_hardware_id_is_the_machine_not_the_host_or_os(monkeypatch):
    """A9: an OS update or another hostname is the same hardware."""
    before = te.hardware_id()
    monkeypatch.setattr(te.platform, "node", lambda: "another-host")
    monkeypatch.setattr(te.platform, "platform", lambda: "Linux-9.9-other")
    assert te.hardware_id() == before
    assert te.cpu_model() in before


def test_hardware_id_names_the_cpu_and_gpus(monkeypatch, tmp_path):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Intel(R) Xeon(R) Platinum 8375C\n")
    real_path = te.Path
    monkeypatch.setattr(te, "Path", lambda p: cpuinfo if p == "/proc/cpuinfo" else real_path(p))
    monkeypatch.setattr(te.platform, "system", lambda: "Linux")
    monkeypatch.setattr(te.platform, "machine", lambda: "x86_64")
    cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                                 get_device_name=lambda i: "NVIDIA L40S")
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    assert te.hardware_id() == "x86_64 | Intel(R) Xeon(R) Platinum 8375C | NVIDIA L40S, NVIDIA L40S"


def _record(**changes):
    env = {"model_revision": REVISION, "tokenizer_revision": REVISION, "precision": "fp32",
           "hardware": "x86_64 | Xeon", "sdk_version": "1.0.0", "source_commit": "b" * 40}
    rec = te.baseline_record(agent_id="org/model", agent_class="self_hosted",
                             input_content_hash="a" * 64, dim=DIM,
                             started_at="2026-09-01T12:00:00Z", finished_at="2026-09-01T12:02:00Z",
                             environment=env)
    return {**rec, **changes}


def test_gap_rounds_up_so_a_valid_gap_never_records_24():
    """A7: 24 h 00 m 15 s is more than 24 h; it records 24.01, not 24.0."""
    comp = _record(started_at="2026-09-02T12:02:15Z", finished_at="2026-09-02T12:04:00Z")
    ev = te.time_evidence(_record(), comp)
    assert ev["gap_hours"] == 24.01
    exact = _record(started_at="2026-09-02T12:02:00Z", finished_at="2026-09-02T12:04:00Z")
    with pytest.raises(te.TimeEvidenceError, match="more than 24"):
        te.time_evidence(_record(), exact)


@pytest.mark.parametrize("field", te.HELD_FIXED)
def test_a_held_fixed_field_missing_from_the_baseline_is_refused(field):
    """A2: the comparison never fills in a value the baseline did not record."""
    base = _record()
    del base[field]
    comp = _record(started_at="2026-09-03T00:00:00Z", finished_at="2026-09-03T00:01:00Z")
    with pytest.raises(te.TimeEvidenceError, match=f"does not record {field}"):
        te.time_evidence(base, comp)


@pytest.mark.parametrize("field", te.HELD_FIXED)
def test_a_held_fixed_field_that_differs_is_refused(field):
    comp = _record(started_at="2026-09-03T00:00:00Z", finished_at="2026-09-03T00:01:00Z")
    comp[field] = 999 if field == "dim" else comp[field] + "-other"
    with pytest.raises(te.TimeEvidenceError, match=f"{field} "):
        te.time_evidence(_record(), comp)


# --------------------------------------------------------------------- capture_time_baseline
@pytest.fixture
def ctb_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(ctb, "build", lambda *a, **k: _APIAgent())
    return tmp_path


def _ctb(mode, d, *extra):
    return ctb.main(["--mode", mode, "--agent", "openai", "--inputs", str(INPUTS),
                     "--limit", "20", "--baseline-dir", str(d), *extra])


def _hosted_report(agent_id="openai/stub-embed", dim=DIM, content_hash=None):
    codes = np.zeros((20, dim // 4), dtype=np.uint8)
    m = metrics.aggregate(codes, codes, dim=dim)
    return report.build_report(
        agent_id=agent_id, input_set="ARI-Bench-v0.1",
        input_content_hash=content_hash or _inputs().content_hash,
        environment={"blas": "provider-internal", "threads": 0, "hardware": "provider-internal",
                     "precision": "provider-internal", "library_versions": {"semq": "1.0.0"}},
        metrics_by_condition={c: m for c in ("same", "proc", "conc")},
        codes_by_condition={c: codes for c in ("same", "proc", "conc")}, agent_class="api",
        fingerprint={"dim": dim, "s": round(2 / dim ** 0.5, 6)})


def test_capture_compare_and_merge(ctb_dir):
    assert _ctb("capture", ctb_dir) == 0
    _age(ctb_dir / "meta.json")
    assert _ctb("compare", ctb_dir) == 0
    cell = json.loads((ctb_dir / "time_cell.json").read_text())["results_per_condition"]["time"]
    assert cell["time_evidence"]["model_revision"] == "stub-embed"
    rp, out = ctb_dir / "r.json", ctb_dir / "out.json"
    rp.write_text(json.dumps(_hosted_report()))
    assert _ctb("compare", ctb_dir, "--report", str(rp), "--out", str(out)) == 0
    merged = json.loads(out.read_text())
    assert check_report(merged) == []
    assert merged["ARI"] == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs,field", [
    ({"agent_id": "cohere/embed-v4.0"}, "agent_id"),
    ({"dim": 128}, "encoder_fingerprint.dim"),
    ({"content_hash": "f" * 64}, "input_content_hash")])
def test_merge_refuses_another_agents_report(ctb_dir, capsys, kwargs, field):
    """A1: nothing is merged into a report for another agent, inputs or dimension."""
    assert _ctb("capture", ctb_dir) == 0
    _age(ctb_dir / "meta.json")
    rp, out = ctb_dir / "r.json", ctb_dir / "out.json"
    rp.write_text(json.dumps(_hosted_report(**kwargs)))
    assert _ctb("compare", ctb_dir, "--report", str(rp), "--out", str(out)) == 1
    assert f"--report {field}" in capsys.readouterr().out
    assert not out.exists()


def test_compare_refuses_a_legacy_baseline_before_encoding(ctb_dir, capsys):
    """A2: the minimal meta refresh_leaderboard used to write in the same store."""
    np.save(ctb_dir / "codes.npy", np.zeros((20, 16), dtype=np.uint8))
    (ctb_dir / "meta.json").write_text(json.dumps({"unix_ts": 0, "scale_s": 0.25, "dim": DIM,
                                                   "content_hash": "a" * 64, "n": 20}))
    assert _ctb("compare", ctb_dir) == 1
    assert "does not record" in capsys.readouterr().out


def test_merge_recomputes_ari_and_ari_r():
    def cell(her, rho):
        return {"detectors": {"semq": {"HER": her, "coordinate_change_rate": rho}}}

    record = {"agent_id": "a", "input_content_hash": "h", "dim": DIM}
    rep = {"agent_id": "a", "input_content_hash": "h", "encoder_fingerprint": {"dim": DIM},
           "results_per_condition": {"proc": cell(0.5, 0.1), "conc": cell(0.5, 0.1),
                                     "time": cell(0.5, 0.1)},
           "ARI": 0.5, "ARI_R": 0.9, "audit_hashes": {}}
    merged = ctb.merge_time_cell(rep, cell(0.2, 0.4), "d", record)
    assert merged["ARI"] == pytest.approx(0.4)
    assert merged["ARI_R"] == pytest.approx(0.8)
    no_rho = {"detectors": {"semq": {"HER": 0.2}}}
    assert "ARI_R" not in ctb.merge_time_cell(rep, no_rho, "d", record)
