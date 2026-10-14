# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The `time` condition must carry the evidence for its gap.

The audit found self-hosted captures that called an immediate repeat `time`.
A schema cannot stop that: the report is well formed either way. These tests
pin the check that can: a `time` cell with an incomplete `time_evidence` block,
with a gap of 24 hours or less, or that does not describe its report, is
rejected, and the rejection names the field. A cell with no block at all predates the block and was not re-signed:
`ari.check_evidence` rejects it by default, `ari.verify_report` warns and
passes. A report with no `time` cell is untouched.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from ari import metrics, report, run  # noqa: E402
from ari.check_evidence import check_report, schema_violations  # noqa: E402
from ari.verify_report import (TIME_EVIDENCE_FIELDS, canonical_json,  # noqa: E402
                               check_time_condition, verify)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "ari" / "verify_report.py"
HASH = "e9ec8b01c62635dee9fbbbdf8127cde5cac094aba66368ee8de306acc3afbe1d"
DIGEST = "0" * 64
COMMIT = "73139c8f0e1d2c3b4a5968778695a4b3c2d1e0f9"
REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
AGENT = "sentence-transformers/all-MiniLM-L6-v2"
HARDWARE = "arm64 | Apple M2"
DIM = 384


def _semq_cell(her=1.0):
    return {"detectors": {"semq": {
        "HER": her, "Hbar": 0.0, "HER_ci": [her, her], "criterion_type": "exact",
        "bytes_of_reference_state": 96, "unit": "code"}}}


def _evidence(gap_hours=30.0):
    """A conformant block: baseline on the 1st, comparison `gap_hours` later."""
    from datetime import datetime, timedelta

    b_start = datetime(2026, 9, 1, 12, 0, 0)
    b_end = b_start + timedelta(minutes=2)
    c_start = b_end + timedelta(hours=gap_hours)
    c_end = c_start + timedelta(minutes=2)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return {
        "baseline_started_at": b_start.strftime(fmt),
        "baseline_finished_at": b_end.strftime(fmt),
        "comparison_started_at": c_start.strftime(fmt),
        "comparison_finished_at": c_end.strftime(fmt),
        "gap_hours": gap_hours,
        "agent_id": AGENT,
        "input_content_hash": HASH,
        "probe_calibration": "ARI-Canonical-v0.2",
        "model_revision": REVISION,
        "tokenizer_revision": REVISION,
        "precision": "fp32",
        "hardware": HARDWARE,
        "sdk_version": "1.0.0",
        "source_commit": COMMIT,
    }


def _report(gap_hours=30.0):
    """A v0.2 self-hosted report whose time cell carries conformant evidence."""
    time_cell = _semq_cell()
    time_cell["time_evidence"] = _evidence(gap_hours)
    return {
        "agent_id": AGENT,
        "ari_version": "0.1",
        "probe_calibration": "ARI-Canonical-v0.2",
        "input_set": "ARI-Bench-v0.1",
        "input_content_hash": HASH,
        "agent_class": "self_hosted",
        "encoder_fingerprint": {"dim": DIM, "s": round(2 / DIM ** 0.5, 6)},
        "environment": {"blas": "torch-default", "threads": 1, "hardware": HARDWARE,
                        "precision": "fp32",
                        "library_versions": {"torch": "2.4.0", "semq": "1.0.0"}},
        "results_per_condition": {"same": _semq_cell(), "proc": _semq_cell(),
                                  "conc": _semq_cell(), "time": time_cell},
        "ARI": 1.0,
        "audit_hashes": {"same": DIGEST, "proc": DIGEST, "conc": DIGEST, "time": DIGEST},
    }


def _ev(rep):
    return rep["results_per_condition"]["time"]["time_evidence"]


def _violations_naming(report, field):
    out = check_time_condition(report)
    assert out, f"expected a violation naming {field}"
    assert any(f"time_evidence.{field}" in v for v in out), out
    return out


def test_conformant_report_passes_schema_and_check():
    assert check_report(_report()) == []


@pytest.mark.parametrize("field", TIME_EVIDENCE_FIELDS)
def test_each_missing_field_is_named(field):
    rep = _report()
    del _ev(rep)[field]
    out = _violations_naming(rep, field)
    assert any(v.endswith(f"{field}: missing") for v in out), out
    # The schema requires the field too, so a jsonschema-only validator agrees.
    assert any(field in v for v in schema_violations(rep))


def test_a_23_hour_gap_is_rejected_naming_gap_hours():
    rep = _report(gap_hours=23.0)
    out = _violations_naming(rep, "gap_hours")
    assert any("23.00 h" in v and "more than 24" in v for v in out), out
    assert any("gap_hours" in v for v in schema_violations(rep))


def test_exactly_24_hours_is_not_more_than_24():
    _violations_naming(_report(gap_hours=24.0), "gap_hours")


def test_the_timestamps_decide_and_gap_hours_must_agree():
    """A7: 24 h 00 m 15 s is a valid gap. Recorded rounded up (24.01) it passes;
    the timestamps, not the recorded number, decide."""
    rep = _report(gap_hours=24 + 15 / 3600)
    _ev(rep)["gap_hours"] = 24.01
    assert check_report(rep) == []
    _ev(rep)["gap_hours"] = 76.0
    out = _violations_naming(rep, "gap_hours")
    assert any("recorded 76.0" in v for v in out), out


def test_a_timestamp_without_utc_marker_is_rejected():
    rep = _report()
    _ev(rep)["comparison_started_at"] = _ev(rep)["comparison_started_at"].rstrip("Z")
    _violations_naming(rep, "comparison_started_at")


SENTINELS = ["main", "unknown", "UNKNOWN ", "latest", "HEAD", "", "pinned", "floating",
             "none", "unavailable", None]


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision", "source_commit",
                                   "hardware", "sdk_version", "agent_id"])
@pytest.mark.parametrize("sentinel", SENTINELS)
def test_placeholders_identify_nothing(field, sentinel):
    rep = _report()
    _ev(rep)[field] = sentinel
    _violations_naming(rep, field)


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision", "source_commit"])
@pytest.mark.parametrize("ref", ["refs/heads/dev", "release-v1", "v1.0", "c9745ed", "abc1234",
                                 "C9745ED1D9F207416BE6D2E6F8DE32D1F16199BF"])
def test_self_hosted_revisions_must_be_full_commit_ids(field, ref):
    """A branch, a tag or a short id is not a commit a reader can fetch."""
    rep = _report()
    _ev(rep)[field] = ref
    out = _violations_naming(rep, field)
    assert any("40-character lowercase hex" in v for v in out), out


def _hosted_report():
    rep = _report()
    rep["agent_id"] = "openai/text-embedding-3-large"
    rep["agent_class"] = "api"
    rep["environment"].update(precision="provider-internal", hardware="provider-internal")
    _ev(rep).update(agent_id=rep["agent_id"], model_revision="text-embedding-3-large",
                    tokenizer_revision="provider-internal", precision="provider-internal",
                    hardware="provider-internal")
    return rep


@pytest.mark.parametrize("agent_class", ["api", "vendor", "rehosted", "platform"])
def test_a_hosted_agent_may_record_provider_internal(agent_class):
    """A hosted API exposes no commit; the model id or the sentinel stands in."""
    rep = _hosted_report()
    rep["agent_class"] = agent_class
    assert check_report(rep) == []
    _ev(rep)["model_revision"] = "provider-internal"
    assert check_report(rep) == []


@pytest.mark.parametrize("sentinel", ["pinned", "floating", "none", "main", "UNKNOWN "])
def test_a_hosted_revision_is_not_a_placeholder(sentinel):
    """A3: every API agent's snapshot attribute is the literal 'pinned'."""
    rep = _hosted_report()
    _ev(rep)["model_revision"] = sentinel
    _violations_naming(rep, "model_revision")


@pytest.mark.parametrize("field", ["sdk_version", "source_commit", "agent_id"])
def test_provider_internal_only_in_the_four_provider_fields(field):
    """A4: a hosted agent may not hide the SDK version, the harness commit or itself."""
    rep = _hosted_report()
    _ev(rep)[field] = "provider-internal"
    out = _violations_naming(rep, field)
    assert any("allowed only in" in v for v in out), out


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision", "hardware",
                                   "precision"])
def test_a_self_hosted_agent_cannot_record_provider_internal(field):
    rep = _report()
    _ev(rep)[field] = "provider-internal"
    out = _violations_naming(rep, field)
    assert any("hosted agent_class" in v for v in out), out


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "30"])
def test_gap_hours_must_be_finite(value):
    rep = _report()
    _ev(rep)["gap_hours"] = value
    out = _violations_naming(rep, "gap_hours")
    assert any("finite" in v for v in out), out


def test_v02_scale_must_be_two_over_sqrt_dim():
    """A5: under ARI-Canonical-v0.2 s carries nothing beyond dim, so it is checked
    against 2/sqrt(dim) rather than against a recorded scale."""
    rep = _report()
    rep["encoder_fingerprint"]["s"] = 0.0731
    out = check_report(rep)
    assert any(v.startswith("encoder_fingerprint.s: 0.0731 is not 2/sqrt(384)") for v in out), out
    rep["encoder_fingerprint"] = {"dim": DIM}
    assert any(v.startswith("encoder_fingerprint.s: None") for v in check_report(rep))
    del rep["encoder_fingerprint"]
    assert any(v.startswith("encoder_fingerprint.dim") for v in check_report(rep))


def test_calibration_scale_is_gone():
    rep = _report()
    _ev(rep)["calibration_scale"] = 0.102062
    assert any("calibration_scale" in v for v in schema_violations(rep))


@pytest.mark.parametrize("field,value", [
    ("agent_id", "other/model"), ("input_content_hash", "f" * 64),
    ("probe_calibration", "ARI-Canonical-v0.1"), ("precision", "bf16"),
    ("hardware", "x86_64 | Intel Xeon"), ("sdk_version", "1.0.1")])
def test_block_must_describe_this_report(field, value):
    """A1: the block is cross-checked against the report it sits in."""
    rep = _report()
    _ev(rep)[field] = value
    out = _violations_naming(rep, field)
    assert any("differs from the report's" in v for v in out), out


def test_sdk_version_is_checked_only_when_the_report_records_semq():
    rep = _report()
    del rep["environment"]["library_versions"]["semq"]
    assert check_report(rep) == []


def test_legacy_flat_cell_with_evidence_validates():
    """The v0.1 flat shape may carry the block as well as the detector shape."""
    rep = _report()
    rep["results_per_condition"]["time"] = {"HER": 1.0, "Hbar": 0.0,
                                            "time_evidence": _evidence()}
    assert check_report(rep) == []


def test_mock_pipeline_time_cell_without_evidence_is_rejected():
    """The mock panel simulates `time` with noise and no gap. Schema-valid, and
    exactly the cell check_evidence exists to refuse by default."""
    # run_mock_panel encodes with the canonical probe, which needs the SDK
    # since the development mock was removed.
    pytest.importorskip("semq")
    rep = run.run_mock_panel()
    assert schema_violations(rep) == []
    out = check_report(rep)
    assert len(out) == 1 and out[0].startswith(
        "results_per_condition.time.time_evidence: missing"), out
    assert check_report(rep, require_time_evidence=False) == []


def _sign(tmp_path, rep):
    """Write `rep` with a manifest and an Ed25519 sidecar that verify_report accepts."""
    import base64
    import hashlib

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    path = tmp_path / "report.json"
    path.write_text(json.dumps(rep) + "\n")
    manifest = canonical_json({"report": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                               "inputs": []})
    (tmp_path / "report.attestation.json").write_bytes(manifest)
    digest = hashlib.sha256(manifest).hexdigest()
    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    (tmp_path / "report.attestation.notary").write_text(json.dumps({
        "magic": "NTRY", "snapshot_sha256": digest,
        "signer_public_key": base64.b64encode(pub).decode(),
        "signature": base64.b64encode(key.sign(bytes.fromhex(digest))).decode(),
        "signer_identity": "test@example.com", "created_at": "2026-09-01T12:00:00Z"}))
    return path


def test_verify_report_warns_but_passes_on_a_legacy_time_cell(tmp_path, capsys):
    """Reports signed before the block existed were not re-signed
    (spec/condition-set.md). The verifier says so and still passes."""
    pytest.importorskip("cryptography")
    rep = _report()
    del rep["results_per_condition"]["time"]["time_evidence"]
    path = _sign(tmp_path, rep)
    assert verify(path) is True
    out = capsys.readouterr().out
    assert "[WARN] time cell carries no capture evidence" in out
    assert "unverified" in out
    assert verify(path, require_time_evidence=True) is False
    assert "time_evidence: missing" in capsys.readouterr().out


def test_verify_report_fails_a_present_block_that_violates(tmp_path, capsys):
    pytest.importorskip("cryptography")
    assert verify(_sign(tmp_path, _report(gap_hours=23.0))) is False
    assert "gap_hours" in capsys.readouterr().out
    assert verify(_sign(tmp_path, _report())) is True


def test_build_report_ari_is_the_mean_of_the_raw_hers():
    """core_ari sees the ConditionMetrics, not the 6 dp HER written per cell."""
    import numpy as np

    hers = {"proc": 0.8512345678, "conc": 0.7623456789, "time": 0.9034567891}
    by_cond = {c: metrics.ConditionMetrics(HER=h, Hbar=1.0, HER_ci=(h, h), n=3)
               for c, h in hers.items()}
    codes = {c: np.zeros((3, 4), dtype=np.uint8) for c in hers}
    rep = report.build_report(agent_id="a", input_set="s", environment={},
                              metrics_by_condition=by_cond, codes_by_condition=codes,
                              fingerprint={"dim": 32})
    raw_mean = round(sum(hers.values()) / 3, 6)
    rounded_mean = round(sum(round(h, 6) for h in hers.values()) / 3, 6)
    assert raw_mean != rounded_mean, "pick HERs where rounding first changes the sixth digit"
    assert rep["ARI"] == raw_mean
    assert round(report.core_ari(rep["results_per_condition"]), 6) == rounded_mean


def test_report_without_time_cell_still_validates():
    # run_mock_panel encodes with the canonical probe, which needs the SDK
    # since the development mock was removed.
    pytest.importorskip("semq")
    rep = run.run_mock_panel()
    del rep["results_per_condition"]["time"]
    del rep["audit_hashes"]["time"]
    assert check_report(rep) == []
    assert check_time_condition(rep) == []


def test_cli_exit_status(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(_report()))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_report(gap_hours=23.0)))
    env = {"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"}
    ok = subprocess.run([sys.executable, "-m", "ari.check_evidence", str(good)],
                        capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "30.0 h gap" in ok.stdout
    fail = subprocess.run([sys.executable, "-m", "ari.check_evidence", str(bad)],
                          capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    assert fail.returncode == 1
    assert "gap_hours" in fail.stdout
    legacy_report = _report()
    del legacy_report["results_per_condition"]["time"]["time_evidence"]
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(legacy_report))
    rejected = subprocess.run([sys.executable, "-m", "ari.check_evidence", str(legacy)],
                              capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    assert rejected.returncode == 1 and "time_evidence: missing" in rejected.stdout
    warned = subprocess.run([sys.executable, "-m", "ari.check_evidence",
                             "--no-require-time-evidence", str(legacy)],
                            capture_output=True, text=True, cwd=REPO_ROOT, env=env)
    assert warned.returncode == 0, warned.stdout + warned.stderr
    assert "[WARN] time cell carries no capture evidence" in warned.stdout


def test_verifier_check_is_standard_library_only():
    """The standalone verifier carries the same check without importing the
    package. This asserts the import list stayed clean; test_attestation.py
    asserts the same on the executed copy when semq is installed."""
    modules = set()
    for node in ast.walk(ast.parse(VERIFIER.read_text())):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert "ari" not in modules and "semq" not in modules, sorted(modules)
    assert modules <= {"argparse", "base64", "hashlib", "json", "math", "re", "sys",
                       "pathlib", "datetime", "cryptography", "__future__"}, sorted(modules)


def test_check_is_the_same_function_in_both_places():
    import ari.verify_report as vr
    assert check_time_condition is vr.check_time_condition


def test_deviations_fail_the_gate_and_warn_in_the_verifier(tmp_path, capsys):
    """A cell with deviations is not a clean canonical capture."""
    rep = _report()
    _ev(rep)["deviations"] = ["baseline recorded one timestamp after its encode"]
    assert schema_violations(rep) == []
    assert any("deviations: 1 recorded" in v for v in check_report(rep))
    assert check_time_condition(rep, allow_deviations=True) == []
    pytest.importorskip("cryptography")
    assert verify(_sign(tmp_path, rep)) is True
    assert "[WARN] time cell deviates from the procedure" in capsys.readouterr().out


def test_unknown_fields_are_rejected():
    rep = _report()
    _ev(rep)["note"] = "x"
    assert any("note" in v for v in schema_violations(rep))
