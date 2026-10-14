# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The `time` condition must carry the evidence for its gap.

The audit found self-hosted captures that called an immediate repeat `time`.
A schema cannot stop that: the report is well formed either way. These tests
pin the check that can: a `time` cell with an incomplete `time_evidence` block,
or with a gap of 24 hours or less, is rejected, and the rejection names the
field. A cell with no block at all predates the block and was not re-signed:
`ari.check_evidence` rejects it by default, `ari.verify_report` warns and
passes. A report with no `time` cell is untouched.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jsonschema")

from ari import metrics, report, run  # noqa: E402
from ari.check_evidence import check_report, schema_violations  # noqa: E402
from ari.verify_report import (CALIBRATION_SCALE_REL_TOL,  # noqa: E402
                               TIME_EVIDENCE_FIELDS, canonical_json,
                               check_time_condition, verify)

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "ari" / "verify_report.py"
HASH = "e9ec8b01c62635dee9fbbbdf8127cde5cac094aba66368ee8de306acc3afbe1d"
DIGEST = "0" * 64
COMMIT = "73139c8f0e1d2c3b4a5968778695a4b3c2d1e0f9"


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
        "probe_calibration": "ARI-Canonical-v0.1",
        "calibration_scale": 0.0731,
        "input_content_hash": HASH,
        "model_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "tokenizer_revision": "c9745ed1d9f207416be6d2e6f8de32d1f16199bf",
        "precision": "fp32",
        "hardware": "arm64 macOS-15.5-arm64-arm-64bit",
        "sdk_version": "1.4.1",
        "source_commit": COMMIT,
    }


def _report(gap_hours=30.0):
    time_cell = _semq_cell()
    time_cell["time_evidence"] = _evidence(gap_hours)
    return {
        "agent_id": "sentence-transformers/all-MiniLM-L6-v2",
        "ari_version": "0.1",
        "probe_calibration": "ARI-Canonical-v0.1",
        "input_set": "ARI-Bench-v0.1",
        "input_content_hash": HASH,
        "agent_class": "self_hosted",
        "encoder_fingerprint": {"dim": 384, "s": 0.0731},
        "environment": {"blas": "torch-default", "threads": 1, "hardware": "arm64",
                        "precision": "fp32", "library_versions": {"torch": "2.4.0"}},
        "results_per_condition": {"same": _semq_cell(), "proc": _semq_cell(),
                                  "time": time_cell},
        "ARI": 1.0,
        "audit_hashes": {"same": DIGEST, "proc": DIGEST, "time": DIGEST},
    }


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
    del rep["results_per_condition"]["time"]["time_evidence"][field]
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


def test_gap_hours_must_agree_with_the_timestamps():
    rep = _report(gap_hours=30.0)
    rep["results_per_condition"]["time"]["time_evidence"]["gap_hours"] = 76.0
    out = _violations_naming(rep, "gap_hours")
    assert any("recorded 76.0" in v for v in out), out


def test_a_timestamp_without_utc_marker_is_rejected():
    rep = _report()
    ev = rep["results_per_condition"]["time"]["time_evidence"]
    ev["comparison_started_at"] = ev["comparison_started_at"].rstrip("Z")
    _violations_naming(rep, "comparison_started_at")


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision", "source_commit"])
@pytest.mark.parametrize("moving", ["main", "unknown", "latest", "HEAD", ""])
def test_moving_refs_are_not_revisions(field, moving):
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"][field] = moving
    out = _violations_naming(rep, field)
    assert any("immutable revision" in v for v in out), out


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision", "source_commit"])
@pytest.mark.parametrize("ref", ["refs/heads/dev", "release-v1", "c9745ed",
                                 "C9745ED1D9F207416BE6D2E6F8DE32D1F16199BF",
                                 "provider-internal", None])
def test_self_hosted_revisions_must_be_full_commit_ids(field, ref):
    """A branch, a tag, a short id or a sentinel is not a commit a reader can fetch."""
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"][field] = ref
    out = _violations_naming(rep, field)
    assert any("40-character lowercase hex" in v for v in out), out


def test_named_refs_and_a_sentinel_commit_are_rejected():
    """The review's reproduction: each of the three fields is named."""
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"].update(
        model_revision="refs/heads/dev", tokenizer_revision="release-v1",
        source_commit="provider-internal")
    out = check_report(rep)
    for field in ("model_revision", "tokenizer_revision", "source_commit"):
        assert any(f"time_evidence.{field}:" in v for v in out), out


def _hosted_report():
    rep = _report()
    rep["agent_class"] = "api"
    rep["environment"]["precision"] = "provider-internal"
    rep["results_per_condition"]["time"]["time_evidence"].update(
        model_revision="pinned", tokenizer_revision="provider-internal",
        precision="provider-internal", hardware="provider-internal")
    return rep


@pytest.mark.parametrize("agent_class", ["api", "vendor", "rehosted", "platform"])
def test_a_hosted_agent_may_record_provider_internal(agent_class):
    """A hosted API exposes no commit; the snapshot id or the sentinel stands in."""
    rep = _hosted_report()
    rep["agent_class"] = agent_class
    assert check_report(rep) == []


@pytest.mark.parametrize("field", ["model_revision", "tokenizer_revision"])
@pytest.mark.parametrize("moving", ["main", "unknown", "latest", "HEAD", ""])
def test_a_hosted_revision_is_not_a_moving_ref(field, moving):
    rep = _hosted_report()
    rep["results_per_condition"]["time"]["time_evidence"][field] = moving
    out = _violations_naming(rep, field)
    assert any("immutable revision" in v for v in out), out


def test_a_hosted_agent_still_records_the_harness_commit():
    rep = _hosted_report()
    rep["results_per_condition"]["time"]["time_evidence"]["source_commit"] = "provider-internal"
    _violations_naming(rep, "source_commit")


@pytest.mark.parametrize("field", ["hardware", "precision"])
def test_a_self_hosted_agent_cannot_record_provider_internal(field):
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"][field] = "provider-internal"
    out = _violations_naming(rep, field)
    assert any("reserved for a hosted agent_class" in v for v in out), out


@pytest.mark.parametrize("field", ["gap_hours", "calibration_scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "30"])
def test_numbers_must_be_finite(field, value):
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"][field] = value
    out = _violations_naming(rep, field)
    assert any("finite" in v for v in out), out


def test_nan_gap_and_scale_are_both_named():
    """The review's reproduction: NaN compares false, so it passed every range check."""
    rep = _report()
    ev = rep["results_per_condition"]["time"]["time_evidence"]
    ev["gap_hours"] = ev["calibration_scale"] = float("nan")
    out = check_report(rep)
    assert any("gap_hours: nan is not a finite number" in v for v in out), out
    assert any("calibration_scale: nan is not a finite positive number" in v for v in out), out


def test_scale_must_match_the_fingerprint():
    """The review's reproduction: a scale of 999 against a fingerprint s of 0.0731."""
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"]["calibration_scale"] = 999.0
    out = _violations_naming(rep, "calibration_scale")
    assert any("differs from encoder_fingerprint.s 0.0731" in v for v in out), out
    assert any("differs from encoder_fingerprint.s" in v for v in check_report(rep))


def test_scale_tolerance_covers_rounding_the_fingerprint_and_no_more():
    """The fingerprint rounds s to 6 decimals; another calibration is far off."""
    s = 0.07314159265
    rep = _report()
    rep["encoder_fingerprint"]["s"] = round(s, 6)
    rep["results_per_condition"]["time"]["time_evidence"]["calibration_scale"] = s
    assert check_report(rep) == []
    rep["results_per_condition"]["time"]["time_evidence"]["calibration_scale"] = (
        round(s, 6) * (1 + 2 * CALIBRATION_SCALE_REL_TOL))
    _violations_naming(rep, "calibration_scale")


def test_v02_fixed_scale_is_checked_the_same_way():
    """ARI-Canonical-v0.2 fixes s at 2/sqrt(dim); the check compares to whatever s is."""
    dim = 768
    rep = _report()
    rep["encoder_fingerprint"] = {"dim": dim, "s": round(2 / dim ** 0.5, 6)}
    rep["results_per_condition"]["time"]["time_evidence"]["calibration_scale"] = 2 / dim ** 0.5
    assert check_report(rep) == []
    rep["results_per_condition"]["time"]["time_evidence"]["calibration_scale"] = 0.0731
    _violations_naming(rep, "calibration_scale")


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), "0.07"])
def test_an_unusable_fingerprint_scale_is_reported(bad):
    rep = _report()
    rep["encoder_fingerprint"]["s"] = bad
    out = check_time_condition(rep)
    assert any("calibration_scale: cannot be checked" in v for v in out), out


def test_a_report_without_a_fingerprint_scale_skips_the_cross_check():
    rep = _report()
    del rep["encoder_fingerprint"]
    assert check_report(rep) == []
    rep["encoder_fingerprint"] = {"dim": 384}
    assert check_report(rep) == []


def test_block_must_describe_this_report():
    rep = _report()
    ev = rep["results_per_condition"]["time"]["time_evidence"]
    ev["input_content_hash"] = "f" * 64
    _violations_naming(rep, "input_content_hash")
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"]["precision"] = "bf16"
    _violations_naming(rep, "precision")
    rep = _report()
    rep["results_per_condition"]["time"]["time_evidence"]["probe_calibration"] = "other"
    _violations_naming(rep, "probe_calibration")


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


def test_deviations_are_allowed_but_nothing_else_is():
    rep = _report()
    ev = rep["results_per_condition"]["time"]["time_evidence"]
    ev["deviations"] = ["baseline recorded one timestamp after its encode"]
    assert check_report(rep) == []
    ev["note"] = "x"
    assert any("note" in v for v in schema_violations(rep))
