"""Assemble and write an ARI report conforming to spec/report-schema.json.

ARI is averaged over the *present* averaged conditions (all of {proc,mach,prec,lib,conc,time}
for self-hosted; the applicable subset {proc,conc,time} for black-box APIs). The `same`
control is reported but excluded from the average — it must be 1.0 by construction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import ConditionMetrics

CANONICAL_CONDITIONS = ["same", "proc", "mach", "prec", "lib", "conc", "time", "batch"]
# Headline ARI averages over the *comparable core* — the conditions measurable for any agent
# class (a black-box API cannot observe mach/prec/lib). Averaging over this core keeps
# self-hosted and API scores apples-to-apples.
CORE_CONDITIONS = ["proc", "conc", "time"]
# Self-hosted-only diagnostics: reported per-condition, NOT folded into ARI. `prec` in
# particular is a reproducibility cliff (fp32 vs bf16) that would sink self-hosted below APIs
# purely because an API can hide it.
DIAGNOSTIC_CONDITIONS = ["mach", "prec", "lib", "batch"]
AVERAGED_CONDITIONS = CORE_CONDITIONS
ARI_VERSION = "0.1"


def condition_digest(codes: np.ndarray) -> str:
    """SHA-256 over a condition's full code matrix (inputs in fixed ARI-Bench order) — a
    compact, order-sensitive audit digest. A third party re-running the protocol on the same
    inputs recomputes the same digest per condition (exactly for a deterministic agent; for a
    non-deterministic API the digest records *our* observed run)."""
    return hashlib.sha256(np.ascontiguousarray(codes, dtype=np.uint8).tobytes()).hexdigest()


def build_report(
    *,
    agent_id: str,
    input_set: str,
    environment: dict,
    metrics_by_condition: dict[str, ConditionMetrics],
    codes_by_condition: dict[str, np.ndarray],
    agent_class: str = "self_hosted",
    input_content_hash: str | None = None,
    probe_calibration: str = "ARI-Canonical-v0.1",
    fingerprint: dict | None = None,
) -> dict:
    present_averaged = [c for c in AVERAGED_CONDITIONS if c in metrics_by_condition]
    if not present_averaged:
        raise ValueError("no averaged conditions present — cannot compute ARI")
    ari = float(np.mean([metrics_by_condition[c].HER for c in present_averaged]))

    # Conditions are born in the multi-detector shape
    # (spec/report-schema.json $defs/conditionResult); the flat pre-Matrix
    # shape exists only in ari/tools/migrate_multidetector.py's input.
    dim = (fingerprint or {}).get("dim")
    if dim is None:
        raise ValueError("fingerprint.dim is required to derive the semq "
                         "detector's bytes_of_reference_state")
    from ari.rpc import semq_bytes_of_reference_state
    bors = semq_bytes_of_reference_state(int(dim))
    results = {}
    for cond, m in metrics_by_condition.items():
        results[cond] = {"detectors": {"semq": {
            "HER": round(m.HER, 6), "Hbar": round(m.Hbar, 6),
            "HER_ci": [round(m.HER_ci[0], 6), round(m.HER_ci[1], 6)],
            "criterion_type": "exact",
            "bytes_of_reference_state": bors,
            "unit": "code",
        }}}

    audit = {cond: condition_digest(codes) for cond, codes in codes_by_condition.items()}

    report = {
        "agent_id": agent_id,
        "ari_version": ARI_VERSION,
        "probe_calibration": probe_calibration,
        "input_set": input_set,
        "agent_class": agent_class,
        "environment": environment,
        "results_per_condition": results,
        "ARI": round(ari, 6),
        "audit_hashes": audit,
    }
    if input_content_hash is not None:
        report["input_content_hash"] = input_content_hash
    if fingerprint is not None:
        report["encoder_fingerprint"] = fingerprint
    return report


def write_report(path: str | Path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path
