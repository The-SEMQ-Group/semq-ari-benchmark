# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
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

from .code_metrics import bits_per_coordinate
from .metrics import ConditionMetrics
from .probe import N_BINS
from .rpc import her, semq_bytes_of_reference_state

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


def _sig(x: float, digits: int = 6) -> float:
    """``x`` to ``digits`` significant digits (a rate near 1e-5 keeps its precision)."""
    return float(f"{x:.{digits}g}")


def semq_condition_entry(m: ConditionMetrics, dim: int) -> dict:
    """One condition in the multi-detector shape (spec/report-schema.json
    $defs/conditionResult) for the `semq` detector."""
    if m.dim is not None and m.dim != int(dim):
        raise ValueError(f"rho was computed over {m.dim} coordinates, but fingerprint.dim is {dim}")
    bors = semq_bytes_of_reference_state(int(dim))
    # Hbar is a bit count, so it needs the code-bit total to become a rate.
    bpc = bits_per_coordinate(N_BINS)
    n_bits = int(dim) * bpc
    return {"detectors": {"semq": {
        "HER": round(m.HER, 6), "Hbar": round(m.Hbar, 6),
        "HER_ci": [round(m.HER_ci[0], 6), round(m.HER_ci[1], 6)],
        "bits_per_coordinate": bpc,
        "n_coordinates": int(dim),
        "n_bits": n_bits,
        "bit_hamming_rate": round(m.Hbar / n_bits, 9),
        # rho is typically 1e-5..1e-3, so keep significant digits, not decimals.
        **({"coordinate_change_rate": _sig(m.rho),
            "coordinate_change_rate_ci": [_sig(m.rho_ci[0]), _sig(m.rho_ci[1])]}
           if m.rho is not None else {}),
        "criterion_type": "exact",
        "bytes_of_reference_state": bors,
        "unit": "code",
    }}}


def _her(value) -> float | None:
    return value.HER if isinstance(value, ConditionMetrics) else her(value)


def _rho(value) -> float | None:
    if isinstance(value, ConditionMetrics):
        return value.rho
    semq = ((value or {}).get("detectors") or {}).get("semq") or {}
    return semq.get("coordinate_change_rate")


def core_ari(by_condition: dict) -> float:
    """Mean semq HER over the present comparable-core conditions.

    Values are ConditionMetrics, whose HER is unrounded, or condition results in
    the report shape, whose HER is rounded to 6 dp. build_report passes the
    former; a tool that only has a written report (capture_time_baseline.py
    merging a `time` cell) passes the latter, and its ARI can differ from the
    original in the sixth digit.
    """
    present = [_her(by_condition[c]) for c in CORE_CONDITIONS if c in by_condition]
    if not present or any(h is None for h in present):
        raise ValueError("no averaged conditions present — cannot compute ARI")
    return float(np.mean(present))


def core_ari_r(by_condition: dict) -> float | None:
    """ARI-R: 1 - mean rho over the present comparable-core conditions, or None
    when any of them lacks rho. Takes the same values as core_ari."""
    rhos = [_rho(by_condition[c]) for c in CORE_CONDITIONS if c in by_condition]
    if not rhos or any(r is None for r in rhos):
        return None
    return 1.0 - float(np.mean(rhos))


def build_report(
    *,
    agent_id: str,
    input_set: str,
    environment: dict,
    metrics_by_condition: dict[str, ConditionMetrics],
    codes_by_condition: dict[str, np.ndarray],
    agent_class: str = "self_hosted",
    input_content_hash: str | None = None,
    probe_calibration: str = "ARI-Canonical-v0.2",
    fingerprint: dict | None = None,
) -> dict:
    if not any(c in metrics_by_condition for c in AVERAGED_CONDITIONS):
        raise ValueError("no averaged conditions present — cannot compute ARI")

    # Conditions are born in the multi-detector shape
    # (spec/report-schema.json $defs/conditionResult); the flat pre-Matrix
    # shape exists only in ari/tools/migrate_multidetector.py's input.
    dim = (fingerprint or {}).get("dim")
    if dim is None:
        raise ValueError("fingerprint.dim is required to derive the semq "
                         "detector's bytes_of_reference_state")
    results = {}
    for cond, m in metrics_by_condition.items():
        results[cond] = semq_condition_entry(m, int(dim))

    audit = {cond: condition_digest(codes) for cond, codes in codes_by_condition.items()}

    report = {
        "agent_id": agent_id,
        "ari_version": ARI_VERSION,
        "probe_calibration": probe_calibration,
        "input_set": input_set,
        "agent_class": agent_class,
        "environment": environment,
        "results_per_condition": results,
        "ARI": round(core_ari(metrics_by_condition), 6),
        "audit_hashes": audit,
    }
    ari_r = core_ari_r(metrics_by_condition)
    if ari_r is not None:
        report["ARI_R"] = round(ari_r, 9)
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
