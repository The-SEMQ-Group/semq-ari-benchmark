# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The `time` condition's evidence: the baseline record and the `time_evidence` block.

A `time` cell compares two captures more than 24 hours apart with everything
else held fixed (spec/condition-set.md, "Time evidence"). The capture tools
(ari/tools/refresh_leaderboard.py for the API panel, selfhosted_conc_time.py
for a local model, capture_time_baseline.py for one API agent) store a
baseline_record next to the baseline codes, build a second record around the
comparison encode, and call time_evidence(). It refuses, rather than notes, a
baseline that lacks a held-fixed field or differs in one.

The decoding layer (ARI-D) has its own form: experiments/arid-dry-run/run_dry.py
records each run in its manifest, and build_arid_submission.py calls
decoding_time_evidence() on two `same` runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .verify_report import MIN_GAP_HOURS, PROVIDER_INTERNAL, HOSTED_CLASSES, identity_problem

PROBE_CALIBRATION = "ARI-Canonical-v0.2"
REPO = Path(__file__).resolve().parents[1]
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
# Fields both captures must share. A difference is another condition (`lib`,
# `prec`, `mach`) or another measurement, not `time`. The harness commit is
# recorded for each capture but not held fixed: in an active repository it
# changes between any two weekly runs.
HELD_FIXED = ("agent_id", "agent_class", "input_content_hash", "dim", "probe_calibration",
              "model_revision", "tokenizer_revision", "precision", "hardware",
              "sdk_version")
# The held-fixed fields the time_evidence block carries (dim is checked
# against encoder_fingerprint instead, agent_class against the report's).
EVIDENCE_FIELDS = ("agent_id", "input_content_hash", "probe_calibration", "model_revision",
                   "tokenizer_revision", "precision", "hardware", "sdk_version")
_IDENTITIES = ("agent_id", "model_revision", "tokenizer_revision", "precision", "hardware",
               "sdk_version", "source_commit")


class TimeEvidenceError(ValueError):
    """The captures cannot support a `time` cell."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TS_FMT)


def source_commit() -> str:
    """The harness checkout's commit, or $GIT_COMMIT outside a checkout."""
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return os.environ.get("GIT_COMMIT", "unknown")


def cpu_model() -> str:
    """The CPU's marketing name: /proc/cpuinfo on Linux, sysctl on macOS."""
    system = platform.system()
    try:
        if system == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
    except OSError:
        pass
    return platform.processor()


def gpu_names() -> list[str]:
    """CUDA device names when torch is installed and sees a GPU, else []."""
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def hardware_id() -> str:
    """Architecture, CPU model and GPU names. Not the hostname and not the OS
    version: two machines of one type are the same hardware, and an OS update
    is not a hardware change."""
    return " | ".join(p for p in (platform.machine(), cpu_model(), ", ".join(gpu_names())) if p)


def _sdk_version() -> str:
    from .semq_compat import sdk_version
    return sdk_version()


def hosted_environment(agent) -> dict:
    """Held-fixed settings of an API agent. The model id is the revision; a
    snapshot id is appended only when the agent has a real one."""
    snapshot = getattr(agent, "snapshot", None)
    rev = agent.agent_id
    if snapshot and identity_problem("model_revision", str(snapshot), True) is None:
        rev = f"{rev}@{snapshot}"
    return {"model_revision": rev, "tokenizer_revision": PROVIDER_INTERNAL,
            "precision": PROVIDER_INTERNAL, "hardware": PROVIDER_INTERNAL,
            "sdk_version": _sdk_version(), "source_commit": source_commit()}


def self_hosted_environment(model, revision: str) -> dict:
    """Held-fixed settings of a loaded sentence-transformers model at Hub commit
    `revision`. Its repository ships the tokenizer, so both pins are one commit."""
    dtype = str(next(model.parameters()).dtype)
    precision = {"torch.float32": "fp32", "torch.float16": "fp16",
                 "torch.bfloat16": "bf16"}.get(dtype, "mixed")
    return {"model_revision": revision, "tokenizer_revision": revision,
            "precision": precision, "hardware": hardware_id(),
            "sdk_version": _sdk_version(), "source_commit": source_commit(),
            "os": platform.platform()}


def check_identities(agent_id: str, agent_class: str, environment: dict) -> None:
    """Raise when an identity does not pin what runs. Capture tools call this
    before encoding, so a bad pin fails now and not after the 24-hour wait."""
    rec = {"agent_id": agent_id, **environment}
    hosted = agent_class in HOSTED_CLASSES
    problems = [f"{f}: {p}" for f in _IDENTITIES
                if (p := identity_problem(f, rec.get(f), hosted))]
    if problems:
        raise TimeEvidenceError("cannot record this capture: " + "; ".join(problems))


def baseline_record(*, agent_id: str, agent_class: str, input_content_hash: str, dim: int,
                    started_at: str, finished_at: str, environment: dict) -> dict:
    """The record of one capture: the held-fixed fields and its UTC start and end."""
    check_identities(agent_id, agent_class, environment)
    return {"agent_id": agent_id, "agent_class": agent_class,
            "input_content_hash": input_content_hash, "dim": int(dim),
            "probe_calibration": PROBE_CALIBRATION,
            "started_at": started_at, "finished_at": finished_at, **environment}


def check_baseline(baseline: dict, current: dict) -> None:
    """Raise when `baseline` lacks a held-fixed field, or differs from `current`
    in one that `current` records."""
    missing = [f for f in HELD_FIXED + ("source_commit",) if f not in baseline]
    if missing:
        raise TimeEvidenceError(f"the baseline does not record {', '.join(missing)}; "
                                f"capture a new baseline")
    differ = [f"{f} {baseline[f]!r} -> {current[f]!r}" for f in HELD_FIXED
              if f in current and current[f] != baseline[f]]
    if differ:
        raise TimeEvidenceError("not a time comparison; differs from the baseline: "
                                + "; ".join(differ))


def _gap_hours(baseline_finished_at: str, comparison_started_at: str) -> float:
    """Hours from the baseline's end to the comparison's start, rounded up to
    0.01 h, so the recorded value exceeds 24 exactly when the timestamps do."""
    end = datetime.strptime(baseline_finished_at, TS_FMT)
    start = datetime.strptime(comparison_started_at, TS_FMT)
    seconds = int((start - end).total_seconds())
    if seconds <= MIN_GAP_HOURS * 3600:
        raise TimeEvidenceError(f"only {seconds / 3600:.2f} h since the baseline finished "
                                f"at {baseline_finished_at}; the time condition needs "
                                f"more than {MIN_GAP_HOURS:g} h")
    return -(-seconds // 36) / 100


def time_evidence(baseline: dict, comparison: dict) -> dict:
    """The `time_evidence` block for a comparison against `baseline`, both
    from baseline_record. The gap runs from the baseline's end to the
    comparison's start and is rounded up to 0.01 h, so the recorded value
    exceeds 24 exactly when the timestamps do."""
    check_baseline(baseline, comparison)
    return {"baseline_started_at": baseline["started_at"],
            "baseline_finished_at": baseline["finished_at"],
            "comparison_started_at": comparison["started_at"],
            "comparison_finished_at": comparison["finished_at"],
            "gap_hours": _gap_hours(baseline["finished_at"], comparison["started_at"]),
            **{f: baseline[f] for f in EVIDENCE_FIELDS},
            "baseline_source_commit": baseline["source_commit"],
            "source_commit": comparison["source_commit"]}


# --------------------------------------------------------------------- decoding layer
# ARI-D (spec/arid-bench-v0.1.md). A capture is one run entry in
# experiments/arid-dry-run/results/manifest.json, written by run_dry.py.
DECODING_PROTOCOL = "arid-bench-v0.1"
# Held fixed between the two runs. The request is held by digest: the manifest
# keeps the full request template, and the digest is what the block carries.
DECODING_HELD_FIXED = ("subject", "class", "model", "condition", "scope",
                       "prompt_content_hash", "protocol", "request_sha256")
# Recorded per run, not held fixed.
DECODING_RECORDED = ("started_at", "finished_at", "source_commit", "model_versions")


def request_digest(request: dict) -> str:
    """SHA-256 of the canonical JSON of a request template."""
    blob = json.dumps(request, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def decoding_time_evidence(baseline: dict, comparison: dict, *, agent_id: str) -> dict:
    """The decoding-layer `time_evidence` block from two manifest run entries.
    Refuses a run that lacks a field, two runs that differ in a held-fixed
    field, a bad harness commit, or a gap of 24 h or less."""
    for name, run in (("baseline", baseline), ("comparison", comparison)):
        missing = [f for f in DECODING_HELD_FIXED + DECODING_RECORDED if f not in run]
        if missing:
            raise TimeEvidenceError(f"{name} run {run.get('transcript')!r} does not record "
                                    f"{', '.join(missing)}; recapture it with run_dry.py")
        if run["request_sha256"] != request_digest(run.get("request")):
            raise TimeEvidenceError(f"{name} run: request_sha256 does not match its request")
        problem = identity_problem("source_commit", run["source_commit"], True)
        if problem:
            raise TimeEvidenceError(f"{name} run source_commit: {problem}")
    differ = [f"{f} {baseline[f]!r} -> {comparison[f]!r}" for f in DECODING_HELD_FIXED
              if baseline[f] != comparison[f]]
    if differ:
        raise TimeEvidenceError("not a time comparison; the runs differ: " + "; ".join(differ))
    return {"baseline_started_at": baseline["started_at"],
            "baseline_finished_at": baseline["finished_at"],
            "comparison_started_at": comparison["started_at"],
            "comparison_finished_at": comparison["finished_at"],
            "gap_hours": _gap_hours(baseline["finished_at"], comparison["started_at"]),
            "agent_id": agent_id,
            "input_content_hash": baseline["prompt_content_hash"],
            "protocol": baseline["protocol"],
            "model": baseline["model"],
            "request_sha256": baseline["request_sha256"],
            "precision": PROVIDER_INTERNAL,
            "hardware": PROVIDER_INTERNAL,
            "baseline_model_versions": baseline["model_versions"],
            "model_versions": comparison["model_versions"],
            "baseline_source_commit": baseline["source_commit"],
            "source_commit": comparison["source_commit"]}
