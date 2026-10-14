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
"""
from __future__ import annotations

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


def time_evidence(baseline: dict, comparison: dict) -> dict:
    """The `time_evidence` block for a comparison against `baseline`, both
    from baseline_record. The gap runs from the baseline's end to the
    comparison's start and is rounded up to 0.01 h, so the recorded value
    exceeds 24 exactly when the timestamps do."""
    check_baseline(baseline, comparison)
    end = datetime.strptime(baseline["finished_at"], TS_FMT)
    start = datetime.strptime(comparison["started_at"], TS_FMT)
    seconds = int((start - end).total_seconds())
    if seconds <= MIN_GAP_HOURS * 3600:
        raise TimeEvidenceError(f"only {seconds / 3600:.2f} h since the baseline finished "
                                f"at {baseline['finished_at']}; the time condition needs "
                                f"more than {MIN_GAP_HOURS:g} h")
    return {"baseline_started_at": baseline["started_at"],
            "baseline_finished_at": baseline["finished_at"],
            "comparison_started_at": comparison["started_at"],
            "comparison_finished_at": comparison["finished_at"],
            "gap_hours": -(-seconds // 36) / 100,
            **{f: baseline[f] for f in EVIDENCE_FIELDS},
            "baseline_source_commit": baseline["source_commit"],
            "source_commit": comparison["source_commit"]}
