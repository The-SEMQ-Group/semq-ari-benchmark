# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""Capture the `time` condition baseline now; compare after more than 24 hours.

  python ari/tools/capture_time_baseline.py --mode capture|compare --agent <api|st> ...

Procedure, arguments and the success check: ari/README.md, "Capture the time condition".
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari.agents import DEFAULT_API_MODELS, SentenceTransformerAgent   # noqa: E402
from ari.hub import UNKNOWN, hub_revision, is_pinned                  # noqa: E402
from ari.inputs import InputSet, load_ari_bench                       # noqa: E402
from ari.probe import load_probe, fixed_scale_codes                   # noqa: E402
from ari import metrics, report                                       # noqa: E402
from ari.verify_report import check_time_condition                    # noqa: E402

API = DEFAULT_API_MODELS
STORE = Path.home() / "ari_time_baseline"
PROBE_CALIBRATION = "ARI-Canonical-v0.1"
PROVIDER_INTERNAL = "provider-internal"
# Same pins as run_report.py, set before torch is imported so both phases encode alike.
PIN = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
# Baseline settings the comparison must reproduce. A difference is a `lib`, `prec` or
# `mach` change, not a `time` measurement.
HELD_FIXED = ("model_revision", "tokenizer_revision", "precision", "sdk_version")
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TS_FMT)


def utc_from_unix(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime(TS_FMT)


def source_commit() -> str:
    r = subprocess.run(["git", "-C", str(HARNESS), "rev-parse", "HEAD"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        return r.stdout.strip()
    return os.environ.get("GIT_COMMIT") or UNKNOWN


def build(agent_name: str, model: str | None, max_workers: int, revision: str | None):
    if agent_name == "st":
        if not model:
            raise SystemExit("--agent st requires --model")
        return SentenceTransformerAgent(model, device="cpu", revision=revision), "self_hosted"
    cls, default = API[agent_name]
    return cls(model_id=model or default, max_workers=max_workers), "api"


def environment_record(agent, agent_class: str) -> dict:
    """The settings the time condition holds fixed, read from the loaded agent."""
    from ari.semq_compat import sdk_version

    if agent_class == "api":
        rev = getattr(agent, "snapshot", None) or PROVIDER_INTERNAL
        return {"model_revision": str(rev), "tokenizer_revision": PROVIDER_INTERNAL,
                "precision": PROVIDER_INTERNAL, "hardware": PROVIDER_INTERNAL,
                "sdk_version": sdk_version(), "source_commit": source_commit()}
    rev = agent.revision or hub_revision(agent.agent_id)
    if not is_pinned(rev):
        raise SystemExit(f"cannot resolve an immutable revision for {agent.agent_id}; "
                         f"pass --revision <hub commit>")
    # A sentence-transformers repository ships its tokenizer, so both pins are one commit.
    return {"model_revision": rev, "tokenizer_revision": rev,
            "precision": agent.precision(),
            "hardware": f"{platform.machine()} {platform.platform()}",
            "sdk_version": sdk_version(), "source_commit": source_commit()}


def time_evidence(meta: dict, started_at: str, finished_at: str, current: dict) -> dict:
    """The `time_evidence` block from the baseline record and this comparison."""
    deviations = []
    if "finished_at" in meta:
        b_start, b_end = meta["started_at"], meta["finished_at"]
    else:
        b_start = b_end = utc_from_unix(meta["unix_ts"])
        deviations.append("baseline recorded one timestamp after its encode; "
                          "baseline_started_at is set equal to baseline_finished_at")
    gap = (datetime.strptime(started_at, TS_FMT) - datetime.strptime(b_end, TS_FMT)
           ).total_seconds() / 3600.0
    ev = {"baseline_started_at": b_start, "baseline_finished_at": b_end,
          "comparison_started_at": started_at, "comparison_finished_at": finished_at,
          "gap_hours": round(gap, 2),
          "probe_calibration": meta.get("probe_calibration", PROBE_CALIBRATION),
          "calibration_scale": float(meta["scale_s"]),
          "input_content_hash": meta["content_hash"]}
    for f in HELD_FIXED + ("hardware", "source_commit"):
        if f in meta:
            ev[f] = meta[f]
        else:
            ev[f] = current[f]
            deviations.append(f"baseline did not record {f}; the comparison's value is shown")
    for f in HELD_FIXED:
        if f in meta and meta[f] != current[f]:
            raise SystemExit(f"{f} differs from the baseline ({meta[f]!r} -> {current[f]!r}); "
                             f"that is not a time comparison")
    if "source_commit" in meta and meta["source_commit"] != current["source_commit"]:
        deviations.append(f"comparison ran at harness commit {current['source_commit']}")
    if deviations:
        ev["deviations"] = deviations
    return ev


def merge_time_cell(rep: dict, cell: dict, digest: str) -> dict:
    merged = dict(rep)
    merged["results_per_condition"] = {**rep["results_per_condition"], "time": cell}
    merged["audit_hashes"] = {**rep.get("audit_hashes", {}), "time": digest}
    merged["ARI"] = round(report.core_ari(merged["results_per_condition"]), 6)
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capture / compare the `time` condition baseline.")
    ap.add_argument("--mode", required=True, choices=["capture", "compare"])
    ap.add_argument("--agent", required=True, choices=list(API) + ["st"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--revision", default=None, help="st: Hub commit to load (default: resolved)")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="directory holding codes.npy and meta.json (default: ~/ari_time_baseline/<slug>)")
    ap.add_argument("--report", type=Path, default=None,
                    help="compare: existing report to merge the time cell into")
    ap.add_argument("--out", type=Path, default=None, help="compare: where to write the merged report")
    args = ap.parse_args(argv)
    if args.report and not args.out:
        ap.error("--report requires --out")

    if args.agent == "st":
        os.environ.update(PIN)

    inputs = load_ari_bench(args.inputs)
    if args.limit:
        inputs = InputSet(inputs.name, inputs.ids[:args.limit], inputs.texts[:args.limit])

    agent, agent_class = build(args.agent, args.model, args.max_workers, args.revision)
    slug = f"{args.agent}_{agent.agent_id.replace('/', '_')}"
    d = args.baseline_dir or STORE / slug
    codes_p, meta_p = d / "codes.npy", d / "meta.json"

    if args.mode == "capture":
        started_at = utc_now()
        vecs = agent.encode(inputs.texts)
        finished_at = utc_now()
        current = environment_record(agent, agent_class)
        probe = load_probe(vecs)
        codes = probe.encode(vecs)
        d.mkdir(parents=True, exist_ok=True)
        np.save(codes_p, codes)
        meta = {"agent_id": agent.agent_id, "agent_class": agent_class,
                "started_at": started_at, "finished_at": finished_at,
                "unix_ts": time.time(), "n": len(inputs),
                "content_hash": inputs.content_hash, "scale_s": float(probe.s),
                "dim": int(vecs.shape[1]), "probe_calibration": PROBE_CALIBRATION, **current}
        meta_p.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"captured {slug}: n={meta['n']}, s={meta['scale_s']:.5f}, "
              f"{started_at} -> {finished_at}, revision {current['model_revision']} -> {d}")
        return 0

    if not codes_p.exists():
        print(f"no baseline for {slug} at {d}"); return 1
    meta = json.loads(meta_p.read_text())
    if meta["content_hash"] != inputs.content_hash:
        print("ERROR: inputs content hash differs from the captured baseline"); return 1
    base_codes = np.load(codes_p)

    started_at = utc_now()
    vecs = agent.encode(inputs.texts)
    finished_at = utc_now()
    current = environment_record(agent, agent_class)
    # the stored scale, not a recalibration, so codes are comparable to the baseline
    time_codes = fixed_scale_codes(vecs, meta["scale_s"], meta["dim"])

    m = metrics.aggregate(base_codes, time_codes)
    cell = report.semq_condition_entry(m, int(meta["dim"]))
    cell["time_evidence"] = time_evidence(meta, started_at, finished_at, current)
    digest = report.condition_digest(time_codes)

    if args.report:
        rep = json.loads(args.report.read_text())
        payload = checked = merge_time_cell(rep, cell, digest)
        where = args.out
    else:
        payload = {"results_per_condition": {"time": cell}, "audit_hashes": {"time": digest}}
        checked = {"results_per_condition": {"time": cell},
                   "input_content_hash": inputs.content_hash,
                   "probe_calibration": cell["time_evidence"]["probe_calibration"],
                   "environment": {"precision": current["precision"]}}
        where = d / "time_cell.json"

    ev = cell["time_evidence"]
    print(f"{slug}: time HER = {m.HER:.4f}  95% CI {m.HER_ci}  H̄={m.Hbar:.2f}  "
          f"(gap {ev['gap_hours']} h, n={len(inputs)})  digest={digest[:12]}")
    for dev in ev.get("deviations", []):
        print(f"  deviation: {dev}")
    violations = check_time_condition(checked)
    for v in violations:
        print(f"  [FAIL] {v}")
    if violations:
        print(f"nothing written: {len(violations)} violation(s)")
        return 1
    report.write_report(where, payload)
    print(f"wrote {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
