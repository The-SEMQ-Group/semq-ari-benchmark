# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""Capture one API agent's `time` baseline now; compare after more than 24 hours.

  python ari/tools/capture_time_baseline.py --mode capture --agent gemini --inputs ari-bench-v0.1.jsonl
  python ari/tools/capture_time_baseline.py --mode compare --agent gemini --inputs ari-bench-v0.1.jsonl \\
      [--report existing.json --out merged.json]

Baseline data lives outside the repo (~/ari_time_baseline/<slug>/): codes.npy + meta.json,
the same store and format as refresh_leaderboard.py. Self-hosted models use
selfhosted_conc_time.py. See ari/README.md, "Capture the time condition".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari import metrics, report                         # noqa: E402
from ari import time_evidence as te                     # noqa: E402
from ari.agents import DEFAULT_API_MODELS               # noqa: E402
from ari.inputs import InputSet, load_ari_bench         # noqa: E402
from ari.probe import load_probe                        # noqa: E402
from ari.verify_report import check_time_condition      # noqa: E402

API = DEFAULT_API_MODELS
STORE = Path.home() / "ari_time_baseline"


def build(agent: str, model: str | None, max_workers: int):
    cls, default = API[agent]
    return cls(model_id=model or default, max_workers=max_workers)


def merge_time_cell(rep: dict, cell: dict, digest: str, record: dict) -> dict:
    """`rep` with `cell` as its `time` result and ARI, ARI-R recomputed. Refuses a
    report for another agent, other inputs or another dimension."""
    fp = rep.get("encoder_fingerprint") or {}
    for field, theirs, ours in (("agent_id", rep.get("agent_id"), record["agent_id"]),
                                ("input_content_hash", rep.get("input_content_hash"),
                                 record["input_content_hash"]),
                                ("encoder_fingerprint.dim", fp.get("dim"), record["dim"])):
        if theirs != ours:
            raise te.TimeEvidenceError(f"--report {field} {theirs!r} is not the baseline's {ours!r}")
    merged = dict(rep)
    merged["results_per_condition"] = {**rep["results_per_condition"], "time": cell}
    merged["audit_hashes"] = {**rep.get("audit_hashes", {}), "time": digest}
    merged["ARI"] = round(report.core_ari(merged["results_per_condition"]), 6)
    # ARI-R moves with the new cell too; drop it rather than keep a stale value when a
    # core cell lacks rho.
    ari_r = report.core_ari_r(merged["results_per_condition"])
    merged.pop("ARI_R", None)
    if ari_r is not None:
        merged["ARI_R"] = round(ari_r, 9)
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capture / compare the `time` condition baseline.")
    ap.add_argument("--mode", required=True, choices=["capture", "compare"])
    ap.add_argument("--agent", required=True, choices=list(API))
    ap.add_argument("--model", default=None)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="directory holding codes.npy and meta.json (default: ~/ari_time_baseline/<slug>)")
    ap.add_argument("--report", type=Path, default=None,
                    help="compare: existing report for this agent to merge the time cell into")
    ap.add_argument("--out", type=Path, default=None, help="compare: where to write the merged report")
    args = ap.parse_args(argv)
    if args.report and not args.out:
        ap.error("--report requires --out")

    inputs = load_ari_bench(args.inputs)
    if args.limit:
        inputs = InputSet(inputs.name, inputs.ids[:args.limit], inputs.texts[:args.limit])

    agent = build(args.agent, args.model, args.max_workers)
    slug = f"{args.agent}_{agent.agent_id.replace('/', '_')}"
    d = args.baseline_dir or STORE / slug
    codes_p, meta_p = d / "codes.npy", d / "meta.json"
    agent_id = f"{args.agent}/{agent.agent_id}"
    env = te.hosted_environment(agent)
    try:
        # Refuse a bad pin or an unusable baseline before encoding.
        te.check_identities(agent_id, "api", env)
        if args.mode == "compare":
            if not codes_p.exists() or not meta_p.exists():
                print(f"no baseline for {slug} at {d}")
                return 1
            meta = json.loads(meta_p.read_text())
            te.check_baseline(meta, {"agent_id": agent_id,
                                     "input_content_hash": inputs.content_hash, **env})
    except te.TimeEvidenceError as e:
        print(f"ERROR: {e}")
        return 1

    started_at = te.utc_now()
    vecs = agent.encode(inputs.texts)
    record = te.baseline_record(agent_id=agent_id, agent_class="api",
                                input_content_hash=inputs.content_hash, dim=vecs.shape[1],
                                started_at=started_at, finished_at=te.utc_now(), environment=env)
    codes = load_probe(vecs).encode(vecs)

    if args.mode == "capture":
        d.mkdir(parents=True, exist_ok=True)
        np.save(codes_p, codes)
        meta_p.write_text(json.dumps({**record, "n": len(inputs)}, indent=2) + "\n")
        print(f"captured {slug}: n={len(inputs)}, "
              f"{record['started_at']} -> {record['finished_at']} -> {d}")
        return 0

    try:
        evidence = te.time_evidence(meta, record)
        m = metrics.aggregate(np.load(codes_p), codes, dim=record["dim"])
        cell = {**report.semq_condition_entry(m, record["dim"]), "time_evidence": evidence}
        digest = report.condition_digest(codes)
        if args.report:
            payload = merge_time_cell(json.loads(args.report.read_text()), cell, digest, record)
            violations = check_time_condition(payload)
            if violations:
                raise te.TimeEvidenceError("; ".join(violations))
            where = args.out
        else:
            payload = {"results_per_condition": {"time": cell}, "audit_hashes": {"time": digest}}
            where = d / "time_cell.json"
    except te.TimeEvidenceError as e:
        print(f"ERROR: {e}\nnothing written")
        return 1
    print(f"{slug}: time HER = {m.HER:.4f}  95% CI {m.HER_ci}  "
          f"(gap {evidence['gap_hours']} h, n={len(inputs)})  digest={digest[:12]}")
    report.write_report(where, payload)
    print(f"wrote {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
