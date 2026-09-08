"""Turn fresh ARI-E rollouts into a leaderboard report.

The metric is not reimplemented here: `ari.harness.harness_report` already
computes self-consistency, cross-harness agreement and the effect
(Δ = mean self-consistency − cross agreement = S − X = D_cross − D_self), and
it is tested. This module only:

  1. reads the trajectories the runner produced (one JSON line per rollout,
     the format `ari.harness.load_trajectories` reads),
  2. runs the report,
  3. maps it onto the shared ARI report envelope (spec §9): `same` and
     `harness` conditions, each an `outcome_agreement` detector, with the
     headline `ARI = same.value − harness.value = Δ`.

It does not sign. Signing is the same standalone step every layer uses
(ari.attest / ARI_SIGNING_KEY), run against the emitted report on the node
that produced it.

    python emit_report.py --trajectories runs/trajectories.jsonl \
        --agent-id qwen/qwen3-coder-30b --endpoint http://... \
        --sweagent-sha <sha> --openhands-sha <sha> \
        --input-content-hash d44f4975...a91b26 --k 5 --out runs/report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ari.harness import harness_report, load_trajectories  # noqa: E402

INPUT_SET = "ARI-E-Bench-v0.1"


def build_report(trajectories_path: Path, *, agent_id: str, endpoint: str,
                 harness_shas: dict[str, str], input_content_hash: str,
                 k: int, measured_at: str, n_resamples: int = 2000) -> dict:
    runs = load_trajectories(trajectories_path)
    rep = harness_report(runs, n_resamples=n_resamples)

    harnesses = sorted({t.harness for t in runs})
    if len(harnesses) != 2:
        raise SystemExit(f"v0.1 expects exactly two harnesses, got {harnesses}")
    label = f"{harnesses[0]} vs {harnesses[1]}"
    if label not in rep.harness_effect:
        raise SystemExit(f"no harness effect for {label}: {rep.notes}")

    # same = mean self-consistency of the two harnesses (S = 1 − D_self);
    # harness = cross-harness agreement (X = 1 − D_cross). The report stores the
    # agreement form; the headline ARI is their difference.
    s_each = [rep.self_consistency[h].outcome_agreement for h in harnesses]
    s_value = sum(s_each) / len(s_each)
    x_value = rep.cross[label].outcome_agreement
    x_ci = list(rep.cross[label].outcome_agreement_ci)
    # The self CI is per-harness; report the mean point with each harness's own
    # CI folded to the wider bound so `same` is not shown tighter than its parts.
    s_cis = [rep.self_consistency[h].outcome_agreement_ci for h in harnesses]
    s_ci = [min(c[0] for c in s_cis), max(c[1] for c in s_cis)]
    effect = rep.harness_effect[label]
    effect_ci = list(rep.harness_effect_ci[label])

    def pass_rate(h: str) -> float:
        outs = [t.outcome for t in runs if t.harness == h]
        return round(sum(outs) / len(outs), 4) if outs else 0.0

    cases = len({t.case_id for t in runs})

    def detector(value: float, ci: list[float]) -> dict:
        return {"outcome_agreement": {
            "value": round(value, 6), "value_ci": [round(ci[0], 6), round(ci[1], 6)],
            "criterion_type": "interval", "bytes_of_reference_state": 0,
            "unit": "instance_pair"}}

    audit = {
        "same": hashlib.sha256(json.dumps(s_each, sort_keys=True).encode()).hexdigest(),
        "harness": hashlib.sha256(f"{label}:{x_value}".encode()).hexdigest(),
    }

    return {
        "agent_id": agent_id,
        "ari_version": "0.1",
        "layer": "environment",
        "probe_calibration": "outcome-level-v0.1",
        "input_set": INPUT_SET,
        "input_content_hash": input_content_hash,
        "environment": {
            "endpoint": endpoint,
            "harnesses": {h: harness_shas.get(h, h) for h in harnesses},
            "pass_rate": {h: pass_rate(h) for h in harnesses},
            "cases": cases,
            "k": k,
        },
        "results_per_condition": {
            "same": {"detectors": detector(s_value, s_ci)},
            "harness": {"detectors": detector(x_value, x_ci)},
        },
        "ARI": round(effect, 6),
        "ARI_ci": [round(effect_ci[0], 6), round(effect_ci[1], 6)],
        "detected": rep.detects(label),
        "audit_hashes": audit,
        "measured_at": measured_at,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Emit an ARI-E report from rollouts.")
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--sweagent-sha", default="sweagent")
    ap.add_argument("--openhands-sha", default="openhands")
    ap.add_argument("--input-content-hash", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--measured-at", required=True, help="ISO-8601 UTC (stamp from caller)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    report = build_report(
        args.trajectories, agent_id=args.agent_id, endpoint=args.endpoint,
        harness_shas={"sweagent": args.sweagent_sha, "openhands": args.openhands_sha},
        input_content_hash=args.input_content_hash, k=args.k,
        measured_at=args.measured_at)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}  ARI(Δ)={report['ARI']:+.4f} "
          f"CI={report['ARI_ci']}  detected={report['detected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
