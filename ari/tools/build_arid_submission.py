"""Build a decoding-layer (ARI-D) submission from dry-run evidence.

    python ari/tools/build_arid_submission.py --subject openai \\
        --agent-id openai/gpt-4o-mini --out leaderboard/submissions/

The formal path (spec/arid-bench-v0.1.md §7, §9): a submission carries the
three core conditions plus the `same` control, each with the sequence-level
detectors, and is only buildable when the line is COMPLETE — the headline is
never computed over a subset, so an incomplete line stays a draft in the
experiment, not a submission. Audit hashes are the digests of the retained
transcripts each condition was computed from, so a verifier holding the
transcripts can rebuild every number.

The report then goes through the same gate as every Representation row:
score.py (schema + decoding invariants), attest_submissions.py (KMS, bound
to the ARI-D prompt set), verify_report.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DRY = REPO / "experiments" / "arid-dry-run" / "results"
ARID_HASH = "af5b1a2bb35b9fa5f8d4e8f05b85a98f495a3edeb93e95da9c9ced6d783e0d6f"

# transcript names per (subject, condition), and the subject's time-pair artifact
SUBJECTS = {
    "openai": {
        "agent_class": "vendor",
        "conditions": {"same": "openai_same.jsonl.gz", "proc": "openai_proc.jsonl.gz",
                       "conc": "openai_conc_b64.jsonl.gz"},
        "time_pair": "time_pair.json",
        "time_transcript": "openai_same_t2.jsonl.gz",
    },
    "together": {
        "agent_class": "rehosted",
        "conditions": {"same": "together_same.jsonl.gz", "proc": "together_proc.jsonl.gz",
                       "conc": "together_conc_b64.jsonl.gz"},
        "time_pair": "time_pair_together.json",
        "time_transcript": "together_same_t2.jsonl.gz",
    },
}


def _generic(value, ci, criterion, unit):
    out = {"value": round(float(value), 6), "criterion_type": criterion,
           "bytes_of_reference_state": 0, "unit": unit}
    if ci is not None:
        out["value_ci"] = [round(float(ci[0]), 6), round(float(ci[1]), 6)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="ARI-D submission from dry-run evidence.")
    ap.add_argument("--subject", required=True, choices=list(SUBJECTS))
    ap.add_argument("--agent-id", required=True,
                    help="e.g. openai/gpt-4o-mini (provider/model, as the board names rows)")
    ap.add_argument("--measured-at", required=True, help="YYYY-MM-DD of the measurement")
    ap.add_argument("--out", type=Path, default=REPO / "leaderboard" / "submissions")
    args = ap.parse_args()

    cfg = SUBJECTS[args.subject]
    analysis = json.loads((DRY / "analysis.json").read_text())
    by_transcript = {c["transcript"]: c for c in analysis["conditions"]}
    manifest = json.loads((DRY / "manifest.json").read_text())
    digests = {r["transcript"]: r["sha256"] for r in manifest["runs"]}

    time_path = DRY / cfg["time_pair"]
    if not time_path.exists():
        sys.exit(f"{cfg['time_pair']} does not exist yet — the line is incomplete, "
                 "and an incomplete line is a draft, not a submission (spec §7).")
    time_pair = json.loads(time_path.read_text())

    rpc = {}
    model = None
    for cond, transcript in cfg["conditions"].items():
        cell = by_transcript.get(transcript)
        if cell is None:
            sys.exit(f"analysis.json has no entry for {transcript}")
        model = model or cell["model"]
        det = {
            "exact_generation": _generic(cell["exact_generation"]["mean"],
                                         cell["exact_generation"]["ci"],
                                         "exact", "completion_pair"),
            "first_divergence": _generic(cell["first_divergence"]["mean"],
                                         cell["first_divergence"]["ci"],
                                         "interval", "byte_position"),
        }
        if "topk_logprob_overlap" in cell:
            det["topk_logprob_overlap"] = _generic(cell["topk_logprob_overlap"]["mean"],
                                                   cell["topk_logprob_overlap"]["ci"],
                                                   "interval", "token_set")
        rpc[cond] = {"detectors": det}

    rpc["time"] = {"detectors": {
        "exact_generation": _generic(time_pair["exact_generation"]["mean"],
                                     time_pair["exact_generation"]["ci"],
                                     "exact", "completion_pair"),
        "first_divergence": _generic(time_pair["first_divergence"]["mean"],
                                     time_pair["first_divergence"]["ci"],
                                     "interval", "byte_position"),
    }}

    ari = round(sum(rpc[c]["detectors"]["exact_generation"]["value"]
                    for c in ("proc", "conc", "time")) / 3, 6)

    audit = {cond: digests[t] for cond, t in cfg["conditions"].items()}
    audit["time"] = digests[cfg["time_transcript"]]

    same_cell = by_transcript[cfg["conditions"]["same"]]
    report = {
        "agent_id": args.agent_id,
        "ari_version": "0.1",
        "layer": "decoding",
        "probe_calibration": "sequence-level-v0.1",
        "input_set": "ARI-D-Bench-v0.1",
        "input_content_hash": ARID_HASH,
        "agent_class": cfg["agent_class"],
        "measured_at": args.measured_at,
        "environment": {
            "blas": "provider-internal",
            "threads": 0,
            "hardware": "provider-internal",
            "precision": "provider-internal",
            "library_versions": {
                "protocol": "arid-bench-v0.1",
                "model_requested": model,
                "fingerprints_observed": str(len(same_cell.get("system_fingerprints", []))),
            },
        },
        "results_per_condition": rpc,
        "ARI": ari,
        "audit_hashes": audit,
    }

    out = args.out / (args.agent_id.replace("/", "_") + ".decoding.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"report -> {out}  (ARI-D={ari}, class={cfg['agent_class']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
