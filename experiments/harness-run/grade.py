"""Grade agent patches with SWE-bench Verified's held-out tests.

The verdict a rollout carries (`resolved in {0,1}`) is the outcome ARI-E measures.
It must come from a checker the agent never saw, so grading runs the instance's own
FAIL_TO_PASS / PASS_TO_PASS suite via the official swebench harness — not an LLM judge.

Two modes:

  # de-risk: grade an instance's OWN gold patch -> must resolve
  python grade.py --gold-check psf__requests-2317

  # real: grade a predictions file the runner produced
  python grade.py --predictions runs/preds.jsonl --run-id smoke

Predictions are JSON Lines, one per rollout:
  {"rollout_id": "...", "instance_id": "...", "model_patch": "<unified diff>"}

Output: JSON Lines with the verdict appended: {..., "resolved": 0|1}.

Note on pinning: swebench loads the dataset (and thus the tests) by name at its current
revision. We froze against c104f840...; that revision is recorded in the output for
provenance. On arm64 the harness builds x86 images under emulation (slow); the block's
Linux x86 node runs them natively.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "SWE-bench/SWE-bench_Verified"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"


def load_instance(instance_id: str) -> dict:
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="test", revision=DATASET_REVISION)
    for r in ds:
        if r["instance_id"] == instance_id:
            return dict(r)
    raise SystemExit(f"instance {instance_id} not in {DATASET}")


def run_swebench(predictions: list[dict], run_id: str, *, workers: int,
                 timeout: int, report_dir: Path) -> dict[str, bool]:
    """Grade a list of {instance_id, model_name_or_path, model_patch}. Returns
    {instance_id: resolved}. One prediction per instance_id (swebench keys on it)."""
    report_dir.mkdir(parents=True, exist_ok=True)
    preds_path = report_dir / f"preds_{run_id}.json"
    preds_path.write_text(json.dumps(predictions))
    instance_ids = [p["instance_id"] for p in predictions]

    from swebench.harness.run_evaluation import main as run_eval
    run_eval(
        dataset_name=DATASET, split="test", instance_ids=instance_ids,
        predictions_path=str(preds_path), max_workers=workers, open_file_limit=4096,
        run_id=run_id, timeout=timeout, rewrite_reports=False, modal=False,
        report_dir=str(report_dir),
    )

    # swebench writes <model_name_or_path>.<run_id>.json with resolved_ids.
    resolved: dict[str, bool] = {i: False for i in instance_ids}
    for rep in report_dir.glob(f"*.{run_id}.json"):
        data = json.loads(rep.read_text())
        for i in data.get("resolved_ids", []):
            resolved[i] = True
    return resolved


def main() -> int:
    ap = argparse.ArgumentParser(description="Grade patches with SWE-bench tests.")
    ap.add_argument("--gold-check", metavar="INSTANCE_ID",
                    help="grade an instance's own gold patch (must resolve)")
    ap.add_argument("--predictions", type=Path, help="JSONL of rollout patches")
    ap.add_argument("--run-id", default="smoke")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--report-dir", type=Path, default=HERE / "runs" / "grading")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.gold_check:
        inst = load_instance(args.gold_check)
        preds = [{"instance_id": args.gold_check, "model_name_or_path": "gold",
                  "model_patch": inst["patch"]}]
        t0 = time.time()
        resolved = run_swebench(preds, args.run_id, workers=args.workers,
                                timeout=args.timeout, report_dir=args.report_dir)
        ok = resolved.get(args.gold_check, False)
        print(f"\ngold-check {args.gold_check}: resolved={ok}  ({time.time()-t0:.0f}s)")
        print("GRADING OK — the harness resolves a known-good patch"
              if ok else "GRADING FAILED — gold patch did not resolve (env/harness issue)")
        return 0 if ok else 1

    if not args.predictions:
        ap.error("pass --gold-check or --predictions")

    rollouts = [json.loads(l) for l in args.predictions.read_text().splitlines() if l.strip()]
    # swebench grades one patch per instance_id, but a rollout set has k patches per
    # instance. Grade each rollout under a unique run_id-scoped model tag so verdicts
    # do not collide, by keying predictions on rollout_id via model_name_or_path.
    verdicts = []
    for ro in rollouts:
        preds = [{"instance_id": ro["instance_id"],
                  "model_name_or_path": ro["rollout_id"],
                  "model_patch": ro.get("model_patch", "")}]
        rid = f"{args.run_id}_{ro['rollout_id']}"
        resolved = run_swebench(preds, rid, workers=1, timeout=args.timeout,
                                report_dir=args.report_dir)
        verdicts.append({**ro, "resolved": int(bool(resolved.get(ro["instance_id"], False)))})

    out = args.out or args.predictions.with_suffix(".graded.jsonl")
    out.write_text("\n".join(json.dumps(v) for v in verdicts) + "\n")
    n_res = sum(v["resolved"] for v in verdicts)
    print(f"graded {len(verdicts)} rollouts -> {out}  ({n_res} resolved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
