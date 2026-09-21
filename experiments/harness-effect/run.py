# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Compute ARI-E for the scaffold and model contrasts in Open-SWE-Traces.

The input is the outcome table that fetch_outcomes.py writes for one pinned
dataset revision. The table has one row per rollout: harness, model,
instance_id, trajectory_id and the grade ``resolved`` (1 pass, 0 fail, -1
ungraded). This script checks the table against its manifest, drops ungraded
rows, keeps the cases with at least two graded rollouts on both sides of a
contrast, and calls ``ari.harness.harness_report``.

The estimator is defined in spec/ari-e-bench-v0.1.md. In short: for each case,
the mean of the two within-condition agreement rates minus the cross-condition
agreement rate; the effect is the mean over cases and its interval is a case
bootstrap of the same per-case vector.

Beside the four contrasts this writes two checks:

* a permutation null for each contrast: the condition labels of the rollouts
  of each case are shuffled, keeping the per-side counts, and the effect is
  recomputed. Under the null the effect has expectation zero;
* a split-half check inside one cell where cases have four or more graded
  rollouts: the rollouts are split at random into two pseudo-conditions of at
  least two each, and the effect between the halves is computed.

Usage, from the repository root:

    python experiments/harness-effect/run.py [--outcomes results/outcomes.<rev>.csv.gz]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from ari.harness import Trajectory, harness_report  # noqa: E402

DATASET = "nvidia/Open-SWE-Traces"
# The dataset revision the retained outcome table was read from. See
# spec/ari-e-bench-v0.1.md for the pin and fetch_outcomes.py for the reader.
PINNED_REVISION = "f967cba3312573981a47fd7a7b80029b53909b5f"
HARNESSES = ("sweagent", "openhands")
MODELS = ("qwen35_122b", "minimax_m25")
DISPLAY = {"sweagent": "SWE-agent", "openhands": "OpenHands",
           "qwen35_122b": "Qwen3.5-122B", "minimax_m25": "Minimax-M2.5"}
MIN_RUNS = 2          # a cell with one run gives no self-consistency control
N_RESAMPLES = 2000
SEED = 0
N_PERMUTATIONS = 50
NULL_RESAMPLES = 1000
MIN_SPLIT_CASES = 30


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_outcomes(csv_path: Path, manifest_path: Path):
    """Outcome rows for each (harness, model) cell, checked against the manifest."""
    manifest = json.loads(manifest_path.read_text())
    digest = sha256_file(csv_path)
    if digest != manifest["output"]["sha256"]:
        raise SystemExit(f"{csv_path.name} does not match its manifest: "
                         f"{digest} != {manifest['output']['sha256']}")
    cells = {(h, m): [] for h in HARNESSES for m in MODELS}
    with gzip.open(csv_path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["harness"], row["model"])
            if key not in cells:
                raise SystemExit(f"unexpected cell {key} in {csv_path.name}")
            cells[key].append({"instance_id": row["instance_id"],
                               "trajectory_id": row["trajectory_id"],
                               "resolved": int(row["resolved"])})
    return cells, manifest


def graded(rows):
    return [r for r in rows if r["resolved"] >= 0]


def by_case(rows) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["instance_id"]].append(r)
    for rs in out.values():
        # Sort by trajectory id so the run numbering does not depend on row order.
        rs.sort(key=lambda x: x["trajectory_id"])
    return out


def to_trajectories(rows, label: str, cases: set[str]) -> list[Trajectory]:
    """Number the rollouts of each instance so they become repeats."""
    out = []
    for case, rs in by_case(rows).items():
        if case in cases:
            for i, r in enumerate(rs):
                out.append(Trajectory(case_id=case, harness=label, run=i,
                                      outcome=r["resolved"] == 1))
    return out


def shared_cases(a, b, min_runs: int = MIN_RUNS) -> set[str]:
    """Instances with enough graded rollouts on both sides to support the control."""
    def enough(rows):
        return {k for k, rs in by_case(rows).items() if len(rs) >= min_runs}
    return enough(a) & enough(b)


def summarize(rep, label: str) -> dict:
    lo, hi = rep.harness_effect_ci.get(label, (float("nan"),) * 2)
    return {
        "label": label,
        "self_consistency": {h: m.outcome_agreement
                             for h, m in rep.self_consistency.items()},
        "self_consistency_ci": {h: list(m.outcome_agreement_ci)
                                for h, m in rep.self_consistency.items()},
        "cross_agreement": rep.cross[label].outcome_agreement,
        "cross_agreement_ci": list(rep.cross[label].outcome_agreement_ci),
        "effect": rep.harness_effect.get(label),
        "effect_ci": [lo, hi],
        "detected": rep.detects(label),
    }


def permutation_null(rows_a, rows_b, name_a, name_b, cases, k: int,
                     rng: np.random.Generator) -> dict:
    """Shuffle condition labels within each case, keeping the per-side counts."""
    ca, cb = by_case(rows_a), by_case(rows_b)
    label = f"{min(name_a, name_b)} vs {max(name_a, name_b)}"
    effects, intervals, excluded = [], [], 0
    for i in range(k):
        runs = []
        for case in sorted(cases):
            pooled = [r["resolved"] == 1 for r in ca[case]] + \
                     [r["resolved"] == 1 for r in cb[case]]
            pooled = [pooled[j] for j in rng.permutation(len(pooled))]
            na = len(ca[case])
            runs += [Trajectory(case, name_a, j, o) for j, o in enumerate(pooled[:na])]
            runs += [Trajectory(case, name_b, j, o) for j, o in enumerate(pooled[na:])]
        rep = harness_report(runs, n_resamples=NULL_RESAMPLES, seed=SEED + i)
        effects.append(rep.harness_effect[label])
        intervals.append(list(rep.harness_effect_ci[label]))
        excluded += int(rep.detects(label))
    return {
        "permutations": k, "n_resamples": NULL_RESAMPLES,
        "effects": effects, "intervals": intervals, "mean": float(np.mean(effects)),
        "sd": float(np.std(effects, ddof=1)) if k > 1 else None,
        "min": float(np.min(effects)), "max": float(np.max(effects)),
        "intervals_excluding_zero": excluded,
    }


def contrast(rows_a, rows_b, name_a: str, name_b: str, title: str,
             rng: np.random.Generator) -> dict:
    ga, gb = graded(rows_a), graded(rows_b)
    cases = shared_cases(ga, gb)
    if not cases:
        return {"title": title, "n_cases": 0, "note": "no shared cases"}
    runs = to_trajectories(ga, name_a, cases) + to_trajectories(gb, name_b, cases)
    rep = harness_report(runs, n_resamples=N_RESAMPLES, seed=SEED)
    label = f"{min(name_a, name_b)} vs {max(name_a, name_b)}"
    out = {"title": title, "n_cases": len(cases), **summarize(rep, label)}

    ca, cb = by_case(ga), by_case(gb)
    shape = Counter(f"{len(ca[c])}/{len(cb[c])}" for c in cases)
    out["repeat_shapes"] = dict(sorted(shape.items()))
    equal = {c for c in cases if len(ca[c]) == len(cb[c])}
    out["equal_repeats"] = {"n_cases": len(equal)}
    if len(equal) >= MIN_SPLIT_CASES:
        rep_eq = harness_report(to_trajectories(ga, name_a, equal)
                                + to_trajectories(gb, name_b, equal),
                                n_resamples=N_RESAMPLES, seed=SEED)
        out["equal_repeats"].update({k: v for k, v in summarize(rep_eq, label).items()
                                     if k in ("effect", "effect_ci", "detected")})

    out["pass_rate"] = {
        name_a: float(np.mean([t.outcome for t in runs if t.harness == name_a])),
        name_b: float(np.mean([t.outcome for t in runs if t.harness == name_b])),
    }
    out["permutation_null"] = permutation_null(ga, gb, name_a, name_b, cases,
                                               N_PERMUTATIONS, rng)
    return out


def split_half(rows, name: str, rng: np.random.Generator) -> dict:
    """Two random halves of one cell, on cases with four or more graded rollouts."""
    cases = {c: rs for c, rs in by_case(graded(rows)).items() if len(rs) >= 2 * MIN_RUNS}
    out = {"cell": name, "n_cases": len(cases)}
    if len(cases) < MIN_SPLIT_CASES:
        out["note"] = f"fewer than {MIN_SPLIT_CASES} cases have {2 * MIN_RUNS}+ rollouts"
        return out
    runs = []
    for case, rs in sorted(cases.items()):
        order = rng.permutation(len(rs))
        half = len(rs) // 2
        for j, idx in enumerate(order[:half]):
            runs.append(Trajectory(case, "half_a", j, rs[idx]["resolved"] == 1))
        for j, idx in enumerate(order[half:]):
            runs.append(Trajectory(case, "half_b", j, rs[idx]["resolved"] == 1))
    rep = harness_report(runs, n_resamples=N_RESAMPLES, seed=SEED)
    out.update(summarize(rep, "half_a vs half_b"))
    return out


def tex_rows(results) -> str:
    """Rows in the format of the paper's tab:arie."""
    lines = []
    for r in results:
        if not r.get("n_cases"):
            continue
        kind, _, rest = r["title"].partition(" effect, ")
        fixed_key, _, fixed_val = rest.partition("=")
        name = f"{kind}, {fixed_key} = {DISPLAY[fixed_val]}"
        selves = np.mean(list(r["self_consistency"].values()))
        lo, hi = r["effect_ci"]
        cases = f"{r['n_cases']:,}".replace(",", "{,}")
        lines.append(f"{name} & {cases} & {selves:.3f} & {r['cross_agreement']:.3f} "
                     f"& \\textbf{{{r['effect']:+.3f}}} & [{lo:+.3f}, {hi:+.3f}] \\\\")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", type=Path,
                    default=RESULTS / f"outcomes.{PINNED_REVISION[:12]}.csv.gz")
    ap.add_argument("--out", type=Path, default=RESULTS / "harness_effect.json",
                    help="report path; the manifest and the table sit beside it")
    args = ap.parse_args()
    manifest_path = args.outcomes.with_name(
        args.outcomes.name.replace(".csv.gz", ".manifest.json"))

    print(f"reading {args.outcomes.name}")
    cells, in_manifest = load_outcomes(args.outcomes, manifest_path)
    revision = in_manifest["revision"]
    ungraded = {}
    for (h, m), rows in cells.items():
        n_ungraded = sum(r["resolved"] < 0 for r in rows)
        ungraded[f"{h}/{m}"] = {"rows": len(rows), "ungraded": n_ungraded,
                                "graded": len(rows) - n_ungraded}
        print(f"  {h:<10} {m:<12} {len(rows) - n_ungraded:>6} graded rows "
              f"({n_ungraded} ungraded dropped)")

    rng = np.random.default_rng(SEED)
    results = []
    print("\n=== Scaffold effect: same model, different scaffold ===")
    for m in MODELS:
        results.append(contrast(cells[("sweagent", m)], cells[("openhands", m)],
                                "sweagent", "openhands",
                                f"scaffold effect, model={m}", rng))
    print("=== Model effect: same scaffold, different model ===")
    for h in HARNESSES:
        results.append(contrast(cells[(h, "qwen35_122b")], cells[(h, "minimax_m25")],
                                "qwen35_122b", "minimax_m25",
                                f"model effect, scaffold={h}", rng))
    splits = [split_half(cells[(h, m)], f"{h}/{m}", rng)
              for h in HARNESSES for m in MODELS]

    hdr = (f"{'contrast':<34}{'cases':>7}{'self':>8}{'cross':>8}"
           f"{'effect':>9}{'95% CI':>20}{'null mean':>11}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        if not r.get("n_cases"):
            print(f"{r['title']:<34}  {r.get('note')}")
            continue
        lo, hi = r["effect_ci"]
        selves = np.mean(list(r["self_consistency"].values()))
        print(f"{r['title']:<34}{r['n_cases']:>7}{selves:>8.3f}"
              f"{r['cross_agreement']:>8.3f}{r['effect']:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}"
              f"{r['permutation_null']['mean']:>+11.4f}")
    for s in splits:
        print(f"split-half {s['cell']:<24} cases {s['n_cases']:>6}  "
              + (f"effect {s['effect']:+.4f} [{s['effect_ci'][0]:+.4f}, "
                 f"{s['effect_ci'][1]:+.4f}]" if "effect" in s else s["note"]))

    scaffold = [r["effect"] for r in results
                if r["title"].startswith("scaffold") and r.get("effect") is not None]
    model = [r["effect"] for r in results
             if r["title"].startswith("model") and r.get("effect") is not None]
    summary = {}
    if scaffold and model:
        summary = {"mean_scaffold_effect": float(np.mean(scaffold)),
                   "mean_model_effect": float(np.mean(model))}
        print(f"\nmean scaffold effect {summary['mean_scaffold_effect']:+.3f}   "
              f"mean model effect {summary['mean_model_effect']:+.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": DATASET, "revision": revision,
        "inputs": {"outcomes": args.outcomes.name,
                   "outcomes_sha256": sha256_file(args.outcomes),
                   "manifest": manifest_path.name,
                   "manifest_sha256": sha256_file(manifest_path)},
        "estimator": "spec/ari-e-bench-v0.1.md",
        "min_runs_per_cell": MIN_RUNS, "n_resamples": N_RESAMPLES, "seed": SEED,
        "rows_per_cell": ungraded,
        "contrasts": results, "split_half": splits, "summary": summary,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    tex = args.out.with_name(args.out.stem.replace("harness_effect", "arie_table") + ".tex")
    tex.write_text(f"% ARI-E rows for tab:arie. Source: {args.out.name}, "
                   f"{DATASET} at {revision}.\n"
                   f"% Generated by experiments/harness-effect/run.py; do not edit.\n"
                   + tex_rows(results))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": DATASET, "revision": revision,
        "inputs": [{"name": p.name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                   for p in (args.outcomes, manifest_path)],
        "outputs": [{"name": p.name, "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                    for p in (args.out, tex)],
        "code": {
            "git_head": _git_head(),
            "files": {rel: sha256_file(ROOT / rel)
                      for rel in ("ari/harness.py", "ari/metrics.py",
                                  "experiments/harness-effect/run.py")},
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "parameters": {"min_runs_per_cell": MIN_RUNS, "n_resamples": N_RESAMPLES,
                       "seed": SEED, "permutations": N_PERMUTATIONS,
                       "null_resamples": NULL_RESAMPLES,
                       "min_split_cases": MIN_SPLIT_CASES},
    }
    mpath = args.out.with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out}\nwrote {tex}\nwrote {mpath}")

    sign_report(args.out, revision)


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def sign_report(report: Path, revision: str) -> None:
    """Attest the report when a signing key is available.

    The key is read from ARI_SIGNING_KEY, a path to a PEM private key. The
    dataset reference binds the revision the outcome table was read from, not
    whatever the Hub serves as ``main`` at signing time.
    """
    key_path = os.environ.get("ARI_SIGNING_KEY")
    if not key_path:
        print("\nnot signed: set ARI_SIGNING_KEY to a PEM Ed25519 private key")
        return

    from cryptography.hazmat.primitives import serialization

    from ari.attest import (attest, build_references, code_ref, config_ref,
                            dataset_ref)

    key = serialization.load_pem_private_key(
        Path(key_path).read_bytes(), password=None)
    references = build_references(
        datasets=[dataset_ref(DATASET, revision, config="v1.0", split=h)
                  for h in HARNESSES],
        config=config_ref({
            "dataset": DATASET, "revision": revision,
            "harnesses": list(HARNESSES), "models": list(MODELS),
            "min_runs": MIN_RUNS, "n_resamples": N_RESAMPLES, "seed": SEED,
        }),
        code=code_ref(packages=["numpy"]),
    )
    outcomes = report.parent / f"outcomes.{revision[:12]}.csv.gz"
    att = attest(metric="ARI-E", report_path=report,
                 input_paths=[outcomes, report.parent / f"outcomes.{revision[:12]}.manifest.json"],
                 references=references,
                 repo_path=report.parent / "attestation-repo", signing_key=key,
                 signer_identity=os.environ.get("ARI_SIGNER", "unnamed"),
                 extra={"dataset": DATASET, "revision": revision})
    print(f"signed: {att.manifest_path.name}, {att.sidecar_path.name}")
    print(f"  verify with: python ari/verify_report.py {report}")


if __name__ == "__main__":
    main()
