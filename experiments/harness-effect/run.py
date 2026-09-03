# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""ARI-E on real harnesses: does the scaffold or the model decide the outcome?

This is the first ARI-E measurement against harnesses nobody here built.

`nvidia/Open-SWE-Traces` ran two agent scaffolds, SWE-agent and OpenHands, with
two models, Qwen3.5-122B and Minimax-M2.5, over about 18,000 SWE-bench-style
instances, with up to three rollouts of each instance in each cell. That is a
crossed design, and it is what makes the comparison below possible:

    harness effect   fix the model, change the scaffold
    model effect     fix the scaffold, change the model

Both are computed on the same instances with the same estimator, so they can be
put side by side. The literature claims scaffold variance can exceed model
variance. This turns that claim into one number, with a control.

**The control is why this is not just a pass-rate difference.** Agents are
stochastic. Two rollouts of one scaffold on one instance often disagree with
each other. ARI-E measures that self-disagreement and subtracts it, so what
remains is the part attributable to the change of scaffold:

    E = mean self-consistency - cross agreement

An outcome is `resolved == 1`, which comes from running the repository's own
tests against the agent's patch. The agent never saw that test.

Run extract.py first only if you want tool sequences. This reads the outcome
columns directly and needs nothing else.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from ari.harness import Trajectory, format_report, harness_report

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DATASET = "nvidia/Open-SWE-Traces"
HARNESSES = ("sweagent", "openhands")
MODELS = ("qwen35_122b", "minimax_m25")
MIN_RUNS = 2          # a cell with one run gives no self-consistency control
N_RESAMPLES = 2000

# The prior artifact bound a sampled trajectories.jsonl export that does not
# reconstruct the report, so this experiment must be RE-RUN, not just re-signed.
# The re-run resolves the dataset revision itself (ari.hub.hub_revision); it is
# cheap, since the metric reads a precomputed dataset on CPU.


def load_cells() -> dict[tuple[str, str], list[dict]]:
    """Outcome records for each (harness, model) cell.

    Only three columns are read. The transcripts in this dataset are large and
    the metric does not need them.
    """
    from datasets import load_dataset

    cells = {}
    for h in HARNESSES:
        for m in MODELS:
            ds = load_dataset(DATASET, h, split=m,
                              columns=["instance_id", "trajectory_id", "resolved"])
            rows = [r for r in ds if int(r["resolved"]) >= 0]
            cells[(h, m)] = rows
            print(f"  {h:<10} {m:<12} {len(rows):>6} graded runs "
                  f"({len(ds) - len(rows)} ungraded dropped)", flush=True)
    return cells


def to_trajectories(rows, label: str, cases: set[str]) -> list[Trajectory]:
    """Number the rollouts of each instance so they become repeats."""
    by_case = defaultdict(list)
    for r in rows:
        if r["instance_id"] in cases:
            by_case[r["instance_id"]].append(r)
    out = []
    for case, rs in by_case.items():
        # Sort by trajectory id so the numbering does not depend on row order.
        for i, r in enumerate(sorted(rs, key=lambda x: str(x["trajectory_id"]))):
            out.append(Trajectory(case_id=case, harness=label, run=i,
                                  outcome=int(r["resolved"]) == 1))
    return out


def shared_cases(a, b, min_runs: int = MIN_RUNS) -> set[str]:
    """Instances with enough rollouts on both sides to support the control."""
    def counts(rows):
        c = defaultdict(int)
        for r in rows:
            c[r["instance_id"]] += 1
        return {k for k, v in c.items() if v >= min_runs}
    return counts(a) & counts(b)


def contrast(rows_a, rows_b, name_a: str, name_b: str, title: str) -> dict:
    cases = shared_cases(rows_a, rows_b)
    if not cases:
        return {"title": title, "n_cases": 0, "note": "no shared cases"}
    runs = (to_trajectories(rows_a, name_a, cases)
            + to_trajectories(rows_b, name_b, cases))
    rep = harness_report(runs, n_resamples=N_RESAMPLES)
    label = f"{min(name_a, name_b)} vs {max(name_a, name_b)}"
    lo, hi = rep.harness_effect_ci.get(label, (float("nan"),) * 2)
    return {
        "title": title,
        "label": label,
        "n_cases": len(cases),
        "self_consistency": {h: m.outcome_agreement
                             for h, m in rep.self_consistency.items()},
        "cross_agreement": rep.cross[label].outcome_agreement,
        "effect": rep.harness_effect.get(label),
        "effect_ci": [lo, hi],
        "detected": rep.detects(label),
        "pass_rate": {
            name_a: float(np.mean([t.outcome for t in runs if t.harness == name_a])),
            name_b: float(np.mean([t.outcome for t in runs if t.harness == name_b])),
        },
        "report": format_report(rep),
    }


def main() -> None:
    print(f"loading {DATASET}")
    cells = load_cells()

    results = []

    print("\n=== Harness effect: same model, different scaffold ===")
    for m in MODELS:
        r = contrast(cells[("sweagent", m)], cells[("openhands", m)],
                     "sweagent", "openhands", f"scaffold effect, model={m}")
        results.append(r)

    print("\n=== Model effect: same scaffold, different model ===")
    for h in HARNESSES:
        r = contrast(cells[(h, "qwen35_122b")], cells[(h, "minimax_m25")],
                     "qwen35_122b", "minimax_m25", f"model effect, scaffold={h}")
        results.append(r)

    hdr = (f"{'contrast':<34}{'cases':>7}{'self':>8}{'cross':>8}"
           f"{'effect':>9}{'95% CI':>20}{'detected':>10}")
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
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{str(r['detected']):>10}")

    scaffold = [r for r in results if r["title"].startswith("scaffold") and r.get("effect")]
    model = [r for r in results if r["title"].startswith("model") and r.get("effect")]
    if scaffold and model:
        s = float(np.mean([r["effect"] for r in scaffold]))
        mo = float(np.mean([r["effect"] for r in model]))
        print(f"\nmean scaffold effect {s:+.3f}   mean model effect {mo:+.3f}")
        print("Larger means that choice decided more of the outcome.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    report = RESULTS / "harness_effect.json"
    report.write_text(json.dumps(
        {"dataset": DATASET, "min_runs_per_cell": MIN_RUNS,
         "n_resamples": N_RESAMPLES, "contrasts": results}, indent=2) + "\n")
    print(f"\nwrote {report}")

    sign_report(report)


def sign_report(report: Path) -> None:
    """Attest the report, if a signing key is available.

    An ARI number is a claim about someone else's system. Signing binds it to
    the inputs it came from, so a reader can check it without taking our word.
    The key is read from ARI_SIGNING_KEY, a path to a PEM private key.
    Generating one here instead would prove nothing, because a verifier checks
    the key against signers it already trusts.
    """
    import os

    key_path = os.environ.get("ARI_SIGNING_KEY")
    if not key_path:
        print("\nnot signed: set ARI_SIGNING_KEY to a PEM Ed25519 private key")
        return

    from cryptography.hazmat.primitives import serialization

    from ari.attest import (attest, build_references, dataset_ref, config_ref,
                            code_ref)
    from ari.hub import hub_revision

    key = serialization.load_pem_private_key(
        Path(key_path).read_bytes(), password=None)
    # Bind the dataset the metric actually reads, at its resolved revision, per
    # (harness config, model split). One repo, one revision. The sampled
    # trajectories.jsonl export is deliberately not bound: it does not
    # reconstruct the report.
    dataset_rev = hub_revision(DATASET, "dataset")
    references = build_references(
        datasets=[dataset_ref(DATASET, dataset_rev, config=h, split=m)
                  for h in HARNESSES for m in MODELS],
        config=config_ref({
            "dataset": DATASET, "harnesses": list(HARNESSES),
            "models": list(MODELS), "min_runs": MIN_RUNS,
            "n_resamples": N_RESAMPLES,
        }),
        code=code_ref(packages=["semq", "numpy", "datasets"]),
    )
    att = attest(metric="ARI-E", report_path=report, input_paths=[],
                 references=references,
                 repo_path=RESULTS / "attestation-repo", signing_key=key,
                 signer_identity=os.environ.get("ARI_SIGNER", "unnamed"),
                 extra={"dataset": DATASET})
    print(f"signed: {att.manifest_path.name}, {att.sidecar_path.name}")
    print(f"  verify with: python ari/verify_report.py {report}")


if __name__ == "__main__":
    main()
