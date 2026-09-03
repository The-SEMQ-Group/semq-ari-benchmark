# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""How many cases and repeats does ARI-E need before it can find anything?

The ARI-E metric is built and tested, but no agent run has fed it yet. Before
spending a day of compute on one, it is worth knowing what such a run could
detect. A first smoke report on 25 cases put the interval on cross-harness
agreement at [0.44, 0.80], which is wide enough that the design might not be
able to answer its own question.

This simulates runs with a known harness effect and asks how often the metric
recovers it. Nothing here is a result about any real harness. It is a property
of the estimator and the sample size, and it is cheap to compute now and
expensive to discover after a real run.

The generative model is deliberately plain. Each harness has a pass
probability per case. A harness that is unreliable passes a case sometimes and
fails it other times, which is what agent stochasticity looks like: tool
results depend on the network and the clock. The true harness effect follows
from those probabilities in closed form, so the simulation has a right answer
to be checked against.

Reported as power: the share of simulated experiments whose interval on the
effect excludes zero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ari.harness import Trajectory, harness_report

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

N_CASES = (10, 25, 50, 100, 200)
N_REPEATS = (2, 3, 5)
# Difference in per-case pass probability between the two harnesses.
GAPS = (0.0, 0.05, 0.10, 0.20, 0.40)
BASE_PASS = 0.75          # how often the stronger harness passes a case
N_TRIALS = 200
N_RESAMPLES = 400
SEED = 0


def true_effect(p1: np.ndarray, p2: np.ndarray) -> float:
    """The harness effect implied by two sets of pass probabilities.

    Self-consistency for one harness on a case is P(two runs agree), which is
    p^2 + (1-p)^2. Cross agreement is p1*p2 + (1-p1)*(1-p2). The effect is the
    mean of the first over both harnesses, minus the second.
    """
    self1 = p1 ** 2 + (1 - p1) ** 2
    self2 = p2 ** 2 + (1 - p2) ** 2
    cross = p1 * p2 + (1 - p1) * (1 - p2)
    return float(np.mean(0.5 * (self1 + self2) - cross))


def simulate(n_cases: int, n_repeats: int, gap: float,
             rng: np.random.Generator) -> tuple[list[Trajectory], float]:
    # Per-case pass probability, jittered so cases are not interchangeable.
    p1 = np.clip(rng.normal(BASE_PASS, 0.12, n_cases), 0.02, 0.98)
    p2 = np.clip(p1 - gap, 0.02, 0.98)

    runs = []
    for r in range(n_repeats):
        for i in range(n_cases):
            case = f"case{i:03d}"
            runs.append(Trajectory(case, "h1", r, bool(rng.random() < p1[i])))
            runs.append(Trajectory(case, "h2", r, bool(rng.random() < p2[i])))
    return runs, true_effect(p1, p2)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    print(f"{N_TRIALS} simulated experiments per cell, base pass rate "
          f"{BASE_PASS}\n")
    header = (f"{'gap':>6} {'true effect':>12} {'repeats':>8}" +
              "".join(f"{n:>9}" for n in N_CASES))
    print(header)
    print("-" * len(header))

    for gap in GAPS:
        for reps in N_REPEATS:
            powers, truths = [], []
            for n in N_CASES:
                hits, te = 0, []
                for _ in range(N_TRIALS):
                    runs, t = simulate(n, reps, gap, rng)
                    rep = harness_report(runs, n_resamples=N_RESAMPLES,
                                         seed=int(rng.integers(1 << 30)))
                    label = "h1 vs h2"
                    hits += int(rep.detects(label))
                    te.append(t)
                powers.append(hits / N_TRIALS)
                truths.append(float(np.mean(te)))
                rows.append({"gap": gap, "repeats": reps, "n_cases": n,
                             "true_effect": truths[-1], "power": powers[-1]})
            print(f"{gap:>6.2f} {np.mean(truths):>12.4f} {reps:>8}" +
                  "".join(f"{p:>9.0%}" for p in powers))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "power.json").write_text(json.dumps(
        {"n_trials": N_TRIALS, "base_pass": BASE_PASS, "seed": SEED,
         "n_cases": list(N_CASES), "repeats": list(N_REPEATS),
         "gaps": list(GAPS), "rows": rows}, indent=2) + "\n")

    print(f"\nwrote {RESULTS / 'power.json'}")

    from ari.attest import (sign_if_configured, build_references, config_ref,
                            code_ref)
    # A pure simulation seeded by SEED: the parameters and the seed fully
    # determine the output, so binding them is the honest provenance.
    references = build_references(
        config=config_ref({
            "n_trials": N_TRIALS, "base_pass": BASE_PASS, "seed": SEED,
            "n_cases": list(N_CASES), "repeats": list(N_REPEATS),
            "gaps": list(GAPS), "n_resamples": N_RESAMPLES,
        }),
        code=code_ref(packages=["numpy"]),
    )
    sign_if_configured(metric="ARI-E-power", report_path=RESULTS / "power.json",
                       references=references)

    print("\nThe gap-0 rows are the false-positive check. Power there is the")
    print("rate at which the metric claims an effect where none exists, and it")
    print("should sit near 5%.")


if __name__ == "__main__":
    main()
