"""ARI-D power simulation: does the frozen design see the effects we care about?

Gate for ARI-D-Bench v0.1 rollout step 1 (spec/arid-bench-v0.1.md,
section 8): before any paid panel run, show that (n_prompts, k) as frozen —
100 prompts, k=8 for same/proc/time, k=12 for conc — detects a drop of 0.10
in `exact_generation` against the `same` arm, at acceptable false-positive
rates, under the same statistics the scorer will use: bootstrap over prompts,
conservative CI-overlap classification.

The generative model
--------------------
A (prompt, condition) cell is modelled by s = P(one call returns the modal
completion). Drift concentrates where the top-2 margin is small, so the prompt
population is a stable mass plus a fragile tail: stable prompts have s = 1,
a FRAGILE_FRACTION of prompts draw s < 1. Two deviation shapes bracket what a
real API can do when a call leaves the modal completion:

- ``bimodal``: the deviation is always the *same* alternative completion (a
  single near-tied argmax flips between two candidates). Pair-identical
  probability is s^2 + (1-s)^2 — bounded BELOW by 0.5, so no single prompt
  can contribute a pair rate under 0.5 and mean effects must spread wide.
- ``unique``: every deviation diverges on its own (flips at many positions,
  compounding). Pair-identical probability is s^2, which can reach 0.

Real serving sits between the two; both are simulated and reported.

Effects are applied two ways, mirroring how a backend change can land:

- ``uniform``: every prompt's s degrades by the same amount.
- ``concentrated``: only the fragile tail degrades (the margin story: a
  precision change moves low-margin prompts first and hardest).

In both cases the degradation is solved numerically so the TRUE mean pair
rate drops by exactly the target delta — power is measured against a known
ground-truth effect, not against a parameter proxy.

The statistic and the test
--------------------------
Per prompt, `exact_generation` is the share of the C(k,2) repeat pairs that
are byte-identical; the arm's value is the mean over prompts. CIs are 95%
percentile bootstrap over prompts (B resamples). A condition is classified
as different from `same` when the two CIs are disjoint — the conservative
overlap rule the Matrix already documents. Power = share of trials where a
true effect is flagged; FPR = share flagged when the true effect is zero.

Determinism: every cell's RNG is seeded from a stable hash of its parameters;
two runs of this script produce identical JSON.

Output: results/power_grid.json + a printed summary table. RESULTS.md holds
the reading.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pathlib
import time

import numpy as np

# ---------------------------------------------------------------- design grid
N_PROMPTS_DESIGN = 100          # frozen by the spec (section 4)
N_PROMPTS_ALT = 50              # "would fewer cases do?" comparison point
K_GRID = (4, 8, 12)             # k=8 frozen for same/proc/time, 12 for conc
FLOORS = (1.0, 0.9, 0.7)        # the provider's own `same` mean pair rate
DELTAS = (0.0, 0.05, 0.10, 0.20)  # true drop in mean pair rate vs `same`
DEV_MODELS = ("bimodal", "unique")
EFFECT_SHAPES = ("uniform", "concentrated")

FRAGILE_FRACTION = 0.2          # share of prompts allowed to be unstable
TRIALS = 200                    # simulated panel runs per cell
BOOT = 500                      # bootstrap resamples per arm
ALPHA_CI = (2.5, 97.5)          # 95% percentile CI

K_SAME = 8                      # the `same` arm is always k=8 (spec, section 6)


def pair_rate(s: np.ndarray, dev_model: str) -> np.ndarray:
    """True P(two independent calls to this prompt are byte-identical)."""
    if dev_model == "bimodal":
        return s ** 2 + (1.0 - s) ** 2
    return s ** 2


def solve_floor_s(floor: float, dev_model: str, rng: np.random.Generator,
                  n: int) -> np.ndarray:
    """Prompt stabilities whose TRUE mean pair rate equals `floor`.

    Stable mass at s=1 plus a fragile tail. The tail's pair rate sits 30% of
    the way down toward the model's own lower bound (0.5 for bimodal, 0 for
    unique) — leaving headroom for effects to degrade it further — and the
    tail WIDTH is solved from the floor: a provider with a low `same` floor
    is unstable on more prompts, not infinitely unstable on a fixed few.
    floor=1.0 means every prompt is stable.
    """
    s = np.ones(n)
    if floor >= 1.0:
        return s
    lo_pr = 0.5 if dev_model == "bimodal" else 0.0
    fragile_pr = lo_pr + (1.0 - lo_pr) * 0.7   # 30% of the way down
    frac = (1.0 - floor) / (1.0 - fragile_pr)
    if frac > 1.0:
        # the whole population must degrade to reach this floor
        frac, fragile_pr = 1.0, floor
    n_fragile = max(int(round(frac * n)), 1)
    idx = rng.choice(n, size=n_fragile, replace=False)
    # solve the tail's pr exactly so the mean lands on the floor
    tail_pr = 1.0 - (1.0 - floor) * n / n_fragile
    s[idx] = invert_pair_rate(max(tail_pr, lo_pr), dev_model)
    return s


def invert_pair_rate(pr: float, dev_model: str) -> float:
    """The s in [0.5, 1] whose pair rate is `pr` (upper branch)."""
    if dev_model == "bimodal":
        # s^2 + (1-s)^2 = pr  ->  s = (1 + sqrt(2 pr - 1)) / 2
        return (1.0 + np.sqrt(max(2.0 * pr - 1.0, 0.0))) / 2.0
    return float(np.sqrt(max(pr, 0.0)))


def apply_effect(s_same: np.ndarray, delta: float, shape: str,
                 dev_model: str) -> np.ndarray:
    """Degrade stabilities so the true mean pair rate drops by exactly delta."""
    if delta <= 0.0:
        return s_same.copy()
    base = float(pair_rate(s_same, dev_model).mean())
    target = base - delta
    lo_pr = 0.5 if dev_model == "bimodal" else 0.0

    if shape == "uniform":
        # shrink every prompt's s toward 0.5/0.0 by a common factor t
        def mean_at(t: float) -> float:
            anchor = 0.5 if dev_model == "bimodal" else 0.0
            s = anchor + (s_same - anchor) * (1.0 - t)
            return float(pair_rate(s, dev_model).mean())
    else:  # concentrated: the low-margin end degrades first, widening as needed
        # A precision change hits low-margin prompts first and hardest; when
        # the current tail cannot absorb the whole effect (its pair rate is
        # already at the model's lower bound), the change recruits the next
        # lowest-margin prompts — the set widens until the effect fits.
        order = np.argsort(s_same)
        n = len(s_same)
        width = max(int(round(FRAGILE_FRACTION * n)), 1)
        while width <= n:
            idx = order[:width]
            reachable = (pair_rate(s_same[idx], dev_model) - lo_pr).sum() / n
            if reachable >= delta - 1e-9:
                break
            width += max(n // 20, 1)
        else:
            raise ValueError(
                f"delta {delta} exceeds what the whole population can lose "
                f"(dev={dev_model})")
        idx = order[:width]

        def mean_at(t: float) -> float:
            anchor = 0.5 if dev_model == "bimodal" else 0.0
            s = s_same.copy()
            s[idx] = anchor + (s[idx] - anchor) * (1.0 - t)
            return float(pair_rate(s, dev_model).mean())

    lo, hi = 0.0, 1.0
    for _ in range(60):  # bisection on t: mean_at is monotone decreasing
        mid = (lo + hi) / 2.0
        if mean_at(mid) > target:
            lo = mid
        else:
            hi = mid
    t = (lo + hi) / 2.0

    anchor = 0.5 if dev_model == "bimodal" else 0.0
    s = s_same.copy()
    if shape == "uniform":
        s = anchor + (s - anchor) * (1.0 - t)
    else:
        s[idx] = anchor + (s[idx] - anchor) * (1.0 - t)  # the widened set
    return s


def simulate_arm(s: np.ndarray, k: int, dev_model: str,
                 rng: np.random.Generator, trials: int) -> np.ndarray:
    """Per-trial per-prompt observed pair-identical share, shape (trials, n)."""
    n = len(s)
    m = rng.binomial(k, s[None, :].repeat(trials, axis=0))  # modal count
    pairs_total = k * (k - 1) / 2.0
    if dev_model == "bimodal":
        agree = m * (m - 1) / 2.0 + (k - m) * (k - m - 1) / 2.0
    else:
        agree = m * (m - 1) / 2.0
    return agree / pairs_total


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """95% percentile bootstrap CI over prompts, per trial. (trials, 2)."""
    trials, n = values.shape
    idx = rng.integers(0, n, size=(BOOT, n))
    # (trials, BOOT): mean over resampled prompts
    boots = values[:, idx].mean(axis=2)
    return np.percentile(boots, ALPHA_CI, axis=1).T


def run_cell(n_prompts: int, k: int, floor: float, delta: float,
             dev_model: str, shape: str) -> dict:
    key = f"{n_prompts}|{k}|{floor}|{delta}|{dev_model}|{shape}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    s_same = solve_floor_s(floor, dev_model, rng, n_prompts)
    s_cond = apply_effect(s_same, delta, shape, dev_model)

    same_vals = simulate_arm(s_same, K_SAME, dev_model, rng, TRIALS)
    cond_vals = simulate_arm(s_cond, k, dev_model, rng, TRIALS)

    ci_same = boot_ci(same_vals, rng)
    ci_cond = boot_ci(cond_vals, rng)

    # conservative CI-overlap rule: flagged iff the intervals are disjoint
    flagged = (ci_cond[:, 1] < ci_same[:, 0]) | (ci_same[:, 1] < ci_cond[:, 0])
    return {
        "n_prompts": n_prompts, "k": k, "floor": floor, "delta": delta,
        "dev_model": dev_model, "shape": shape,
        "flag_rate": float(flagged.mean()),
        "mean_ci_width_cond": float((ci_cond[:, 1] - ci_cond[:, 0]).mean()),
        "true_same": float(pair_rate(s_same, dev_model).mean()),
        "true_cond": float(pair_rate(s_cond, dev_model).mean()),
    }


def main() -> None:
    t0 = time.time()
    out = []
    grid = list(itertools.product(
        (N_PROMPTS_DESIGN, N_PROMPTS_ALT), K_GRID, FLOORS, DELTAS,
        DEV_MODELS, EFFECT_SHAPES))
    skipped = []
    for n, k, floor, delta, dev, shape in grid:
        # shape is meaningless at delta 0 and at floor 1.0 the fragile tail
        # does not exist yet for `concentrated` to act on
        if delta == 0.0 and shape == "concentrated":
            continue
        try:
            out.append(run_cell(n, k, floor, delta, dev, shape))
        except ValueError as exc:
            skipped.append({"cell": [n, k, floor, delta, dev, shape],
                            "reason": str(exc)})
    res_dir = pathlib.Path(__file__).parent / "results"
    res_dir.mkdir(exist_ok=True)
    # No timestamps or runtimes inside the artifact: the JSON must be
    # byte-identical across runs and machines so a reviewer can re-run the
    # script and diff the file.
    payload = {
        "design": {"trials": TRIALS, "bootstrap": BOOT,
                   "fragile_fraction": FRAGILE_FRACTION, "k_same": K_SAME},
        "cells": out, "skipped": skipped,
    }
    (res_dir / "power_grid.json").write_text(json.dumps(payload, indent=1))

    print(f"{len(out)} cells, {len(skipped)} skipped, "
          f"{time.time() - t0:.1f}s\n")
    print("power / FPR at the frozen design (n=100):")
    print(f"{'floor':>6} {'k':>3} {'dev':>8} {'shape':>13} "
          f"{'Δ=0':>6} {'Δ=.05':>6} {'Δ=.10':>6} {'Δ=.20':>6}")
    for floor in FLOORS:
        for k in K_GRID:
            for dev in DEV_MODELS:
                for shape in EFFECT_SHAPES:
                    row = []
                    for delta in DELTAS:
                        cell = [c for c in out if
                                c["n_prompts"] == 100 and c["k"] == k and
                                c["floor"] == floor and c["delta"] == delta and
                                c["dev_model"] == dev and
                                (c["shape"] == shape or delta == 0.0)]
                        row.append(f"{cell[0]['flag_rate']:.2f}"
                                   if cell else "  —  ")
                    print(f"{floor:>6} {k:>3} {dev:>8} {shape:>13} "
                          f"{row[0]:>6} {row[1]:>6} {row[2]:>6} {row[3]:>6}")


if __name__ == "__main__":
    main()
