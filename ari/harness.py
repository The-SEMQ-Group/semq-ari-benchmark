"""ARI-E metrics: how much of an agent's outcome is decided by its harness?

ARI-R asks whether the same input gives the same embedding. ARI-D asks whether
the same context gives the same token. ARI-E asks the question the first two
cannot reach: holding the model, the endpoint and the task fixed, how much does
the *scaffold around the model* decide what happens?

The published evidence says: a great deal. Two harnesses on the same model and
the same endpoint have been reported to differ by several tasks on the same
benchmark, and the literature now argues harness-induced variance can exceed
model-induced variance, up to reversing a model ranking. If that is right, ARI-E
is the largest of the three terms and the one ARI has so far not measured.

**The control is the whole design.** A cross-harness disagreement means nothing
until you know how often a *single* harness disagrees with itself. Agents are
stochastic: tool results depend on the network and the clock, and a harness may
not reproduce its own trajectory. So this module always measures two things
together:

    HER_self(h)      P(two runs of harness h on one case give the same outcome)
    HER_cross(h1,h2) P(h1 and h2 give the same outcome on one case)

and reports the difference between them. If harnesses were interchangeable, the
two would be equal and the difference would be zero. The difference is the
harness effect, and it is the number ARI-E exists to produce:

    E(h1, h2) = mean_h HER_self(h) - HER_cross(h1, h2)

This is a variance decomposition, not a quality score. A high harness effect
does not say which harness is better. It says the choice of harness decided the
answer, which is what makes a leaderboard entry that omits the harness
uninterpretable.

Outcome agreement is deliberately coarse -- two harnesses can reach the same
verdict by wholly different routes. So trajectory agreement is reported beside
it, over the sequence of tool calls, together with the step at which two runs
first diverge. Reporting only the outcome would hide the case where two
harnesses agree by luck.

This module defines the metric and reads trajectories from a plain file. It
does not run agents, and it deliberately depends on nothing but numpy.

An earlier design took its input from a companion repository that snapshotted
transformer KV caches at each decision. That coupling is gone. ARI-E needs a
verdict and an ordered list of actions, and neither requires model state: KV
snapshots are lossy and diverge within a few tokens, so they never supported
the replay they were added for. Any runner that can emit the JSON Lines format
below can feed this metric.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .metrics import _bootstrap_ci


@dataclass(frozen=True)
class Trajectory:
    """One run of one harness on one case.

    `outcome` is the objective verdict — for the octobench corpus, whether the
    held-out gold test passed. It must come from a checker the agent never saw,
    otherwise this measures the judge and not the harness.

    `tool_calls` is the ordered sequence of actions, each reduced to a
    comparable label. What counts as a label is a decision with consequences:
    raw shell strings make two runs look different whenever a path differs,
    while a bare tool name makes them look identical whenever the tool matches.
    Record the choice in the report.
    """

    case_id: str
    harness: str
    run: int
    outcome: bool
    tool_calls: tuple[str, ...] = ()
    checkpoints: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairMetrics:
    """Agreement between two sets of runs over the same cases."""

    label: str
    n_cases: int
    outcome_agreement: float
    outcome_agreement_ci: tuple[float, float]
    trajectory_agreement: float
    mean_common_prefix: float
    mean_first_divergence: float
    both_passed: float
    both_failed: float
    disagreed: float


@dataclass
class HarnessReport:
    """ARI-E for one comparison, with the self-consistency control beside it."""

    self_consistency: dict[str, PairMetrics] = field(default_factory=dict)
    cross: dict[str, PairMetrics] = field(default_factory=dict)
    harness_effect: dict[str, float] = field(default_factory=dict)
    harness_effect_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def detects(self, label: str) -> bool:
        """Does the interval for this comparison exclude zero?

        A point estimate of the harness effect is not a finding on its own.
        Case counts in agent benchmarks are small, so the interval decides
        whether an effect separates from run-to-run noise.
        """
        lo, hi = self.harness_effect_ci.get(label, (0.0, 0.0))
        return not (lo <= 0.0 <= hi)


# ---------------------------------------------------------------------------
# Trajectory comparison
# ---------------------------------------------------------------------------

def common_prefix(a: Sequence[str], b: Sequence[str]) -> int:
    """Number of leading tool calls two runs share.

    The prefix length matters more than total overlap: agents compound. Once
    two runs take a different action the states diverge, and everything after
    is a consequence rather than an independent disagreement.
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def trajectory_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """1.0 for identical sequences, 0.0 for sequences sharing nothing.

    Normalised Levenshtein over tool-call labels. Two empty trajectories count
    as identical, which is correct: a harness that took no action twice did
    reproduce itself.
    """
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1.0 - prev[lb] / max(la, lb)


def _index(runs: Iterable[Trajectory]) -> dict[tuple[str, str, int], Trajectory]:
    return {(t.harness, t.case_id, t.run): t for t in runs}


def _per_case_agreement(pairs_of_runs, cases) -> np.ndarray:
    """Mean agreement for each case, averaged over the given run pairings.

    Resampling has to be over *cases*, because runs of one case are not
    independent of each other. Collapsing to one value per case first is what
    makes the bootstrap valid.
    """
    acc = {c: [] for c in cases}
    for left, right in pairs_of_runs:
        by_case_r = {t.case_id: t for t in right}
        for l in left:
            r = by_case_r.get(l.case_id)
            if r is not None:
                acc[l.case_id].append(float(l.outcome == r.outcome))
    return np.array([np.mean(acc[c]) if acc[c] else np.nan for c in cases])


def _mean_and_ci(per_case: np.ndarray, n_resamples: int, seed: int):
    """Mean over the cases that have a value, with a bootstrap interval on it.

    Point estimate and interval come from the same vector, so the interval
    always describes the number printed beside it.
    """
    v = per_case[~np.isnan(per_case)]
    if not len(v):
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_resamples, len(v)))
    means = v[idx].mean(axis=1)
    return (float(v.mean()),
            (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))))


def compare_runs(left: Sequence[Trajectory], right: Sequence[Trajectory],
                 label: str, n_resamples: int = 1000, seed: int = 0) -> PairMetrics:
    """Agreement between two aligned run sets, matched on case_id."""
    by_case_r = {t.case_id: t for t in right}
    pairs = [(l, by_case_r[l.case_id]) for l in left if l.case_id in by_case_r]
    if not pairs:
        raise ValueError(f"no shared cases between the two run sets ({label})")

    same_outcome = np.array([a.outcome == b.outcome for a, b in pairs], float)
    traj = np.array([trajectory_similarity(a.tool_calls, b.tool_calls)
                     for a, b in pairs], float)
    prefix = np.array([common_prefix(a.tool_calls, b.tool_calls)
                       for a, b in pairs], float)
    # A run that never diverges reports its own length, so that "identical" and
    # "diverged at the last step" are not conflated.
    first_div = np.array([
        common_prefix(a.tool_calls, b.tool_calls)
        if common_prefix(a.tool_calls, b.tool_calls) < max(len(a.tool_calls),
                                                           len(b.tool_calls))
        else max(len(a.tool_calls), len(b.tool_calls))
        for a, b in pairs], float)

    both_pass = np.mean([a.outcome and b.outcome for a, b in pairs])
    both_fail = np.mean([not a.outcome and not b.outcome for a, b in pairs])

    return PairMetrics(
        label=label,
        n_cases=len(pairs),
        outcome_agreement=float(same_outcome.mean()),
        outcome_agreement_ci=_bootstrap_ci(same_outcome, n_resamples, seed),
        trajectory_agreement=float(traj.mean()),
        mean_common_prefix=float(prefix.mean()),
        mean_first_divergence=float(first_div.mean()),
        both_passed=float(both_pass),
        both_failed=float(both_fail),
        disagreed=float(1.0 - same_outcome.mean()),
    )


# ---------------------------------------------------------------------------
# ARI-E
# ---------------------------------------------------------------------------

def harness_report(runs: Sequence[Trajectory], n_resamples: int = 1000,
                   seed: int = 0) -> HarnessReport:
    """Self-consistency per harness, cross-harness agreement, and the effect.

    Requires at least two runs per (harness, case) to produce the control. If a
    harness has only one run per case its self-consistency is unmeasurable, and
    that is recorded as a note rather than silently assumed to be 1.0 —
    assuming it is exactly the error this metric exists to avoid.
    """
    rep = HarnessReport()
    harnesses = sorted({t.harness for t in runs})
    by_hr: dict[str, dict[int, list[Trajectory]]] = {}
    for t in runs:
        by_hr.setdefault(t.harness, {}).setdefault(t.run, []).append(t)

    cases = sorted({t.case_id for t in runs})

    def mean_over_pairs(pairs, label):
        """Average agreement over several run pairings.

        Comparing a single pair of runs makes the estimate depend on which run
        happened to be numbered 0. Averaging over the available pairings
        removes that arbitrary choice and lowers the variance.

        The interval has to describe the same statistic. Taking it from the
        first pairing would leave the estimate averaged over every pairing and
        the interval measured on one of them, so relabelling the runs would
        move the interval while the estimate stood still. Collapse each case to
        one agreement value across the pairings, then resample cases.
        """
        ms = [compare_runs(a, b, label, n_resamples, seed) for a, b in pairs]
        per_case = _per_case_agreement(pairs, cases)
        agreement, ci = _mean_and_ci(per_case, n_resamples, seed)
        return PairMetrics(
            label=label,
            n_cases=ms[0].n_cases,
            outcome_agreement=agreement,
            outcome_agreement_ci=ci,
            trajectory_agreement=float(
                np.mean([m.trajectory_agreement for m in ms])),
            mean_common_prefix=float(np.mean([m.mean_common_prefix for m in ms])),
            mean_first_divergence=float(
                np.mean([m.mean_first_divergence for m in ms])),
            both_passed=float(np.mean([m.both_passed for m in ms])),
            both_failed=float(np.mean([m.both_failed for m in ms])),
            disagreed=float(np.mean([m.disagreed for m in ms])),
        )

    for h in harnesses:
        rl = sorted(by_hr[h])
        if len(rl) < 2:
            rep.notes.append(
                f"harness '{h}' has one run per case, so its self-consistency "
                f"is unmeasured. The harness effect against it is not computed")
            continue
        pairs = [(by_hr[h][a], by_hr[h][b]) for a, b in combinations(rl, 2)]
        rep.self_consistency[h] = mean_over_pairs(pairs, f"{h} vs itself")

    for h1, h2 in combinations(harnesses, 2):
        label = f"{h1} vs {h2}"
        pairs = [(by_hr[h1][a], by_hr[h2][b])
                 for a in sorted(by_hr[h1]) for b in sorted(by_hr[h2])]
        rep.cross[label] = mean_over_pairs(pairs, label)

        # The control is the mean self-consistency of the two harnesses.
        # Falling back to 1.0 when it is missing would charge every bit of
        # agent stochasticity to the harness, which is the error this metric
        # exists to avoid.
        if h1 in rep.self_consistency and h2 in rep.self_consistency:
            # The control weights the two harnesses equally. Pooling every
            # within-harness pair instead would weight each harness by how many
            # runs it happens to have, so a harness with three runs would count
            # three times against one with two, and the interval would describe
            # a different quantity from the effect printed next to it.
            s_each = [_per_case_agreement(
                [(by_hr[h][a], by_hr[h][b])
                 for a, b in combinations(sorted(by_hr[h]), 2)], cases)
                for h in (h1, h2)]
            cross_pairs = [(by_hr[h1][a], by_hr[h2][b])
                           for a in sorted(by_hr[h1]) for b in sorted(by_hr[h2])]
            c_case = _per_case_agreement(cross_pairs, cases)

            # One vector, one estimand: per case, the 50/50 control minus the
            # cross agreement. Both the effect and its interval come from this,
            # which is what keeps the estimate inside its own interval.
            ok = ~(np.isnan(s_each[0]) | np.isnan(s_each[1]) | np.isnan(c_case))
            s_case = (s_each[0] + s_each[1]) / 2.0
            d = s_case[ok] - c_case[ok]
            if len(d):
                effect, ci = _mean_and_ci(d, n_resamples, seed)
                rep.harness_effect[label] = effect
                rep.harness_effect_ci[label] = ci
        else:
            rep.notes.append(
                f"no self-consistency control for {label}. Harness effect "
                f"not computed")
    return rep


def format_report(rep: HarnessReport) -> str:
    """Human-readable summary, control first."""
    out = ["ARI-E — harness reproducibility", ""]

    out.append("Self-consistency (the control: does one harness repeat itself?)")
    if not rep.self_consistency:
        out.append("  none measured — every harness has a single run per case")
    for h, m in sorted(rep.self_consistency.items()):
        lo, hi = m.outcome_agreement_ci
        out.append(f"  {h:<16} outcome {m.outcome_agreement:.3f} "
                   f"[{lo:.3f}, {hi:.3f}]   trajectory {m.trajectory_agreement:.3f}"
                   f"   n={m.n_cases}")

    out += ["", "Cross-harness agreement"]
    for label, m in sorted(rep.cross.items()):
        lo, hi = m.outcome_agreement_ci
        out.append(f"  {label:<24} outcome {m.outcome_agreement:.3f} "
                   f"[{lo:.3f}, {hi:.3f}]   trajectory {m.trajectory_agreement:.3f}"
                   f"   diverges at step {m.mean_first_divergence:.1f}")

    out += ["", "Harness effect = mean self-consistency - cross agreement"]
    if not rep.harness_effect:
        out.append("  not computable without a self-consistency control")
    for label, e in sorted(rep.harness_effect.items()):
        lo, hi = rep.harness_effect_ci.get(label, (float("nan"),) * 2)
        mark = "detected" if rep.detects(label) else "not distinguishable from 0"
        out.append(f"  {label:<24} {e:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {mark}")
    out.append("")
    out.append("  0 means the harnesses are interchangeable on these cases.")
    out.append("  Large means the scaffold decided the answer, not the model.")

    if rep.notes:
        out += ["", "Notes"] + [f"  - {n}" for n in rep.notes]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Input format
# ---------------------------------------------------------------------------

def load_trajectories(path) -> list[Trajectory]:
    """Read runs from JSON Lines. One object per run:

        {"case_id": "go/gin-1", "harness": "octomind", "run": 0,
         "outcome": true, "tool_calls": ["read", "edit", "test"]}

    ``outcome`` must come from a checker the agent never saw. A verdict from
    an LLM judge measures the judge.

    ``tool_calls`` is optional and defaults to empty, which supports outcome
    agreement but not trajectory agreement. Compare a run that has no actions
    against another that has none and they score as an exact match, so a
    missing sequence is not the same as a matching one. Count how many runs
    arrived empty before reading a trajectory number.
    """
    import json
    from pathlib import Path

    out = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {e}") from e
        missing = {"case_id", "harness", "run", "outcome"} - set(d)
        if missing:
            raise ValueError(f"{path}:{line_no} is missing {sorted(missing)}")
        out.append(Trajectory(
            case_id=str(d["case_id"]),
            harness=str(d["harness"]),
            run=int(d["run"]),
            outcome=bool(d["outcome"]),
            tool_calls=tuple(str(c) for c in d.get("tool_calls", ())),
            checkpoints=tuple(str(c) for c in d.get("checkpoints", ())),
        ))
    if not out:
        raise ValueError(f"{path} contains no runs")
    return out


def save_trajectories(trajectories, path) -> None:
    """Write runs as JSON Lines, in the format load_trajectories reads."""
    import json
    from pathlib import Path

    lines = [json.dumps({
        "case_id": t.case_id, "harness": t.harness, "run": t.run,
        "outcome": t.outcome, "tool_calls": list(t.tool_calls),
    }, sort_keys=True) for t in trajectories]
    Path(path).write_text("\n".join(lines) + "\n")
