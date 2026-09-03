"""Tests for the ARI-E harness metric.

The metric exists to answer one question: did the scaffold decide the outcome,
or did the agent's own stochasticity? Every test here targets a way that
question can be answered wrongly.

The cases that matter most are the ones where a naive metric reports a large
harness effect that is not real:

  * two harnesses that disagree only as much as one harness disagrees with
    itself (the effect must be 0, not the raw disagreement rate)
  * a harness with one run per case, where self-consistency is unknown and
    must not be assumed perfect
"""

from __future__ import annotations

import random

import pytest

from ari.harness import (
    Trajectory,
    common_prefix,
    compare_runs,
    format_report,
    harness_report,
    trajectory_similarity,
)


def traj(case, harness, run, outcome, calls=()):
    return Trajectory(case_id=case, harness=harness, run=run,
                      outcome=outcome, tool_calls=tuple(calls))


# ---------------------------------------------------------------------------
# Trajectory comparison
# ---------------------------------------------------------------------------

def test_identical_trajectories_score_one():
    a = ("read", "edit", "test")
    assert trajectory_similarity(a, a) == 1.0
    assert common_prefix(a, a) == 3


def test_two_empty_trajectories_are_identical():
    # A harness that took no action twice did reproduce itself.
    assert trajectory_similarity((), ()) == 1.0


def test_disjoint_trajectories_score_zero():
    assert trajectory_similarity(("a", "b"), ("c", "d")) == 0.0


def test_prefix_stops_at_first_difference():
    # Agents compound: what follows a divergence is a consequence, not an
    # independent disagreement. A later match must not extend the prefix.
    assert common_prefix(("a", "b", "c"), ("a", "x", "c")) == 1


def test_similarity_is_symmetric_and_length_normalised():
    a, b = ("a", "b", "c", "d"), ("a", "b")
    assert trajectory_similarity(a, b) == trajectory_similarity(b, a)
    assert 0.0 < trajectory_similarity(a, b) < 1.0


# ---------------------------------------------------------------------------
# The control: a metric that ignores self-consistency overstates the effect
# ---------------------------------------------------------------------------

def _noisy_harness(name, n=20, flaky=(), runs=2, seed=0):
    """Several runs of one harness, flaky on the named cases.

    A flaky case draws its outcome independently on every run, which is how
    agent stochasticity actually behaves. Making the runs correlated would
    hide the very effect the control is meant to remove.
    """
    rng = random.Random(f"{name}-{seed}")
    out = []
    for r in range(runs):
        for i in range(n):
            c = f"case{i}"
            out.append(traj(c, name, r,
                            rng.random() < 0.5 if c in flaky else True))
    return out


def test_interchangeable_harnesses_give_zero_effect():
    """Two harnesses that always agree must score exactly 0."""
    runs = _noisy_harness("h1") + _noisy_harness("h2")
    rep = harness_report(runs, n_resamples=200)
    assert rep.cross["h1 vs h2"].outcome_agreement == 1.0
    assert rep.harness_effect["h1 vs h2"] == pytest.approx(0.0)


def test_effect_discounts_agent_stochasticity():
    """The headline case.

    Both harnesses are flaky on the same 8 of 40 cases and identical
    otherwise. They disagree with each other, but only as often as each
    disagrees with itself. A metric reading raw cross-harness disagreement
    would report a harness effect around 0.10. The attributable effect is 0.
    """
    flaky = {f"case{i}" for i in range(8)}
    runs = (_noisy_harness("h1", n=40, flaky=flaky, runs=6, seed=1)
            + _noisy_harness("h2", n=40, flaky=flaky, runs=6, seed=2))
    rep = harness_report(runs, n_resamples=200)

    # Both harnesses are genuinely unreliable, and the metric says so.
    assert rep.self_consistency["h1"].outcome_agreement < 0.95
    # The raw cross-harness number looks like a harness difference.
    assert rep.cross["h1 vs h2"].outcome_agreement < 0.95
    # After the control, none of it is attributable to the harness.
    assert rep.harness_effect["h1 vs h2"] == pytest.approx(0.0, abs=0.03)


def test_real_harness_difference_is_detected():
    """A harness that genuinely fails cases the other passes scores positive."""
    h1 = [t for i in range(20)
          for t in (traj(f"case{i}", "h1", 0, True),
                    traj(f"case{i}", "h1", 1, True))]
    # h2 is self-consistent but fails a quarter of the cases outright.
    h2 = [t for i in range(20)
          for t in (traj(f"case{i}", "h2", 0, i >= 5),
                    traj(f"case{i}", "h2", 1, i >= 5))]
    rep = harness_report(h1 + h2, n_resamples=200)

    assert rep.self_consistency["h1"].outcome_agreement == 1.0
    assert rep.self_consistency["h2"].outcome_agreement == 1.0
    assert rep.harness_effect["h1 vs h2"] == pytest.approx(0.25)


def test_single_run_per_harness_refuses_to_assume_perfection():
    """With one run per case the control is unmeasurable.

    The metric must say so rather than default self-consistency to 1.0, which
    would charge all agent stochasticity to the harness.
    """
    runs = [traj(f"case{i}", "h1", 0, True) for i in range(10)]
    runs += [traj(f"case{i}", "h2", 0, i % 2 == 0) for i in range(10)]
    rep = harness_report(runs, n_resamples=200)

    assert rep.self_consistency == {}
    assert "h1 vs h2" not in rep.harness_effect
    assert any("lower bound" in n or "not computed" in n for n in rep.notes)
    # The raw agreement is still reported, because it is still a measurement.
    assert rep.cross["h1 vs h2"].outcome_agreement == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Outcome agreement alone can hide a difference
# ---------------------------------------------------------------------------

def test_same_verdict_by_different_route_shows_in_trajectory():
    """Two harnesses both pass every case, by wholly different actions.

    Outcome agreement is 1.0 and says the harnesses are interchangeable.
    Trajectory agreement is 0.0 and says they are not. Reporting only the
    outcome would hide this.
    """
    h1 = [traj(f"case{i}", "h1", r, True, ("read", "edit", "test"))
          for i in range(10) for r in (0, 1)]
    h2 = [traj(f"case{i}", "h2", r, True, ("search", "patch", "verify"))
          for i in range(10) for r in (0, 1)]
    rep = harness_report(h1 + h2, n_resamples=200)

    cross = rep.cross["h1 vs h2"]
    assert cross.outcome_agreement == 1.0
    assert cross.trajectory_agreement == 0.0
    assert cross.mean_first_divergence == 0.0
    assert rep.harness_effect["h1 vs h2"] == pytest.approx(0.0)


def test_divergence_step_locates_where_harnesses_split():
    h1 = [traj(f"case{i}", "h1", r, True, ("read", "edit", "test", "done"))
          for i in range(6) for r in (0, 1)]
    h2 = [traj(f"case{i}", "h2", r, True, ("read", "edit", "lint", "done"))
          for i in range(6) for r in (0, 1)]
    rep = harness_report(h1 + h2, n_resamples=200)
    assert rep.cross["h1 vs h2"].mean_first_divergence == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------

def test_no_shared_cases_raises():
    left = [traj("a", "h1", 0, True)]
    right = [traj("b", "h2", 0, True)]
    with pytest.raises(ValueError, match="no shared cases"):
        compare_runs(left, right, "h1 vs h2")


def test_unequal_case_sets_intersect_rather_than_fail():
    left = [traj(f"case{i}", "h1", 0, True) for i in range(5)]
    right = [traj(f"case{i}", "h2", 0, True) for i in range(3)]
    m = compare_runs(left, right, "h1 vs h2", n_resamples=100)
    assert m.n_cases == 3


def test_report_formats_without_a_control():
    runs = [traj("a", "h1", 0, True), traj("a", "h2", 0, False)]
    text = format_report(harness_report(runs, n_resamples=50))
    assert "not computable" in text
    assert "Self-consistency" in text


def test_three_harnesses_produce_all_pairs():
    runs = []
    for h in ("a", "b", "c"):
        runs += _noisy_harness(h, n=5)
    rep = harness_report(runs, n_resamples=100)
    assert set(rep.cross) == {"a vs b", "a vs c", "b vs c"}
