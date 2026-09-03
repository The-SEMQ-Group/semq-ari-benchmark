"""The two things an interval on ARI-E has to do.

It has to describe the number printed beside it, and it must not depend on how the runs
happened to be numbered. Both failed before, in ways that only appear with three or more
repeats or with unequal repeats between the harnesses, so both get a test at those shapes.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from ari.harness import Trajectory, harness_report  # noqa: E402


def build(spec):
    """spec: {harness: {run_id: [outcome per case]}} -> list[Trajectory]."""
    return [Trajectory(case_id=f"c{i}", harness=h, run=rid, outcome=bool(o))
            for h, rs in spec.items()
            for rid, outcomes in rs.items()
            for i, o in enumerate(outcomes)]


# A noisy harness on three runs. Run 1 agrees with run 3 more than with run 2, so the
# three pairings genuinely differ and picking one of them is visible.
A3 = {1: [1] * 12,
      2: [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
      3: [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]}
B2 = {1: [1] * 12, 2: [1] * 12}


def test_relabelling_runs_moves_neither_estimate_nor_interval():
    """Run numbering is arbitrary, so nothing reported may depend on it.

    Relabelling 2 and 3 rather than 1 and 2 matters: `combinations` yields the same
    unordered pair first either way, so swapping 1 and 2 hides the bug this guards.
    """
    original = harness_report(build({"A": A3, "B": B2}), n_resamples=4000, seed=0)
    relabelled = harness_report(
        build({"A": {1: A3[1], 2: A3[3], 3: A3[2]}, "B": B2}),
        n_resamples=4000, seed=0)

    a, b = original.self_consistency["A"], relabelled.self_consistency["A"]
    assert a.outcome_agreement == pytest.approx(b.outcome_agreement)
    assert a.outcome_agreement_ci == pytest.approx(b.outcome_agreement_ci)


def test_every_pairing_order_gives_the_same_answer():
    """Stronger form: all six relabellings of three runs agree."""
    from itertools import permutations

    seen = set()
    for perm in permutations([1, 2, 3]):
        spec = {"A": {i + 1: A3[p] for i, p in enumerate(perm)}, "B": B2}
        m = harness_report(build(spec), n_resamples=2000, seed=0).self_consistency["A"]
        seen.add((round(m.outcome_agreement, 12),
                  round(m.outcome_agreement_ci[0], 12),
                  round(m.outcome_agreement_ci[1], 12)))
    assert len(seen) == 1, f"run numbering changed the answer: {seen}"


def test_the_interval_describes_the_estimate_beside_it():
    """With three repeats, the estimate must sit inside its own interval."""
    rep = harness_report(build({"A": A3, "B": B2}), n_resamples=4000, seed=0)
    m = rep.self_consistency["A"]
    lo, hi = m.outcome_agreement_ci
    assert lo <= m.outcome_agreement <= hi


def test_asymmetric_repeats_keep_the_effect_inside_its_interval():
    """The 2-vs-3 case. This is what put the point estimate outside its own CI."""
    A = {1: [1] * 12, 2: [0] * 6 + [1] * 6, 3: [0] * 6 + [1] * 6}   # 3 runs, noisy
    B = {1: [1] * 12, 2: [1] * 12}                                   # 2 runs, clean
    rep = harness_report(build({"A": A, "B": B}), n_resamples=6000, seed=0)

    label = "A vs B"
    effect = rep.harness_effect[label]
    lo, hi = rep.harness_effect_ci[label]
    assert lo <= effect <= hi, f"{effect} outside ({lo}, {hi})"


def test_the_control_weights_the_two_harnesses_equally():
    """Not by how many runs each happens to have.

    A contributes three within-harness pairs and B contributes one. Weighting by pair
    count would pull the control toward A; the effect must reflect a 50/50 control.
    """
    A = {1: [1] * 12, 2: [0] * 6 + [1] * 6, 3: [0] * 6 + [1] * 6}
    B = {1: [1] * 12, 2: [1] * 12}
    rep = harness_report(build({"A": A, "B": B}), n_resamples=2000, seed=0)

    selves = [rep.self_consistency[h].outcome_agreement for h in ("A", "B")]
    cross = rep.cross["A vs B"].outcome_agreement
    fifty_fifty = float(np.mean(selves)) - cross
    by_pair_count = (selves[0] * 3 + selves[1] * 1) / 4 - cross

    assert rep.harness_effect["A vs B"] == pytest.approx(fifty_fifty, abs=1e-9)
    assert rep.harness_effect["A vs B"] != pytest.approx(by_pair_count, abs=1e-9)


def test_two_equal_repeats_still_behave():
    """The shape the published runs use. Guards against fixing one case by breaking it."""
    A = {1: [1] * 10, 2: [1] * 5 + [0] * 5}
    B = {1: [1] * 10, 2: [1] * 10}
    rep = harness_report(build({"A": A, "B": B}), n_resamples=2000, seed=0)

    assert rep.self_consistency["A"].outcome_agreement == pytest.approx(0.5)
    assert rep.self_consistency["B"].outcome_agreement == pytest.approx(1.0)
    lo, hi = rep.harness_effect_ci["A vs B"]
    assert lo <= rep.harness_effect["A vs B"] <= hi


def test_a_single_run_per_case_is_recorded_not_assumed():
    """Unchanged behaviour: no control means no effect, and a note saying so."""
    rep = harness_report(build({"A": {1: [1] * 8}, "B": {1: [1] * 8}}),
                         n_resamples=500, seed=0)
    assert rep.self_consistency == {}
    assert rep.harness_effect == {}
    assert any("unmeasured" in n for n in rep.notes)


def test_missing_cases_do_not_silently_reweight_the_effect():
    """A case only one harness covers is dropped from both sides, not counted on one."""
    A = {1: [1] * 12, 2: [0] * 6 + [1] * 6}
    B = {1: [1] * 12, 2: [1] * 12}
    runs = build({"A": A, "B": B})
    # Drop one case from B entirely.
    runs = [t for t in runs if not (t.harness == "B" and t.case_id == "c0")]
    rep = harness_report(runs, n_resamples=2000, seed=0)

    lo, hi = rep.harness_effect_ci["A vs B"]
    assert lo <= rep.harness_effect["A vs B"] <= hi
