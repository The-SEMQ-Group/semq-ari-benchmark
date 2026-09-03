"""The reassociation probe must only ask the question it can answer.

Reversing the vocabulary axis reorders a reduction, which is the point for KL, JS and the
norms. For a statistic whose value is an index it permutes the index space instead, so
there is no reduction to reassociate and any verdict would really be about tie-breaking.
These tests pin that boundary, and pin that the guard is not a dead branch again.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINES = REPO_ROOT / "experiments" / "decoding-reproducibility" / "baselines.py"


@pytest.fixture(scope="module")
def bl():
    spec = importlib.util.spec_from_file_location("_baselines", BASELINES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The SEMQ statistic needs the SEMQ package, which CI does not have. It is
    # not what these tests are about, and skipping the whole file over it would
    # take the index-valued regression with it.
    try:
        import semq  # noqa: F401
    except ImportError:
        mod.STATS = {k: v for k, v in mod.STATS.items() if k != "SEMQ Hbar"}
    return mod


@pytest.fixture(scope="module")
def logits():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((24, 512)).astype(np.float32) * 3.0
    cur = ref + rng.standard_normal(ref.shape).astype(np.float32) * 1e-3
    return ref, cur


def test_index_valued_statistics_are_excluded_with_a_reason(bl, logits):
    out = bl.reassociation(*logits)
    for name in bl.INDEX_VALUED:
        assert name in out, f"{name} should appear, marked not applicable"
        assert out[name]["bit_identical"] is None
        assert out[name]["not_applicable"]


def test_index_valued_entries_carry_no_numeric_verdict(bl, logits):
    """The old code emitted a bit_identical for these, which read as a result."""
    out = bl.reassociation(*logits)
    for name in bl.INDEX_VALUED:
        assert "max_abs_diff" not in out[name]
        assert "max_rel_diff" not in out[name]


def test_reduction_statistics_still_get_a_real_verdict(bl, logits):
    out = bl.reassociation(*logits)
    for name in ("KL", "JS", "L2 |dlogit|", "max |dlogit|"):
        assert isinstance(out[name]["bit_identical"], bool)
        assert isinstance(out[name]["max_rel_diff"], float)


def test_the_guard_actually_guards(bl, logits):
    """Regression on the dead branch: both arms of the old `if` were identical.

    Emptying INDEX_VALUED must change the output. If it does not, the guard is
    decorative again.
    """
    out_guarded = bl.reassociation(*logits)
    original = bl.INDEX_VALUED
    try:
        bl.INDEX_VALUED = ()
        out_unguarded = bl.reassociation(*logits)
    finally:
        bl.INDEX_VALUED = original

    assert out_guarded != out_unguarded
    for name in original:
        assert out_guarded[name]["bit_identical"] is None
        assert isinstance(out_unguarded[name]["bit_identical"], bool)


def test_a_reversal_verdict_on_token_flip_would_have_been_about_ties(bl):
    """Why the exclusion is right, not merely tidy.

    With a deliberate tie, argmax picks the first maximum, so reversing the axis picks the
    other one and `token flip` changes. That is tie-breaking, not reassociation.
    """
    # Indices 1 and 3 of 8. Not symmetric about the centre, so reversing moves
    # which of the tied maxima comes first. (2 and 5 would reverse onto each
    # other and hide the effect.)
    ref = np.zeros((1, 8), dtype=np.float32)
    ref[0, 1] = ref[0, 3] = 1.0
    cur = ref.copy()

    forward = bl.stat_token_flip(ref, cur)
    rev = slice(None, None, -1)
    reversed_ = bl.stat_token_flip(np.ascontiguousarray(ref[:, rev]),
                                   np.ascontiguousarray(cur[:, rev]))
    # Identical inputs, so no flip either way: the statistic is fine. What moves
    # under reversal is which index argmax selects, which is what the old probe
    # would have been reporting on.
    assert forward == reversed_ == 0.0
    assert ref.argmax() != ref[:, rev].argmax()
