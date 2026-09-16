"""How informative the movement bounds are, and against what denominator.

The load-bearing case is the canonical probe, where the bounds turn out to say
very little. That is a measurement of this probe under this intervention, not
a property of the method.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from ari.bound_quality import (cost_record, resolve_threshold,
                               summarize_bounds, timed)
from ari.change_profile import profile, reference_bytes, supported
from ari.code_metrics import code_diff

pytest.importorskip("semq")

from ari.semq_compat import quant_context  # noqa: E402


def _regions_available() -> bool:
    ctx = quant_context(8, n_bins=2, scale_max=1.0)
    try:
        return supported(ctx)
    finally:
        ctx.close()


pytestmark = pytest.mark.skipif(
    not _regions_available(),
    reason="this SEMQ build has no Context.quant_regions")


def regions_displacement(scale, **kw):
    regions, *_, sa, sb = _drifted(scale=scale, **kw)
    return regions.displacement(sa, sb)


def _drifted(dim=384, n=64, n_bins=2, scale=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = X + rng.normal(scale=scale, size=X.shape).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    ctx = quant_context(dim, n_bins=n_bins)
    try:
        ctx.calibrate(np.ascontiguousarray(X), percentile=0.99)
        regions = ctx.quant_regions()
        ca = np.asarray(ctx.batch_encode(np.ascontiguousarray(X)))
        cb = np.asarray(ctx.batch_encode(np.ascontiguousarray(Y)))
        sa = ctx.unpack_codes(ca, dim)
        sb = ctx.unpack_codes(cb, dim)
    finally:
        ctx.close()
    return regions, X, Y, ca, cb, sa, sb


# ------------------------------------------------------- the key invariant


@pytest.mark.parametrize("scale", [1e-3, 1e-2, 5e-2, 2e-1])
def test_a_positive_lower_bound_needs_a_move_of_two_or_more_regions(scale):
    """Adjacent regions touch, so a one-region move admits zero movement.

    Only a move that skips a whole region puts a floor under it. The two
    counts must agree exactly, not approximately.
    """
    regions, *_, sa, sb = _drifted(scale=scale)
    d = regions.displacement(sa, sb)
    q = summarize_bounds(d, over_changed_only=True)
    assert q.n_lower_positive == int((d.regions_moved >= 2).sum())


def test_the_canonical_probe_resolves_almost_nothing_at_realistic_drift():
    """A measurement of this probe and this drift, not a constant.

    At n_bins=2 under 1e-3 drift every detected change is a single-region
    move, so every lower bound is zero and no threshold can separate the
    changes. Recorded so the limitation is visible rather than implied.
    """
    regions, *_, sa, sb = _drifted(scale=1e-3)
    d = regions.displacement(sa, sb)
    q = summarize_bounds(d, over_changed_only=True)
    assert q.n_changed > 0
    assert q.n_lower_positive == 0
    assert (d.regions_moved[d.regions_moved > 0] == 1).all()
    r = resolve_threshold(d, 0.01, over_changed_only=True)
    assert r.n_definitely_below == 0 and r.n_definitely_above == 0
    assert r.n_unresolved == q.n_changed


# ----------------------------------------------------------- denominators


def test_both_denominators_are_reported_and_differ():
    regions, *_, sa, sb = _drifted(scale=1e-2)
    d = summarize_bounds(regions.displacement(sa, sb), over_changed_only=True).as_dict()
    assert d["n_coordinates"] > d["n_changed"] > 0
    # the same numerator against two denominators
    assert d["upper_finite_rate_of_all"] != d["upper_finite_rate_of_changed"]
    assert 0 <= d["upper_finite_rate_of_changed"] <= 1
    assert d["n_upper_finite"] + d["n_upper_unbounded"] == d["n_coordinates"]


def test_every_rate_over_changed_stays_within_its_denominator():
    """A rate over changed coordinates needs a numerator restricted to them.

    lower_positive_rate_of_changed divided an unrestricted count by n_changed.
    It reads correctly only while a positive lower bound implies a move, which
    is a property of the SDK rather than of this module.
    """
    for scale in (1e-2, 5e-2, 2e-1):
        q = summarize_bounds(regions_displacement(scale), over_changed_only=True)
        d = q.as_dict()
        assert q.n_lower_positive_changed <= q.n_changed
        assert q.n_upper_finite_changed <= q.n_changed
        for key in ("lower_positive_rate_of_changed",
                    "upper_finite_rate_of_changed"):
            assert 0.0 <= d[key] <= 1.0, (scale, key, d[key])


def test_the_width_quantiles_record_which_coordinates_they_came_from():
    """Restricting the widths changes their meaning, so the basis is emitted."""
    regions, *_, sa, sb = _drifted(scale=5e-2)
    d = regions.displacement(sa, sb)
    over_all = summarize_bounds(d).as_dict()
    over_changed = summarize_bounds(d, over_changed_only=True).as_dict()

    assert over_all["finite_width_basis"] == "all"
    assert over_changed["finite_width_basis"] == "changed"
    assert over_all["n_width_samples"] > over_changed["n_width_samples"] > 0
    assert over_all["finite_width_quantiles"] != over_changed["finite_width_quantiles"]


def test_no_change_gives_none_rather_than_zero_for_changed_rates():
    """Nothing changed is not the same as everything being uninformative."""
    regions, *_, sa, _ = _drifted(scale=1e-3)
    d = regions.displacement(sa, sa.copy())
    out = summarize_bounds(d, over_changed_only=True).as_dict()
    assert out["n_changed"] == 0
    assert out["lower_positive_rate_of_changed"] is None
    assert out["upper_finite_rate_of_changed"] is None
    assert out["lower_positive_rate_of_all"] == 0.0        # a real zero
    assert out["finite_width_quantiles"] == {}
    assert out["finite_width_basis"] == "changed"          # still stated


# ------------------------------------------------------------- thresholds


def test_an_unbounded_coordinate_is_never_definitely_below():
    regions, *_, sa, sb = _drifted(scale=2e-1)
    d = regions.displacement(sa, sb)
    unbounded = np.isinf(d.max_abs_movement)
    assert unbounded.any()
    huge = resolve_threshold(d, 1e6)
    # every unbounded coordinate must land in above-or-unresolved
    assert huge.n_definitely_below <= int((~unbounded).sum())


def test_threshold_counts_partition_the_coordinates():
    regions, *_, sa, sb = _drifted(scale=5e-2)
    d = regions.displacement(sa, sb)
    for t in (0.0, 1e-3, 0.05, 10.0):
        r = resolve_threshold(d, t)
        assert (r.n_definitely_below + r.n_definitely_above
                + r.n_unresolved) == d.regions_moved.size


def test_a_negative_threshold_is_rejected():
    regions, *_, sa, sb = _drifted()
    d = regions.displacement(sa, sb)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            resolve_threshold(d, bad)


# -------------------------------- generic and norm-constrained kept apart


def test_norm_bounds_are_reported_separately_from_generic_ones():
    regions, X, Y, ca, cb, sa, sb = _drifted(scale=1e-2)
    # bounded_displacement bounds one pair of vectors, so the norm block
    # describes a single input rather than the batch the generic block covers.
    #
    # float32 normalization lands within ~7.1e-8 of 1, never on it, so the
    # honest claim is a narrow interval rather than exact unit norm. Passing
    # 1.0 for the minimum is rejected, which is the point of the check.
    eps = 1e-6
    nb = regions.bounded_displacement(
        sa[0], sb[0], ref_norm_max=1.0 + eps, cur_norm_max=1.0 + eps,
        ref_norm_min=1.0 - eps, cur_norm_min=1.0 - eps,
        ref_vectors=X[0], cur_vectors=Y[0])
    p = profile(code_diff(ca, cb, n_bins=2, dim=384), n_bins=2,
                ref_symbols=sa, cur_symbols=sb, regions=regions,
                thresholds=(0.01,), norm_bounds=nb)
    out = p.as_dict()
    assert "bound_quality" in out and "norm_bound_quality" in out
    assert out["bound_quality"]["basis"] == "generic"
    # the norm block carries the assumption it rests on
    assert out["norm_bound_quality"]["assumptions"]


def test_an_exact_unit_claim_is_rejected_for_float32_normalized_input():
    """Normalizing in float32 does not produce norm 1.0 exactly."""
    regions, X, Y, _, _, sa, sb = _drifted(scale=1e-2)
    with pytest.raises(ValueError, match="norm_min"):
        regions.bounded_displacement(
            sa[0], sb[0], ref_norm_max=1.0, cur_norm_max=1.0,
            ref_norm_min=1.0, cur_norm_min=1.0,
            ref_vectors=X[0], cur_vectors=Y[0])


def test_norm_block_is_absent_when_no_assumption_was_declared():
    regions, _, _, ca, cb, sa, sb = _drifted(scale=1e-2)
    p = profile(code_diff(ca, cb, n_bins=2, dim=384), n_bins=2,
                ref_symbols=sa, cur_symbols=sb, regions=regions)
    assert p.as_dict().get("norm_bound_quality") is None


# ------------------------------- serialization and independent recomputation


def test_profile_with_bounds_survives_json_and_recomputes():
    regions, _, _, ca, cb, sa, sb = _drifted(scale=1e-2)

    def build():
        return profile(code_diff(ca, cb, n_bins=2, dim=384), n_bins=2,
                       ref_symbols=sa, cur_symbols=sb, regions=regions,
                       thresholds=(0.001, 0.05))
    first = build().as_dict()
    assert json.loads(json.dumps(first)) == first
    assert build().as_dict() == first          # deterministic from the codes
    assert [t["threshold"] for t in first["thresholds"]] == [0.001, 0.05]


# ------------------------------------------------------------------ cost


def test_cost_record_names_its_conditions():
    regions, _, _, ca, cb, sa, sb = _drifted(scale=1e-2)
    d = code_diff(ca, cb, n_bins=2, dim=384)
    p, seconds = timed(profile, d, n_bins=2, ref_symbols=sa,
                       cur_symbols=sb, regions=regions)
    rec = cost_record(n_inputs=64, n_coordinates=384, n_bins=2,
                      reference_bytes_per_unit=reference_bytes(384, 2)["semq_code"],
                      report_bytes=len(json.dumps(p.as_dict()).encode()),
                      seconds=seconds)
    assert rec["reference_state_bytes_per_unit"] == 96
    assert rec["reference_state_bytes_total"] == 96 * 64
    assert rec["report_bytes"] > 0
    assert rec["compute_seconds"] >= 0
    assert rec["machine"]["platform"] and rec["machine"]["numpy"]
    assert json.loads(json.dumps(rec)) == rec


# ----------------------------------------------- the schema does the enforcing


def _change_profile_schema() -> dict:
    """The change_profile subschema, wherever it sits under $defs."""
    root = json.loads(
        (Path(__file__).resolve().parents[1] / "spec" / "report-schema.json")
        .read_text())
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "change_profile" in node.get("properties", {}):
                return node["properties"]["change_profile"]
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(n for n in node if isinstance(n, (dict, list)))
    raise AssertionError("no change_profile in the report schema")


def test_a_profile_with_every_new_block_validates():
    jsonschema = pytest.importorskip("jsonschema")
    regions, X, Y, ca, cb, sa, sb = _drifted(scale=1e-2)
    eps = 1e-6
    nb = regions.bounded_displacement(
        sa[0], sb[0], ref_norm_max=1.0 + eps, cur_norm_max=1.0 + eps,
        ref_norm_min=1.0 - eps, cur_norm_min=1.0 - eps,
        ref_vectors=X[0], cur_vectors=Y[0])
    out = profile(code_diff(ca, cb, n_bins=2, dim=384), n_bins=2,
                  ref_symbols=sa, cur_symbols=sb, regions=regions,
                  thresholds=(0.01, 0.5), norm_bounds=nb).as_dict()
    assert {"bound_quality", "norm_bound_quality", "thresholds"} <= set(out)
    jsonschema.validate(out, _change_profile_schema())


def test_the_schema_rejects_a_rate_that_does_not_name_its_denominator():
    """The blocks are closed, so an unqualified rate is a validation error."""
    jsonschema = pytest.importorskip("jsonschema")
    regions, _, _, ca, cb, sa, sb = _drifted(scale=1e-2)
    out = profile(code_diff(ca, cb, n_bins=2, dim=384), n_bins=2,
                  ref_symbols=sa, cur_symbols=sb, regions=regions).as_dict()
    out["bound_quality"]["lower_positive_rate"] = 0.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(out, _change_profile_schema())


def test_the_schema_rejects_a_norm_block_with_no_declared_assumption():
    jsonschema = pytest.importorskip("jsonschema")
    regions, _, _, ca, cb, sa, sb = _drifted(scale=1e-2)
    out = profile(code_diff(ca, cb, n_bins=2, dim=384), n_bins=2,
                  ref_symbols=sa, cur_symbols=sb, regions=regions).as_dict()
    out["norm_bound_quality"] = {"l2_lower": 0.0, "l2_upper": 1.0,
                                 "assumptions": {}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(out, _change_profile_schema())


# ------------------------------------- the two spellings of "serialize me"


def test_norm_bounds_accept_either_serialization_spelling():
    """The SDK returns to_dict(); everything in this repo exposes as_dict()."""
    regions, _, _, ca, cb, sa, sb = _drifted(scale=1e-2)
    diff = code_diff(ca, cb, n_bins=2, dim=384)
    expected = {"l2_lower": 0.0, "l2_upper": 1.0, "assumptions": {"norm": "unit"}}

    class ViaAsDict:
        def as_dict(self):
            return expected

    class ViaToDict:
        def to_dict(self):
            return expected

    for obj in (ViaAsDict(), ViaToDict(), expected):
        out = profile(diff, n_bins=2, ref_symbols=sa, cur_symbols=sb,
                      regions=regions, norm_bounds=obj).as_dict()
        assert out["norm_bound_quality"] == expected

    with pytest.raises(TypeError, match="as_dict"):
        profile(diff, n_bins=2, ref_symbols=sa, cur_symbols=sb,
                regions=regions, norm_bounds=object())
