"""Extent, quantized movement and location, and what they cost.

The fixtures put known movement at known coordinates so the reported
location and bounds can be checked against the truth rather than against
another run of the same code.
"""
import json

import numpy as np
import pytest

from ari.change_profile import (ChangeProfile, capabilities, profile,
                                reference_bytes, supported)
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


def _encode(ctx, X):
    return np.asarray(ctx.batch_encode(np.ascontiguousarray(X, np.float32)))


def _fixture(dim=256, n_bins=4, scale_max=1.0, moves=((3, 0.1, 0.9),)):
    """Vectors with movement placed at known coordinates.

    ``moves`` is ``(index, from_value, to_value)``. Everything else is
    identical between the two vectors, so any reported change that is not
    at one of those indices is a defect.
    """
    a = np.zeros((1, dim), np.float32)
    b = np.zeros((1, dim), np.float32)
    for idx, lo, hi in moves:
        a[0, idx], b[0, idx] = np.float32(lo), np.float32(hi)
    ctx = quant_context(dim, n_bins=n_bins, scale_max=scale_max)
    return ctx, a, b


# ------------------------------------------------------- extent, location


def test_location_is_reported_at_the_coordinate_that_moved():
    dim, n_bins = 256, 4
    ctx, a, b = _fixture(dim, n_bins, moves=((3, 0.1, 0.9), (200, -0.1, -0.9)))
    try:
        ca, cb = _encode(ctx, a), _encode(ctx, b)
        regions = ctx.quant_regions()
        sa = ctx.unpack_codes(ca, dim)
        sb = ctx.unpack_codes(cb, dim)
    finally:
        ctx.close()
    d = code_diff(ca, cb, n_bins=n_bins, dim=dim)
    assert d.changed_coordinates[0].tolist() == [3, 200]

    p = profile(d, n_bins=n_bins, group_size=64,
                ref_symbols=sa, cur_symbols=sb, regions=regions)
    assert p.coordinate_change_rate == pytest.approx(2 / dim)
    # blocks 0 and 3 hold coordinates 3 and 200
    assert p.group_change_counts == [1, 0, 0, 1]
    assert p.group_sizes == [64, 64, 64, 64]
    assert sum(p.group_change_counts) == 2


def test_a_short_final_block_keeps_its_own_denominator():
    dim, n_bins = 100, 4
    ctx, a, b = _fixture(dim, n_bins, moves=((99, 0.1, 0.9),))
    try:
        d = code_diff(_encode(ctx, a), _encode(ctx, b), n_bins=n_bins, dim=dim)
    finally:
        ctx.close()
    p = profile(d, n_bins=n_bins, group_size=32)
    assert p.group_sizes == [32, 32, 32, 4]
    assert p.group_change_counts == [0, 0, 0, 1]


# ------------------------------------------------------ quantized movement


def test_reported_bounds_contain_the_movement_that_was_placed():
    """Three placed moves, one of which the code cannot see.

    At n_bins=8 and scale_max=1 a region is 0.125 wide, so 0.4 -> 0.45
    stays inside one region and changes no symbol. The bounds must still
    contain it, and the profile must not count it as a change: that is
    the coordinate-level form of equality not meaning identical.
    """
    dim, n_bins, scale_max = 64, 8, 1.0
    visible = ((5, 0.1, 0.9), (17, -0.2, 0.2))
    within_region = (33, 0.4, 0.45)
    ctx, a, b = _fixture(dim, n_bins, scale_max, visible + (within_region,))
    try:
        ca, cb = _encode(ctx, a), _encode(ctx, b)
        regions = ctx.quant_regions()
        sa, sb = ctx.unpack_codes(ca, dim), ctx.unpack_codes(cb, dim)
    finally:
        ctx.close()
    d = regions.displacement(sa, sb)
    for idx, lo, hi in visible + (within_region,):
        true = abs(float(np.float32(hi)) - float(np.float32(lo)))
        assert d.min_abs_movement[0, idx] <= true <= d.max_abs_movement[0, idx]

    idx = within_region[0]
    assert sa[0, idx] == sb[0, idx]
    assert d.regions_moved[0, idx] == 0
    assert d.max_abs_movement[0, idx] >= 0.125      # a region width, not zero

    p = profile(code_diff(ca, cb, n_bins=n_bins, dim=dim), n_bins=n_bins,
                ref_symbols=sa, cur_symbols=sb, regions=regions)
    assert p.displacement["n_coordinates_moved"] == len(visible)
    assert p.displacement["n_crossed_zero"] == 1          # the -0.2 -> 0.2 move


def test_movement_is_omitted_rather_than_guessed_without_regions():
    dim, n_bins = 64, 4
    ctx, a, b = _fixture(dim, n_bins)
    try:
        d = code_diff(_encode(ctx, a), _encode(ctx, b), n_bins=n_bins, dim=dim)
    finally:
        ctx.close()
    p = profile(d, n_bins=n_bins)
    assert p.displacement is None
    assert "displacement" not in p.as_dict()


def test_the_canonical_probe_cannot_bound_many_of_its_own_changes():
    """n_bins=2 gives four regions, two of them outermost.

    Measured rather than assumed: this is the honest limit on the
    magnitude claim at the probe ARI actually ships.
    """
    dim = 384
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = X + rng.normal(scale=1e-3, size=X.shape).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    ctx = quant_context(dim, n_bins=2)
    try:
        ctx.calibrate(np.ascontiguousarray(X), percentile=0.99)
        regions = ctx.quant_regions()
        sa = ctx.unpack_codes(_encode(ctx, X), dim)
        sb = ctx.unpack_codes(_encode(ctx, Y), dim)
    finally:
        ctx.close()
    d = regions.displacement(sa, sb)
    moved = d.regions_moved > 0
    assert moved.any()
    unbounded = np.isinf(d.max_abs_movement[moved]).mean()
    # A third or more of detected changes have no upper bound at n_bins=2.
    assert unbounded > 0.3, unbounded
    # The lower bound is always available, which is the half that survives.
    assert np.isfinite(d.min_abs_movement).all()


# --------------------------------------------------------- what it costs


def test_reference_bytes_put_the_hash_far_below_the_code():
    b = reference_bytes(n_coordinates=1024, n_bins=2)
    assert b["semq_code"] == 256              # 1024 coords at 2 bits
    assert b["sha256"] == 32
    assert b["raw_fp32"] == 4096
    assert b["semq_code"] / b["sha256"] == 8.0


def test_capabilities_do_not_claim_the_hash_is_worse_at_detection():
    caps = capabilities()
    assert caps["sha256"]["detects_change"]
    assert not caps["sha256"]["locates_change"]
    assert caps["semq_code"]["locates_change"]
    assert not caps["semq_code"]["exact_movement"]
    assert caps["raw_fp32"]["exact_movement"]


# ------------------------------- serialization, independent recomputation


def test_profile_survives_a_json_round_trip():
    dim, n_bins = 128, 4
    ctx, a, b = _fixture(dim, n_bins, moves=((7, 0.1, 0.9),))
    try:
        ca, cb = _encode(ctx, a), _encode(ctx, b)
        regions = ctx.quant_regions()
        sa, sb = ctx.unpack_codes(ca, dim), ctx.unpack_codes(cb, dim)
    finally:
        ctx.close()
    p = profile(code_diff(ca, cb, n_bins=n_bins, dim=dim), n_bins=n_bins,
                ref_symbols=sa, cur_symbols=sb, regions=regions)
    restored = json.loads(json.dumps(p.as_dict()))
    assert restored == p.as_dict()
    assert isinstance(restored["group_change_counts"][0], int)


def test_the_profile_recomputes_from_the_stored_codes_alone():
    """A report is only checkable if its numbers come back from the codes."""
    dim, n_bins, scale_max = 128, 4, 1.0
    ctx, a, b = _fixture(dim, n_bins, scale_max, moves=((7, 0.1, 0.9), (70, -0.9, 0.1)))
    try:
        ca, cb = _encode(ctx, a), _encode(ctx, b)
        regions = ctx.quant_regions()
        first = profile(code_diff(ca, cb, n_bins=n_bins, dim=dim), n_bins=n_bins,
                        ref_symbols=ctx.unpack_codes(ca, dim),
                        cur_symbols=ctx.unpack_codes(cb, dim), regions=regions)
    finally:
        ctx.close()

    # Round-trip the codes through bytes, then rebuild from a fresh context
    # carrying only the declared settings.
    stored_a, stored_b = ca.tobytes(), cb.tobytes()
    again_ctx = quant_context(dim, n_bins=n_bins, scale_max=scale_max)
    try:
        ra = np.frombuffer(stored_a, np.uint8).reshape(ca.shape)
        rb = np.frombuffer(stored_b, np.uint8).reshape(cb.shape)
        second = profile(code_diff(ra, rb, n_bins=n_bins, dim=dim), n_bins=n_bins,
                         ref_symbols=again_ctx.unpack_codes(ra, dim),
                         cur_symbols=again_ctx.unpack_codes(rb, dim),
                         regions=again_ctx.quant_regions())
    finally:
        again_ctx.close()
    assert second.as_dict() == first.as_dict()


def test_group_summaries_cost_an_order_of_magnitude_less_than_indices():
    """Why the report carries blocks and not every index.

    The indices stay available on the comparison for a caller who wants
    them; serializing them into every condition of every report is what
    the block counts avoid. Measured at the canonical probe: roughly
    450-550 bytes per condition against 5-20 kB.
    """
    dim, n = 384, 200          # the count the recorded figures were measured at
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = X + rng.normal(scale=1e-3, size=X.shape).astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)

    ctx = quant_context(dim, n_bins=2)
    try:
        ctx.calibrate(np.ascontiguousarray(X), percentile=0.99)
        ca, cb = _encode(ctx, X), _encode(ctx, Y)
        regions = ctx.quant_regions()
        sa, sb = ctx.unpack_codes(ca, dim), ctx.unpack_codes(cb, dim)
    finally:
        ctx.close()
    d = code_diff(ca, cb, n_bins=2, dim=dim)
    p = profile(d, n_bins=2, group_size=64,
                ref_symbols=sa, cur_symbols=sb, regions=regions)

    block = len(json.dumps(p.as_dict()).encode())
    with_indices = len(json.dumps(
        {**p.as_dict(),
         "changed_coordinates": [i.tolist() for i in d.changed_coordinates]}
    ).encode())
    # The block is flat in the number of inputs; the index lists are not,
    # which is the whole reason the report carries one and not the other.
    assert block < 2000
    assert with_indices > 5 * block
