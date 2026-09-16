"""Scoring a probe too wide for one context.

The pilot's logit probe is 32,000 coordinates and modern vocabularies are
wider still, so the chunking path is the one that has to hold.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("semq")

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_pilot", MODULE_PATH)
run_pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_pilot)


def _pair(n=24, dim=3072, seed=0):
    rng = np.random.default_rng(seed)
    r = (rng.standard_normal((n, dim)) * 4).astype(np.float32)
    return r, r + (rng.standard_normal((n, dim)) * 0.3).astype(np.float32)


@pytest.mark.parametrize("max_dim", [1536, 1024, 768, 512, 384])
def test_the_score_does_not_depend_on_how_the_probe_was_chunked(monkeypatch,
                                                                max_dim):
    """Chunking is an encoding detail. It must not move a reported number."""
    import ari.semq_compat as semq_compat

    r, c = _pair()
    whole, whole_meta = run_pilot._ari_scores(r, c, 8)
    assert whole_meta["n_chunks"] == 1

    monkeypatch.setattr(semq_compat, "MAX_DIM", max_dim)
    chunked, chunked_meta = run_pilot._ari_scores(r, c, 8)

    assert chunked_meta["n_chunks"] > 1
    # one calibration over every coordinate, not one per chunk
    assert chunked_meta["scale"] == whole_meta["scale"]
    for key, value in whole.items():
        assert np.array_equal(chunked[key], value), key


def test_calibration_is_global_and_not_per_chunk(monkeypatch):
    """Per-chunk calibration is the thing that does not compose.

    Coordinates are not exchangeable across a logit vector, so a chunk
    calibrated on its own slice gets a different scale and its codes stop
    being comparable with the next chunk's.
    """
    import ari.semq_compat as semq_compat
    from ari.semq_compat import quant_context

    # a probe whose second half is on a different scale from its first
    rng = np.random.default_rng(3)
    left = (rng.standard_normal((16, 512)) * 1.0).astype(np.float32)
    right = (rng.standard_normal((16, 512)) * 9.0).astype(np.float32)
    r = np.ascontiguousarray(np.hstack([left, right]))

    monkeypatch.setattr(semq_compat, "MAX_DIM", 512)
    _, meta = run_pilot._ari_scores(r, r.copy(), 8)
    assert meta["n_chunks"] == 2

    with quant_context(512, n_bins=8) as ctx:
        per_chunk_left = float(ctx.calibrate(left, percentile=0.99))
    # the global scale is pulled up by the wider half; a per-chunk scale is not
    assert meta["scale"] > per_chunk_left * 2


def test_chunk_width_divides_evenly_and_packs_whole_bytes():
    for dim, max_dim in ((128256, 65536), (256128, 65536), (3072, 1024),
                         (32000, 65536)):
        width = run_pilot._chunk_width(dim, max_dim, 2)
        assert width <= max_dim
        assert dim % width == 0, (dim, width)
        assert width % 2 == 0


def test_an_unchunkable_width_is_refused_rather_than_silently_padded():
    """A prime wider than the context has no equal, byte-aligned split."""
    with pytest.raises(ValueError, match="no chunk width"):
        run_pilot._chunk_width(65539, 65536, 2)     # 65539 is prime
