"""Detector scores over a reference/current vector pair, and their byte costs.

Every score takes two arrays of identical shape and returns one float per row,
so an alarm unit is a row and the aggregation to prompt level happens outside.
Higher means more different, for all of them, so a threshold is always an upper
bound and the calibration code never has to special-case direction.

Three rules the study depends on:

* Bytes are counted as serialized, rounded up, with metadata. A method that
  needs a scale, a seed or a codebook pays for it.
* ARI is scored three ways, because the published quantity counts packed bytes
  rather than symbols and those differ by the coordinates-per-byte factor. See
  PROTOCOL.md §0.4.
* KL and JS are computed in float64 log-space. The instability reported for
  them elsewhere was measured on a float32 implementation and is a property of
  that implementation, not of the statistic.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------
# canonical bytes
# --------------------------------------------------------------------------


def canonical_bytes(x: np.ndarray, dtype: str = "float32") -> bytes:
    """Bytes that identify a vector, with every ambiguity pinned.

    dtype, little-endian order and C-contiguity are fixed; negative zero is
    normalized to positive zero, since -0.0 and 0.0 compare equal but hash
    differently and a hash that disagrees with equality is a bug. Nonfinite
    values are rejected rather than hashed, because NaN != NaN would make the
    hash unreproducible for the same array.
    """
    a = np.ascontiguousarray(x, dtype=np.dtype(dtype).newbyteorder("<"))
    if not np.isfinite(a).all():
        raise ValueError("nonfinite value in vector; refusing to hash")
    a = a + 0.0  # -0.0 -> 0.0
    return a.tobytes()


def sha256_rows(x: np.ndarray, dtype: str = "float32") -> list[str]:
    return [hashlib.sha256(canonical_bytes(r, dtype)).hexdigest() for r in np.atleast_2d(x)]


# --------------------------------------------------------------------------
# vector distances
# --------------------------------------------------------------------------


def max_abs_diff(r, c):
    return np.abs(c - r).max(axis=1)


def rel_l2(r, c):
    """Relative L2. Zero-norm reference yields the absolute norm instead.

    Returning 0 or inf there would either hide a real change or saturate the
    score on a degenerate row.
    """
    num = np.linalg.norm(c - r, axis=1)
    den = np.linalg.norm(r, axis=1)
    out = np.empty_like(num)
    nz = den > 0
    out[nz] = num[nz] / den[nz]
    out[~nz] = num[~nz]
    return out


def coord_mismatch(r, c):
    return (c != r).mean(axis=1)


def cosine_distance(r, c):
    """1 - cosine. A zero-norm row has no direction; identical rows score 0."""
    nr = np.linalg.norm(r, axis=1)
    nc = np.linalg.norm(c, axis=1)
    dot = (r * c).sum(axis=1)
    out = np.empty(len(r), dtype=np.float64)
    ok = (nr > 0) & (nc > 0)
    out[ok] = 1.0 - dot[ok] / (nr[ok] * nc[ok])
    both_zero = (nr == 0) & (nc == 0)
    out[both_zero] = 0.0
    out[~ok & ~both_zero] = 1.0
    return np.clip(out, 0.0, 2.0)


# --------------------------------------------------------------------------
# logit-only statistics, float64 log-space
# --------------------------------------------------------------------------


def block_hash_rows(x: np.ndarray, n_blocks: int, digest_bytes: int,
                    dtype: str = "float32") -> np.ndarray:
    """Truncated SHA-256 per contiguous block, one row of digests per vector.

    The budget-matched answer to "a hash, but able to localise". Splitting a
    vector into blocks and truncating each digest to ``digest_bytes`` spends
    the same storage a code would and buys block-level location with it.
    Without this baseline the comparison hands localisation to the code for
    free.

    Truncation is a real weakening: a ``digest_bytes``-byte digest collides at
    roughly ``2**(-8 * digest_bytes)`` per block, which the caller accounts for
    rather than ignores.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be at least 1, got {n_blocks}")
    if not 1 <= digest_bytes <= 32:
        raise ValueError(f"digest_bytes must be in [1, 32], got {digest_bytes}")
    rows = np.atleast_2d(x)
    bounds = np.linspace(0, rows.shape[1], n_blocks + 1).astype(int)
    out = np.empty((rows.shape[0], n_blocks, digest_bytes), dtype=np.uint8)
    for i, row in enumerate(rows):
        for b, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
            digest = hashlib.sha256(canonical_bytes(row[lo:hi], dtype)).digest()
            out[i, b] = np.frombuffer(digest[:digest_bytes], dtype=np.uint8)
    return out


def block_hash_mismatch(r: np.ndarray, c: np.ndarray, n_blocks: int,
                        digest_bytes: int) -> np.ndarray:
    """Fraction of blocks whose digest differs, per vector."""
    a = block_hash_rows(r, n_blocks, digest_bytes)
    b = block_hash_rows(c, n_blocks, digest_bytes)
    return (a != b).any(axis=2).mean(axis=1)


def float16_roundtrip_delta(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Max absolute difference after a float16 round trip, per vector.

    The larger-storage accuracy anchor: two bytes per coordinate, against a
    fraction of one for the compressed methods. Not budget-matched, and bounds
    what the compressed methods give up.
    """
    a = np.asarray(r, dtype=np.float16).astype(np.float64)
    b = np.asarray(c, dtype=np.float16).astype(np.float64)
    return np.abs(b - a).max(axis=-1)


def _reject_non_distribution(name: str) -> None:
    raise ValueError(
        f"{name} is defined for probability distributions, so for the logit "
        "probe only. Applying it to signed embedding coordinates is undefined, "
        "not merely awkward; pass logits or use a vector distance."
    )


def _log_softmax64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kl_div(r, c):
    """KL(P_r || P_c) in nats, float64 log-space.

    Nonnegative by construction up to rounding; a negative value here would
    indicate a numerical fault rather than a property of the statistic.
    """
    lr, lc = _log_softmax64(r), _log_softmax64(c)
    return (np.exp(lr) * (lr - lc)).sum(axis=-1)


def js_div(r, c):
    """Jensen-Shannon divergence in nats, via a log-space mixture."""
    lr, lc = _log_softmax64(r), _log_softmax64(c)
    lm = np.logaddexp(lr, lc) - math.log(2.0)
    return 0.5 * (np.exp(lr) * (lr - lm)).sum(-1) + 0.5 * (np.exp(lc) * (lc - lm)).sum(-1)


def _top2_margin(x):
    p = np.partition(np.asarray(x, dtype=np.float64), -2, axis=-1)
    return p[..., -1] - p[..., -2]


def margin_delta(r, c):
    return np.abs(_top2_margin(c) - _top2_margin(r))


def token_flip(r, c):
    return (c.argmax(-1) != r.argmax(-1)).astype(np.float64)


def topk_set_change(r, c, k: int = 20):
    """1 - Jaccard over the top-k index sets. Ties broken by index order."""
    out = np.empty(len(r), dtype=np.float64)
    for i in range(len(r)):
        a = set(np.argsort(-r[i], kind="stable")[:k].tolist())
        b = set(np.argsort(-c[i], kind="stable")[:k].tolist())
        out[i] = 1.0 - len(a & b) / len(a | b)
    return out


# --------------------------------------------------------------------------
# sketch
# --------------------------------------------------------------------------


def random_projection_scorer(dim: int, k: int = 64, seed: int = 0):
    """Seeded Gaussian projection. The seed is part of its storage cost."""
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((dim, k)) / math.sqrt(k)

    def score(r, c):
        return np.linalg.norm((c - r) @ P, axis=1)

    score.meta = {"k": k, "seed": seed, "dim": dim}
    return score


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Budget:
    """Serialized reference bytes for one example, plus amortized shared state."""

    per_example: int
    shared: int
    note: str

    @property
    def total_at(self) -> Callable[[int], float]:
        return lambda n: self.per_example + self.shared / max(1, n)


def budget_raw_fp32(dim: int) -> Budget:
    return Budget(dim * 4, 0, "full fp32 reference vector")


def budget_sha256() -> Budget:
    return Budget(32, 0, "digest only; equality but no severity or location")


def budget_ari(dim: int, bits_per_dim: int) -> Budget:
    # Packed, rounded up to whole bytes, plus one float32 scale shared across
    # examples of the same model.
    return Budget(math.ceil(dim * bits_per_dim / 8), 4, f"{bits_per_dim}-bit codes + scale")


def budget_uniform_quant(dim: int, bits_per_dim: int) -> Budget:
    return Budget(math.ceil(dim * bits_per_dim / 8), 8, f"{bits_per_dim}-bit uniform + min/max")


def budget_margin_fp32() -> Budget:
    return Budget(4, 0, "one float32 top-2 margin")


def budget_block_hash(n_blocks: int, digest_bytes: int) -> Budget:
    return Budget(n_blocks * digest_bytes, 8,
                  f"{n_blocks} x {digest_bytes}-byte truncated digests + layout")


def budget_float16(dim: int) -> Budget:
    return Budget(dim * 2, 0, "float16 reference vector")


def budget_projection(k: int) -> Budget:
    # The projection is regenerable from the seed, so the matrix is not stored;
    # the seed and shape are.
    return Budget(k * 4, 12, "k float32 sketch + seed/shape")
