"""Extent, quantized movement and location of a representation change.

Hash equality answers one question: did anything change. A 32-byte
SHA-256 answers it for every change, at a fraction of the storage a code
costs, so equality is not where a code earns its size. What a hash cannot
report is how much of the vector moved, how far, and where. This module
reports those three, and reports what they cost.

Each quantity is kept separate and named for its unit, because they
answer different questions and are not interchangeable:

* **Extent** — the fraction of coordinates whose symbol changed, and the
  count per block of coordinates. Location, at whatever resolution the
  caller asks for.
* **Quantized movement** — how far a coordinate moved, in regions and as
  bounds on the value, from :mod:`semq.regions`.
* **Functional harm** — not measured here at all. A large movement in a
  coordinate nothing depends on may change no downstream result, and a
  one-region movement may change one.

The bounds are honest about what a code cannot say. The outermost region
on each side of the QUANT alphabet absorbs everything past the
calibration, so a coordinate there has no upper bound on its movement.
At the canonical probe (``n_bins=2``, four regions, two of them
outermost) that is a large share of the changes, and
``displacement.n_unbounded`` reports it rather than substituting a
region width that does not exist.

The report carries block counts rather than every changed index. At the
canonical probe over 200 inputs the profile serializes to 449 bytes at
dim 384 and 547 at dim 1024, against 5.2 kB and 20.3 kB with the full
index lists, and costs 6.4 ms and 12.7 ms to compute. The indices stay
on the comparison for a caller who wants them.

Requires a SEMQ build exposing ``Context.quant_regions``. Where it is
absent the profile is omitted rather than approximated, since a bound
this module cannot verify is worse than no bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ari.bound_quality import (BoundQuality, ThresholdResolution,
                               resolve_threshold, summarize_bounds)
from ari.code_metrics import CodeDiff, bits_per_coordinate

# Storage a monitor must retain per encoded unit, for the alternatives a
# code is worth comparing against. Serialized bytes, not theoretical.
SHA256_BYTES = 32


def supported(ctx: Any) -> bool:
    """Whether this SEMQ build can report quantized movement."""
    return hasattr(ctx, "quant_regions")


@dataclass(frozen=True)
class ChangeProfile:
    """What changed, how far it moved, and where."""

    n_coordinates: int
    bits_per_coordinate: int
    coordinate_change_rate: float
    group_size: int
    group_change_counts: list[int]
    group_sizes: list[int]
    displacement: dict[str, float | int] | None
    # Generic codec bounds and norm-constrained bounds are separate fields so
    # a declared assumption cannot reach a figure that did not declare it.
    bound_quality: BoundQuality | None = None
    norm_bound_quality: dict[str, Any] | None = None
    thresholds: list[ThresholdResolution] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_coordinates": self.n_coordinates,
            "bits_per_coordinate": self.bits_per_coordinate,
            "coordinate_change_rate": self.coordinate_change_rate,
            "group_size": self.group_size,
            "group_change_counts": self.group_change_counts,
            "group_sizes": self.group_sizes,
        }
        if self.displacement is not None:
            out["displacement"] = self.displacement
        if self.bound_quality is not None:
            out["bound_quality"] = self.bound_quality.as_dict()
        if self.norm_bound_quality is not None:
            out["norm_bound_quality"] = self.norm_bound_quality
        if self.thresholds:
            out["thresholds"] = [t.as_dict() for t in self.thresholds]
        return out


def reference_bytes(n_coordinates: int, n_bins: int) -> dict[str, int]:
    """Bytes per encoded unit for each detector, and what each can report.

    Counted as serialized, rounded up to whole bytes. The comparison is
    the point: a hash is far smaller and detects every change, so a code
    has to justify its size with what the hash cannot answer.
    """
    bits = bits_per_coordinate(n_bins)
    return {
        "semq_code": -(-n_coordinates * bits // 8),
        "sha256": SHA256_BYTES,
        "uniform_quant_8bit": n_coordinates,
        "raw_fp32": n_coordinates * 4,
    }


def capabilities() -> dict[str, dict[str, bool]]:
    """What each detector in :func:`reference_bytes` can report.

    Stated from the construction of each method, not measured. A hash
    detects any change and locates none; a quantized reference locates
    changes and bounds movement to its own resolution.
    """
    return {
        "semq_code": {"detects_change": True, "locates_change": True,
                      "bounds_movement": True, "exact_movement": False},
        "sha256": {"detects_change": True, "locates_change": False,
                   "bounds_movement": False, "exact_movement": False},
        "uniform_quant_8bit": {"detects_change": True, "locates_change": True,
                               "bounds_movement": True, "exact_movement": False},
        "raw_fp32": {"detects_change": True, "locates_change": True,
                     "bounds_movement": True, "exact_movement": True},
    }


def _mapping(obj: Any) -> dict[str, Any] | None:
    """Serialize a bounds object. The SDK spells it to_dict, this repo as_dict."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    for name in ("as_dict", "to_dict"):
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn()
    raise TypeError(
        f"norm_bounds needs as_dict() or to_dict(), got {type(obj).__name__}")


def profile(diff: CodeDiff, *, n_bins: int, group_size: int = 64,
            ref_symbols: np.ndarray | None = None,
            cur_symbols: np.ndarray | None = None,
            regions: Any = None,
            thresholds: tuple[float, ...] = (),
            norm_bounds: Any = None) -> ChangeProfile:
    """Build a profile from a comparison, and movement when regions are given.

    ``regions`` is a ``semq.regions.QuantRegions``. Without it the
    profile carries extent and location only, since movement cannot be
    bounded without the value intervals.
    """
    n = diff.n_coordinates
    n_groups = -(-n // group_size)
    counts = np.zeros(n_groups, dtype=np.int64)
    for idx in diff.changed_coordinates:
        if idx.size:
            np.add.at(counts, idx // group_size, 1)
    sizes = np.full(n_groups, group_size, dtype=np.int64)
    if n % group_size:
        sizes[-1] = n % group_size

    disp = quality = resolved = None
    if regions is not None:
        if ref_symbols is None or cur_symbols is None:
            raise ValueError("regions need ref_symbols and cur_symbols")
        d = regions.displacement(ref_symbols, cur_symbols)
        disp = d.summary()
        quality = summarize_bounds(d, over_changed_only=True)
        resolved = [resolve_threshold(d, t, over_changed_only=True)
                    for t in thresholds] or None

    return ChangeProfile(
        n_coordinates=n,
        bits_per_coordinate=bits_per_coordinate(n_bins),
        coordinate_change_rate=float(diff.coordinate_change_rate.mean()),
        group_size=group_size,
        group_change_counts=counts.tolist(),
        group_sizes=sizes.tolist(),
        displacement=disp,
        bound_quality=quality,
        norm_bound_quality=_mapping(norm_bounds),
        thresholds=resolved,
    )
