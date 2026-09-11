"""How informative the movement bounds are, separately from their existence.

:mod:`ari.change_profile` reports that a coordinate moved and bounds how far.
A lower bound of zero rules nothing out and a finite upper bound can be too
loose to act on, so the width of the interval is reported alongside it.

Every rate names its denominator. ``_of_all`` covers every coordinate
compared; ``_of_changed`` only those whose symbol changed. Rates over changed
coordinates are ``None`` when nothing changed, not zero.

Generic codec bounds and norm-constrained bounds are kept apart: the generic
bounds hold for any input, the norm-constrained ones only under a declared
assumption about vector norms (see ``QuantRegions.bounded_displacement``).

Nothing here measures functional harm. A threshold on movement is a threshold
on the representation.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BoundQuality", "ThresholdResolution", "cost_record",
           "resolve_threshold", "summarize_bounds", "timed"]


def _rate(num: int, den: int) -> float | None:
    """A rate, or None when the denominator is empty."""
    return float(num / den) if den else None


@dataclass(frozen=True)
class BoundQuality:
    """Informativeness of the per-coordinate movement bounds.

    Counts are absolute; every rate states which denominator it used.
    ``n_upper_unbounded`` stays a count because the fraction it implies
    depends on which denominator the reader wants.
    """

    n_coordinates: int
    n_changed: int
    n_lower_positive: int
    n_lower_positive_changed: int
    n_upper_finite: int
    n_upper_finite_changed: int
    n_upper_unbounded: int
    width_quantiles: dict[str, float]
    n_width_samples: int
    width_basis: str
    basis: str = "generic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "n_coordinates": self.n_coordinates,
            "n_changed": self.n_changed,
            "n_lower_positive": self.n_lower_positive,
            "n_upper_finite": self.n_upper_finite,
            "n_upper_unbounded": self.n_upper_unbounded,
            # over every coordinate compared
            "lower_positive_rate_of_all": _rate(self.n_lower_positive,
                                                self.n_coordinates),
            "upper_finite_rate_of_all": _rate(self.n_upper_finite,
                                              self.n_coordinates),
            # over the coordinates that actually changed
            "lower_positive_rate_of_changed": _rate(
                self.n_lower_positive_changed, self.n_changed),
            "upper_finite_rate_of_changed": _rate(self.n_upper_finite_changed,
                                                  self.n_changed),
            "finite_width_quantiles": dict(self.width_quantiles),
            "finite_width_basis": self.width_basis,
            "n_width_samples": self.n_width_samples,
        }


def summarize_bounds(displacement: Any, *, over_changed_only: bool = False,
                     quantiles: tuple[float, ...] = (0.5, 0.9, 0.99),
                     basis: str = "generic") -> BoundQuality:
    """Summarize how informative a :class:`semq.regions.Displacement` is.

    ``over_changed_only`` restricts the width quantiles to coordinates whose
    symbol changed, and is recorded as ``width_basis`` so the quantiles are
    not read against the wrong denominator. The counts always cover every
    coordinate, so both denominators stay recoverable either way.
    """
    moved = np.asarray(displacement.regions_moved) > 0
    lo = np.asarray(displacement.min_abs_movement, dtype=np.float64)
    hi = np.asarray(displacement.max_abs_movement, dtype=np.float64)
    finite = np.isfinite(hi)
    positive = lo > 0

    width_mask = finite & moved if over_changed_only else finite
    widths = hi[width_mask] - lo[width_mask]
    qs = {f"p{int(q * 100)}": float(np.quantile(widths, q)) for q in quantiles} \
        if widths.size else {}

    return BoundQuality(
        n_coordinates=int(lo.size),
        n_changed=int(moved.sum()),
        n_lower_positive=int(positive.sum()),
        n_lower_positive_changed=int((positive & moved).sum()),
        n_upper_finite=int(finite.sum()),
        n_upper_finite_changed=int((finite & moved).sum()),
        n_upper_unbounded=int((~finite).sum()),
        width_quantiles=qs,
        n_width_samples=int(widths.size),
        width_basis="changed" if over_changed_only else "all",
        basis=basis,
    )


@dataclass(frozen=True)
class ThresholdResolution:
    """How a movement threshold resolves against the bounds.

    A coordinate is *definitely below* when even its largest possible
    movement is under the threshold, *definitely above* when even its
    smallest possible movement is over, and *unresolved* when the interval
    straddles it. An unbounded upper bound can never be definitely below.

    This is a threshold on the representation, not on downstream behaviour.
    """

    threshold: float
    n_definitely_below: int
    n_definitely_above: int
    n_unresolved: int
    basis: str

    def as_dict(self) -> dict[str, Any]:
        total = self.n_definitely_below + self.n_definitely_above + self.n_unresolved
        return {
            "threshold": self.threshold,
            "basis": self.basis,
            "n_definitely_below": self.n_definitely_below,
            "n_definitely_above": self.n_definitely_above,
            "n_unresolved": self.n_unresolved,
            "definitely_below_rate": _rate(self.n_definitely_below, total),
            "definitely_above_rate": _rate(self.n_definitely_above, total),
            "unresolved_rate": _rate(self.n_unresolved, total),
        }


def resolve_threshold(displacement: Any, threshold: float, *,
                      over_changed_only: bool = False) -> ThresholdResolution:
    """Split coordinates by where ``threshold`` falls in their interval."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError(
            f"threshold must be finite and non-negative, got {threshold}")
    lo = np.asarray(displacement.min_abs_movement, dtype=np.float64)
    hi = np.asarray(displacement.max_abs_movement, dtype=np.float64)
    if over_changed_only:
        keep = np.asarray(displacement.regions_moved) > 0
        lo, hi = lo[keep], hi[keep]

    below = hi < threshold
    above = lo > threshold
    return ThresholdResolution(
        threshold=float(threshold),
        n_definitely_below=int(below.sum()),
        n_definitely_above=int(above.sum()),
        n_unresolved=int((~below & ~above).sum()),
        basis="changed" if over_changed_only else "all",
    )


def cost_record(*, n_inputs: int, n_coordinates: int, n_bins: int,
                reference_bytes_per_unit: int, report_bytes: int,
                seconds: float) -> dict[str, Any]:
    """Storage and runtime, each named, with the conditions they hold under.

    A byte count or a timing means nothing without the input count and the
    machine it was taken on, so both travel with the numbers.
    """
    return {
        "n_inputs": int(n_inputs),
        "n_coordinates": int(n_coordinates),
        "n_bins": int(n_bins),
        "reference_state_bytes_per_unit": int(reference_bytes_per_unit),
        "reference_state_bytes_total": int(reference_bytes_per_unit) * int(n_inputs),
        "report_bytes": int(report_bytes),
        "compute_seconds": float(seconds),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    """Run ``fn`` and return its result with the elapsed seconds."""
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start
