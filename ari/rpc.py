"""Read access to `results_per_condition` entries, across both report shapes.

v0.1 reports carried a single implicit result per condition:
    {"HER": 0.86, "Hbar": 1.2, "HER_ci": [0.84, 0.88]}

The multi-detector shape nests one or more named detectors under `detectors`, each
carrying its own result plus `criterion_type` / `bytes_of_reference_state` / `unit`:
    {"detectors": {"semq": {"HER": 0.86, "Hbar": 1.2, ...}}}

Everything that reads a condition's numbers (the scorer, the leaderboard builder, ad-hoc
tools) should go through here instead of indexing the dict directly, so it works on old
and migrated submissions alike. `semq` is the implicit detector for the legacy shape.
"""
from __future__ import annotations

import math

DEFAULT_DETECTOR = "semq"

# QBIN n_bins=2 packs ~2 bits per embedding dimension (spec/ari-canonical-v0.1.md §1:
# "n_bins = 2 (~2 bits per embedding dimension / 4 symbols after calibration)"). This is the
# bit budget of one SEMQ code, independent of any particular submission.
SEMQ_BITS_PER_DIM = 2.0


def detector(entry: dict | None, name: str = DEFAULT_DETECTOR) -> dict | None:
    """The named detector's result dict from one `results_per_condition[cond]` entry.

    Handles both shapes: if `entry` has a `detectors` map, look the name up there;
    otherwise treat `entry` itself as the (legacy, implicitly-`semq`) result."""
    if not entry:
        return None
    if "detectors" in entry:
        return entry["detectors"].get(name)
    if name == DEFAULT_DETECTOR:
        return entry
    return None


def her(entry: dict | None, name: str = DEFAULT_DETECTOR) -> float | None:
    d = detector(entry, name)
    return d.get("HER") if d else None


def hbar(entry: dict | None, name: str = DEFAULT_DETECTOR) -> float | None:
    d = detector(entry, name)
    return d.get("Hbar") if d else None


def her_ci(entry: dict | None, name: str = DEFAULT_DETECTOR) -> list | None:
    d = detector(entry, name)
    return d.get("HER_ci") if d else None


def semq_bytes_of_reference_state(dim: int) -> int:
    """Bytes of packed SEMQ-code state that must be retained per item to evaluate the
    `semq` detector on it, derived from the encoder's dimensionality (never hardcoded per
    submission) at the frozen probe's fixed bit budget (`SEMQ_BITS_PER_DIM`)."""
    return math.ceil(dim * SEMQ_BITS_PER_DIM / 8)
