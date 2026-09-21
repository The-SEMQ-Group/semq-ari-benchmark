# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""One way to build a SEMQ quantizer context across SDK versions.

The harness requires semq 1.5.1 or newer. ``Context.compare_codes``,
``bits_per_coordinate`` and ``quant_regions`` were added after 1.5.0, so their
presence is evidence of a new enough build, and the module refuses an older
build below.

The operator is ``SEMQ_OP_QUANT`` with ``quant_n_bins`` and
``quant_scale_max``. Version 1.4.1 named it ``SEMQ_OP_QBIN``; the guard above
refuses builds that old. The name is read here rather than in each script
because the first GPU run failed on it: the scripts were developed against a
local build and failed against the published wheel, after the encoding had
already finished.

``SEMQ_MAX_DIM`` is read the same way, because a caller that chunks a long
vector needs the limit and should not hard-code 65536.
"""

from __future__ import annotations

import semq

MIN_SDK = "1.5.1"
_REQUIRED = ("compare_codes", "bits_per_coordinate", "quant_regions")
_missing = [n for n in _REQUIRED if not hasattr(semq.Context, n)]
if _missing:
    raise ImportError(
        f"semq {getattr(semq, '__version__', 'unknown')} lacks Context."
        + ", Context.".join(_missing)
        + f"; the ARI harness needs semq>={MIN_SDK}. Install a newer wheel."
    )

if not hasattr(semq, "SEMQ_OP_QUANT"):  # pragma: no cover
    raise ImportError(
        "this SEMQ build does not expose SEMQ_OP_QUANT; "
        f"it has {[n for n in dir(semq) if n.startswith('SEMQ_OP')]}")
QUANT_OP = semq.SEMQ_OP_QUANT
_BINS_KW, _SCALE_KW = "quant_n_bins", "quant_scale_max"

MAX_DIM = int(getattr(semq, "SEMQ_MAX_DIM", 65536))


def quant_context(max_dim: int, n_bins: int = 8, scale_max: float | None = None):
    """A Context on the magnitude-binning operator.

    Pass ``scale_max`` to fix the calibration scale instead of calling
    ``calibrate``. Fixing it explicitly is what lets one scale cover several
    chunks of a vector too long for a single context.

    The two are **not** interchangeable. ``calibrate`` takes the percentile in
    the core in float32; ``numpy.percentile`` interpolates in float64, and the
    scales differ by up to 2e-4 relative at the 0.999 percentile. That moves
    coordinates sitting near a boundary across it: measured at dim 384, 60 of
    38,400 code bytes differ at ``n_bins=8``. Prefer ``calibrate`` — it is the
    definition every language binding shares — and treat a switch between the
    two as a change that needs results regenerated, not a refactor.
    """
    kwargs = {"max_dim": max_dim, "op": QUANT_OP, _BINS_KW: n_bins}
    if scale_max is not None:
        kwargs[_SCALE_KW] = float(scale_max)
    return semq.Context(**kwargs)


def sdk_version() -> str:
    return str(getattr(semq, "__version__", "unknown"))
