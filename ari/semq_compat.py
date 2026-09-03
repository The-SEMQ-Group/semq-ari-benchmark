"""One way to build a SEMQ quantizer context across SDK versions.

The magnitude-binning operator was renamed. Version 1.4.1, the newest wheel on
CodeArtifact, exposes it as ``SEMQ_OP_QBIN`` with ``qbin_n_bins`` and
``qbin_scale_max``. Later builds expose the same operator as ``SEMQ_OP_QUANT``
with ``quant_n_bins`` and ``quant_scale_max``.

An experiment written against one name raises ImportError on the other. That
happened on the first GPU run: the scripts were developed against a local
build and failed against the published wheel, after the encoding had already
finished. A reproducibility benchmark that only runs against its author's
working copy is not reproducible, so the compatibility lives here rather than
in each script.

``SEMQ_MAX_DIM`` is read the same way, because a caller that chunks a long
vector needs the limit and should not hard-code 65536.
"""

from __future__ import annotations

import semq

# The operator, under whichever name this SDK uses.
if hasattr(semq, "SEMQ_OP_QUANT"):
    QUANT_OP = semq.SEMQ_OP_QUANT
    _BINS_KW, _SCALE_KW = "quant_n_bins", "quant_scale_max"
elif hasattr(semq, "SEMQ_OP_QBIN"):
    QUANT_OP = semq.SEMQ_OP_QBIN
    _BINS_KW, _SCALE_KW = "qbin_n_bins", "qbin_scale_max"
else:  # pragma: no cover
    raise ImportError(
        "this SEMQ build exposes neither SEMQ_OP_QUANT nor SEMQ_OP_QBIN; "
        f"it has {[n for n in dir(semq) if n.startswith('SEMQ_OP')]}")

MAX_DIM = int(getattr(semq, "SEMQ_MAX_DIM", 65536))


def quant_context(max_dim: int, n_bins: int = 8, scale_max: float | None = None):
    """A Context on the magnitude-binning operator.

    Pass ``scale_max`` to fix the calibration scale instead of calling
    ``calibrate``. The two are equivalent: an explicit scale of
    ``percentile(abs(x), 99)`` gives codes bit-identical to
    ``calibrate(x, percentile=0.99)``. Fixing it explicitly is what lets one
    scale cover several chunks of a vector too long for a single context.
    """
    kwargs = {"max_dim": max_dim, "op": QUANT_OP, _BINS_KW: n_bins}
    if scale_max is not None:
        kwargs[_SCALE_KW] = float(scale_max)
    return semq.Context(**kwargs)


def sdk_version() -> str:
    return str(getattr(semq, "__version__", "unknown"))
