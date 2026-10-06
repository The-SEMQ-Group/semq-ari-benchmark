# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The one place the harness calls the public SEMQ SDK (``semq`` 1.x).

The harness requires semq 1.0.0 or newer within major version 1, the first
public release (https://github.com/The-SEMQ-Group/semq). Earlier private
builds (1.2–1.5) exposed a different interface — ``semq.Context``,
``calibrate``, ``scale_max`` — and are no longer supported; this module
refuses them below.

What changed with the public SDK, and why it matters for ARI:

* **No calibration.** Public quant bins magnitudes over the fixed range
  ``M = 2 / sqrt(dim)`` (``CodecConfig.max_magnitude``). There is no
  percentile calibration and no ``scale_max``: the range is a function of the
  dimension alone, so every party computes the same codes from the same
  vectors without sharing a scale. ARI v0.2 adopts this probe
  (spec/ari-canonical-v0.2.md).
* **Unit-norm input.** ``encode`` rejects rows whose squared norm is more than
  ``2^-10`` from 1. :func:`encode_packed` renormalises every row in binary64
  before casting to float32, the same deterministic step for every condition,
  so a provider that returns unnormalised vectors is still measurable.
* **Same row layout.** Quant symbols are ``sign * bins + bin`` packed
  least-significant-bit first, exactly as before, so ``ari.code_metrics`` and
  every consumer of packed codes is unchanged.
"""

from __future__ import annotations

import numpy as np
import semq

MIN_SDK = "1.0.0"

if not hasattr(semq, "Codec") or hasattr(semq, "Context"):  # pragma: no cover
    raise ImportError(
        f"semq {getattr(semq, '__version__', 'unknown')} is a pre-release build; "
        f"the ARI harness needs the public SDK, semq>={MIN_SDK},<2 "
        "(pip install semq).")

MAX_DIM = 65536  # SEMQ_MAX_DIM in include/semq.h


def quant_codec(dim: int, n_bins: int) -> semq.Codec:
    """The public quant codec: fixed range ``2 / sqrt(dim)``, ``n_bins`` magnitude bins."""
    return semq.Codec.quant(int(dim), int(n_bins))


def max_magnitude(dim: int) -> float:
    """The range quant bins over at ``dim``: ``(float)(2 / sqrt(dim))``.

    This replaces the calibrated scale ``s``. Reports keep recording it under
    ``fingerprint.s`` so the report schema is unchanged.
    """
    return float(semq.CodecConfig.quant(int(dim), 2).max_magnitude)


def unit_rows(X: np.ndarray) -> np.ndarray:
    """Rows rescaled to unit L2 norm (binary64 norm, float32 result).

    A zero row has no direction and cannot be encoded; it raises here rather
    than inside the SDK so the message names the row.
    """
    X = np.asarray(X)
    norms = np.linalg.norm(X.astype(np.float64), axis=1, keepdims=True)
    zero = np.flatnonzero(norms[:, 0] == 0)
    if zero.size:
        raise ValueError(f"row {int(zero[0])} is all zeros; it has no direction to encode")
    return np.ascontiguousarray(X / norms, dtype=np.float32)


def encode_packed(X: np.ndarray, n_bins: int) -> np.ndarray:
    """Encode ``X`` (n, dim) to packed quant rows, (n, bytes_per_vector) uint8, in input order."""
    X = unit_rows(X)
    codec = quant_codec(X.shape[1], n_bins)
    enc = codec.encode(X, ids=list(range(len(X))))
    # ids are 0..n-1 and an Encoding sorts by id, so rows come back in input order.
    return np.array(enc.rows, dtype=np.uint8, copy=True)


def unpack(packed: np.ndarray, dim: int, n_bins: int) -> np.ndarray:
    """The SDK's own unpacking of packed quant rows, (n, dim) symbols."""
    codec = quant_codec(dim, n_bins)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    enc = semq.Encoding(list(range(len(packed))), packed, codec.config)
    return np.asarray(codec.unpack(enc))


def sdk_version() -> str:
    return str(getattr(semq, "__version__", "unknown"))
