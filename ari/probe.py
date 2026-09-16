# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library that is subject to a
# commercial license owned by The SEMQ Group Inc. and is patent pending.
# The SDK is not covered by the Apache License.
"""SEMQ QBIN probe — the canonical ARI probe. **Codes must be bit-comparable across runs.**

Binds to the `semq` SDK. There is no substitute: the operators are distributed only in
that package, and this repository carries no reimplementation of them (CONTRIBUTING.md).
Without the SDK, `load_probe` raises and nothing here can produce a code.

See ../../spec/ari-canonical-v0.1.md.
"""
from __future__ import annotations

import numpy as np

N_BINS = 2
PERCENTILE = 99.0


class Probe:
    """Calibrated QBIN probe. `encode` maps an (n, d) float matrix to an (n, d)
    uint8 code matrix, deterministically."""

    backend = "abstract"

    def encode(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class _SemqProbe(Probe):
    """The canonical probe via the `semq` SDK (verified against semq==1.2.0).

    Flow: open a QBIN Context with n_bins=2, `calibrate(ref, 0.99)` to fix the scale `s`
    (it persists on the context and is returned), then `batch_encode` to bit-packed uint8
    codes. Discrete-attractor (`encode(reconstruct(c)) == c`, deterministic) holds.
    """

    backend = "semq"

    def __init__(self, ctx, s: float):
        self._ctx = ctx
        self.s = float(s)

    @classmethod
    def calibrate(cls, X: np.ndarray) -> "_SemqProbe":
        # Built through semq_compat, which resolves the operator name across SDK
        # versions. Constructing the Context here instead is what broke the
        # 2026-09-08 run: the published wheel renamed the operator to
        # SEMQ_OP_QUANT and this line still asked for SEMQ_OP_QBIN.
        from ari.semq_compat import quant_context

        dim = int(X.shape[1])
        ref = np.ascontiguousarray(X, dtype=np.float32)
        ctx = quant_context(max_dim=dim, n_bins=N_BINS)
        s = ctx.calibrate(ref, percentile=PERCENTILE / 100.0)
        return cls(ctx, s)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            self._ctx.batch_encode(np.ascontiguousarray(X, dtype=np.float32)), dtype=np.uint8
        )

    def close(self):
        if getattr(self, "_ctx", None) is not None:
            self._ctx.close()
            self._ctx = None

    def __del__(self):  # best-effort native cleanup
        try:
            self.close()
        except Exception:
            pass


def load_probe(calibration_vectors: np.ndarray) -> Probe:
    """Calibrate the canonical probe on `calibration_vectors`. Requires the `semq` SDK."""
    try:
        return _SemqProbe.calibrate(calibration_vectors)
    except ImportError as exc:
        raise RuntimeError(
            "the `semq` SDK is not installed and this repository has no substitute for it; "
            "see ari/README.md for how to obtain it"
        ) from exc


def fixed_scale_codes(vectors: np.ndarray, s: float, dim: int) -> np.ndarray:
    """Encode with a **fixed** QBIN scale `s` (no re-calibration), so codes are bit-comparable
    across separate runs / machines against a baseline calibrated once. This is the shared
    encode path for the capture tools (proc / conc / time / mach). Requires the `semq` SDK."""
    from ari.semq_compat import quant_context

    ctx = quant_context(max_dim=dim, n_bins=N_BINS, scale_max=s)
    try:
        return np.ascontiguousarray(
            ctx.batch_encode(np.ascontiguousarray(vectors, dtype=np.float32)), dtype=np.uint8)
    finally:
        ctx.close()
