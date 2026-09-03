"""The ARI canonical probe (SEMQ QBIN n=2, 99th-percentile calibration).

At run time this binds to the `semq` SDK. When the SDK is not importable (local
dry-runs, CI), a deterministic reference **mock** stands in so the whole harness is
testable end-to-end. The mock is NOT the canonical probe — it only reproduces the
discrete-attractor behaviour (`encode(v) == encode(v)` bit-exact) needed to exercise
the pipeline. Real numbers require the SDK.

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
        import semq  # SEMQ SDK — private index today, public PyPI once the SDK ships

        dim = int(X.shape[1])
        ref = np.ascontiguousarray(X, dtype=np.float32)
        ctx = semq.Context(max_dim=dim, op=semq.SEMQ_OP_QBIN, qbin_n_bins=N_BINS)
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


class _MockProbe(Probe):
    """Deterministic quantile-bin reference. Global scale `s` at the 99th percentile
    of |component|; symbols from edges {-s, 0, +s} (4 symbols ≈ 2 bits/dim), matching
    the canonical n=2 setting's bit budget. Deterministic ⇒ discrete-attractor holds."""

    backend = "mock"

    def __init__(self, s: float):
        self.s = float(s)
        self._edges = np.array([-self.s, 0.0, self.s])

    @classmethod
    def calibrate(cls, X: np.ndarray) -> "_MockProbe":
        s = float(np.percentile(np.abs(X), PERCENTILE))
        return cls(s if s > 0 else 1.0)

    def encode(self, X: np.ndarray) -> np.ndarray:
        return np.digitize(X, self._edges).astype(np.uint8)


def load_probe(calibration_vectors: np.ndarray, backend: str = "auto") -> Probe:
    """Calibrate and return a probe. `backend`: 'auto' (semq if available, else mock),
    'semq' (require the SDK), or 'mock' (force the reference mock)."""
    if backend in ("auto", "semq"):
        try:
            return _SemqProbe.calibrate(calibration_vectors)
        except ImportError:
            if backend == "semq":
                raise RuntimeError(
                    "backend='semq' requested but the `semq` SDK is not installed. "
                    "Install it from the private index (pre-launch) or PyPI (post-launch)."
                )
    return _MockProbe.calibrate(calibration_vectors)


def fixed_scale_codes(vectors: np.ndarray, s: float, dim: int) -> np.ndarray:
    """Encode with a **fixed** QBIN scale `s` (no re-calibration), so codes are bit-comparable
    across separate runs / machines against a baseline calibrated once. This is the shared
    encode path for the capture tools (proc / conc / time / mach). Requires the `semq` SDK."""
    import semq

    ctx = semq.Context(max_dim=dim, op=semq.SEMQ_OP_QBIN, qbin_n_bins=N_BINS, qbin_scale_max=s)
    try:
        return np.ascontiguousarray(
            ctx.batch_encode(np.ascontiguousarray(vectors, dtype=np.float32)), dtype=np.uint8)
    finally:
        ctx.close()
