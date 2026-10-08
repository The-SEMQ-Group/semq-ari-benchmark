# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The ARI canonical probe (SEMQ QUANT n=2, fixed range 2/sqrt(dim)).

Binds to the public `semq` SDK (1.x). There is no substitute: the operators are
distributed only in that package, and this repository carries no reimplementation
of them (CONTRIBUTING.md). Without the SDK, `load_probe` raises and nothing here
can produce a code.

ARI v0.2 changed the probe: v0.1 calibrated its scale at the 99th percentile of a
reference set; v0.2 uses the public SDK's fixed range, a function of the dimension
alone. Codes from the two are not comparable. See ../spec/ari-canonical-v0.2.md.
"""
from __future__ import annotations

import numpy as np

N_BINS = 2


class Probe:
    """QUANT probe. `encode` maps an (n, d) float matrix to an (n, bytes_per_vector)
    uint8 packed code matrix, deterministically."""

    backend = "abstract"

    def encode(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class _SemqProbe(Probe):
    """The canonical probe via the public `semq` SDK.

    Rows are renormalised to unit length, then encoded with `Codec.quant(dim, 2)`.
    `s` is the fixed range `2 / sqrt(dim)`: it depends on nothing but `dim`, so two
    parties encoding the same vectors always share it.
    """

    backend = "semq"

    def __init__(self, dim: int):
        from ari import semq_compat

        self.dim = int(dim)
        self.s = semq_compat.max_magnitude(self.dim)
        self.version = semq_compat.sdk_version()

    def encode(self, X: np.ndarray) -> np.ndarray:
        from ari.semq_compat import encode_packed

        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {X.shape}")
        return encode_packed(X, N_BINS)


def load_probe(reference_vectors: np.ndarray) -> Probe:
    """The canonical probe for vectors shaped like `reference_vectors`. Requires the `semq` SDK.

    Only the dimension is read: the v0.2 probe has no calibration. The argument keeps the
    v0.1 call shape, so capture tools pass their baseline vectors as before.
    """
    try:
        return _SemqProbe(int(np.asarray(reference_vectors).shape[1]))
    except ImportError as exc:
        raise RuntimeError(
            "the `semq` SDK is not installed and this repository has no substitute for it; "
            "install it with `pip install semq` (see ari/README.md)"
        ) from exc


def fixed_scale_codes(vectors: np.ndarray, s: float, dim: int) -> np.ndarray:
    """Encode `vectors` against a stored baseline's scale `s`. Requires the `semq` SDK.

    The shared encode path for the capture tools (proc / conc / time / mach), which store
    `s` with each baseline. Under v0.2 `s` is fixed by `dim`, so a stored `s` that differs
    is a v0.1 calibrated baseline: its codes are not comparable with these, and this raises
    rather than silently comparing the two.
    """
    probe = _SemqProbe(dim)
    if not np.isclose(float(s), probe.s, rtol=0, atol=1e-9):
        raise ValueError(
            f"baseline scale s={float(s):.9g} is not the v0.2 range {probe.s:.9g} for dim {dim}; "
            "it was calibrated by the v0.1 probe. Recapture the baseline with this harness.")
    return probe.encode(vectors)
