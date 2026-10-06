# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The ARI-D white-box logit probe, v0.2 (spec/arid-logit-probe-v0.2.md).

A logit vector is not an embedding: it is not unit-norm, it carries an offset
the softmax ignores, and a vocabulary can be wider than one SEMQ row (65,536).
The v0.1 probe handled all three with one calibrated scale shared by every
chunk. The public SDK has no shared scale and accepts only unit-norm rows, so
v0.2 defines the encoding as:

1. **Centre** each row over the full vocabulary (subtract its mean, binary64).
   Softmax is invariant to a constant shift, so the offset carries no
   information about the output distribution; centring removes it before it
   can dominate the normalisation.
2. **Chunk** at fixed boundaries, consecutive widths of at most 65,536
   (``ari.code_metrics.chunk_widths``). The layout depends on the vocabulary
   size alone.
3. **Encode** each chunk with ``semq_compat.encode_packed``, which rescales it
   to unit norm and applies QUANT with ``n_bins = 8`` over ``2 / sqrt(width)``.

Because each chunk is normalised on its own, codes depend on the chunk layout.
That is why the layout is fixed by the vocabulary size rather than left to the
caller: two runs of one model always share it, so they are comparable.
"""

from __future__ import annotations

import numpy as np

from ari.code_metrics import chunk_widths
from ari.semq_compat import MAX_DIM, encode_packed

N_BINS = 8
PROBE_ID = "ARI-D-Logit-v0.2"


def centre_rows(X: np.ndarray) -> np.ndarray:
    """Rows minus their mean over the full vocabulary (binary64 mean, float32 result)."""
    X64 = np.asarray(X, dtype=np.float64)
    return np.ascontiguousarray(X64 - X64.mean(axis=1, keepdims=True), dtype=np.float32)


def logit_widths(vocab: int) -> list[int]:
    """The fixed chunk layout for a vocabulary of ``vocab`` logits."""
    return chunk_widths(int(vocab), MAX_DIM)


def encode_logit_chunks(X: np.ndarray, n_bins: int = N_BINS) -> list[tuple[int, np.ndarray]]:
    """Encode logits (steps, vocab) as ``(width, packed codes)`` per chunk, unjoined.

    Each chunk pads its own final byte, so callers that compare coordinates
    should keep chunks apart or use ``chunked_code_diff`` with the widths.
    """
    C = centre_rows(X)
    out, off = [], 0
    for w in logit_widths(C.shape[1]):
        out.append((w, encode_packed(C[:, off:off + w], n_bins)))
        off += w
    return out


def encode_logits(X: np.ndarray, n_bins: int = N_BINS) -> tuple[np.ndarray, list[int]]:
    """Encode logits (steps, vocab) to concatenated packed codes and their chunk widths."""
    parts = encode_logit_chunks(X, n_bins)
    codes = parts[0][1] if len(parts) == 1 else np.concatenate([p for _, p in parts], axis=1)
    return codes, [w for w, _ in parts]
