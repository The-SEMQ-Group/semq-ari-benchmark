# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""Scoring a probe too wide for one SEMQ row.

Modern vocabularies are wider than 65,536 logits, so the chunked path of the
ARI-D v0.2 logit probe is the one that has to hold.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("semq")

MODULE_PATH = Path(__file__).resolve().parents[1] / "run_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_pilot", MODULE_PATH)
run_pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_pilot)


def _pair(n=6, vocab=70000, seed=0):
    rng = np.random.default_rng(seed)
    r = (rng.standard_normal((n, vocab)) * 4).astype(np.float32)
    return r, r + (rng.standard_normal((n, vocab)) * 0.3).astype(np.float32)


def test_a_wide_vocabulary_uses_the_fixed_layout():
    r, c = _pair()
    scores, meta = run_pilot._ari_scores(r, c, 8)
    assert meta["chunk_widths"] == [65536, 4464]
    assert meta["n_chunks"] == 2
    assert meta["probe"] == "ARI-D-Logit-v0.2"
    for key in ("ari_byte_mismatch", "ari_code_hamming", "ari_symbol_mismatch"):
        assert scores[key].shape == (len(r),)
        assert (scores[key] > 0).all(), key


def test_identical_logits_score_zero():
    r, _ = _pair()
    scores, _ = run_pilot._ari_scores(r, r.copy(), 8)
    for key, value in scores.items():
        assert not value.any(), key


def test_the_byte_rate_is_not_smaller_than_the_symbol_rate():
    """Two symbols share a byte at 4 bits, so a byte rate cannot be smaller. PROTOCOL §0.4."""
    r, c = _pair()
    s, _ = run_pilot._ari_scores(r, c, 8)
    assert (s["ari_byte_mismatch"] >= s["ari_symbol_mismatch"] - 1e-12).all()
    assert (s["ari_code_hamming"] <= s["ari_symbol_mismatch"] + 1e-12).all()
