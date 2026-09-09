"""Pilot: score every detector on the legacy decoding cache.

Pilot, not confirmatory. The arrays are TinyLlama-1.1B on CPU over 12 prompts
(PROTOCOL.md §0.1-0.2), so the finest achievable prompt-level FPR is 1/12 and a
flawless control run bounds the FPR only below 26%. Nothing here can support
the 5% target. What it can do is exercise every scorer on real logits, produce
the storage frontier, and show which methods separate at all.

Controls are repeats at the same nominal configuration. Interventions are
declared single-axis changes. ARI disagreement never defines a positive.

  python run_pilot.py --cache <dir> --out results/pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scorers import (  # noqa: E402
    budget_ari, budget_margin_fp32, budget_projection, budget_raw_fp32,
    budget_sha256, budget_uniform_quant, canonical_bytes, coord_mismatch,
    cosine_distance, js_div, kl_div, margin_delta, max_abs_diff,
    random_projection_scorer, rel_l2, token_flip, topk_set_change,
)

# Same nominal configuration as the reference => control. Everything else is a
# declared single-axis intervention.
CONTROLS = ["proc"]
INTERVENTIONS = ["threads1", "batched", "bf16", "int8"]
QUANT_BINS_LOGIT = 8          # PROTOCOL §0.3: the logit probe, 4 bits/dim


def _ari_scores(r, c, bins):
    """Three distinct ARI quantities. PROTOCOL §0.4.

    The published score counts packed bytes; symbol mismatch and bit Hamming
    are what the coordinate-wise baselines are actually comparable against.
    """
    from ari.semq_compat import MAX_DIM, quant_context

    scale = float(np.percentile(np.abs(r), 99.0))
    dim = r.shape[1]
    bounds = list(range(0, dim, MAX_DIM)) + [dim]

    def enc(X):
        out = []
        for lo, hi in zip(bounds, bounds[1:]):
            ctx = quant_context(hi - lo, n_bins=bins, scale_max=scale)
            out.append(np.asarray(ctx.batch_encode(
                np.ascontiguousarray(X[:, lo:hi], np.float32))))
            ctx.close()
        return out[0] if len(out) == 1 else np.concatenate(out, axis=1)

    a, b = enc(r), enc(c)
    byte_mismatch = (a != b).mean(axis=1)
    bits_per_dim = int(round(a.shape[1] * 8 / dim))
    per_byte = 8 // bits_per_dim                     # coordinates packed per byte
    xor = np.bitwise_xor(a, b)
    hamming = np.unpackbits(xor, axis=1).sum(axis=1) / (dim * bits_per_dim)
    # Unpack to symbols so a changed coordinate counts once, not once per byte.
    sym_a = np.unpackbits(a, axis=1).reshape(len(a), -1, bits_per_dim)
    sym_b = np.unpackbits(b, axis=1).reshape(len(b), -1, bits_per_dim)
    symbol_mismatch = (sym_a != sym_b).any(axis=2).mean(axis=1)
    return {
        "ari_byte_mismatch": byte_mismatch,
        "ari_code_hamming": hamming,
        "ari_symbol_mismatch": symbol_mismatch,
    }, {"bits_per_dim": bits_per_dim, "coords_per_byte": per_byte, "scale": scale}


def _hash_scores(r, c, roundings):
    out = {"sha256_raw": np.array(
        [float(hashlib.sha256(canonical_bytes(a)).hexdigest()
               != hashlib.sha256(canonical_bytes(b)).hexdigest())
         for a, b in zip(r, c)])}
    for bits in roundings:
        step = (np.abs(r).max() * 2) / (2 ** bits)
        qa = np.round(r / step).astype(np.int64)
        qb = np.round(c / step).astype(np.int64)
        out[f"sha256_uniform_{bits}bit"] = (qa != qb).any(axis=1).astype(float)
    return out


def score_pair(r, c, proj):
    s = {
        "max_abs_diff": max_abs_diff(r, c),
        "rel_l2": rel_l2(r, c),
        "coord_mismatch": coord_mismatch(r, c),
        "cosine_distance": cosine_distance(r, c),
        "kl_fp64": kl_div(r, c),
        "js_fp64": js_div(r, c),
        "margin_delta": margin_delta(r, c),
        "token_flip": token_flip(r, c),
        "topk20_change": topk_set_change(r, c, k=20),
        "sketch_rp64": proj(r, c),
    }
    ari, meta = _ari_scores(r, c, QUANT_BINS_LOGIT)
    s.update(ari)
    s.update(_hash_scores(r, c, roundings=(4, 8)))
    return s, meta


def to_prompt(scores, lengths):
    """Mean over the steps of a prompt. Frozen aggregation, PROTOCOL §2."""
    out, i = [], 0
    for L in lengths:
        out.append(float(np.mean(scores[i:i + L])))
        i += L
    return np.array(out)


def tpr_at_fpr(ctrl, pos, nominal=0.05):
    """Threshold from controls at the conservative side of a tie."""
    if len(ctrl) == 0:
        return float("nan"), float("nan"), float("nan")
    thr = np.quantile(ctrl, 1.0 - nominal, method="higher")
    achieved = float((ctrl > thr).mean())
    tpr = float((pos > thr).mean())
    return tpr, achieved, float(thr)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=str(HERE / "results" / "pilot.json"))
    a = ap.parse_args()
    cache = Path(a.cache)

    ref = np.load(cache / "reference.npz", allow_pickle=True)
    lengths = [int(x) for x in ref["lengths"]]
    R = ref["logits"].astype(np.float64)
    dim = R.shape[1]
    proj = random_projection_scorer(dim, k=64, seed=0)

    report = {
        "provenance": {
            "cache": str(cache),
            "model_inferred": "TinyLlama/TinyLlama-1.1B-Chat-v1.0 (by elimination)",
            "device_inferred": "cpu",
            "vocab": dim,
            "n_prompts": len(lengths),
            "n_steps": int(R.shape[0]),
            "caveat": "legacy flat cache, no manifest; pilot only, not confirmatory",
        },
        "sizing": {
            "finest_fpr": 1.0 / len(lengths),
            "note": "12 prompts cannot support the 5% primary target",
        },
        "env": {"python": platform.python_version(), "numpy": np.__version__},
        "conditions": {},
    }

    per_cond = {}
    t0 = time.time()
    for name in CONTROLS + INTERVENTIONS:
        f = cache / f"{name}.npz"
        if not f.exists():
            report["conditions"][name] = {"status": "missing"}
            continue
        C = np.load(f, allow_pickle=True)["logits"].astype(np.float64)
        if C.shape != R.shape:
            report["conditions"][name] = {"status": "shape_mismatch", "shape": list(C.shape)}
            continue
        s, meta = score_pair(R, C, proj)
        per_cond[name] = {k: to_prompt(v, lengths) for k, v in s.items()}
        identical = bool(np.array_equal(R, C))
        report["conditions"][name] = {
            "status": "scored",
            "role": "control" if name in CONTROLS else "intervention",
            "arrays_identical": identical,
            "ari_meta": meta,
            "changed_steps": int((np.abs(R - C).max(axis=1) > 0).sum()),
        }
    report["scoring_seconds"] = round(time.time() - t0, 1)

    ctrl_name = CONTROLS[0]
    methods = sorted(per_cond[ctrl_name].keys()) if ctrl_name in per_cond else []
    report["detection"] = {}
    for m in methods:
        ctrl = per_cond[ctrl_name][m]
        row = {"control_mean": float(np.mean(ctrl)),
               "control_max": float(np.max(ctrl))}
        for iv in INTERVENTIONS:
            if iv not in per_cond:
                continue
            pos = per_cond[iv][m]
            tpr, ach, thr = tpr_at_fpr(ctrl, pos, 0.05)
            lo, hi = wilson(int(round(tpr * len(pos))), len(pos))
            row[iv] = {"tpr": tpr, "tpr_ci": [round(lo, 3), round(hi, 3)],
                       "achieved_fpr": ach, "threshold": thr,
                       "pos_mean": float(np.mean(pos))}
        report["detection"][m] = row

    report["storage_bytes"] = {
        "raw_fp32": budget_raw_fp32(dim).__dict__,
        "sha256": budget_sha256().__dict__,
        "ari_4bit": budget_ari(dim, 4).__dict__,
        "ari_2bit": budget_ari(dim, 2).__dict__,
        "uniform_4bit": budget_uniform_quant(dim, 4).__dict__,
        "uniform_8bit": budget_uniform_quant(dim, 8).__dict__,
        "margin_fp32": budget_margin_fp32().__dict__,
        "sketch_rp64": budget_projection(64).__dict__,
    }

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    npz_out = out.with_suffix(".scores.npz")
    np.savez_compressed(npz_out, **{f"{c}__{m}": v
                                    for c, d in per_cond.items() for m, v in d.items()})
    print(f"wrote {out} and {npz_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
