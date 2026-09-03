# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""Does SEMQ beat the cheap alternatives at the decoding layer?

The ARI-D result shows SEMQ detects a precision change at steps where the
emitted token is unchanged. That is only interesting if a simpler statistic
does not do the same job. KL divergence over the logprob distribution, or the
top-2 margin, are both obvious and much cheaper to compute.

This asks the question directly, on three axes:

  1. Sensitivity. Perturb reference logits by a controlled sigma and see which
     statistic responds first and most steeply. A monitor is only useful in
     the regime where it separates signal from its own noise floor, so the
     number that matters is response per unit of noise, not raw response.

  2. Reference storage. A monitor has to keep something from the reference run
     to compare against. Some statistics need the whole distribution; SEMQ
     needs a code; the token-level baseline needs one integer.

  3. Reassociation stability. Whether the statistic is reproducible bit-exactly
     when computed two algebraically equivalent ways. This is the same question
     the probe-choice analysis (docs/analysis/probe-validation.md) asks about
     probes, applied to the monitors themselves. It
     matters for attestation -- publishing a value a third party recomputes --
     and NOT for detection, where any of these would be thresholded anyway.

Run after run_matrix.py, which produces the cached logits this reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _cache() -> Path:
    """The fingerprinted cache directory run_matrix.py wrote.

    Resolved through run_matrix rather than rebuilt here, so there is one
    definition of what identifies a cached array. Reading a flat results/cache
    would silently pick up whichever model ran last.
    """
    import importlib.util
    import sys

    name = "_ari_d_run_matrix"
    spec = importlib.util.spec_from_file_location(name, HERE / "run_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    # run_matrix defines a dataclass at module level, and @dataclass resolves
    # cls.__module__ through sys.modules. Loading it without registering it
    # first fails with a bare AttributeError from inside dataclasses.
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    d = mod.cache_dir()
    if not d.is_dir():
        raise SystemExit(
            f"no cache at {d}\n"
            f"Run run_matrix.py first, under the same model, device and "
            f"library versions. The directory name is a fingerprint of those.")
    return d



QUANT_BINS = 8
CALIBRATION_PERCENTILE = 0.99
SIGMAS = (1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)
TOP_K_API = 20          # what a hosted API typically exposes
N_NOISE_TRIALS = 5
SEED = 0


# ---------------------------------------------------------------------------
# Statistics. Each maps (reference logits, current logits) -> per-step value.
# ---------------------------------------------------------------------------

def _logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def stat_kl(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """KL(P_ref || P_cur), the obvious distributional baseline."""
    lp, lq = _logsoftmax(r), _logsoftmax(c)
    p = np.exp(lp)
    return (p * (lp - lq)).sum(axis=1)


def stat_kl_expanded(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Same quantity, regrouped: sum(p*log p) - sum(p*log q)."""
    lp, lq = _logsoftmax(r), _logsoftmax(c)
    p = np.exp(lp)
    return (p * lp).sum(axis=1) - (p * lq).sum(axis=1)


def stat_js(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence; symmetric and bounded."""
    p, q = np.exp(_logsoftmax(r)), np.exp(_logsoftmax(c))
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * (np.log(p) - np.log(m)), 0.0).sum(axis=1)
        kq = np.where(q > 0, q * (np.log(q) - np.log(m)), 0.0).sum(axis=1)
    return 0.5 * (kp + kq)


def _top2_margin(x: np.ndarray) -> np.ndarray:
    part = np.partition(x, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def stat_margin_delta(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Change in the top-2 margin. Two numbers per step to store."""
    return np.abs(_top2_margin(c) - _top2_margin(r))


def stat_max_abs_dlogit(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.abs(c - r).max(axis=1)


def stat_l2_dlogit(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.linalg.norm(c - r, axis=1)


def stat_topk_jaccard(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """1 - Jaccard over the top-k token sets. This is what a monitor limited
    to a hosted API's top-k logprobs could compute."""
    tr = np.argpartition(-r, TOP_K_API, axis=1)[:, :TOP_K_API]
    tc = np.argpartition(-c, TOP_K_API, axis=1)[:, :TOP_K_API]
    out = np.empty(len(r))
    for i in range(len(r)):
        a, b = set(tr[i].tolist()), set(tc[i].tolist())
        out[i] = 1.0 - len(a & b) / len(a | b)
    return out


def stat_token_flip(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The thing SEMQ has to beat: did the emitted token change?"""
    return (r.argmax(axis=1) != c.argmax(axis=1)).astype(float)


def _semq(X: np.ndarray, scale: float) -> np.ndarray:
    from ari.semq_compat import MAX_DIM, quant_context

    dim = X.shape[1]
    bounds = list(range(0, dim, MAX_DIM)) + [dim]
    out = []
    for lo, hi in zip(bounds, bounds[1:]):
        ctx = quant_context(hi - lo, n_bins=QUANT_BINS, scale_max=scale)
        out.append(np.asarray(ctx.batch_encode(
            np.ascontiguousarray(X[:, lo:hi], np.float32))))
        ctx.close()
    return out[0] if len(out) == 1 else np.concatenate(out, axis=1)


def stat_semq(r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-step SEMQ symbol disagreement, at a scale frozen from reference."""
    scale = float(np.percentile(np.abs(r), CALIBRATION_PERCENTILE * 100.0))
    return (_semq(c, scale) != _semq(r, scale)).mean(axis=1)


STATS = {
    "SEMQ Hbar": stat_semq,
    "KL": stat_kl,
    "JS": stat_js,
    "top-2 margin delta": stat_margin_delta,
    "max |dlogit|": stat_max_abs_dlogit,
    "L2 |dlogit|": stat_l2_dlogit,
    f"top-{TOP_K_API} set change": stat_topk_jaccard,
    "token flip": stat_token_flip,
}

# Bytes of reference state a monitor must retain, per decoding step, to be
# able to compute the statistic later against a new run.
def storage_bytes(vocab: int) -> dict[str, int]:
    return {
        "SEMQ Hbar": vocab // 2,          # uint8 code, 2 dims per byte
        "KL": vocab * 4,                  # full fp32 distribution
        "JS": vocab * 4,
        "top-2 margin delta": 8,          # two fp32
        "max |dlogit|": vocab * 4,
        "L2 |dlogit|": vocab * 4,
        f"top-{TOP_K_API} set change": TOP_K_API * 4,
        "token flip": 4,                  # one token id
    }


# ---------------------------------------------------------------------------
# Axis 1: sensitivity under a controlled perturbation
# ---------------------------------------------------------------------------

def sensitivity(ref: np.ndarray, rng: np.random.Generator) -> dict:
    """Response of each statistic to Gaussian logit noise of magnitude sigma.

    Reported as response relative to the statistic's own value at the smallest
    sigma, so statistics on different scales are comparable. A monitor needs a
    large ratio between the perturbed and the quiet case, not a large absolute
    number.
    """
    rows = {name: [] for name in STATS}
    for s in SIGMAS:
        acc = {name: [] for name in STATS}
        for _ in range(N_NOISE_TRIALS):
            noise = rng.standard_normal(ref.shape).astype(np.float32) * s
            cur = ref + noise
            for name, fn in STATS.items():
                acc[name].append(float(np.mean(fn(ref, cur))))
        for name in STATS:
            rows[name].append(float(np.mean(acc[name])))
    return rows


# ---------------------------------------------------------------------------
# Axis 3: is the statistic reproducible bit-exactly?
# ---------------------------------------------------------------------------

# Statistics whose value is an index or a set of indices. Reversing the
# vocabulary axis permutes the index space rather than reordering a sum, so it
# does not pose the reassociation question to these at all: there is no
# reduction to reassociate. Whatever it did report would be about tie-breaking
# under argmax and argpartition, which is a different property with a different
# name. They are excluded and the exclusion is recorded, so a reader sees a
# stated reason rather than a missing row.
INDEX_VALUED = (f"top-{TOP_K_API} set change", "token flip")


def reassociation(ref: np.ndarray, cur: np.ndarray) -> dict:
    """Compute each statistic two algebraically equivalent ways and compare.

    For KL the two forms are sum(p*(log p - log q)) and
    sum(p*log p) - sum(p*log q). For the reductions, the summation order is
    reversed, which is the same class of change a different BLAS or lane layout
    makes. Index-valued statistics are excluded; see INDEX_VALUED.
    """
    out = {}

    a, b = stat_kl(ref, cur), stat_kl_expanded(ref, cur)
    out["KL"] = {
        "bit_identical": bool(np.array_equal(a, b)),
        "max_abs_diff": float(np.abs(a - b).max()),
        "max_rel_diff": float(np.abs((a - b) / np.where(a == 0, 1, a)).max()),
    }

    rev = slice(None, None, -1)
    for name, fn in STATS.items():
        if name == "KL":
            continue
        if name in INDEX_VALUED:
            out[name] = {
                "bit_identical": None,
                "not_applicable": "index-valued: reversing the vocabulary axis "
                                  "permutes indices rather than reordering a "
                                  "reduction, so reassociation is undefined here",
            }
            continue
        x = fn(ref, cur)
        y = fn(np.ascontiguousarray(ref[:, rev]), np.ascontiguousarray(cur[:, rev]))
        out[name] = {
            "bit_identical": bool(np.array_equal(x, y)),
            "max_abs_diff": float(np.abs(x - y).max()),
            "max_rel_diff": float(
                np.abs((x - y) / np.where(x == 0, 1, x)).max()),
        }
    return out


# ---------------------------------------------------------------------------

def tail_only(ref: np.ndarray, rng: np.random.Generator,
              sigma: float = 0.1) -> dict:
    """Response when only the tail of the distribution moves.

    The sigma sweep uses isotropic noise, which hits the top of the
    distribution and the tail alike. It therefore cannot tell a statistic that
    reads the whole vector from one that reads two entries. Here the top
    `keep` entries are left untouched by construction, so the top-2 margin and
    the emitted token cannot change. A statistic that still responds is
    reading something the cheap ones cannot see.
    """
    order = np.argsort(-ref, axis=1)
    out = {}
    for keep in (0, 2, 20, 100):
        cur = ref.copy()
        if keep == 0:
            cur = cur + rng.standard_normal(ref.shape).astype(np.float32) * sigma
        else:
            mask = np.zeros_like(ref, dtype=bool)
            np.put_along_axis(mask, order[:, keep:], True, axis=1)
            cur[mask] += rng.standard_normal(int(mask.sum())).astype(np.float32) * sigma
        out[str(keep)] = {name: float(np.mean(fn(ref, cur)))
                          for name, fn in STATS.items()}
    return out


def main() -> None:
    CACHE = _cache()
    ref = np.load(CACHE / "reference.npz", allow_pickle=True)["logits"]
    vocab = ref.shape[1]
    rng = np.random.default_rng(SEED)
    print(f"reference logits {ref.shape}, vocabulary {vocab}\n")

    # --- real conditions ---
    conds = sorted(p.stem for p in CACHE.glob("*.npz") if p.stem != "reference")
    print("Per-step statistic, mean over steps, against the fp32 reference")
    hdr = f"{'statistic':<22}" + "".join(f"{c:>14}" for c in conds)
    print(hdr)
    print("-" * len(hdr))
    real = {}
    for name, fn in STATS.items():
        vals = {}
        for c in conds:
            cur = np.load(CACHE / f"{c}.npz", allow_pickle=True)["logits"]
            vals[c] = float(np.mean(fn(ref, cur)))
        real[name] = vals
        print(f"{name:<22}" + "".join(f"{vals[c]:>14.6g}" for c in conds))

    # --- sensitivity sweep ---
    print("\nResponse to Gaussian logit noise of magnitude sigma "
          f"({N_NOISE_TRIALS} trials)")
    sens = sensitivity(ref, rng)
    hdr = f"{'statistic':<22}" + "".join(f"{s:>12.0e}" for s in SIGMAS)
    print(hdr)
    print("-" * len(hdr))
    for name in STATS:
        print(f"{name:<22}" + "".join(f"{v:>12.4g}" for v in sens[name]))

    print("\nDynamic range = response at the largest sigma / at the smallest.")
    print("A larger number means the statistic separates a big change from a")
    print("small one more cleanly; it does not mean it detects sooner.")
    dyn = {}
    for name in STATS:
        lo, hi = sens[name][0], sens[name][-1]
        dyn[name] = float(hi / lo) if lo > 0 else float("inf")
    for name, v in sorted(dyn.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22} {v:>12.4g}")

    # --- storage ---
    print("\nReference state a monitor must retain, per decoding step")
    store = storage_bytes(vocab)
    for name, b in sorted(store.items(), key=lambda kv: kv[1]):
        print(f"  {name:<22} {b:>10,} bytes")

    # --- reassociation ---
    bf = CACHE / "bf16.npz"
    reas = {}
    if bf.exists():
        print("\nIs the statistic reproducible bit-exactly when the same value")
        print("is computed two algebraically equivalent ways?")
        reas = reassociation(ref, np.load(bf, allow_pickle=True)["logits"])
        print(f"  {'statistic':<22} {'bit-identical':>14} {'max rel diff':>14}")
        print("  " + "-" * 50)
        for name, d in reas.items():
            if d.get("not_applicable"):
                print(f"  {name:<22} {'n/a':>14} {'n/a':>14}")
                continue
            print(f"  {name:<22} {str(d['bit_identical']):>14} "
                  f"{d['max_rel_diff']:>14.3g}")
        for name, d in reas.items():
            if d.get("not_applicable"):
                print(f"    n/a — {name}: {d['not_applicable']}")

    print("\nResponse when only the tail moves (sigma 0.1). '0' perturbs")
    print("everything; 'below rank k' leaves the top k entries untouched.")
    tail = tail_only(ref, np.random.default_rng(SEED))
    keeps = list(tail)
    hdr = f"{'statistic':<22}" + "".join(
        f"{('all' if k=='0' else '<rank '+k):>12}" for k in keeps)
    print(hdr)
    print("-" * len(hdr))
    for name in STATS:
        print(f"{name:<22}" + "".join(f"{tail[k][name]:>12.4g}" for k in keeps))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(
        {"vocab": int(vocab), "sigmas": list(SIGMAS), "tail_only": tail,
         "conditions": conds, "real": real, "sensitivity": sens,
         "dynamic_range": dyn, "storage_bytes": store,
         "reassociation": reas}, indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'baselines.json'}")

    from ari.attest import (sign_if_configured, build_references, config_ref,
                            code_ref)
    # A simulation over synthetic logits: no dataset or model, so the honest
    # binding is the parameters and the seed that fully determine the output.
    references = build_references(
        config=config_ref({
            "sigmas": list(SIGMAS), "top_k_api": TOP_K_API,
            "n_noise_trials": N_NOISE_TRIALS, "quant_bins": QUANT_BINS,
            "calibration_percentile": CALIBRATION_PERCENTILE, "seed": SEED,
            "vocab": int(vocab),
        }),
        code=code_ref(packages=["semq", "numpy"]),
    )
    sign_if_configured(metric="ARI-D-baselines",
                       report_path=RESULTS / "baselines.json",
                       references=references,
                       extra={"vocab": int(vocab)})


if __name__ == "__main__":
    main()
