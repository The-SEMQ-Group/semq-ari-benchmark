# Drift Sensitivity Benchmark

The reference experiment behind ARI. It characterises how an embedding model's SEMQ code
responds to a controlled perturbation of magnitude **σ**, and shows that the response is a
clean, universal power law — the property that makes SEMQ usable as a reproducibility
instrument.

- **Protocol:** this document.
- **Results:** [`RESULTS.md`](RESULTS.md).
- **Machine-readable data:** [`results/`](results/) (CSV).

## What it measures

For each `(model, σ)` pair, on a fixed corpus:

1. Encode the corpus to FP32 and to SEMQ codes (canonical probe — see
   [`../../spec/ari-canonical-v0.1.md`](../../spec/ari-canonical-v0.1.md)).
2. Add isotropic Gaussian noise of magnitude σ to every embedding.
3. Re-encode the noisy embeddings to SEMQ codes.
4. Record the **mean Hamming distance** between clean and noisy codes.
5. Compute **Recall@10** of the noisy retrieval against the clean ranking (the baseline
   instrument, for contrast).

Then fit the power law `Hamming = a · σ^b` on the linear-regime window σ ∈ [1e-5, 1e-3].

## Setup

| axis | value |
| --- | --- |
| **Models** | 6 embedding models spanning three families — BERT-style 1024-d encoders (`BAAI/bge-large-en-v1.5`, `BAAI/bge-m3`, `intfloat/multilingual-e5-large`), 7B decoder embedders (`intfloat/e5-mistral-7b-instruct` 4096-d, `Alibaba-NLP/gte-Qwen2-7B-instruct` 3584-d), and a closed-source API (`openai/text-embedding-3-large` 3072-d). |
| **Corpus** | BEIR MS MARCO, 3,000-passage subset (primary run). Corpus extension adds BEIR NFCorpus (3,633 docs) and SciFact (5,183 docs). |
| **Probe** | SEMQ QBIN, `n_bins = 2`, 99th-percentile calibration. Single probe — this characterises the instrument, not a probe comparison. |
| **σ grid** | 11 points log-spaced over [1e-7, 1e-2]. The lower bound sits below the BLAS-thread-scheduling regime (~1e-6); the upper bound is where Hamming approaches its max-entropy ceiling (0.5). |
| **Fit window** | σ ∈ [1e-5, 1e-3] — the clean linear regime. Below it, discrete-code quantization noise dominates the per-cell mean; above it, the power law bends toward saturation. |
| **Bootstrap** | 1,000 resamples per `(model, σ)` cell for the Recall@10 95% CI. Hamming CIs are computed analytically from the per-passage Hamming distribution. |

Primary run: **66 cells** (6 models × 11 σ). Corpus extension: **132 cells**
(6 models × 11 σ × 2 datasets).

## How to reproduce

The measurement is model-agnostic and cheap once embeddings are cached (the σ sweep is a
few seconds per model on CPU). Any team can reproduce it independently — that is the point
of a public benchmark. The only special dependency is the **SEMQ QBIN probe**, provided by
the `semq` package (`pip install semq`; public on PyPI once the SEMQ SDK ships).

### Reference algorithm

```python
import numpy as np
from sentence_transformers import SentenceTransformer
from semq import qbin_calibrate, qbin_encode   # the canonical ARI probe

SIGMAS = np.logspace(-7, -2, 11)
FIT_LO, FIT_HI = 1e-5, 1e-3

def drift_sensitivity(model_id, passages, seed=0):
    rng = np.random.default_rng(seed)
    enc = SentenceTransformer(model_id)
    X = enc.encode(passages, normalize_embeddings=True)          # FP32 embeddings
    probe = qbin_calibrate(X, n_bins=2, percentile=0.99)          # 99th-pct calibration
    clean = qbin_encode(X, probe)                                 # clean SEMQ codes
    mean_norm = np.linalg.norm(X, axis=1).mean()

    rows = []
    for sigma in SIGMAS:
        noisy = X + rng.normal(0.0, sigma * mean_norm, size=X.shape)
        codes = qbin_encode(noisy, probe)
        hamming = (codes != clean).mean()                         # fraction of bits flipped
        rows.append((sigma, hamming))

    # fit Hamming = a * sigma^b on the linear-regime window
    s = np.array([r[0] for r in rows]); h = np.array([r[1] for r in rows])
    m = (s >= FIT_LO) & (s <= FIT_HI)
    b, log_a = np.polyfit(np.log(s[m]), np.log(h[m]), 1)
    a = np.exp(log_a)
    kappa = a / np.sqrt(2 * X.shape[1] / np.pi)                   # concentration scalar
    return {"slope_b": b, "prefactor_a": a, "kappa": kappa, "curve": rows}
```

The Recall@10 contrast (step 5) re-ranks the noisy corpus against the clean top-K and
reports the intersection — see `RESULTS.md` for why it is a *threshold detector* and SEMQ
Hamming is a *continuous instrument*.

### Reproducibility requirements

- Pin BLAS threads (`OMP_NUM_THREADS=1` etc.) before importing the numerical stack, so the
  measured drift comes from the injected σ and not from the host's own non-determinism.
- Fix the RNG seed for noise generation and bootstrap.
- Record the environment (BLAS, threads, hardware, precision, library versions) alongside
  the numbers — ARI is a *system* property, not a model-in-isolation property.

### Our reference numbers

The values in [`results/`](results/) are our reference measurements. Independent runs
should reproduce the **slope** `b ≈ 1` and the per-model **κ** to within a few percent; the
absolute Hamming per cell depends on the environment and is expected to vary.
