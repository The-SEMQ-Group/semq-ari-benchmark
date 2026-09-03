# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""How exposed is a frozen product quantizer to floating-point reassociation?

A vector-quantizer probe assigns a code by argmin over floating-point
distances. Floating-point addition is not associative, so two algebraically
identical ways of computing the same distance can differ in the final bits.
Where two centroids are nearly equidistant from a point, that difference can
flip the assignment, and the probe's reading then depends on the library that
produced it rather than on the subject being measured.

Whether that matters is an empirical question about codebook geometry, and it
must be answered on real codebooks. This script fits a product quantizer the
way a shipped one is fitted -- k-means over real embeddings of a real corpus --
and sweeps the number of subspaces, because a finer split puts the same number
of centroids into a lower-dimensional block and should crowd them together.

For each setting it reports three things:

  1. The share of assignments whose two best distances differ by less than a
     threshold, and the median margin.
  2. Whether the direct form (b - c)^2 and the expanded form
     |b|^2 - 2b.c + |c|^2 disagree on the resulting codes. The expanded form
     is the one production libraries use, because its cross term is a matmul.
  3. Whether encode(reconstruct(c)) = c survives, under each form.

Results and their reading: RESULTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
N_DOCS = 5000
K_CENTROIDS = 256
SEED = 0
THRESHOLDS = (1e-6, 1e-5, 1e-4)
# 384 dims split into subvectors of 12, 6, 3, 2 and 1 dimensions.
SUBSPACE_COUNTS = (32, 64, 128, 192, 384)

RESULTS = Path(__file__).with_name("results") / "near_tie_sweep.json"


def embed_corpus() -> np.ndarray:
    ds = load_dataset("BeIR/scifact", "corpus", split="corpus")
    texts = [
        (t + " " + x).strip()
        for t, x in zip(ds["title"][:N_DOCS], ds["text"][:N_DOCS])
    ]
    model = SentenceTransformer(ENCODER)
    X = model.encode(texts, batch_size=128, show_progress_bar=False,
                     normalize_embeddings=True)
    return np.ascontiguousarray(X, dtype=np.float32)


def distances(block: np.ndarray, C: np.ndarray, expanded: bool) -> np.ndarray:
    if expanded:
        return ((block * block).sum(1)[:, None]
                - 2.0 * (block @ C.T)
                + (C * C).sum(1)[None, :])
    return ((block[:, None, :] - C[None, :, :]) ** 2).sum(-1)


def evaluate(train: np.ndarray, probe: np.ndarray, m: int) -> dict:
    sub = train.shape[1] // m
    book = [
        np.ascontiguousarray(
            KMeans(n_clusters=K_CENTROIDS, n_init=4, random_state=SEED)
            .fit(train[:, i * sub:(i + 1) * sub]).cluster_centers_,
            dtype=np.float32,
        )
        for i in range(m)
    ]

    def encode(Y: np.ndarray, expanded: bool) -> np.ndarray:
        out = np.empty((Y.shape[0], m), dtype=np.int32)
        for i in range(m):
            out[:, i] = distances(
                Y[:, i * sub:(i + 1) * sub], book[i], expanded).argmin(1)
        return out

    margins = []
    for i in range(m):
        d = distances(probe[:, i * sub:(i + 1) * sub], book[i], expanded=False)
        top2 = np.partition(d, 1, axis=1)[:, :2]
        top2.sort(axis=1)
        margins.append(top2[:, 1] - top2[:, 0])
    g = np.concatenate(margins)

    cd, ce = encode(probe, False), encode(probe, True)
    res = {
        "subspaces": m,
        "dims_per_subvector": sub,
        "near_tie_rate": {f"{t:.0e}": float((g < t).mean()) for t in THRESHOLDS},
        "median_margin": float(np.median(g)),
        "min_margin": float(g.min()),
        "form_disagreement_symbols": float((cd != ce).mean()),
        "form_disagreement_rows": float((cd != ce).any(1).mean()),
    }
    for expanded, label in ((False, "direct"), (True, "expanded")):
        c = encode(probe, expanded)
        r = np.concatenate([book[i][c[:, i]] for i in range(m)], axis=1)
        res[f"attractor_break_{label}"] = float((c != encode(r, expanded)).mean())
    return res


def main() -> None:
    X = embed_corpus()
    n, dim = X.shape
    train, probe = X[: n // 2], X[n // 2:]
    print(f"corpus {n} SciFact abstracts, encoder {ENCODER}, dim {dim}")
    print(f"codebook k-means, {K_CENTROIDS} centroids per subspace, seed {SEED}\n")

    hdr = (f"{'subspaces':>10} {'dims':>5} {'tie<1e-6':>10} {'tie<1e-5':>10} "
           f"{'median':>11} {'sym diff':>10} {'row diff':>10} "
           f"{'attr direct':>12} {'attr expand':>12}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for m in SUBSPACE_COUNTS:
        r = evaluate(train, probe, m)
        rows.append(r)
        print(f"{r['subspaces']:>10} {r['dims_per_subvector']:>5} "
              f"{r['near_tie_rate']['1e-06']:>9.4%} {r['near_tie_rate']['1e-05']:>9.4%} "
              f"{r['median_margin']:>11.3e} {r['form_disagreement_symbols']:>9.4%} "
              f"{r['form_disagreement_rows']:>9.4%} "
              f"{r['attractor_break_direct']:>11.4%} "
              f"{r['attractor_break_expanded']:>11.4%}")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"encoder": ENCODER, "n_docs": N_DOCS, "dim": int(dim),
         "centroids": K_CENTROIDS, "seed": SEED, "sweep": rows},
        indent=2) + "\n")
    print(f"\nwrote {RESULTS}")

    from ari.attest import (sign_if_configured, build_references, dataset_ref,
                            model_ref, config_ref, code_ref)
    from ari.hub import hub_revision
    references = build_references(
        datasets=[dataset_ref("BeIR/scifact", hub_revision("BeIR/scifact", "dataset"),
                              config="corpus", split="corpus")],
        models=[model_ref(ENCODER, hub_revision(ENCODER, "model"))],
        config=config_ref({
            "encoder": ENCODER, "n_docs": N_DOCS, "centroids": K_CENTROIDS,
            "seed": SEED, "thresholds": list(THRESHOLDS),
            "subspace_counts": list(SUBSPACE_COUNTS),
        }),
        code=code_ref(packages=["semq", "numpy", "sentence-transformers",
                                "datasets", "scikit-learn"]),
    )
    sign_if_configured(metric="probe-verifiability", report_path=RESULTS,
                       references=references,
                       extra={"encoder": ENCODER, "centroids": K_CENTROIDS})


if __name__ == "__main__":
    main()
