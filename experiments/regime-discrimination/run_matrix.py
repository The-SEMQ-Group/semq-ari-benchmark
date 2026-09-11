# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""Can retrieval metrics tell you which serving configuration you are in?

The ARI probe-choice analysis claims that discrete serving changes -- a precision
change, a backend swap, an audit-on-CPU/serve-on-GPU gap -- are "invisible to
cosine/retrieval" while SEMQ reads them clearly. The GPU-determinism
experiment measured the SEMQ half of that claim and asserted the other half
without measuring it. This experiment measures both halves side by side.

Each condition changes one thing about how the encoder runs, then encodes the
same corpus and the same queries. Every condition is compared against a
reference run (fp32, CPU, fixed thread count and batch size) on:

  * mean and 1st-percentile cosine similarity of the embeddings
  * Recall@10 and nDCG@10 against real relevance judgements
  * SEMQ hash-equality rate (HER) and mean Hamming drift, at a calibration
    scale frozen from the reference run

Real qrels matter here. On a clean corpus Recall@10 saturates at 1.0 and any
comparison against it measures the saturation, not the instrument -- that
mistake is recorded in docs/retractions.md.

Conditions that need a GPU are declared but skipped on a CPU host. Run with
--list to see which apply.

Usage:
    python run_matrix.py                 # run every applicable condition
    python run_matrix.py --list
    python run_matrix.py --worker bf16 --out /tmp/bf16.npz   # internal
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ari.code_metrics import code_diff

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = RESULTS / "cache"

# Overridable so the same conditions can be run against another encoder
# (the probe sweep needs a second dimension). The default is the encoder
# every committed result in this experiment was produced with.
ENCODER = os.environ.get("ARI_ENCODER",
                         "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = 10
QUANT_BINS = 8
CALIBRATION_PERCENTILE = 0.99


@dataclass(frozen=True)
class Condition:
    """One way of running the encoder, and what a difference would mean."""

    name: str
    axis: str          # matches spec/condition-set.md where one applies
    dtype: str = "float32"
    threads: int = 4
    batch: int = 32
    quantize: bool = False
    device: str = "cpu"
    tf32: bool | None = None
    needs_gpu: bool = False
    note: str = ""


CONDITIONS: tuple[Condition, ...] = (
    Condition("reference", "same", note="fp32 / CPU / 4 threads / batch 32"),
    Condition("proc", "proc", note="identical settings, fresh process"),
    Condition("bf16", "prec", dtype="bfloat16", note="bf16 weights"),
    # fp16 is a GPU condition. PyTorch has no optimised CPU fp16 kernels, so
    # running it here would measure the fallback path, not a serving regime.
    Condition("fp16", "prec", dtype="float16", device="cuda", needs_gpu=True,
              note="fp16 weights (GPU only)"),
    Condition("int8", "prec", quantize=True, note="dynamic int8 on Linear layers"),
    Condition("threads1", "proc", threads=1, note="single-threaded BLAS"),
    Condition("batch8", "batch", batch=8, note="batch 8 instead of 32"),
    Condition("batch128", "batch", batch=128, note="batch 128 instead of 32"),
    # Declared for the GPU run; skipped on a CPU host.
    Condition("gpu_tf32_off", "mach", device="cuda", tf32=False, needs_gpu=True,
              note="GPU fp32, TF32 disabled"),
    Condition("gpu_tf32_on", "mach", device="cuda", tf32=True, needs_gpu=True,
              note="GPU fp32, TF32 enabled (Ampere default pre-2.0)"),
    Condition("gpu_bf16", "prec", device="cuda", dtype="bfloat16", needs_gpu=True,
              note="GPU bf16"),
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def load_scifact() -> tuple[
        list[str], list[str], list[str], list[str], dict[str, set[str]]]:
    """Return (doc_ids, doc_texts, query_ids, query_texts, qrels) for SciFact."""
    from datasets import load_dataset

    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/scifact-qrels", split="test")

    qrels: dict[str, set[str]] = {}
    for row in qrels_ds:
        if int(row["score"]) > 0:
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))

    doc_ids = [str(i) for i in corpus["_id"]]
    doc_texts = [(t + " " + x).strip()
                 for t, x in zip(corpus["title"], corpus["text"])]

    # Keep only queries that have a judgement, so Recall is well defined.
    q_ids, q_texts = [], []
    for qid, text in zip(queries["_id"], queries["text"]):
        if str(qid) in qrels:
            q_ids.append(str(qid))
            q_texts.append(text)
    return doc_ids, doc_texts, q_ids, q_texts, qrels


# ---------------------------------------------------------------------------
# Encoding under one condition
# ---------------------------------------------------------------------------

def encode(cond: Condition, doc_texts: list[str], q_texts: list[str]) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(cond.threads)
    if cond.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = cond.tf32
        torch.backends.cudnn.allow_tf32 = cond.tf32

    model = SentenceTransformer(ENCODER, device=cond.device)
    if cond.quantize:
        # A qengine has to be selected explicitly; the default is NoQEngine on
        # some builds. qnnpack covers arm64, fbgemm covers x86_64.
        engines = torch.backends.quantized.supported_engines
        for name in ("qnnpack", "fbgemm", "onednn"):
            if name in engines:
                torch.backends.quantized.engine = name
                break
        else:
            raise RuntimeError(f"no usable qengine; have {engines}")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    elif cond.dtype != "float32":
        model = model.to(getattr(torch, cond.dtype))

    def run(texts: list[str]) -> np.ndarray:
        v = model.encode(texts, batch_size=cond.batch, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True)
        return np.ascontiguousarray(v, dtype=np.float32)

    return {"docs": run(doc_texts), "queries": run(q_texts)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def retrieval_metrics(docs: np.ndarray, queries: np.ndarray, doc_ids: list[str],
                      q_ids: list[str], qrels: dict[str, set[str]]) -> dict:
    scores = queries @ docs.T
    top = np.argpartition(-scores, TOP_K, axis=1)[:, :TOP_K]
    order = np.take_along_axis(scores, top, 1).argsort(axis=1)[:, ::-1]
    top = np.take_along_axis(top, order, 1)

    recalls, ndcgs = [], []
    discounts = 1.0 / np.log2(np.arange(2, TOP_K + 2))
    for row, qid in zip(top, q_ids):
        rel = qrels[qid]
        hits = np.array([doc_ids[i] in rel for i in row], dtype=np.float64)
        recalls.append(hits.sum() / len(rel))
        dcg = float((hits * discounts).sum())
        ideal = float(discounts[:min(len(rel), TOP_K)].sum())
        ndcgs.append(dcg / ideal if ideal else 0.0)
    return {"recall_at_10": float(np.mean(recalls)),
            "ndcg_at_10": float(np.mean(ndcgs)),
            "per_query_recall": np.asarray(recalls),
            "per_query_ndcg": np.asarray(ndcgs),
            "top10_ids": top}


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n: int = 10000,
                        seed: int = 0) -> tuple[float, float]:
    """95% CI on mean(b - a), resampling queries. Zero inside means the
    difference is not distinguishable from noise at this query count."""
    rng = np.random.default_rng(seed)
    d = b - a
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def semq_codes(X: np.ndarray, scale_source: np.ndarray | None = None):
    """Encode X with SEMQ QUANT, calibrated on scale_source (or on X)."""
    from ari.semq_compat import quant_context

    ctx = quant_context(X.shape[1], n_bins=QUANT_BINS)
    ctx.calibrate(scale_source if scale_source is not None else X,
                  percentile=CALIBRATION_PERCENTILE)
    codes = np.asarray(ctx.batch_encode(np.ascontiguousarray(X, np.float32)))
    ctx.close()
    return codes


def compare(ref: dict, cur: dict, doc_ids, q_ids, qrels, ref_codes) -> dict:
    cos = (ref["docs"] * cur["docs"]).sum(1)
    r_ref = retrieval_metrics(ref["docs"], ref["queries"], doc_ids, q_ids, qrels)
    r_cur = retrieval_metrics(cur["docs"], cur["queries"], doc_ids, q_ids, qrels)

    codes = semq_codes(cur["docs"], scale_source=ref["docs"])
    diff = code_diff(ref_codes, codes, n_bins=QUANT_BINS,
                     dim=cur["docs"].shape[1])

    r_lo, r_hi = paired_bootstrap_ci(
        r_ref["per_query_recall"], r_cur["per_query_recall"])
    n_lo, n_hi = paired_bootstrap_ci(
        r_ref["per_query_ndcg"], r_cur["per_query_ndcg"])

    return {
        "cosine_mean": float(cos.mean()),
        "cosine_p01": float(np.percentile(cos, 1)),
        "recall_at_10": r_cur["recall_at_10"],
        "recall_delta": r_cur["recall_at_10"] - r_ref["recall_at_10"],
        "recall_delta_ci95": [r_lo, r_hi],
        "recall_delta_significant": not (r_lo <= 0.0 <= r_hi),
        "ndcg_at_10": r_cur["ndcg_at_10"],
        "ndcg_delta": r_cur["ndcg_at_10"] - r_ref["ndcg_at_10"],
        "ndcg_delta_ci95": [n_lo, n_hi],
        "ndcg_delta_significant": not (n_lo <= 0.0 <= n_hi),
        "top10_identical": float(
            (r_cur["top10_ids"] == r_ref["top10_ids"]).all(1).mean()),
        "semq_her": float(diff.codes_equal.mean()),
        "semq_coord_change": float(diff.coordinate_change_rate.mean()),
        "semq_byte_change_legacy": float(diff.byte_change_rate.mean()),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _hub_revision(repo_id: str, repo_type: str = "model") -> str:
    """Resolved commit for a Hub repo, or `unknown` when it cannot be had.

    A name is not an identity: `all-MiniLM-L6-v2` and `BeIR/scifact` can both
    change under the same string. Delegates to ari.hub.hub_revision, the one
    resolver shared with the attestation references, so the cache fingerprint
    and the manifest name one revision.
    """
    from ari.hub import hub_revision
    return hub_revision(repo_id, repo_type)


def _versions() -> dict:
    """Versions from package metadata, so the driver need not import torch."""
    out = {}
    for pkg in ("torch", "transformers", "sentence-transformers"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def _identity() -> dict:
    """What a cached embedding array depends on.

    Keyed on the condition name alone, `reference.npz` survived a change of
    encoder, of dataset revision, or of library, and was then reported under the
    new one.
    """
    return {
        "encoder": ENCODER,
        "encoder_revision": _hub_revision(ENCODER),
        "dataset": "BeIR/scifact",
        "dataset_revision": _hub_revision("BeIR/scifact", "dataset"),
        "qrels_revision": _hub_revision("BeIR/scifact-qrels", "dataset"),
        "top_k": TOP_K,
        "quant_bins": QUANT_BINS,
        "calibration_percentile": CALIBRATION_PERCENTILE,
        **_versions(),
    }


def fingerprint() -> str:
    blob = json.dumps(_identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_dir() -> Path:
    return CACHE / fingerprint()


def cache_path(name: str) -> Path:
    return cache_dir() / f"{name}.npz"


def write_cache_manifest() -> None:
    """Record what this fingerprint stands for, and refuse a mismatch."""
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.json"
    identity = _identity()
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise SystemExit(
                f"cache manifest at {path} does not match this run.\n"
                f"  cached : {json.dumps(existing, sort_keys=True)}\n"
                f"  current: {json.dumps(identity, sort_keys=True)}\n"
                f"Delete the directory to re-measure.")
        return
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")


def run_worker(name: str, out: Path, limit: int | None = None) -> None:
    cond = next(c for c in CONDITIONS if c.name == name)
    _, doc_texts, _, q_texts, _ = load_scifact()
    # A subset is enough for a study of coordinate statistics rather than
    # retrieval quality, and the corpus is long enough that the full encode
    # costs hours on CPU. Retrieval metrics need the whole corpus, so this
    # stays opt-in and the committed results are unaffected.
    if limit is not None:
        doc_texts = doc_texts[:limit]
    v = encode(cond, doc_texts, q_texts)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, docs=v["docs"], queries=v["queries"])


def gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int,
                    help="encode only the first N documents")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker, args.out, args.limit)
        return

    have_gpu = gpu_available()
    applicable = [c for c in CONDITIONS if not c.needs_gpu or have_gpu]
    skipped = [c for c in CONDITIONS if c.needs_gpu and not have_gpu]

    # A GPU host is the wrong place to measure a CPU precision condition. The
    # instance CPU here has AVX2 and no AVX-512, so torch runs bf16 through a
    # slow emulation path: one condition took over 13 minutes and measured the
    # instance rather than the regime. ARI_R_CONDITIONS restricts the run to a
    # named set, so a GPU box can do the reference plus the GPU conditions and
    # reuse the CPU numbers already measured on a CPU host.
    only = os.environ.get("ARI_R_CONDITIONS", "").strip()
    if only:
        want = {n.strip() for n in only.split(",") if n.strip()}
        applicable = [c for c in applicable if c.name in want]
        skipped = [c for c in CONDITIONS if c.name not in want]

    if args.list:
        for c in CONDITIONS:
            mark = "skip (no GPU)" if c in skipped else "run"
            print(f"{c.name:<14} {c.axis:<6} {mark:<14} {c.note}")
        return

    failed: dict[str, str] = {}
    print(f"encoder {ENCODER}   GPU {'yes' if have_gpu else 'no'}")
    print(f"encoder revision {_hub_revision(ENCODER)}")
    # Print the cache namespace. A run that reuses arrays should say whose.
    write_cache_manifest()
    print(f"cache {cache_dir().relative_to(HERE)}")
    print(f"running {len(applicable)} conditions, skipping {len(skipped)}\n")

    # Each condition runs in its own process. That is required for the `proc`
    # comparison to mean anything, and it keeps a dtype or thread change from
    # leaking into the next condition.
    for c in applicable:
        p = cache_path(c.name)
        if p.exists():
            print(f"  {c.name:<14} cached")
            continue
        print(f"  {c.name:<14} encoding...", flush=True)
        r = subprocess.run([sys.executable, str(HERE / "run_matrix.py"),
                            "--worker", c.name, "--out", str(p)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            # One unsupported condition must not lose the rest of the matrix.
            last = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
            failed[c.name] = last[-1] if last else f"exit {r.returncode}"
            print(f"  {c.name:<14} FAILED — {failed[c.name]}")

    applicable = [c for c in applicable if c.name not in failed]

    doc_ids, _, q_ids, _, qrels = load_scifact()
    ref = dict(np.load(cache_path("reference")))
    ref_codes = semq_codes(ref["docs"])

    rows = []
    for c in applicable:
        cur = dict(np.load(cache_path(c.name)))
        m = compare(ref, cur, doc_ids, q_ids, qrels, ref_codes)
        m["condition"], m["axis"], m["note"] = c.name, c.axis, c.note
        rows.append(m)

    hdr = (f"{'condition':<14} {'axis':<6} {'cos mean':>10} {'cos p01':>9} "
           f"{'R@10':>7} {'dR@10':>8} {'dR@10 95% CI':>22} {'sig':>5} "
           f"{'top10 same':>11} {'SEMQ HER':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        lo, hi = r["recall_delta_ci95"]
        print(f"{r['condition']:<14} {r['axis']:<6} {r['cosine_mean']:>10.6f} "
              f"{r['cosine_p01']:>9.6f} {r['recall_at_10']:>7.4f} "
              f"{r['recall_delta']:>+8.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>22} "
              f"{('yes' if r['recall_delta_significant'] else 'no'):>5} "
              f"{r['top10_identical']:>10.2%} {r['semq_her']:>10.4f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "regime_matrix.json").write_text(json.dumps(
        {"encoder": ENCODER, "top_k": TOP_K, "quant_bins": QUANT_BINS,
         "gpu": have_gpu, "n_queries": len(q_ids), "n_docs": len(doc_ids),
         "skipped": [c.name for c in skipped], "failed": failed,
         "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'regime_matrix.json'}")

    from ari.attest import (sign_if_configured, build_references, dataset_ref,
                            model_ref, config_ref, code_ref)
    from ari.hub import hub_revision
    scifact_rev = hub_revision("BeIR/scifact", "dataset")
    references = build_references(
        datasets=[
            dataset_ref("BeIR/scifact", scifact_rev,
                        config="corpus", split="corpus"),
            dataset_ref("BeIR/scifact", scifact_rev,
                        config="queries", split="queries"),
            dataset_ref("BeIR/scifact-qrels",
                        hub_revision("BeIR/scifact-qrels", "dataset"),
                        split="test"),
        ],
        models=[model_ref(ENCODER, hub_revision(ENCODER, "model"))],
        config=config_ref({
            "encoder": ENCODER, "top_k": TOP_K, "quant_bins": QUANT_BINS,
            "calibration_percentile": CALIBRATION_PERCENTILE,
            "conditions": [c.name for c in CONDITIONS],
        }),
        code=code_ref(packages=["semq", "torch", "sentence-transformers",
                                "datasets", "numpy"]),
    )
    sign_if_configured(metric="ARI-R", report_path=RESULTS / "regime_matrix.json",
                       references=references,
                       extra={"encoder": ENCODER, "top_k": TOP_K, "gpu": have_gpu})
    if skipped:
        print("skipped (need a GPU): " + ", ".join(c.name for c in skipped))


if __name__ == "__main__":
    main()
