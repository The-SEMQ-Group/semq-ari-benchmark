# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0 and patent pending. The SDK is not covered by the
# Apache License.
"""The `conc` and `time` conditions for a self-hosted model — completing the comparable core.

  - `conc`: re-encode under a concurrent multi-threaded burst (8 workers, small batches) vs a
    calm single pass on the same day. This is the test of thread-scheduling / dynamic-batch
    determinism — the BLAS-thread concern.
  - `time`: re-encode after a real wall-clock gap and compare with a baseline captured at least
    24h earlier on the same machine (spec/condition-set.md). Deterministic local compute is
    expected to be gap-invariant, but that is what the measurement checks, so the baseline is
    stored and the gap is recorded rather than assumed.

Codes use the v0.2 fixed-range probe, so no registry scale is involved. Measured values are
recorded as measured.

  # day 0, on the machine that will run the comparison
  python selfhosted_conc_time.py --mode baseline --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl
  # >=24h later, same machine
  python selfhosted_conc_time.py --mode measure  --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari import metrics, report                          # noqa: E402
from ari.probe import load_probe                          # noqa: E402
from ari.inputs import load_ari_bench                     # noqa: E402

STORE = Path.home() / "ari_selfhosted_conc_time"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure self-hosted `conc` and `time`.")
    ap.add_argument("--mode", choices=["baseline", "measure"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--burst-workers", type=int, default=8)
    ap.add_argument("--min-gap-hours", type=float, default=24.0)
    args = ap.parse_args(argv)

    from ari.st_load import load_sentence_transformer
    inputs = load_ari_bench(args.inputs)
    texts = inputs.texts[:args.n]
    m = load_sentence_transformer(args.model, device="cpu", trust_remote_code=True)

    def enc(chunk, bs):
        return np.asarray(m.encode(chunk, normalize_embeddings=True, convert_to_numpy=True,
                                   batch_size=bs), dtype=np.float32)

    calm_vecs = enc(texts, 32)
    probe = load_probe(calm_vecs)
    calm = probe.encode(calm_vecs)
    d = STORE / args.model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        np.save(d / "baseline_codes.npy", calm)
        (d / "baseline_meta.json").write_text(json.dumps({
            "model": args.model, "n": len(texts), "dim": int(calm_vecs.shape[1]), "s": probe.s,
            "semq": probe.version, "content_hash": inputs.content_hash,
            "host": platform.node(), "unix_ts": time.time()}, indent=2))
        print(f"{args.model}: baseline saved -> {d}")
        return 0

    meta = json.loads((d / "baseline_meta.json").read_text())
    if meta["n"] != len(texts) or meta["content_hash"] != inputs.content_hash:
        print("ERROR: baseline was captured on different inputs"); return 1
    gap_h = (time.time() - meta["unix_ts"]) / 3600
    if gap_h < args.min_gap_hours:
        print(f"ERROR: only {gap_h:.1f}h since the baseline; `time` needs >= {args.min_gap_hours}h")
        return 1
    if meta["host"] != platform.node():
        print(f"WARNING: baseline host {meta['host']} differs from {platform.node()}")
    base = np.load(d / "baseline_codes.npy")

    # conc: workers encode shards concurrently, small batches (dynamic-batch + thread contention)
    shards = [list(x) for x in np.array_split(np.arange(len(texts)), args.burst_workers)]
    with ThreadPoolExecutor(max_workers=args.burst_workers) as ex:
        parts = list(ex.map(lambda idx: enc([texts[i] for i in idx], 8), shards))
    conc = probe.encode(np.vstack(parts))

    cm = metrics.aggregate(calm, conc)
    tm = metrics.aggregate(base, calm)
    result = {"model": args.model, "n": len(texts), "semq": probe.version, "s": probe.s,
              "time_gap_hours": round(gap_h, 2), "baseline_host": meta["host"],
              "measure_host": platform.node(),
              "conc": {"HER": round(cm.HER, 6), "Hbar": round(cm.Hbar, 6),
                       "HER_ci": [round(cm.HER_ci[0], 6), round(cm.HER_ci[1], 6)],
                       "digest": report.condition_digest(conc)},
              "time": {"HER": round(tm.HER, 6), "Hbar": round(tm.Hbar, 6),
                       "HER_ci": [round(tm.HER_ci[0], 6), round(tm.HER_ci[1], 6)],
                       "digest": report.condition_digest(calm)}}
    (d / "conc_time_result.json").write_text(json.dumps(result, indent=2))
    print(f"{args.model}: conc HER={cm.HER:.4f} (H̄={cm.Hbar:.3f})  "
          f"time HER={tm.HER:.4f} (H̄={tm.Hbar:.3f}) after {gap_h:.1f}h  n={len(texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
