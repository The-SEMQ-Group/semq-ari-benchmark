"""The `conc` and `time` conditions for a self-hosted model — completing the comparable core.

For a local, deterministic model the two axes test different things:
  - `conc`: re-encode under a concurrent multi-threaded burst (8 workers, small batches) vs a
    calm single pass. This is the real test of thread-scheduling / dynamic-batch determinism —
    the BLAS-thread concern. Compared against the calm baseline.
  - `time`: re-encode again (deterministic local compute has no time-varying external state, so
    a wall-clock gap cannot change the output — a >24h re-run is confirmatory only). Compared
    against the calm baseline.

Uses each model's canonical registry `s` so codes match its leaderboard submission.

  python selfhosted_conc_time.py --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl --n 200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari import metrics, report                          # noqa: E402
from ari.probe import fixed_scale_codes                   # noqa: E402
from ari.inputs import load_ari_bench                     # noqa: E402

REGISTRY = HARNESS / "spec" / "fingerprints-v0.1.csv"
STORE = Path.home() / "ari_selfhosted_conc_time"


def _registry_s(model_id: str):
    for r in csv.DictReader(REGISTRY.open()):
        if r["model_id"] == model_id and r["s"]:
            return float(r["s"]), int(r["dim"])
    return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure self-hosted `conc` and `time`.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--burst-workers", type=int, default=8)
    args = ap.parse_args(argv)

    from sentence_transformers import SentenceTransformer

    s, dim = _registry_s(args.model)
    if s is None:
        print(f"ERROR: no registry s for {args.model}"); return 1
    inputs = load_ari_bench(args.inputs)
    texts = inputs.texts[:args.n]

    m = SentenceTransformer(args.model, device="cpu", trust_remote_code=True)

    def enc(chunk, bs):
        return np.asarray(m.encode(chunk, normalize_embeddings=True, convert_to_numpy=True,
                                   batch_size=bs), dtype=np.float32)

    # calm baseline
    base = fixed_scale_codes(enc(texts, 32), s, dim)

    # conc: 8 workers encode shards concurrently, small batches (dynamic-batch + thread contention)
    shards = [list(x) for x in np.array_split(np.arange(len(texts)), args.burst_workers)]
    with ThreadPoolExecutor(max_workers=args.burst_workers) as ex:
        parts = list(ex.map(lambda idx: enc([texts[i] for i in idx], 8), shards))
    conc = fixed_scale_codes(np.vstack(parts), s, dim)

    # time: re-encode (deterministic; gap-invariant for local compute)
    tcodes = fixed_scale_codes(enc(texts, 32), s, dim)

    cm = metrics.aggregate(base, conc)
    tm = metrics.aggregate(base, tcodes)
    slug = args.model.replace("/", "_")
    d = STORE / slug; d.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "n": len(texts),
              "conc": {"HER": round(cm.HER, 6), "Hbar": round(cm.Hbar, 6),
                       "HER_ci": [round(cm.HER_ci[0], 6), round(cm.HER_ci[1], 6)],
                       "digest": report.condition_digest(conc)},
              "time": {"HER": round(tm.HER, 6), "Hbar": round(tm.Hbar, 6),
                       "HER_ci": [round(tm.HER_ci[0], 6), round(tm.HER_ci[1], 6)],
                       "digest": report.condition_digest(tcodes)}}
    (d / "conc_time_result.json").write_text(json.dumps(result, indent=2))
    print(f"{args.model}: conc HER={cm.HER:.4f} (H̄={cm.Hbar:.3f})  "
          f"time HER={tm.HER:.4f} (H̄={tm.Hbar:.3f})  n={len(texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
