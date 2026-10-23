# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""The `conc` and `time` conditions for a self-hosted model — completing the comparable core.

  - `conc`: re-encode under a concurrent multi-threaded burst (8 workers, small batches) vs a
    calm single pass on the same day. This is the test of thread-scheduling / dynamic-batch
    determinism — the BLAS-thread concern.
  - `time`: re-encode after a real wall-clock gap and compare with a baseline captured more than
    24h earlier on the same hardware (spec/condition-set.md). Deterministic local compute is
    expected to be gap-invariant, but that is what the measurement checks, so the baseline is
    stored and the gap is recorded rather than assumed. The baseline's meta records the
    held-fixed settings (ari/time_evidence.py); the result's `time` carries `time_evidence`.

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
from ari import time_evidence as te                       # noqa: E402
from ari.hub import hub_revision                          # noqa: E402
from ari.probe import load_probe                          # noqa: E402
from ari.inputs import InputSet, load_ari_bench           # noqa: E402

STORE = Path.home() / "ari_selfhosted_conc_time"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure self-hosted `conc` and `time`.")
    ap.add_argument("--mode", choices=["baseline", "measure"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--burst-workers", type=int, default=8)
    ap.add_argument("--revision", default=None,
                    help="Hub commit to load (default: the commit the local cache resolves)")
    args = ap.parse_args(argv)

    from ari.st_load import load_sentence_transformer
    inputs = load_ari_bench(args.inputs)
    # Hash what is encoded, so the evidence's input hash matches a report on these inputs.
    inputs = InputSet(inputs.name, inputs.ids[:args.n], inputs.texts[:args.n])
    texts = inputs.texts
    m = load_sentence_transformer(args.model, device="cpu", trust_remote_code=True,
                                  revision=args.revision)
    env = te.self_hosted_environment(m, args.revision or hub_revision(args.model))
    try:
        te.check_identities(args.model, "self_hosted", env)
    except te.TimeEvidenceError as e:
        print(f"ERROR: {e}")
        return 1

    def enc(chunk, bs):
        return np.asarray(m.encode(chunk, normalize_embeddings=True, convert_to_numpy=True,
                                   batch_size=bs), dtype=np.float32)

    started_at = te.utc_now()
    calm_vecs = enc(texts, 32)
    record = te.baseline_record(agent_id=args.model, agent_class="self_hosted",
                                input_content_hash=inputs.content_hash, dim=calm_vecs.shape[1],
                                started_at=started_at, finished_at=te.utc_now(), environment=env)
    probe = load_probe(calm_vecs)
    calm = probe.encode(calm_vecs)
    d = STORE / args.model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)

    if args.mode == "baseline":
        np.save(d / "baseline_codes.npy", calm)
        (d / "baseline_meta.json").write_text(json.dumps({
            "model": args.model, "n": len(texts), "s": probe.s, "semq": probe.version,
            "content_hash": inputs.content_hash, "host": platform.node(), "unix_ts": time.time(),
            **record}, indent=2))
        print(f"{args.model}: baseline saved -> {d}")
        return 0

    meta = json.loads((d / "baseline_meta.json").read_text())
    if meta["n"] != len(texts) or meta["content_hash"] != inputs.content_hash:
        print("ERROR: baseline was captured on different inputs"); return 1
    try:
        evidence = te.time_evidence(meta, record)
    except te.TimeEvidenceError as e:
        print(f"ERROR: {e}")
        return 1
    gap_h = evidence["gap_hours"]
    base = np.load(d / "baseline_codes.npy")

    # conc: workers encode shards concurrently, small batches (dynamic-batch + thread contention)
    shards = [list(x) for x in np.array_split(np.arange(len(texts)), args.burst_workers)]
    with ThreadPoolExecutor(max_workers=args.burst_workers) as ex:
        parts = list(ex.map(lambda idx: enc([texts[i] for i in idx], 8), shards))
    conc = probe.encode(np.vstack(parts))

    dim = record["dim"]
    cm = metrics.aggregate(calm, conc, dim=dim)
    tm = metrics.aggregate(base, calm, dim=dim)
    result = {"model": args.model, "n": len(texts), "semq": probe.version, "s": probe.s,
              "time_gap_hours": gap_h, "baseline_host": meta["host"],
              "measure_host": platform.node(),
              "conc": {"HER": round(cm.HER, 6), "Hbar": round(cm.Hbar, 6),
                       "HER_ci": [round(cm.HER_ci[0], 6), round(cm.HER_ci[1], 6)],
                       "digest": report.condition_digest(conc)},
              "time": {"HER": round(tm.HER, 6), "Hbar": round(tm.Hbar, 6),
                       "HER_ci": [round(tm.HER_ci[0], 6), round(tm.HER_ci[1], 6)],
                       "digest": report.condition_digest(calm), "time_evidence": evidence}}
    (d / "conc_time_result.json").write_text(json.dumps(result, indent=2))
    print(f"{args.model}: conc HER={cm.HER:.4f} (H̄={cm.Hbar:.3f})  "
          f"time HER={tm.HER:.4f} (H̄={tm.Hbar:.3f}) after {gap_h:.1f}h  n={len(texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
