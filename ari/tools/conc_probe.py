# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
#
# This file calls the SEMQ SDK, a separate library licensed under the PolyForm
# Noncommercial License 1.0.0. The SDK is not covered by the Apache License.
"""The `conc` condition for a black-box API: re-encode the inputs under a burst (high
concurrency) and against a calm (low concurrency) pass — both now, so the difference isolates
concurrency (does load route to a different backend?) rather than time. Reads the baseline
that capture_time_baseline.py or refresh_leaderboard.py wrote, to check that the inputs,
dimension and probe match the submission; the v0.2 probe's range depends on the dimension
alone, so no stored scale is needed.

  python conc_probe.py --agent gemini --inputs data/ari-bench-v0.1.jsonl
  # calm=2 workers, burst=48 workers by default; writes conc_result.json next to the baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari.agents import DEFAULT_API_MODELS   # noqa: E402
from ari.probe import load_probe            # noqa: E402
from ari.inputs import InputSet, load_ari_bench  # noqa: E402
from ari import metrics, report              # noqa: E402
from ari import time_evidence as te          # noqa: E402

API = DEFAULT_API_MODELS
STORE = Path.home() / "ari_time_baseline"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the `conc` (burst vs calm) condition.")
    ap.add_argument("--agent", required=True, choices=list(API))
    ap.add_argument("--model", default=None)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--calm-workers", type=int, default=2)
    ap.add_argument("--burst-workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--baseline-dir", type=Path, default=None,
                    help="directory holding meta.json (default: ~/ari_time_baseline/<slug>)")
    args = ap.parse_args(argv)

    cls, default = API[args.agent]
    model = args.model or default
    inputs = load_ari_bench(args.inputs)
    if args.limit:
        inputs = InputSet(inputs.name, inputs.ids[:args.limit], inputs.texts[:args.limit])
    slug = f"{args.agent}_{model.replace('/', '_')}"
    d = args.baseline_dir or STORE / slug
    meta_p = d / "meta.json"
    if not meta_p.exists():
        print(f"no baseline for {slug} at {d}"); return 1
    meta = json.loads(meta_p.read_text())
    missing = [k for k in ("input_content_hash", "dim", "probe_calibration") if k not in meta]
    if missing:
        print(f"ERROR: the baseline at {d} does not record {', '.join(missing)}; "
              "recapture it with capture_time_baseline.py --mode capture"); return 1
    if meta["probe_calibration"] != te.PROBE_CALIBRATION:
        print(f"ERROR: the baseline uses probe {meta['probe_calibration']}, not {te.PROBE_CALIBRATION}"); return 1
    if meta["input_content_hash"] != inputs.content_hash:
        print("ERROR: inputs differ from the captured baseline"); return 1
    dim = int(meta["dim"])

    codes = {}
    for name, workers in (("calm", args.calm_workers), ("burst", args.burst_workers)):
        vecs = cls(model_id=model, max_workers=workers).encode(inputs.texts)
        if vecs.shape[1] != dim:
            print(f"ERROR: {name} vectors have dimension {vecs.shape[1]}, the baseline {dim}"); return 1
        codes[name] = load_probe(vecs).encode(vecs)
    calm, burst = codes["calm"], codes["burst"]

    m = metrics.aggregate(calm, burst, dim=dim)
    result = {"HER": round(m.HER, 6), "Hbar": round(m.Hbar, 6),
              "HER_ci": [round(m.HER_ci[0], 6), round(m.HER_ci[1], 6)],
              "digest": report.condition_digest(burst),
              "calm_workers": args.calm_workers, "burst_workers": args.burst_workers,
              "n": len(inputs)}
    (d / "conc_result.json").write_text(json.dumps(result, indent=2))
    print(f"{slug}: conc HER = {m.HER:.4f}  95% CI {m.HER_ci}  H̄={m.Hbar:.2f}  "
          f"(calm={args.calm_workers} vs burst={args.burst_workers}, n={len(inputs)})  "
          f"digest={result['digest'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
