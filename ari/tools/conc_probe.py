"""The `conc` condition for a black-box API: re-encode the inputs under a burst (high
concurrency) and against a calm (low concurrency) pass — both now, so the difference isolates
concurrency (does load route to a different backend?) rather than time. Uses the fixed
calibration scale from the captured baseline so codes are comparable to the submission.

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
from ari.probe import fixed_scale_codes     # noqa: E402
from ari.inputs import load_ari_bench        # noqa: E402
from ari import metrics, report              # noqa: E402

API = DEFAULT_API_MODELS
STORE = Path.home() / "ari_time_baseline"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure the `conc` (burst vs calm) condition.")
    ap.add_argument("--agent", required=True, choices=list(API))
    ap.add_argument("--model", default=None)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--calm-workers", type=int, default=2)
    ap.add_argument("--burst-workers", type=int, default=48)
    args = ap.parse_args(argv)

    cls, default = API[args.agent]
    model = args.model or default
    inputs = load_ari_bench(args.inputs)
    slug = f"{args.agent}_{model.replace('/', '_')}"
    d = STORE / slug
    meta = json.loads((d / "meta.json").read_text())
    if meta["content_hash"] != inputs.content_hash:
        print("ERROR: inputs differ from the captured baseline"); return 1
    s, dim = meta["scale_s"], meta["dim"]

    calm = fixed_scale_codes(cls(model_id=model, max_workers=args.calm_workers).encode(inputs.texts), s, dim)
    burst = fixed_scale_codes(cls(model_id=model, max_workers=args.burst_workers).encode(inputs.texts), s, dim)

    m = metrics.aggregate(calm, burst)
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
