"""The `time` condition: capture an agent's SEMQ codes now, re-measure after a wall-clock
gap (≥24h) to get the `time` HER — the most real-world drift axis (silent infra/snapshot
updates over time). Also gives temporal replication of the drift rate.

  # now: encode the frozen inputs, save the baseline codes + calibration scale
  python capture_time_baseline.py --mode capture --agent gemini --inputs ari-bench-v0.1.jsonl
  # ≥24h later: re-encode, compare to the saved baseline
  python capture_time_baseline.py --mode compare --agent gemini --inputs ari-bench-v0.1.jsonl

Baseline data lives outside the repo (~/ari_time_baseline/<slug>/): codes.npy + meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HARNESS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARNESS))

from ari.agents import DEFAULT_API_MODELS               # noqa: E402
from ari.inputs import load_ari_bench                   # noqa: E402
from ari.probe import load_probe, fixed_scale_codes     # noqa: E402
from ari import metrics, report                         # noqa: E402

API = DEFAULT_API_MODELS
STORE = Path.home() / "ari_time_baseline"


def build(agent: str, model: str | None, max_workers: int):
    cls, default = API[agent]
    return cls(model_id=model or default, max_workers=max_workers)


def encode_codes(agent, texts, probe):
    return probe.encode(agent.encode(texts))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Capture / compare the `time` condition baseline.")
    ap.add_argument("--mode", required=True, choices=["capture", "compare"])
    ap.add_argument("--agent", required=True, choices=list(API))
    ap.add_argument("--model", default=None)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs)
    if args.limit:
        from ari.inputs import InputSet
        inputs = InputSet(inputs.name, inputs.ids[:args.limit], inputs.texts[:args.limit])

    agent = build(args.agent, args.model, args.max_workers)
    slug = f"{args.agent}_{agent.agent_id.replace('/', '_')}"
    d = STORE / slug
    codes_p, meta_p = d / "codes.npy", d / "meta.json"

    if args.mode == "capture":
        vecs = agent.encode(inputs.texts)
        probe = load_probe(vecs, backend="semq")
        codes = probe.encode(vecs)
        d.mkdir(parents=True, exist_ok=True)
        np.save(codes_p, codes)
        meta = {"agent_id": agent.agent_id, "unix_ts": time.time(), "n": len(inputs),
                "content_hash": inputs.content_hash, "scale_s": float(probe.s),
                "dim": int(vecs.shape[1])}
        meta_p.write_text(json.dumps(meta, indent=2))
        print(f"captured {slug}: n={meta['n']}, s={meta['scale_s']:.5f} -> {d}")
        return 0

    # compare mode
    if not codes_p.exists():
        print(f"no baseline for {slug} at {d}"); return 1
    meta = json.loads(meta_p.read_text())
    if meta["content_hash"] != inputs.content_hash:
        print("ERROR: inputs content hash differs from the captured baseline"); return 1
    base_codes = np.load(codes_p)

    # re-encode with the stored scale so codes are comparable to the baseline
    time_codes = fixed_scale_codes(agent.encode(inputs.texts), meta["scale_s"], meta["dim"])

    m = metrics.aggregate(base_codes, time_codes)
    hours = (time.time() - meta["unix_ts"]) / 3600
    result = {"HER": round(m.HER, 6), "Hbar": round(m.Hbar, 6),
              "HER_ci": [round(m.HER_ci[0], 6), round(m.HER_ci[1], 6)],
              "digest": report.condition_digest(time_codes),
              "gap_hours": round(hours, 2), "n": len(inputs)}
    (d / "time_result.json").write_text(json.dumps(result, indent=2))
    print(f"{slug}: time HER = {m.HER:.4f}  95% CI {m.HER_ci}  H̄={m.Hbar:.2f}  "
          f"(gap {hours:.1f}h, n={len(inputs)})  digest={result['digest'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
