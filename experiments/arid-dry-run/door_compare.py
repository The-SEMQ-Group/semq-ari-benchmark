"""The door-to-door comparison table, reproducible from the transcripts.

  python door_compare.py    # writes results/door_compare.json and prints the table

Byte-identical PAIR rates at fixed prefix horizons (length control: the
platform ignores max_tokens, so its completions run longer and full-string
exactness mechanically drops; a fixed horizon compares like with like), plus
mean completion lengths and the fingerprint overlap between doors.
"""
from __future__ import annotations

import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
HORIZONS = [64, 128, 256, 512]

DOORS = {
    "direct_seed": "openai_same.jsonl.gz",
    "direct_noseed": "openai_same_noseed.jsonl.gz",
    "platform": "salesforce_same.jsonl.gz",
}


def load(name):
    by = defaultdict(list)
    for line in gzip.open(RESULTS / "transcripts" / name, "rt"):
        r = json.loads(line)
        by[r["input_id"]].append((r["rep"], r["parsed"]))
    return by


def stats(by):
    prefix = {h: [] for h in HORIZONS}
    full, lens = [], []
    fps = set()
    for pid, recs in by.items():
        parsed = [p for _, p in sorted(recs)]
        texts = [p["text"].encode() for p in parsed]
        fps |= {p.get("system_fingerprint") for p in parsed} - {None}
        lens += [len(t) for t in texts]
        pairs = list(itertools.combinations(texts, 2))
        for h in HORIZONS:
            prefix[h].append(sum(a[:h] == b[:h] for a, b in pairs) / len(pairs))
        full.append(sum(a == b for a, b in pairs) / len(pairs))
    n = len(full)
    return {
        "prefix_pair_rate": {h: round(sum(v) / n, 4) for h, v in prefix.items()},
        "full_pair_rate": round(sum(full) / n, 4),
        "mean_completion_bytes": round(sum(lens) / len(lens)),
        "fingerprints": sorted(fps),
    }


def main() -> int:
    out = {door: stats(load(t)) for door, t in DOORS.items()}
    direct = set(out["direct_seed"]["fingerprints"])
    platform = set(out["platform"]["fingerprints"])
    out["fingerprint_overlap"] = {
        "direct_seed": len(direct),
        "platform": len(platform),
        "shared": len(direct & platform),
        "shared_ids": sorted(direct & platform),
    }
    (RESULTS / "door_compare.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'horizon':>10} " + " ".join(f"{d:>14}" for d in DOORS))
    for h in HORIZONS:
        print(f"{h:>9}B " + " ".join(f"{out[d]['prefix_pair_rate'][h]:>14}" for d in DOORS))
    print(f"{'full':>10} " + " ".join(f"{out[d]['full_pair_rate']:>14}" for d in DOORS))
    print(f"{'len (B)':>10} " + " ".join(f"{out[d]['mean_completion_bytes']:>14}" for d in DOORS))
    o = out["fingerprint_overlap"]
    print(f"fingerprints: direct={o['direct_seed']} platform={o['platform']} shared={o['shared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
