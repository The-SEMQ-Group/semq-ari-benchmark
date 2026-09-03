"""The `time` condition: pairwise agreement between two `same` batches
across a >24 h gap (spec §6), computed from the two retained transcripts.

  python time_pair.py    # writes results/time_pair.json and prints the cell

Per prompt, every cross-batch pair (k x k) is compared; `exact_generation`
and `first_divergence` follow §7's definitions; the CI is a bootstrap over
prompts. Both batches' fingerprint sets are recorded so a reader can tell
drift under a disclosed backend change from drift under a silent one —
half the point of the condition.
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
B = 2000

import argparse

_ap = argparse.ArgumentParser(description="time condition from two same batches")
_ap.add_argument("--batch-a", default="openai_same.jsonl.gz")
_ap.add_argument("--batch-b", default="openai_same_t2.jsonl.gz")
_ap.add_argument("--out", default="time_pair.json")
_ARGS = _ap.parse_args()
PAIR = (_ARGS.batch_a, _ARGS.batch_b)


def load(name):
    by = defaultdict(list)
    for line in gzip.open(RESULTS / "transcripts" / name, "rt"):
        r = json.loads(line)
        by[r["input_id"]].append((r["rep"], r["parsed"]))
    return by


def first_divergence(a: bytes, b: bytes) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return d / max(len(a), len(b))


def boot_ci(vals, tag):
    mean = sum(vals) / len(vals)
    rng = random.Random(int(hashlib.sha256(tag.encode()).hexdigest()[:12], 16))
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return round(mean, 4), [round(means[int(0.025 * B)], 4), round(means[int(0.975 * B)], 4)]


def main() -> int:
    a, b = (load(t) for t in PAIR)
    assert set(a) == set(b), "the two batches cover different prompts"
    per_exact, per_fd = [], []
    fps_a, fps_b = set(), set()
    for pid in sorted(a):
        ta = [p["text"].encode() for _, p in sorted(a[pid])]
        tb = [p["text"].encode() for _, p in sorted(b[pid])]
        fps_a |= {p.get("system_fingerprint") for _, p in a[pid]} - {None}
        fps_b |= {p.get("system_fingerprint") for _, p in b[pid]} - {None}
        pairs = list(itertools.product(ta, tb))
        per_exact.append(sum(x == y for x, y in pairs) / len(pairs))
        per_fd.append(sum(first_divergence(x, y) for x, y in pairs) / len(pairs))

    e_mean, e_ci = boot_ci(per_exact, "time:exact")
    f_mean, f_ci = boot_ci(per_fd, "time:fd")
    out = {
        "condition": "time",
        "batches": list(PAIR),
        "cross_pairs_per_prompt": "k x k",
        "exact_generation": {"mean": e_mean, "ci": e_ci},
        "first_divergence": {"mean": f_mean, "ci": f_ci},
        "fingerprints_batch_a": sorted(fps_a),
        "fingerprints_batch_b": sorted(fps_b),
        "fingerprints_shared": len(fps_a & fps_b),
    }
    (RESULTS / _ARGS.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"time: exact={e_mean} {e_ci}  fd={f_mean} {f_ci}")
    print(f"fingerprints: dia1={len(fps_a)} dia2={len(fps_b)} compartidos={out['fingerprints_shared']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
