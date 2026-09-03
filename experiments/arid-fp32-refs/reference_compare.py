"""The reference axis: compare a rehosted subject's transcripts against a
reference generation (spec §7: `reference_match` exact,
`reference_first_divergence` interval).

  python reference_compare.py --reference results/refs_llama33_fp32.jsonl \\
      --transcript ../arid-dry-run/results/transcripts/together_same.jsonl.gz

Per prompt, every completion in the subject batch is compared against the
single reference completion: `reference_match` is the byte-identical share,
`reference_first_divergence` the normalized position of the first differing
byte (prefix rule; identical = 1.0). Means are over prompts with a bootstrap
CI, same statistics discipline as the dry-run analyzer.

Reading rule (§7): a faithful half-precision serving sits WELL BELOW 1.000 on
the exact share — the honest-serving ceiling — so the exact number alone
never convicts. The quantization signature is the graded companion collapsing
(divergence immediately after the first tokens) rather than a low exact share
with late divergence.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

B = 2000


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
    ap = argparse.ArgumentParser(description="reference_match / reference_first_divergence")
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--transcript", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ref = {}
    for line in args.reference.open():
        r = json.loads(line)
        ref[r["input_id"]] = r["text"].encode()

    subject = defaultdict(list)
    for line in gzip.open(args.transcript, "rt"):
        r = json.loads(line)
        subject[r["input_id"]].append(r["parsed"]["text"].encode())

    missing = set(subject) - set(ref)
    if missing:
        raise SystemExit(f"{len(missing)} prompts in the transcript have no reference "
                         f"(e.g. {sorted(missing)[:2]}) — generate the full reference first")

    per_match, per_fd = [], []
    for pid in sorted(subject):
        texts = subject[pid]
        r = ref[pid]
        per_match.append(sum(t == r for t in texts) / len(texts))
        per_fd.append(sum(first_divergence(t, r) for t in texts) / len(texts))

    m_mean, m_ci = boot_ci(per_match, "refmatch")
    f_mean, f_ci = boot_ci(per_fd, "reffd")
    result = {
        "reference": args.reference.name,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "transcript": args.transcript.name,
        "n_prompts": len(subject),
        "reference_match": {"mean": m_mean, "ci": m_ci},
        "reference_first_divergence": {"mean": f_mean, "ci": f_ci},
    }
    out = args.out or (Path(__file__).parent / "results" /
                       f"compare_{args.transcript.stem.replace('.jsonl', '')}_vs_{args.reference.stem}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"reference_match={m_mean} {m_ci}  reference_first_divergence={f_mean} {f_ci}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
