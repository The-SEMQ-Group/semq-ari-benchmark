"""Freeze the ARI-Bench v0.1 input slice.

ARI-Bench v0.1 is a fixed, hash-pinned N-item slice of BEIR (standard data — no authored
prompts; see ../../spec/ari-bench-v0.1.md). This tool samples it deterministically and
writes a JSONL of {input_id, text} plus the content hash that pins the slice.

Determinism: for each corpus we sort candidate ids, seed a fixed RNG, sample without
replacement, then sort the selection by id. Same (corpora, n, seed) → byte-identical output
→ identical content hash, anywhere.

Real run (on AWS / with network + `datasets`):
    python build_ari_bench.py --out ari_bench_v0.1.jsonl
Logic check (no download):
    python build_ari_bench.py --dry --n 30 --out /tmp/dry.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# v0.1 composition: a balanced slice across three small BEIR corpora, domain-diverse
# (medical / scientific / financial). Small corpora keep the build light via streaming.
DEFAULT_CORPORA = ["nfcorpus", "scifact", "fiqa"]
DEFAULT_N = 1000
DEFAULT_SEED = 20260701
STREAM_CAP = 20000  # per corpus: cap streamed docs before sampling
MAX_CHARS = 512     # truncate each item to a typical RAG-chunk length (keeps encoding cheap;
                    # reproducibility is length-independent)


def _rng(seed: int):
    import random

    r = random.Random()
    r.seed(seed)
    return r


def sample_slice(candidates: list[tuple[str, str]], n: int, seed: int) -> list[tuple[str, str]]:
    """Deterministic sample of `n` (id, text) pairs from `candidates`."""
    ordered = sorted(candidates, key=lambda p: p[0])
    if n >= len(ordered):
        return ordered
    idx = sorted(_rng(seed).sample(range(len(ordered)), n))
    return [ordered[i] for i in idx]


def load_beir(name: str, stream_cap: int = STREAM_CAP) -> list[tuple[str, str]]:
    """Stream a BEIR corpus as (doc_id, text) via Hugging Face `datasets` (title + body), up
    to `stream_cap` docs. Streaming avoids downloading full corpora; the deterministic sample
    then picks from the first `stream_cap` docs (a fixed, reproducible candidate set)."""
    from datasets import load_dataset

    ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus", streaming=True)
    out = []
    for i, row in enumerate(ds):
        if i >= stream_cap:
            break
        text = (row.get("title", "") + " " + row.get("text", "")).strip()[:MAX_CHARS]
        if text:
            out.append((f"{name}:{row['_id']}", text))
    return out


def dry_corpus(name: str, size: int = 200) -> list[tuple[str, str]]:
    """Tiny synthetic stand-in so the sampling + hashing logic can be validated offline."""
    return [(f"{name}:{k:05d}", f"{name} reproducibility probe document {k}") for k in range(size)]


def build(corpora: list[str], n: int, seed: int, dry: bool) -> list[dict]:
    per = n // len(corpora)
    remainder = n - per * len(corpora)
    rows: list[tuple[str, str]] = []
    for i, name in enumerate(corpora):
        want = per + (1 if i < remainder else 0)
        candidates = dry_corpus(name) if dry else load_beir(name)
        picked = sample_slice(candidates, want, seed + i)
        rows.extend(picked)
    rows.sort(key=lambda p: p[0])  # global stable order
    return [{"input_id": f"ari-bench-v0.1:{k:06d}", "source_id": sid, "text": txt}
            for k, (sid, txt) in enumerate(rows)]


def content_hash(items: list[dict]) -> str:
    h = hashlib.sha256()
    for it in items:
        h.update(it["input_id"].encode()); h.update(b"\x00")
        h.update(it["text"].encode()); h.update(b"\n")
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze the ARI-Bench v0.1 slice.")
    ap.add_argument("--corpora", nargs="+", default=DEFAULT_CORPORA)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry", action="store_true", help="use a synthetic corpus (no download)")
    args = ap.parse_args(argv)

    items = build(args.corpora, args.n, args.seed, args.dry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    chash = content_hash(items)
    tag = " (DRY — synthetic corpus, not real BEIR)" if args.dry else ""
    print(f"wrote {len(items)} items -> {args.out}{tag}")
    print(f"corpora={args.corpora} seed={args.seed}")
    print(f"content_hash={chash}")
    print("→ record this hash as ARI-Bench-v0.1's pin (spec/ari-bench-v0.1.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
