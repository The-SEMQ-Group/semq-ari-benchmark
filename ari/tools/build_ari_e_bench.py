"""Freeze the ARI-E-Bench v0.1 task slice.

ARI-E-Bench v0.1 is a fixed, hash-pinned slice of SWE-bench Verified (standard data
-- no authored tasks; see ../../spec/ari-e-bench-v0.1.md §4). This tool selects it
deterministically and writes a JSONL of {input_id, instance_id, repo, base_commit,
text} plus the content hash that pins the slice.

The frozen record pins the identity every rollout runs on and the problem statement
the agent sees. It does NOT copy the held-out tests: grading loads each instance from
the SWE-bench Verified dataset at the pinned revision (recorded below and in the
data README), so the tests are frozen by that revision, not re-stated here.

Determinism: cap each repo (so django's 231 instances do not dominate, and its slow
test setups do not eat the run), pool the caps, seed a fixed RNG, sample without
replacement to N, then sort the selection by instance_id. Same (revision, cap, n,
seed) -> byte-identical output -> identical content hash, anywhere.

    python ari/tools/build_ari_e_bench.py --out data/ari-e-bench-v0.1.jsonl

The hash scheme is ARI-Bench's (SHA-256 over ordered `input_id\0text\n` records), so
every prefix hash is also valid and a reduced run can prove it measured the first N
as `prefix:N`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

DATASET = "SWE-bench/SWE-bench_Verified"
# Pinned 2026-09-07. The canonical current dataset: it carries the prebuilt eval
# `image` and `eval_script` the swebench 5.x harness grades against, and its problem
# statements + base/env commits are byte-identical to the older princeton-nlp mirror
# (verified 500/500). The grader loads instances (and their held-out tests) from this
# exact revision, so bumping it re-freezes the bench.
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"

DEFAULT_N = 200          # spec §4 target; power sweet spot (200 cases -> 0.965 at a 20pt gap)
DEFAULT_REPO_CAP = 25    # max instances per repo before sampling (diversity + wall-time)
DEFAULT_SEED = 20260907

# Fields carried into the frozen record. `text` (problem_statement) is what the agent
# sees and what the hash covers; the rest identify the instance for the grader.
CARRY = ("instance_id", "repo", "base_commit", "version", "environment_setup_commit")


def _rng(seed: int):
    import random

    r = random.Random()
    r.seed(seed)
    return r


def select(rows: list[dict], n: int, repo_cap: int, seed: int) -> list[dict]:
    """Deterministic, repo-capped sample of `n` instances, sorted by instance_id."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r["repo"]].append(r)
    pool: list[dict] = []
    for repo in sorted(by_repo):
        capped = sorted(by_repo[repo], key=lambda r: r["instance_id"])[:repo_cap]
        pool.extend(capped)
    pool.sort(key=lambda r: r["instance_id"])
    if n < len(pool):
        rng = _rng(seed)
        pool = rng.sample(pool, n)
    return sorted(pool, key=lambda r: r["instance_id"])


def content_hash(records: list[dict]) -> str:
    h = hashlib.sha256()
    for rec in records:
        h.update(rec["input_id"].encode() + b"\x00" + rec["text"].encode() + b"\n")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Freeze the ARI-E-Bench v0.1 task slice.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--repo-cap", type=int, default=DEFAULT_REPO_CAP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET, split="test", revision=DATASET_REVISION)
    rows = [dict(r) for r in ds]
    chosen = select(rows, args.n, args.repo_cap, args.seed)

    records = []
    for i, r in enumerate(chosen):
        rec = {"input_id": f"ari-e-bench-v0.1:{i:06d}"}
        rec.update({k: r[k] for k in CARRY})
        rec["text"] = r["problem_statement"]
        records.append(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    digest = content_hash(records)
    repos = defaultdict(int)
    for r in chosen:
        repos[r["repo"]] += 1
    print(f"wrote {len(records)} tasks -> {args.out}")
    print(f"content_hash = {digest}")
    print(f"dataset = {DATASET}@{DATASET_REVISION[:12]}")
    print(f"repos ({len(repos)}): " + ", ".join(f"{k.split('/')[-1]}:{v}"
                                                 for k, v in sorted(repos.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
