# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""Pull ARI-E trajectories out of nvidia/Open-SWE-Traces.

That dataset is a crossed design and it is the reason this experiment can run
at all. Two agent scaffolds, SWE-agent and OpenHands, were each run with two
models, Minimax-M2.5 and Qwen3.5-122B, over about 20,000 SWE-bench-style
instances, with roughly ten rollouts per instance. Neither scaffold was built
by us, so the harness effect measured here is not an artifact of our own design
choices.

The structure maps onto ARI-E directly:

    config (openhands | sweagent)  ->  harness
    split  (minimax_m25 | qwen35_122b)  ->  model
    instance_id                    ->  case
    trajectory_id                  ->  one run of that case
    resolved == 1                  ->  outcome

`resolved` comes from running the repository's own tests against the agent's
patch, so it is a checker the agent never saw. That is the property ARI-E needs
and the reason this dataset works where a judge score would not.

The full transcripts are large and this keeps none of them. Each row collapses
to an outcome and an ordered list of tool names, which is all the metric reads.

Usage:
    python extract.py                 # bounded sample, good for a first look
    python extract.py --max-rows 0    # everything, slow
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

DATASET = "nvidia/Open-SWE-Traces"
HARNESSES = ("sweagent", "openhands")
MODELS = ("qwen35_122b", "minimax_m25")


def tool_sequence(trajectory) -> list[str]:
    """Ordered tool names the agent called.

    Only the function name is kept. Arguments hold repository paths and shell
    text that differ between runs for reasons that have nothing to do with the
    scaffold, so including them would make every pair of runs look different.
    """
    out = []
    for msg in trajectory or ():
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or ():
            if isinstance(call, dict):
                fn = call.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else None
                if name:
                    out.append(str(name))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=12000,
                    help="rows per harness/model cell, 0 for all")
    ap.add_argument("--out", type=Path, default=RESULTS / "trajectories.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tools_seen = defaultdict(Counter)
    kept = Counter()
    unknown_outcome = Counter()

    with args.out.open("w") as fh:
        for harness in HARNESSES:
            for model in MODELS:
                ds = load_dataset(DATASET, harness, split=model, streaming=True)
                n = 0
                for row in ds:
                    resolved = int(row.get("resolved", -1))
                    if resolved < 0:
                        # -1 means the grade was never recorded. An unknown
                        # outcome is not a failure, so it cannot be counted as
                        # one.
                        unknown_outcome[(harness, model)] += 1
                        n += 1
                        if args.max_rows and n >= args.max_rows:
                            break
                        continue
                    seq = tool_sequence(row.get("trajectory"))
                    tools_seen[harness].update(seq)
                    fh.write(json.dumps({
                        "case_id": row["instance_id"],
                        "harness": harness,
                        "model": model,
                        "trajectory_id": row["trajectory_id"],
                        "outcome": resolved == 1,
                        "tool_calls": seq,
                    }) + "\n")
                    kept[(harness, model)] += 1
                    n += 1
                    if args.max_rows and n >= args.max_rows:
                        break
                print(f"  {harness:<10} {model:<12} kept {kept[(harness, model)]:>6}"
                      f"  unknown-outcome {unknown_outcome[(harness, model)]:>5}",
                      flush=True)

    print(f"\nwrote {args.out}  ({sum(kept.values())} runs)")

    print("\nTool vocabulary per harness. The two scaffolds do not share one,")
    print("so a cross-harness trajectory comparison needs a mapping first.")
    for h, c in tools_seen.items():
        top = ", ".join(f"{name} ({n})" for name, n in c.most_common(8))
        print(f"  {h:<10} {len(c):>3} distinct: {top}")

    (RESULTS / "extract_summary.json").write_text(json.dumps({
        "dataset": DATASET,
        "max_rows_per_cell": args.max_rows,
        "kept": {f"{h}/{m}": n for (h, m), n in kept.items()},
        "unknown_outcome": {f"{h}/{m}": n for (h, m), n in unknown_outcome.items()},
        "tool_vocabulary": {h: dict(c.most_common()) for h, c in tools_seen.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
