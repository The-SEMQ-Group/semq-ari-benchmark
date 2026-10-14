# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Generate the episode plan for the selectivity study (PREREGISTRATION.md).

Deterministic: the same seed writes the same episodes.json. The plan is committed
with the pre-registration, before any capture.

    python plan.py            # writes episodes.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261007

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Snowflake/snowflake-arctic-embed-l",
    "google/embeddinggemma-2",
]

# Reference captures, one per (model, reference kind).
REFERENCES = {
    "cpu": {"box": "cpu", "device": "cpu", "dtype": "fp32", "threads": 4, "batch": 32, "tf32": "off"},
    "a10g": {"box": "a10g", "device": "cuda", "dtype": "fp32", "threads": 4, "batch": 32, "tf32": "off"},
}

# condition -> (label, box, settings, reference kind). Labels are the pre-registered ones.
CONDITIONS = {
    # benign: same precision, scheduling or placement differs
    "proc": ("benign", "cpu", {"device": "cpu", "dtype": "fp32", "threads": 4, "batch": 32}, "cpu"),
    "threads1": ("benign", "cpu", {"device": "cpu", "dtype": "fp32", "threads": 1, "batch": 32}, "cpu"),
    "batch_cpu": ("benign", "cpu", {"device": "cpu", "dtype": "fp32", "threads": 4, "batch": [8, 128]}, "cpu"),
    "batch_gpu": ("benign", "a10g", {"device": "cuda", "dtype": "fp32", "tf32": "off", "batch": [1, 8, 128]}, "a10g"),
    "a10g_vs_t4": ("benign", "t4", {"device": "cuda", "dtype": "fp32", "tf32": "off", "batch": 32}, "a10g"),
    "cpu_vs_a10g": ("benign", "a10g", {"device": "cuda", "dtype": "fp32", "tf32": "off", "batch": 32}, "cpu"),
    # material: the precision or quantization of the computation changes
    "tf32": ("material", "a10g", {"device": "cuda", "dtype": "fp32", "tf32": "on", "batch": 32}, "a10g"),
    "bf16": ("material", "a10g", {"device": "cuda", "dtype": "bf16", "tf32": "off", "batch": 32}, "a10g"),
    "fp16": ("material", "a10g", {"device": "cuda", "dtype": "fp16", "tf32": "off", "batch": 32}, "a10g"),
    "int8": ("material", "cpu", {"device": "cpu", "dtype": "fp32", "int8": True, "threads": 4, "batch": 32}, "cpu"),
}
PER_LABEL = 300


def main() -> None:
    rng = random.Random(SEED)
    refs = [{"id": f"ref-{kind}-{i:02d}", "model": m, "reference": kind, **REFERENCES[kind]}
            for i, m in enumerate(MODELS) for kind in REFERENCES]
    episodes = []
    for label in ("benign", "material"):
        conds = [c for c, v in CONDITIONS.items() if v[0] == label]
        per_cond = PER_LABEL // len(conds)
        for c in conds:
            _, box, settings, ref_kind = CONDITIONS[c]
            # spread per_cond episodes over the models as evenly as possible, in random order
            order = MODELS * (per_cond // len(MODELS) + 1)
            models = order[:per_cond]
            rng.shuffle(models)
            for k, m in enumerate(models):
                s = dict(settings)
                if isinstance(s["batch"], list):
                    s["batch"] = rng.choice(s["batch"])
                s.setdefault("threads", 4)
                s.setdefault("tf32", "off")
                episodes.append({"id": f"{c}-{k:03d}", "model": m, "condition": c, "label": label,
                                 "box": box, "reference": f"ref-{ref_kind}-{MODELS.index(m):02d}", **s})
    rng.shuffle(episodes)
    plan = {"seed": SEED, "n_inputs": 1000, "inputs": "data/ari-bench-v0.1.jsonl",
            "references": refs, "episodes": episodes}
    (HERE / "episodes.json").write_text(json.dumps(plan, indent=1) + "\n")
    counts = {}
    for e in episodes:
        counts[(e["label"], e["condition"])] = counts.get((e["label"], e["condition"]), 0) + 1
    for k in sorted(counts):
        print(k, counts[k])
    print("total", len(episodes), "references", len(refs))


if __name__ == "__main__":
    main()
