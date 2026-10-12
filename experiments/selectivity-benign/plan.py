# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Generate the capture plan for the benign-alarm study (PREREGISTRATION.md).

Deterministic: one capture per (model, condition) cell plus two references per model.
The plan is committed with the pre-registration, before any capture.

    python plan.py            # writes episodes.json
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The nine models of the first study, then sixteen more. Each model's CPU captures run on
# one CPU host (cpu_a or cpu_b), so CPU references and CPU episodes share a machine.
MODELS = [
    ("sentence-transformers/all-MiniLM-L6-v2", "cpu_a"),
    ("sentence-transformers/all-mpnet-base-v2", "cpu_b"),
    ("BAAI/bge-large-en-v1.5", "cpu_a"),
    ("BAAI/bge-m3", "cpu_b"),
    ("intfloat/multilingual-e5-large", "cpu_a"),
    ("nomic-ai/nomic-embed-text-v1.5", "cpu_b"),
    ("mixedbread-ai/mxbai-embed-large-v1", "cpu_a"),
    ("Snowflake/snowflake-arctic-embed-l", "cpu_b"),
    ("google/embeddinggemma-2", "cpu_a"),
    ("sentence-transformers/all-MiniLM-L12-v2", "cpu_b"),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "cpu_a"),
    ("sentence-transformers/multi-qa-mpnet-base-dot-v1", "cpu_b"),
    ("BAAI/bge-small-en-v1.5", "cpu_a"),
    ("BAAI/bge-base-en-v1.5", "cpu_b"),
    ("intfloat/e5-small-v2", "cpu_a"),
    ("intfloat/e5-base-v2", "cpu_b"),
    ("intfloat/e5-large-v2", "cpu_a"),
    ("intfloat/multilingual-e5-base", "cpu_b"),
    ("thenlper/gte-small", "cpu_a"),
    ("thenlper/gte-base", "cpu_b"),
    ("thenlper/gte-large", "cpu_a"),
    ("Snowflake/snowflake-arctic-embed-s", "cpu_b"),
    ("Snowflake/snowflake-arctic-embed-m", "cpu_a"),
    ("ibm-granite/granite-embedding-small-english-r2", "cpu_b"),
    ("Qwen/Qwen3-Embedding-0.6B", "cpu_a"),
]

CPU = {"device": "cpu", "dtype": "fp32", "threads": 4, "batch": 32, "tf32": "off"}
GPU = {"device": "cuda", "dtype": "fp32", "threads": 4, "batch": 32, "tf32": "off"}

# condition -> (label, box ("cpu" means the model's CPU host), settings, reference kind)
CONDITIONS = {
    # benign: same precision; scheduling, shapes or device differ
    "proc": ("benign", "cpu", CPU, "cpu"),
    "threads1": ("benign", "cpu", {**CPU, "threads": 1}, "cpu"),
    "batch8_cpu": ("benign", "cpu", {**CPU, "batch": 8}, "cpu"),
    "batch128_cpu": ("benign", "cpu", {**CPU, "batch": 128}, "cpu"),
    "proc_gpu": ("benign", "a10g", GPU, "a10g"),
    "batch1_gpu": ("benign", "a10g", {**GPU, "batch": 1}, "a10g"),
    "batch8_gpu": ("benign", "a10g", {**GPU, "batch": 8}, "a10g"),
    "batch128_gpu": ("benign", "a10g", {**GPU, "batch": 128}, "a10g"),
    "batch512_gpu": ("benign", "a10g", {**GPU, "batch": 512}, "a10g"),
    "cpu_vs_a10g": ("benign", "a10g", GPU, "cpu"),
    "cpu_vs_t4": ("benign", "t4", GPU, "cpu"),
    "cpu_vs_l4": ("benign", "l4", GPU, "cpu"),
    "a10g_vs_t4": ("benign", "t4", GPU, "a10g"),
    "a10g_vs_l4": ("benign", "l4", GPU, "a10g"),
    # material (secondary): does each threshold still catch a real change on new models?
    "tf32": ("material", "a10g", {**GPU, "tf32": "on"}, "a10g"),
    "bf16": ("material", "a10g", {**GPU, "dtype": "bf16"}, "a10g"),
}


def main() -> None:
    refs, episodes = [], []
    for i, (m, cpu_box) in enumerate(MODELS):
        refs.append({"id": f"ref-cpu-{i:02d}", "model": m, "reference": "cpu", "box": cpu_box, **CPU})
        refs.append({"id": f"ref-a10g-{i:02d}", "model": m, "reference": "a10g", "box": "a10g", **GPU})
        for c, (label, box, settings, ref_kind) in CONDITIONS.items():
            episodes.append({"id": f"{c}-{i:02d}", "model": m, "condition": c, "label": label,
                             "box": cpu_box if box == "cpu" else box,
                             "reference": f"ref-{ref_kind}-{i:02d}", **settings})
    plan = {"n_inputs": 1000, "inputs": "data/ari-bench-v0.1.jsonl", "references": refs, "episodes": episodes}
    (HERE / "episodes.json").write_text(json.dumps(plan, indent=1) + "\n")
    boxes = {}
    for e in refs + episodes:
        boxes[e["box"]] = boxes.get(e["box"], 0) + 1
    print("cells", len(episodes), "benign", sum(e["label"] == "benign" for e in episodes),
          "material", sum(e["label"] == "material" for e in episodes), "references", len(refs), "per box", boxes)


if __name__ == "__main__":
    main()
