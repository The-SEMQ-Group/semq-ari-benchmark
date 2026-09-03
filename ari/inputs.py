"""ARI-Bench input loading.

ARI-Bench v0.1 is a fixed, hash-pinned slice of BEIR (standard data — no authored
prompts; see ../../spec/ari-bench-v0.1.md). The slice is identified by a content hash over
the ordered item texts, so any submitter can confirm they ran the exact same inputs in the
exact same order (required for comparable audit hashes).

`load_ari_bench` loads a JSONL of `{"input_id": ..., "text": ...}` rows. `sample_inputs`
returns a tiny deterministic set for tests / dry-runs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InputSet:
    name: str
    ids: list[str]
    texts: list[str]

    @property
    def content_hash(self) -> str:
        """SHA-256 over the ordered (id, text) pairs — the pin for this slice."""
        h = hashlib.sha256()
        for i, t in zip(self.ids, self.texts):
            h.update(i.encode())
            h.update(b"\x00")
            h.update(t.encode())
            h.update(b"\n")
        return h.hexdigest()

    def __len__(self) -> int:
        return len(self.ids)


def load_ari_bench(path: str | Path, name: str = "ARI-Bench-v0.1") -> InputSet:
    ids, texts = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["input_id"]))
            texts.append(str(row["text"]))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate input_id in input set — ordering/hashing would be ambiguous")
    return InputSet(name=name, ids=ids, texts=texts)


def sample_inputs(n: int = 64) -> InputSet:
    """Deterministic placeholder inputs for dry-runs. NOT ARI-Bench — the real slice is a
    fixed BEIR selection produced by the (mechanical) freezing step."""
    ids = [f"sample_{k:04d}" for k in range(n)]
    texts = [f"reproducibility probe input number {k}" for k in range(n)]
    return InputSet(name="dryrun-sample", ids=ids, texts=texts)
