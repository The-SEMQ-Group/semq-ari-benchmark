"""Subprocess encode worker for the self-hosted pilot's `proc` condition.

Loads the model fresh in this process and encodes the given texts, so the parent can test
**cross-process** reproducibility (not just in-process). Thread pinning, if wanted, is set
by the parent in this process's environment *before* it starts (so it takes effect before
torch imports here). Input/output paths are passed as a JSON payload file.
"""
import json
import sys

import numpy as np


def main() -> int:
    payload = json.load(open(sys.argv[1]))
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(payload["model"], device="cpu", trust_remote_code=True)
    vecs = model.encode(payload["texts"], normalize_embeddings=True, convert_to_numpy=True)
    np.save(payload["out"], np.asarray(vecs, dtype=np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
