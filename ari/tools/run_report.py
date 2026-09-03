"""Generate a schema-valid ARI report for one agent and (optionally) add it to the
leaderboard. Runs baseline E0 + the conditions we can measure now (`same`, `proc`), builds
the report, scores it with the leaderboard scorer, and appends a row.

  # API (key from env, e.g. OPENAI_API_KEY)
  python run_report.py --agent openai --inputs data/ari-bench-v0.1.jsonl --submit
  # self-hosted (local; proc = fresh subprocess, BLAS threads pinned)
  python run_report.py --agent bge --model BAAI/bge-large-en-v1.5 --inputs data/ari-bench-v0.1.jsonl --submit
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]   # the `ari` package directory
REPO = PKG.parent
sys.path.insert(0, str(REPO))

from ari import metrics, report                                              # noqa: E402
from ari.agents import DEFAULT_API_MODELS                                    # noqa: E402
from ari.inputs import InputSet, sample_inputs, load_ari_bench              # noqa: E402
from ari.probe import load_probe                                            # noqa: E402

# BLAS thread pinning for self-hosted runs, so the `proc` cross-process test measures the
# model's reproducibility, not BLAS thread-scheduling noise. Set before torch imports and
# inherited by the fresh subprocess.
PIN = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
WORKER = PKG / "tools" / "_encode_worker.py"
API = DEFAULT_API_MODELS


def api_codes(agent, probe, texts):
    """`same`/`proc` codes for a black-box API. The baseline (E0) is computed by the caller
    and reused, so this does not re-encode the corpus a third (billed) time."""
    same = probe.encode(agent.encode(texts))          # within-session re-encode (persistent client)
    proc = probe.encode(agent.encode_fresh(texts))    # fresh client/connection
    return {"same": same, "proc": proc}


def _load_st(model_id):
    # trust_remote_code=True is required by some encoders (e.g. nomic). It executes code from
    # the model repo — acceptable for the vetted, widely-used models in the panel; a stricter
    # deployment should pin/allowlist model revisions.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, device="cpu", trust_remote_code=True)


def selfhosted_codes(model_id, probe_backend, texts):
    model = _load_st(model_id)

    def enc(ts):
        return np.asarray(model.encode(ts, normalize_embeddings=True, convert_to_numpy=True),
                          dtype=np.float32)

    base_vecs = enc(texts)
    dim = int(base_vecs.shape[1])
    probe = load_probe(base_vecs, backend=probe_backend)
    base = probe.encode(base_vecs)
    same = probe.encode(enc(texts))                   # in-process re-encode
    with tempfile.TemporaryDirectory() as d:          # proc: fresh subprocess (real cross-process)
        out, pj = f"{d}/v.npy", f"{d}/payload.json"
        json.dump({"model": model_id, "texts": texts, "out": out}, open(pj, "w"))
        subprocess.run([sys.executable, str(WORKER), pj], check=True, env={**os.environ})
        proc = probe.encode(np.load(out))
    return probe, base, {"same": same, "proc": proc}, dim


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate + score an ARI report for one agent.")
    ap.add_argument("--agent", required=True,
                    choices=["openai", "voyage", "cohere", "mistral", "gemini", "bge"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--dimensions", type=int, default=None, help="OpenAI: output dimensions")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--inputs", type=Path)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N inputs of --inputs")
    ap.add_argument("--probe-backend", choices=["auto", "semq", "mock"], default="semq")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--submit", action="store_true", help="score + append to leaderboard")
    args = ap.parse_args(argv)

    inputs = load_ari_bench(args.inputs) if args.inputs else sample_inputs(args.n)
    if args.limit and len(inputs.ids) > args.limit:
        inputs = InputSet(inputs.name, inputs.ids[:args.limit], inputs.texts[:args.limit])

    if args.agent == "bge":
        os.environ.update(PIN)   # pin BLAS threads (parent + inherited subprocess) before torch
        model_id = args.model or "BAAI/bge-large-en-v1.5"
        probe, base_codes, cond_codes, dim = selfhosted_codes(model_id, args.probe_backend, inputs.texts)
        agent_id, agent_class = model_id, "self_hosted"
        import torch
        environment = {"blas": "torch-default", "threads": torch.get_num_threads(),
                       "hardware": os.uname().machine, "precision": "fp32",
                       "library_versions": {"probe_backend": probe.backend, "threads_pinned": "true"}}
    else:
        cls, default_model = API[args.agent]
        kwargs = {"model_id": args.model or default_model, "max_workers": args.max_workers}
        if args.agent == "openai" and args.dimensions:
            kwargs["dimensions"] = args.dimensions
        agent = cls(**kwargs)
        base_vecs = agent.encode(inputs.texts)
        dim = int(base_vecs.shape[1])
        probe = load_probe(base_vecs, backend=args.probe_backend)
        base_codes = probe.encode(base_vecs)
        cond_codes = api_codes(agent, probe, inputs.texts)
        agent_id, agent_class = f"{args.agent}/{agent.agent_id}", "api"
        environment = {"blas": "provider-internal", "threads": 0, "hardware": "provider-internal",
                       "precision": "provider-internal",
                       "library_versions": {"probe_backend": probe.backend, "snapshot": str(agent.snapshot)}}

    metrics_by = {c: metrics.aggregate(base_codes, codes) for c, codes in cond_codes.items()}
    rep = report.build_report(
        agent_id=agent_id, input_set=inputs.name, input_content_hash=inputs.content_hash,
        environment=environment, metrics_by_condition=metrics_by, codes_by_condition=cond_codes,
        agent_class=agent_class,
        fingerprint={"dim": dim, "s": round(float(getattr(probe, "s", 0.0)), 6)})

    sub_dir = REPO / "leaderboard" / "submissions"
    sub_dir.mkdir(parents=True, exist_ok=True)
    rep_path = sub_dir / f"{agent_id.replace('/', '_')}.json"
    rep_path.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"report -> {rep_path}  (ARI={rep['ARI']:.4f}, agent_class={agent_class}, n={len(inputs)})")

    if args.submit:
        return subprocess.call([sys.executable, str(REPO / "leaderboard" / "scoring" / "score.py"),
                                str(rep_path), "--append", str(REPO / "leaderboard" / "leaderboard.json")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
