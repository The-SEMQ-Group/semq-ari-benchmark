"""ARI-R capture session: Salesforce/SFR-Embedding-2_R, self-hosted on GPU.

Mirrors the protocol that produced the 8 self-hosted leaderboard rows
(run_report selfhosted path + selfhosted_conc_time + the prec fp32-vs-bf16
contrast of commit 2465eb8), adapted for a 7B model on an L40S:

- weights load once in the checkpoint dtype (bf16) and are upcast to fp32 on
  device for the canonical passes (lossless: bf16 values are exact in fp32);
  the prec pass casts back to bf16 (roundtrip-exact for bf16-born weights)
- determinism config: CUBLAS_WORKSPACE_CONFIG set before CUDA init, TF32 off,
  deterministic algorithms on (warn_only), BLAS threads pinned
- every condition's RAW FLOAT MATRIX is retained (the float-gate lesson:
  keep the evidence so float detectors can run retroactively)

Conditions, all compared against the fp32 base E0:
  same  in-process fp32 re-encode
  proc  fresh-subprocess fp32 encode (worker below)
  conc  8 threads encode shards concurrently, batch 8 (vs calm batch 16)
  time  fp32 re-encode at the end of the session (confirmatory for local
        deterministic compute, per the panel's documented semantics)
  prec  bf16 encode (diagnostic, not in the comparable core)
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
PIN = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
os.environ.update(PIN)

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ari import metrics, report          # noqa: E402
from ari.inputs import load_ari_bench    # noqa: E402
from ari.probe import load_probe         # noqa: E402

MODEL_ID = "Salesforce/SFR-Embedding-2_R"
REVISION = "f62d15f411ca97b66acc0f34da2a65f3420b55b0"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

WORKER_SRC = '''
import json, os, sys
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)
from sentence_transformers import SentenceTransformer
p = json.load(open(sys.argv[1]))
m = SentenceTransformer(p["model"], device="cuda", trust_remote_code=True,
                        revision=p["revision"],
                        model_kwargs={"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True})
m.to(torch.float32)
v = m.encode(p["texts"], normalize_embeddings=True, convert_to_numpy=True, batch_size=16)
np.save(p["out"], np.asarray(v, dtype=np.float32))
'''


def main() -> int:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    inputs = load_ari_bench(HERE / "data" / "ari-bench-v0.1.jsonl")
    texts = inputs.texts
    print(f"inputs: {len(texts)}  hash={inputs.content_hash[:16]}", flush=True)

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(
        MODEL_ID, device="cuda", trust_remote_code=True, revision=REVISION,
        model_kwargs={"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True})
    model.to(torch.float32)
    print(f"model loaded+fp32 in {time.time()-t0:.0f}s", flush=True)

    def enc(ts, bs=16):
        return np.asarray(model.encode(ts, normalize_embeddings=True,
                                       convert_to_numpy=True, batch_size=bs),
                          dtype=np.float32)

    def timed(name, fn):
        t = time.time()
        v = fn()
        np.save(OUT / f"vecs_{name}.npy", v)   # float-gate lesson: retain floats
        print(f"{name}: {v.shape} in {time.time()-t:.0f}s", flush=True)
        return v

    base_vecs = timed("base", lambda: enc(texts))
    dim = int(base_vecs.shape[1])
    probe = load_probe(base_vecs, backend="semq")
    base = probe.encode(base_vecs)
    print(f"probe: s={probe.s:.6f} dim={dim}", flush=True)

    vecs = {}
    vecs["same"] = timed("same", lambda: enc(texts))

    # conc: 8 threads, shards, small batches
    def conc_pass():
        shards = [list(x) for x in np.array_split(np.arange(len(texts)), 8)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            parts = list(ex.map(lambda idx: enc([texts[i] for i in idx], bs=8), shards))
        return np.vstack(parts)
    vecs["conc"] = timed("conc", conc_pass)

    # prec: bf16 diagnostic (cast back to fp32 afterwards; bf16-born weights roundtrip exactly)
    model.to(torch.bfloat16)
    vecs["prec"] = timed("prec", lambda: enc(texts))
    model.to(torch.float32)

    # time: end-of-session re-encode (confirmatory for local deterministic compute)
    vecs["time"] = timed("time", lambda: enc(texts))

    # proc: fresh subprocess with its own CUDA context
    wpath = OUT / "_worker.py"
    wpath.write_text(WORKER_SRC)
    payload = OUT / "_payload.json"
    pout = OUT / "vecs_proc.npy"
    payload.write_text(json.dumps({"model": MODEL_ID, "revision": REVISION,
                                   "texts": texts, "out": str(pout)}))
    t = time.time()
    subprocess.run([sys.executable, str(wpath), str(payload)], check=True,
                   env={**os.environ})
    vecs["proc"] = np.load(pout)
    print(f"proc: {vecs['proc'].shape} in {time.time()-t:.0f}s", flush=True)

    cond_codes = {c: probe.encode(v) for c, v in vecs.items()}
    metrics_by = {c: metrics.aggregate(base, codes) for c, codes in cond_codes.items()}
    for c, m in metrics_by.items():
        print(f"{c}: HER={m.HER:.6f} Hbar={m.Hbar:.6f} CI={m.HER_ci}", flush=True)

    environment = {
        "blas": "torch-default", "threads": torch.get_num_threads(),
        "hardware": "x86_64+L40S(g6e.xlarge)", "precision": "fp32",
        "library_versions": {
            "probe_backend": probe.backend, "threads_pinned": "true",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "tf32": "off", "deterministic_algorithms": "true",
            "model_revision": REVISION,
        },
    }
    rep = report.build_report(
        agent_id=MODEL_ID, input_set=inputs.name, input_content_hash=inputs.content_hash,
        environment=environment, metrics_by_condition=metrics_by,
        codes_by_condition=cond_codes, agent_class="self_hosted",
        fingerprint={"dim": dim, "s": round(float(probe.s), 6)})
    (OUT / "report.json").write_text(json.dumps(rep, indent=2) + "\n")
    meta = {"model": MODEL_ID, "revision": REVISION, "s": float(probe.s), "dim": dim,
            "n": len(texts), "input_content_hash": inputs.content_hash,
            "gpu": torch.cuda.get_device_name(0)}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ARI={rep['ARI']:.4f} -> {OUT/'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
