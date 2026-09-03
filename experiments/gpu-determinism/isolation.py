"""Isolation 2x2 (TF32 on/off × determinism on/off) at fp32, to attribute the GPU
cross-process (`proc`) drift to TF32 vs nondeterministic kernels — and to record the
framework's *default* allow_tf32 (settles the "by default" question). See RESULTS.md.

  python isolation.py --n 512 --out ~/iso_out
"""
import argparse, subprocess, sys, os
from pathlib import Path

S = {
    "BAAI/bge-large-en-v1.5": 0.077546, "BAAI/bge-m3": 0.082476,
    "intfloat/multilingual-e5-large": 0.076236,
    "sentence-transformers/all-mpnet-base-v2": 0.096253,
    "sentence-transformers/all-MiniLM-L6-v2": 0.132349,
    "nomic-ai/nomic-embed-text-v1.5": 0.093399,
    "mixedbread-ai/mxbai-embed-large-v1": 0.078308,
    "Snowflake/snowflake-arctic-embed-l": 0.082256,
}
# (tag, tf32, deterministic) — the 2x2. Two are new; two re-confirm the prior run, all in one
# internally-consistent session.
CONFIGS = [("tf32on_nondet", "on", False), ("tf32off_nondet", "off", False),
           ("tf32on_det", "on", True), ("tf32off_det", "off", True)]
HERE = Path(__file__).resolve().parent
TOOL = HERE.parent.parent / "ari" / "tools" / "gpu_capture.py"
INPUTS = HERE.parent.parent / "data" / "ari-bench-v0.1.jsonl"


def cap(model, tag, tf32, det, s, n, out):
    slug = model.replace("/", "_")
    cmd = [sys.executable, str(TOOL), "capture", "--model", model, "--device", "cuda",
           "--dtype", "fp32", "--tf32", tf32, "--scale-s", str(s), "--conditions", "same,proc",
           "--inputs", str(INPUTS), "--n", str(n), "--outdir", str(Path(out) / f"{slug}__{tag}"),
           "--tag", f"{slug}__{tag}"]
    if det:
        cmd += ["--deterministic"]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default=os.path.expanduser("~/iso_out"))
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)

    # framework defaults, BEFORE any flag is touched — this is what a real deployment gets
    import torch
    print(f"TORCH {torch.__version__} | DEFAULT matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"| cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} | GPU={torch.cuda.get_device_name(0)}",
          flush=True)

    for m in S:
        for tag, tf32, det in CONFIGS:
            print(f">>> {m} :: {tag}", flush=True)
            cap(m, tag, tf32, det, S[m], a.n, a.out)

    tarball = f"{a.out}.tar.gz"
    subprocess.run(["tar", "czf", tarball, "-C", str(Path(a.out).parent), Path(a.out).name], check=True)
    print(f"DONE -> {tarball}", flush=True)


if __name__ == "__main__":
    main()
