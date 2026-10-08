# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""On-instance driver for the GPU-determinism matrix. Runs GPU captures for the 8 encoders
(+ one 7B) under the controlled config cells, then tars the outputs for S3 upload. CPU-fp32
reference captures are produced locally (off-instance); `mach` is compared offline.

  python run_matrix.py --label a10g --tf32-available true  --n 512 --out ~/gpu_out
  python run_matrix.py --label t4   --tf32-available false --n 512 --out ~/gpu_out
"""
import argparse, subprocess, sys, os
from pathlib import Path

# The encoders. Under the v0.2 probe every capture of a model quantizes over the same
# fixed range 2/sqrt(dim), so bin edges match across machines without a registry scale.
S = (  # v0.2 probe: the range is 2/sqrt(dim), so no per-model scale
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Snowflake/snowflake-arctic-embed-l",
)
SEVENB = "intfloat/e5-mistral-7b-instruct"   # fp32 OOMs a 24GB card -> bf16 only
HERE = Path(__file__).resolve().parent
TOOL = HERE.parent.parent / "ari" / "tools" / "gpu_capture.py"
INPUTS = HERE.parent.parent / "data" / "ari-bench-v0.1.jsonl"


FAILED: list[str] = []


def cap(model, tag, device, dtype, tf32, det, s, conds, n, out):
    slug = model.replace("/", "_")
    d = Path(out) / f"{slug}__{tag}"
    cmd = [sys.executable, str(TOOL), "capture", "--model", model, "--device", device,
           "--dtype", dtype, "--tf32", tf32, "--inputs", str(INPUTS), "--n", str(n),
           "--outdir", str(d), "--tag", f"{slug}__{tag}"]
    if s is not None:
        cmd += ["--scale-s", str(s)]
    if det:
        cmd += ["--deterministic"]
    if conds:
        cmd += ["--conditions", conds]
    print(f">>> {tag}: {model}", flush=True)
    # One model failing (e.g. a .bin-only checkpoint on an old torch) must not
    # discard the rest of the matrix; failures are reported at the end.
    if subprocess.run(cmd).returncode != 0:
        FAILED.append(tag + " " + model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)              # a10g | t4
    ap.add_argument("--tf32-available", default="false")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default=os.path.expanduser("~/gpu_out"))
    ap.add_argument("--skip-7b", action="store_true", help="skip the 7B (won't fit small GPUs)")
    a = ap.parse_args()
    tf32_ok = a.tf32_available == "true"
    Path(a.out).mkdir(parents=True, exist_ok=True)

    for m in S:
        s = None  # gpu_capture uses the probe's fixed range
        # fp32 at true precision (TF32 off), deterministic — the GPU<->GPU mach reference + H5
        cap(m, f"{a.label}_fp32_det", "cuda", "fp32", "off", True, s, "same,proc", a.n, a.out)
        if tf32_ok:
            # fp32 with TF32 on (Ampere default) — the hidden-cliff (H3) + H1 default
            cap(m, f"{a.label}_fp32_tf32on", "cuda", "fp32", "on", False, s, "same,proc", a.n, a.out)
        # bf16 — the precision cliff on GPU (H2)
        cap(m, f"{a.label}_bf16", "cuda", "bf16", "off", False, s, "", a.n, a.out)

    # 7B decoder embedder: calibrate on this machine (no fixed s in the registry yet), bf16 only.
    # `same` only (its cross-process `proc` needs the two-capture method to avoid a 2-copy OOM).
    if not a.skip_7b:
        cap(SEVENB, f"{a.label}_bf16", "cuda", "bf16", "off", False, None, "same", a.n, a.out)

    tarball = f"{a.out}_{a.label}.tar.gz"
    subprocess.run(["tar", "czf", tarball, "-C", str(Path(a.out).parent), Path(a.out).name], check=True)
    print(f"DONE -> {tarball}", flush=True)


if __name__ == "__main__":
    main()
    if FAILED:
        print("FAILED captures:\n  " + "\n  ".join(FAILED), flush=True)
        sys.exit(1)
