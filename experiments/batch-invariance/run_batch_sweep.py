"""On-instance driver for the batch-invariance matrix. See README.md for the design.

Runs three cells over five batch sizes, holding the model, the process configuration and the
QBIN scale fixed. Inside each cell every batch size is compared against batch 32, which is the
batch the published panel used.

  python run_batch_sweep.py --label h100 --n 1000 --out ~/batch_out
  python run_batch_sweep.py --label h100 --models all --publish-s3 s3://bucket/prefix

Each cell writes its comparisons as soon as it finishes. Pass --publish-s3 on any instance
that terminates on a timer, so a shutdown cannot take the only copy of the run with it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Canonical QBIN scales from spec/fingerprints-v0.1.csv, the same fixed values the
# gpu-determinism matrix uses. A per-capture calibration would move the bin edges with the
# data and could hide or invent disagreement, so the scale is fixed everywhere.
S = {
    "sentence-transformers/all-MiniLM-L6-v2": 0.132349,
    "BAAI/bge-large-en-v1.5": 0.077546,
    "BAAI/bge-m3": 0.082476,
    "intfloat/multilingual-e5-large": 0.076236,
    "sentence-transformers/all-mpnet-base-v2": 0.096253,
    "nomic-ai/nomic-embed-text-v1.5": 0.093399,
    "mixedbread-ai/mxbai-embed-large-v1": 0.078308,
    "Snowflake/snowflake-arctic-embed-l": 0.082256,
}
DEFAULT_MODELS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5"]

BATCHES = [1, 8, 32, 128, 512]
REFERENCE_BATCH = 32

# (cell, tf32, deterministic). Cell A is the T1 test: everything fixed except the batch shape.
# B removes the deterministic pin, so a difference between A and B is kernel and algorithm
# selection rather than the batch itself. C adds TF32, the lever the panel already showed.
CELLS = [("A", "off", True), ("B", "off", False), ("C", "on", False)]

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent.parent / "ari" / "tools" / "gpu_capture.py"
INPUTS = HERE.parent.parent / "data" / "ari-bench-v0.1.jsonl"


def capture(model, tag, tf32, deterministic, batch, n, outdir):
    d = Path(outdir) / tag
    cmd = [sys.executable, str(TOOL), "capture", "--model", model, "--device", "cuda",
           "--dtype", "fp32", "--tf32", tf32, "--scale-s", str(S[model]),
           "--inputs", str(INPUTS), "--n", str(n), "--batch-size", str(batch),
           "--outdir", str(d), "--tag", tag]
    if deterministic:
        cmd.append("--deterministic")
    print(f">>> {tag}", flush=True)
    subprocess.run(cmd, check=True)
    return d


def compare(dir_a, dir_b):
    """HER between two captures. Parses the tool's one-line report."""
    r = subprocess.run([sys.executable, str(TOOL), "compare", str(dir_a), str(dir_b)],
                       check=True, capture_output=True, text=True)
    line = r.stdout.strip().splitlines()[0]
    her = float(line.split("HER =")[1].split()[0])
    return her, r.stdout.strip()


def publish(path: Path, dest: str | None, region: str = "us-east-2"):
    """Best effort copy to S3. An upload failure must never stop the sweep.

    The region is explicit. The results bucket lives in us-east-2 while these
    runs launch in us-east-1, and an s3 cp that inherits the instance region
    fails with PermanentRedirect against a bucket in another one.
    """
    if not dest:
        return
    cmd = ["aws", "s3", "cp", str(path), f"{dest.rstrip('/')}/{path.name}",
           "--region", region, "--only-show-errors"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode:
            print(f"  WARNING: publish {path.name} failed: {(r.stderr or '').strip()[:200]}")
    except Exception as e:
        print(f"  WARNING: publish {path.name} errored: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="hardware tag, for example h100")
    ap.add_argument("--models", default="default", help="'default', 'all', or one model id")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=os.path.expanduser("~/batch_out"))
    ap.add_argument("--publish-s3", default=None)
    ap.add_argument("--publish-region", default="us-east-2",
                    help="region of the results bucket, not of the instance")
    ap.add_argument("--batches", default=",".join(str(b) for b in BATCHES),
                    help="comma list; must contain the reference batch 32")
    a = ap.parse_args()

    if a.models == "all":
        models = list(S)
    elif a.models == "default":
        models = DEFAULT_MODELS
    else:
        models = [a.models]
    for m in models:
        if m not in S:
            print(f"ERROR: no canonical scale for {m}. Add it from "
                  f"spec/fingerprints-v0.1.csv before running."); return 1

    batches = [int(x) for x in a.batches.split(",") if x.strip()]
    if REFERENCE_BATCH not in batches:
        print(f"ERROR: --batches must contain the reference batch "
              f"{REFERENCE_BATCH}; got {batches}"); return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"label": a.label, "n": a.n, "reference_batch": REFERENCE_BATCH,
               "batches": batches, "cells": {}}
    spath = out / "summary.json"

    for model in models:
        slug = model.replace("/", "_")
        for cell, tf32, det in CELLS:
            key = f"{slug}__cell{cell}"
            dirs = {}
            for b in batches:
                tag = f"{slug}__{a.label}_cell{cell}_b{b}"
                dirs[b] = capture(model, tag, tf32, det, b, a.n, out)
                # The code matrix is the capture. A summary records what this
                # run concluded; only codes.npy lets a later run compare this
                # hardware against another one, which is what the mach
                # condition needs. The 2026-09-08 run published the summary
                # alone and the matrices went with the volume, so Hopper could
                # not be added to the cross-GPU table without repeating it.
                for f in ("codes.npy", "meta.json"):
                    publish(dirs[b] / f, (a.publish_s3 + "/captures/" + tag)
                            if a.publish_s3 else None, a.publish_region)

            rows = []
            for b in batches:
                if b == REFERENCE_BATCH:
                    continue
                her, report = compare(dirs[REFERENCE_BATCH], dirs[b])
                rows.append({"batch": b, "vs_batch": REFERENCE_BATCH, "HER": her})
                print(f"  batch {b:>3} vs {REFERENCE_BATCH}: HER = {her:.4f}")

            summary["cells"][key] = {
                "model": model, "cell": cell, "tf32": tf32, "deterministic": det,
                "min_HER": min(r["HER"] for r in rows), "comparisons": rows,
            }
            # Written and published per cell, not at the end. A run that only saves on
            # completion loses everything to a shutdown that lands mid-sweep.
            spath.write_text(json.dumps(summary, indent=2) + "\n")
            publish(spath, a.publish_s3, a.publish_region)

    print(f"\n-> {spath}")
    for key, c in summary["cells"].items():
        verdict = "invariant" if c["min_HER"] == 1.0 else "batch changes the codes"
        print(f"  {key}: min HER {c['min_HER']:.4f}  ({verdict})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
