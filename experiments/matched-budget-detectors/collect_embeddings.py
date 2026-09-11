"""Collect repeated embedding episodes under controls and single-axis changes.

One episode is one fresh process embedding the frozen input set under one
declared configuration. Controls repeat the nominal configuration; each
intervention changes exactly one axis. Controls and interventions are
interleaved by the caller so drift in machine state cannot align with
condition.

Effective settings are recorded, not requested ones: `torch.backends` is read
back after being set, because a flag that did not take would otherwise be
reported as though it had.

  python collect_embeddings.py --condition control --episode 0 --out ~/mbd_collect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CONDITIONS = {
    # name          dtype    tf32   batch  threads
    "control":     ("fp32",  False, 32,    None),
    "tf32_on":     ("fp32",  True,  32,    None),
    "batch_1":     ("fp32",  False, 1,     None),
    "batch_256":   ("fp32",  False, 256,   None),
    "fp16":        ("fp16",  False, 32,    None),
    "bf16":        ("bf16",  False, 32,    None),
    "threads_1":   ("fp32",  False, 32,    1),
}


def _episode_paths(root: Path, condition: str, episode: int) -> tuple[Path, Path]:
    stem = root / condition / f"ep{episode:03d}"
    return stem.with_suffix(".npz"), stem.with_suffix(".json")


def _inputs_sha256(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def _reuse_existing_episode(paths: tuple[Path, Path], expected: dict) -> bool:
    """Validate and reuse a complete episode; reject partial/corrupt output."""
    data_path, manifest_path = paths
    present = [path.exists() for path in paths]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(f"incomplete episode artifacts: {data_path} / {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text())
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"existing episode mismatch for {key}: "
                    f"{manifest.get(key)!r} != {value!r}"
                )
        with np.load(data_path, allow_pickle=False) as archive:
            embeddings = np.ascontiguousarray(archive["embeddings"], dtype=np.float32)
        digest = hashlib.sha256(embeddings.tobytes()).hexdigest()
        if manifest.get("embeddings_sha256") != digest:
            raise RuntimeError("existing episode checksum mismatch")
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid existing episode artifacts: {exc}") from exc
    return True


def publish_files(paths: tuple[Path, ...], destination: str, region: str) -> None:
    """Upload an episode's artifacts, failing if any upload is not durable."""
    failures = []
    for path in paths:
        cmd = ["aws", "s3", "cp", str(path), f"{destination}/{path.name}",
               "--region", region, "--only-show-errors"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=600)
        except Exception as exc:
            failures.append(f"{path.name}: {exc}")
            continue
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()
            failures.append(f"{path.name}: {detail[:200]}")
    if failures:
        raise RuntimeError("episode publication failed: " + "; ".join(failures))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--publish-s3", default=None,
                    help="s3://bucket/prefix to copy each episode to as it is written")
    ap.add_argument("--publish-region", default="us-east-2",
                    help="region of the results bucket, not of the instance")
    a = ap.parse_args()

    dtype_name, tf32, batch, threads = CONDITIONS[a.condition]
    texts = []
    with open(a.inputs) as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            texts.append(rec.get("text") or rec.get("input") or "")
            if len(texts) >= a.n:
                break

    data_path, manifest_path = _episode_paths(Path(a.out), a.condition, a.episode)
    expected = {
        "condition": a.condition,
        "episode": a.episode,
        "model": a.model,
        "n_inputs": len(texts),
        "inputs_sha256": _inputs_sha256(texts),
    }
    if _reuse_existing_episode((data_path, manifest_path), expected):
        print(f"{a.condition} ep{a.episode}: existing validated artifacts reused",
              flush=True)
        return 0

    import torch
    if threads is not None:
        torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32

    from sentence_transformers import SentenceTransformer


    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(a.model, device=dev)
    if dtype_name == "fp16":
        model = model.half()
    elif dtype_name == "bf16":
        model = model.bfloat16()

    t0 = time.time()
    emb = model.encode(texts, batch_size=batch, convert_to_numpy=True,
                       normalize_embeddings=False, show_progress_bar=False)
    encode_s = time.time() - t0
    emb = np.ascontiguousarray(emb, dtype=np.float32)

    out_dir = Path(a.out) / a.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"ep{a.episode:03d}"
    np.savez_compressed(stem.with_suffix(".npz"), embeddings=emb)

    manifest = {
        "condition": a.condition,
        "episode": a.episode,
        "model": a.model,
        "n_inputs": len(texts),
        "dim": int(emb.shape[1]),
        "requested": {"dtype": dtype_name, "tf32": tf32, "batch": batch,
                      "threads": threads},
        # Read back, not assumed.
        "effective": {
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "torch_threads": int(torch.get_num_threads()),
            "device": dev,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "param_dtype": str(next(model.parameters()).dtype),
            "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "encode_seconds": round(encode_s, 3),
        "embeddings_sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
        "inputs_sha256": _inputs_sha256(texts),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "pid": os.getpid(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=Path(__file__).parent).stdout.strip(),
    }
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")

    # Published as each episode lands, not at the end of the run. A collection
    # that keeps its only copy on an instance volume is one shutdown away from
    # nothing, and a 200-episode run outlives any shutdown timer worth setting.
    # This is not hypothetical: a full collection was lost this way on
    # 2026-09-09 because the driver wrote only to local disk.
    if a.publish_s3:
        dest = a.publish_s3.rstrip("/") + f"/{a.condition}"
        publish_files((stem.with_suffix(".npz"), stem.with_suffix(".json")),
                      dest, a.publish_region)
    print(f"{a.condition} ep{a.episode}: {emb.shape} sha={manifest['embeddings_sha256'][:12]} "
          f"tf32={manifest['effective']['matmul_tf32']} dtype={manifest['effective']['param_dtype']} "
          f"{encode_s:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
