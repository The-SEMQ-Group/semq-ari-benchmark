"""GPU-determinism capture/compare for the self-hosted condition matrix.

Encodes a SentenceTransformer model under a controlled (device, dtype, TF32, deterministic)
config, quantizes with a **fixed** QBIN scale `s` (so codes are comparable across machines),
and either computes in-machine conditions (`same`/`proc`) or stores the code matrix + meta for
a cross-machine `mach` compare. See experiments/gpu-determinism/README.md.

  # reference calibration (CPU fp32) -> prints the canonical scale s to reuse everywhere
  python gpu_capture.py capture --model BAAI/bge-large-en-v1.5 --device cpu --dtype fp32 \
      --inputs data/ari-bench-v0.1.jsonl --n 1000 --outdir ~/gpu_run/bge_cpu_fp32

  # GPU capture reusing the fixed s (comparable codes), with same/proc measured in-machine
  python gpu_capture.py capture --model BAAI/bge-large-en-v1.5 --device cuda --dtype fp32 \
      --tf32 on --scale-s 0.077546 --conditions same,proc --inputs ... --n 1000 \
      --outdir ~/gpu_run/bge_a10g_fp32_tf32on

  # mach: HER between two stored captures (same inputs, same fixed s)
  python gpu_capture.py compare ~/gpu_run/bge_cpu_fp32 ~/gpu_run/bge_a10g_fp32_tf32on
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Must be set before any CUDA context is created (deterministic cuBLAS).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
sys.path.insert(0, str(REPO))

from ari import metrics                            # noqa: E402
from ari.inputs import InputSet, load_ari_bench    # noqa: E402
from ari.probe import load_probe, fixed_scale_codes  # noqa: E402


def _apply_torch_flags(tf32: bool, deterministic: bool):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _load(model_id, device, dtype):
    import torch
    from sentence_transformers import SentenceTransformer
    # Load directly in the target dtype via model_kwargs so a large model never materialises
    # in fp32 first (a 7B in fp32 is ~28GB and OOMs a 24GB card before .bfloat16() can run).
    mk = {"fp16": torch.float16, "bf16": torch.bfloat16}
    kwargs = {"model_kwargs": {"torch_dtype": mk[dtype]}} if dtype in mk else {}
    return SentenceTransformer(model_id, device=device, trust_remote_code=True, **kwargs)


def _encode(model, texts, batch_size=32):
    return np.asarray(model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                   batch_size=batch_size), dtype=np.float32)


def _codes(vecs, scale_s, dim):
    """Encode and return (codes, s). If scale_s is None, calibrate on these vecs (the reference)
    and return the fitted scale; otherwise encode with the FIXED scale so codes are comparable
    across machines."""
    if scale_s is None:
        probe = load_probe(vecs, backend="semq")
        return probe.encode(vecs), float(probe.s)
    return fixed_scale_codes(vecs, scale_s, dim), scale_s


def _sm_arch():
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            return f"{torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})"
    except Exception:
        pass
    return "cpu"


def _worker_src():
    # standalone fresh-process encoder for the `proc` condition
    return (
        "import json,sys,numpy as np,torch\n"
        "from sentence_transformers import SentenceTransformer\n"
        "p=json.load(open(sys.argv[1]))\n"
        "d=p['dtype']; mk={'fp16':torch.float16,'bf16':torch.bfloat16}\n"
        "kw={'model_kwargs':{'torch_dtype':mk[d]}} if d in mk else {}\n"
        "m=SentenceTransformer(p['model'],device=p['device'],trust_remote_code=True,**kw)\n"
        "v=np.asarray(m.encode(p['texts'],normalize_embeddings=True,convert_to_numpy=True,batch_size=p.get('batch',32)),dtype=np.float32)\n"
        "np.save(p['out'],v)\n"
    )


def do_capture(args):
    inputs = load_ari_bench(args.inputs)
    if args.n and len(inputs.ids) > args.n:
        inputs = InputSet(inputs.name, inputs.ids[:args.n], inputs.texts[:args.n])

    _apply_torch_flags(args.tf32 == "on", args.deterministic)
    bs = args.batch_size
    model = _load(args.model, args.device, args.dtype)
    base_vecs = _encode(model, inputs.texts, bs)
    dim = int(base_vecs.shape[1])
    base_codes, s = _codes(base_vecs, args.scale_s, dim)

    conditions = [c for c in (args.conditions.split(",") if args.conditions else []) if c]
    cond = {}
    if "same" in conditions:
        c2, _ = _codes(_encode(model, inputs.texts, bs), s, dim)
        cond["same"] = _agg(base_codes, c2)
    if "proc" in conditions:
        with tempfile.TemporaryDirectory() as d:
            out, pj, wk = f"{d}/v.npy", f"{d}/p.json", f"{d}/w.py"
            Path(wk).write_text(_worker_src())
            json.dump({"model": args.model, "texts": inputs.texts, "out": out,
                       "device": args.device, "dtype": args.dtype, "batch": bs}, open(pj, "w"))
            subprocess.run([sys.executable, wk, pj], check=True, env={**os.environ})
            c3, _ = _codes(np.load(out), s, dim)
            cond["proc"] = _agg(base_codes, c3)

    out = Path(os.path.expanduser(args.outdir))
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "codes.npy", base_codes)
    meta = {"model": args.model, "device": args.device, "sm_arch": _sm_arch(),
            "dtype": args.dtype, "tf32": args.tf32, "deterministic": bool(args.deterministic),
            "scale_s": float(s), "dim": dim, "n": len(inputs),
            "content_hash": inputs.content_hash, "conditions": cond, "tag": args.tag}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    line = f"[{args.tag or out.name}] {args.model} {args.device}/{args.dtype} tf32={args.tf32} " \
           f"det={args.deterministic} s={s:.6f} dim={dim} n={len(inputs)}"
    for k, v in cond.items():
        line += f" | {k} HER={v['HER']:.4f} Hbar={v['Hbar']:.2f}"
    print(line)
    return 0


def _agg(a, b):
    m = metrics.aggregate(a, b)
    return {"HER": round(m.HER, 6), "Hbar": round(m.Hbar, 6),
            "HER_ci": [round(m.HER_ci[0], 6), round(m.HER_ci[1], 6)]}


def do_compare(args):
    ma = json.loads((Path(os.path.expanduser(args.dir_a)) / "meta.json").read_text())
    mb = json.loads((Path(os.path.expanduser(args.dir_b)) / "meta.json").read_text())
    if ma["content_hash"] != mb["content_hash"]:
        print("ERROR: the two captures used different inputs (content_hash differs)"); return 1
    if abs(ma["scale_s"] - mb["scale_s"]) > 1e-9:
        print(f"WARNING: scale_s differs ({ma['scale_s']} vs {mb['scale_s']}) — codes not "
              "strictly comparable; pass --scale-s to fix it on both captures.")
    a = np.load(Path(os.path.expanduser(args.dir_a)) / "codes.npy")
    b = np.load(Path(os.path.expanduser(args.dir_b)) / "codes.npy")
    r = _agg(a, b)
    print(f"mach HER = {r['HER']:.4f}  95% CI {r['HER_ci']}  H̄={r['Hbar']:.2f}\n"
          f"  A: {ma['tag']} [{ma['sm_arch']} {ma['dtype']} tf32={ma['tf32']}]\n"
          f"  B: {mb['tag']} [{mb['sm_arch']} {mb['dtype']} tf32={mb['tf32']}]")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GPU-determinism capture/compare.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture")
    c.add_argument("--model", required=True)
    c.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    c.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    c.add_argument("--tf32", default="off", choices=["on", "off"])
    c.add_argument("--deterministic", action="store_true")
    c.add_argument("--scale-s", type=float, default=None, help="fixed QBIN scale (omit to calibrate)")
    c.add_argument("--conditions", default="", help="comma list: same,proc")
    c.add_argument("--inputs", type=Path, required=True)
    c.add_argument("--n", type=int, default=1000)
    c.add_argument("--batch-size", type=int, default=32)
    c.add_argument("--outdir", required=True)
    c.add_argument("--tag", default=None)
    c.set_defaults(fn=do_capture)

    p = sub.add_parser("compare")
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.set_defaults(fn=do_compare)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
