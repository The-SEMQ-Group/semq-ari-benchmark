"""fp32 reference generation for the ARI-D rehosted class (spec §7, rollout item 4).

Generates, under our control, the completions the published weights produce in
fp32 for every prompt in the frozen set — the yardstick `reference_match` and
`reference_first_divergence` compare a rehosted provider against. The same
session also produces bf16 and fp16 passes: exploratory dry-run data (spec
§13 item 4), not published detectors.

  python generate_refs.py --model llama33 --dtype fp32
  python generate_refs.py --model llama33 --dtype bf16
  python generate_refs.py --model qwen25 --dtype fp32

Protocol identity with the harness is what makes the comparison meaningful:

- the PROMPT TEMPLATE is imported from the dry-run harness, byte for byte —
  a one-byte template difference at temperature 0 produces a different
  completion from identical weights and would read as infidelity (§7);
- greedy decoding (temperature 0 ≡ argmax), max 128 new tokens, generation
  stops at the model's end-of-turn token, which is stripped from the text —
  matching what a raw completions endpoint returns with that stop sequence;
- batch size 1: the reference is single-stream compute, the cleanest baseline;
- each dtype is a FRESH LOAD in that dtype, never an in-place cast — casting
  a loaded model crushes fp32-born buffers (rotary frequencies) and poisons
  every later pass, a failure mode this project has already measured;
- TF32 off, deterministic algorithms on, HF revision pinned below.

Sizing: 70B fp32 is ~282 GB of weights — an 8×L40S node (g6e.48xlarge,
384 GB) fits it; the pass is memory-bandwidth-bound at roughly a few tokens/s
single-stream, ~1-2 h per dtype pass.

The pass RESUMES. A dtype pass on a 70B costs an hour or more on a node
billed by the hour, so a crash at prompt 87 must not repay the first 86.
Completed prompts are read back from the output file and skipped; --restart
forces the pass to begin again. A truncated final line (the crash itself) is
dropped on resume rather than parsed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "arid-dry-run"))

from providers import LLAMA3_TEMPLATE  # noqa: E402  (byte-identical to the harness)

PROMPTS = REPO / "data" / "arid-bench-v0.1.jsonl"
FROZEN_HASH = "af5b1a2bb35b9fa5f8d4e8f05b85a98f495a3edeb93e95da9c9ced6d783e0d6f"

# Qwen's official single-turn template (ChatML), for the second panel model.
QWEN_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

MODELS = {
    "llama33": {
        "hf_id": "meta-llama/Llama-3.3-70B-Instruct",
        "revision": "6f6073b423013f6a7d4d9f39144961bfbfbc386b",  # pinned 2026-09-01
        "template": LLAMA3_TEMPLATE,
        "stop_token": "<|eot_id|>",
    },
    "qwen25": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "revision": "495f39366efef23836d0cfae4fbe635880d2be31",  # pinned 2026-09-01
        "template": QWEN_TEMPLATE,
        "stop_token": "<|im_end|>",
    },
}
DTYPES = ("fp32", "bf16", "fp16")
MAX_NEW_TOKENS = 128


def load_prompts() -> list[dict]:
    items, h = [], hashlib.sha256()
    for line in PROMPTS.open():
        it = json.loads(line)
        h.update(it["input_id"].encode() + b"\x00" + it["text"].encode() + b"\n")
        items.append(it)
    if h.hexdigest() != FROZEN_HASH:
        sys.exit(f"prompt set does not match the frozen pin ({h.hexdigest()})")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate reference completions.")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--dtype", required=True, choices=DTYPES)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    ap.add_argument("--outdir", type=Path, default=HERE / "results")
    ap.add_argument("--restart", action="store_true",
                    help="discard a partial pass instead of resuming it")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    cfg = MODELS[args.model]
    dt = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    # Fetch before loading. Llama-3.3 is a gated repo, so an unset or
    # unaccepted HF token fails here -- seconds into the session, with the
    # node already billing -- rather than after the first pass has run.
    from huggingface_hub import snapshot_download
    t0 = time.time()
    # `original/` holds the consolidated .pth weights that Meta ships beside
    # the safetensors -- another ~141 GB of the same 70B, which transformers
    # never reads. Fetching it doubles the download and overruns the volume.
    local = snapshot_download(cfg["hf_id"], revision=cfg["revision"],
                              ignore_patterns=["original/**", "*.pth"])
    print(f"weights present in {time.time()-t0:.0f}s ({local})", flush=True)

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(local)
    # transformers renamed `torch_dtype` to `dtype` in 4.56, keeping the old
    # name as a deprecated alias. `from_pretrained` takes **kwargs, so the one
    # this wheel does not know is not rejected -- it is swallowed, and the
    # model loads in its default precision. That failure is silent and would
    # publish, say, a bf16 pass labelled fp32. Pick by version, then verify
    # against the loaded weights, because the verification is what actually
    # rules it out.
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(
        local, device_map="auto", low_cpu_mem_usage=True, **{dtype_kw: dt})
    model.eval()

    got = next(model.parameters()).dtype
    if got != dt:
        sys.exit(f"asked for {args.dtype} ({dt}) and the weights loaded as {got} "
                 f"-- transformers {transformers.__version__} did not accept "
                 f"`{dtype_kw}`. Refusing to write a mislabelled reference.")
    print(f"loaded {cfg['hf_id']} [{args.dtype}] in {time.time()-t0:.0f}s "
          f"over {torch.cuda.device_count()} GPU(s)", flush=True)

    # 70B fp32 is ~282 GB against the node's 384 GB, so `device_map="auto"`
    # has room -- but when it does not, it does not fail. It silently spills
    # layers to CPU or to disk, and the pass still completes: orders of
    # magnitude slower, through a different arithmetic path, and the result
    # is no longer the single-stream GPU reference the spec asks for. A
    # yardstick that is wrong in a way nobody can see is worse than no
    # yardstick, so refuse rather than measure the wrong thing.
    offloaded = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()
                        if str(d) in ("cpu", "disk")})
    if offloaded:
        sys.exit(f"model offloaded to {', '.join(offloaded)} -- this node cannot hold "
                 f"{args.dtype} for {cfg['hf_id']}. Use a larger node; do not "
                 f"publish a reference generated through an offload path.")

    stop_id = tok.convert_tokens_to_ids(cfg["stop_token"])
    eos_ids = [i for i in {stop_id, tok.eos_token_id} if i is not None]

    args.outdir.mkdir(parents=True, exist_ok=True)
    scope = f"prefix{args.limit}" if args.limit else "full"
    out = args.outdir / f"refs_{args.model}_{args.dtype}{'' if scope == 'full' else '_' + scope}.jsonl"
    # Resume: keep the completed prompts, drop a truncated tail. Rewriting the
    # file from the records that parsed is what makes the append below safe --
    # a half-written line from the crash would otherwise sit in the middle of
    # the reference and break every later read of it.
    done: set[str] = set()
    if out.exists() and not args.restart:
        kept = []
        for line in out.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break          # the crash; nothing after it is trustworthy
            kept.append(line)
            done.add(rec["input_id"])
        out.write_text("".join(l + "\n" for l in kept))
        if done:
            print(f"resuming: {len(done)} prompts already generated", flush=True)
    elif out.exists():
        out.unlink()

    todo = [p for p in prompts if p["input_id"] not in done]
    t0 = time.time()
    with out.open("a") as f, torch.no_grad():
        for i, p in enumerate(todo):
            text_in = cfg["template"].format(prompt=p["text"])
            ids = tok(text_in, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
            gen = model.generate(ids, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                 eos_token_id=eos_ids, pad_token_id=eos_ids[0])
            new = gen[0][ids.shape[1]:]
            # An immediate end-of-turn leaves nothing generated; indexing [-1]
            # would raise and lose the whole pass over an empty completion.
            if new.numel() and new[-1].item() in eos_ids:
                finish, new = "stop", new[:-1]
            else:
                finish = "length" if new.numel() else "empty"
            f.write(json.dumps({
                "input_id": p["input_id"],
                "text": tok.decode(new, skip_special_tokens=False),
                "token_ids": new.tolist(),
                "finish_reason": finish,
            }) + "\n")
            f.flush()          # a resume is only as good as what reached disk
            if (i + 1) % 10 == 0:
                print(f"  {len(done)+i+1}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "model": cfg["hf_id"], "revision": cfg["revision"], "dtype": args.dtype,
        "scope": scope, "template_source": "arid-dry-run/providers.py",
        "max_new_tokens": MAX_NEW_TOKENS, "decoding": "greedy, batch=1",
        "tf32": "off", "deterministic_algorithms": True,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "reference_sha256": digest,
        "prompt_content_hash": FROZEN_HASH,
    }
    mpath = args.outdir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {"references": []}
    # Key the entry on what identifies the pass, not on its digest. Digest
    # dedup only collapses a rerun that produced identical bytes, so a
    # regenerated pass -- new driver, new node, a fixed bug -- left the
    # superseded entry sitting beside it with nothing to say which was current.
    key = (meta["model"], meta["dtype"], meta["scope"])
    manifest["references"] = [r for r in manifest["references"]
                              if (r["model"], r["dtype"], r["scope"]) != key]
    manifest["references"].append(meta)
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out.name}  sha256={digest[:16]}...  ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
