# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""ARI-D: does a serving change move the logits before it moves the tokens?

ARI-R measures whether the same input gives the same embedding. This asks the
same question one layer up: given the same context, does the same token come
out, and does SEMQ see a change in the logit vector at steps where the emitted
token is still identical?

The claim under test is that a logit vector behaves like an embedding. A
precision change shifts token probabilities long before it flips an argmax,
exactly as it shifts cosine long before it flips a ranking. If that holds, a
SEMQ trace over logits is an early-warning signal that token-level output
comparison cannot give, because token comparison only fires after the decision
has already changed.

Two arms:

  * Teacher-forced. Every condition is fed the *reference* token sequence and
    scored in one forward pass. Position i is therefore evaluated against an
    identical context in every condition, which isolates the per-step
    computation from compounding divergence. This is the arm that tests the
    claim.

  * Free-running. Every condition generates on its own, greedily. This is what
    production does, and it measures how fast a per-step difference compounds
    into a different answer.

Sampling is greedy throughout (do_sample=False), so nothing here measures
temperature noise. Any difference is infrastructure.

Usage:
    python run_matrix.py
    python run_matrix.py --list
    python run_matrix.py --worker bf16 --out /tmp/bf16.npz   # internal
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ari.code_metrics import chunked_code_diff

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CACHE = RESULTS / "cache"

# SEMQ's Context caps max_dim at 65,536. TinyLlama's 32,000-token vocabulary
# fits directly; Qwen2.5 (151,936), Llama-3 (128,256) and Gemma (256,000) are
# handled by chunking in semq_codes(). Override with ARI_D_MODEL.
MODEL = os.environ.get("ARI_D_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
DEVICE = os.environ.get("ARI_D_DEVICE", "cpu")
N_NEW_TOKENS = 48
QUANT_BINS = 8
CALIBRATION_PERCENTILE = 0.99

PROMPTS = (
    "Explain why the sky appears blue.",
    "Write a Python function that reverses a linked list.",
    "What is the difference between a process and a thread?",
    "Summarise the causes of the 1929 stock market crash.",
    "Describe how a hash table handles collisions.",
    "Give three arguments for and against nuclear power.",
    "What does the CAP theorem say about distributed systems?",
    "Translate to French: the weather is cold today.",
    "Explain floating-point rounding error to a beginner.",
    "List the steps to bisect a bug in a git history.",
    "Why does quantisation reduce model quality?",
    "Describe the halting problem in two sentences.",
    "Explain the difference between TCP and UDP.",
    "Write a SQL query that finds duplicate rows in a table.",
    "What is a race condition? Give one example.",
    "Describe how garbage collection works in a managed runtime.",
    "Convert 98 degrees Fahrenheit to Celsius and show the working.",
    "Explain what a Merkle tree is and one place it is used.",
    "Write a bash one-liner that counts lines in every .py file.",
    "Why is UTF-8 backwards compatible with ASCII?",
    "Summarise the tradeoff between latency and throughput.",
    "Explain public key cryptography without using the word key.",
    "What does a load balancer do when a backend stops responding?",
    "Write a regular expression that matches an IPv4 address.",
    "Describe the difference between a mutex and a semaphore.",
    "What is the birthday paradox and why does it matter for hashing?",
    "Explain database normalisation to third normal form.",
    "Write a Python generator that yields the Fibonacci sequence.",
    "What causes a memory leak in a language with reference counting?",
    "Explain the CAP theorem tradeoff a bank would choose, and why.",
    "Describe how DNS resolves a hostname, step by step.",
    "What is the difference between authentication and authorisation?",
    "Write a function that checks whether a string is a palindrome.",
    "Explain why floating point cannot represent 0.1 exactly.",
    "Describe two ways to prevent SQL injection.",
    "What is idempotency and why do HTTP verbs care about it?",
    "Explain the difference between a process and a container.",
    "Write a git command sequence that undoes the last commit safely.",
    "What is backpressure in a streaming system?",
    "Describe the tradeoff between B-trees and LSM trees.",
    "Explain what a JIT compiler does that an interpreter does not.",
    "Why does adding more threads sometimes make a program slower?",
    "Describe how a bloom filter works and what it cannot do.",
    "Explain the difference between latency and jitter.",
    "What is the purpose of a write-ahead log?",
    "Describe how TLS establishes a shared secret.",
    "Explain tail latency and why averages hide it.",
    "What is the difference between a cache miss and a page fault.",
)


@dataclass(frozen=True)
class Condition:
    name: str
    axis: str
    dtype: str = "float32"
    threads: int = 4
    quantize: bool = False
    batched: bool = False
    tf32: bool | None = None
    gpu_only: bool = False
    note: str = ""


CONDITIONS: tuple[Condition, ...] = (
    Condition("reference", "same", note="fp32 / CPU / 4 threads / unbatched"),
    Condition("proc", "proc", note="identical settings, fresh process"),
    Condition("threads1", "proc", threads=1, note="single-threaded BLAS"),
    Condition("batched", "batch", batched=True,
              note="scored in a padded batch of 4 rather than alone"),
    Condition("bf16", "prec", dtype="bfloat16", note="bf16 weights"),
    Condition("int8", "prec", quantize=True,
              note="dynamic int8 on Linear layers"),
    Condition("fp16", "prec", dtype="float16", gpu_only=True,
              note="fp16 weights (GPU only; no optimised CPU kernels)"),
    Condition("tf32_on", "mach", tf32=True, gpu_only=True,
              note="fp32 weights, TF32 matmul enabled"),
    Condition("tf32_off", "mach", tf32=False, gpu_only=True,
              note="fp32 weights, TF32 matmul disabled"),
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build(cond: Condition):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(cond.threads)
    if cond.tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = cond.tf32
        torch.backends.cudnn.allow_tf32 = cond.tf32
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # from_pretrained materialises the whole model in CPU RAM before any .to().
    # A 3B model in fp32 is about 12.4 GiB, and a g5.xlarge has 16 GiB, so that
    # path thrashes the host into unresponsiveness. device_map streams the
    # weights straight to the GPU instead.
    if DEVICE != "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float32, device_map=DEVICE,
            low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    if cond.quantize and DEVICE != "cpu":
        raise RuntimeError("dynamic int8 quantisation is CPU-only in torch; "
                           "run the int8 condition on a CPU host")
    if cond.quantize:
        engines = torch.backends.quantized.supported_engines
        for name in ("qnnpack", "fbgemm", "onednn"):
            if name in engines:
                torch.backends.quantized.engine = name
                break
        else:
            raise RuntimeError(f"no usable qengine; have {engines}")
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    elif cond.dtype != "float32":
        model = model.to(getattr(torch, cond.dtype))
    return tok, model


def format_prompt(tok, prompt: str) -> str:
    """Apply the chat template. Without it the model emits EOS after a few
    tokens, which leaves too few decoding steps to measure anything."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def free_generate(tok, model, prompt: str) -> list[int]:
    """Greedy generation, the condition running on its own."""
    import torch

    ids = tok(format_prompt(tok, prompt), return_tensors="pt").input_ids
    ids = ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N_NEW_TOKENS,
                             do_sample=False, num_beams=1,
                             pad_token_id=tok.pad_token_id)
    return out[0, ids.shape[1]:].cpu().tolist()


def teacher_forced_logits(tok, model, prompt: str, cont: list[int],
                          batched: bool) -> np.ndarray:
    """Logits at each continuation position, given the reference context.

    Returns (len(cont), vocab). Position i holds the distribution the model
    would decode token cont[i] from, so argmax(row i) is the token this
    condition *would* have emitted with identical history.
    """
    import torch

    p_ids = tok(format_prompt(tok, prompt),
                return_tensors="pt").input_ids[0].tolist()
    full = torch.tensor([p_ids + cont]).to(DEVICE)

    if batched:
        # Pad three filler rows to the same length so the real row is scored
        # inside a batch. Left padding keeps the real tokens right-aligned.
        filler = tok("The", return_tensors="pt").input_ids[0].tolist()
        rows, attn = [], []
        for r in [p_ids + cont] + [filler] * 3:
            pad = full.shape[1] - len(r)
            rows.append([tok.pad_token_id] * pad + r)
            attn.append([0] * pad + [1] * len(r))
        inp = torch.tensor(rows).to(DEVICE)
        mask = torch.tensor(attn).to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=mask).logits[0]
    else:
        with torch.no_grad():
            out = model(input_ids=full).logits[0]

    # Row j predicts token j+1, so the distribution for cont[i] is at
    # index len(prompt) + i - 1.
    start = len(p_ids) - 1
    return out[start:start + len(cont)].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def semq_codes(X: np.ndarray, scale_source: np.ndarray):
    """SEMQ codes and chunk widths, at a scale frozen from scale_source.

    SEMQ's Context caps max_dim at 65,536, which is smaller than the
    vocabulary of every model in current use except the small ones. A
    vocabulary above the cap is split into consecutive chunks and the codes
    are concatenated.

    The scale is computed once over the whole reference and passed to every
    chunk explicitly. Calibrating each chunk on its own slice would give each
    one a different scale and make the concatenated codes incomparable.

    Passing quant_scale_max = percentile(|scale_source|, 99) is bit-identical
    to calibrate(scale_source, percentile=0.99), so a chunked run stays
    directly comparable with an unchunked one.

    The chunk widths are returned with the codes because every chunk pads
    its own final byte: a coordinate-level comparison has to split the
    concatenated buffer back at the same boundaries.
    """
    from ari.semq_compat import MAX_DIM, quant_context

    scale = float(np.percentile(np.abs(scale_source),
                                CALIBRATION_PERCENTILE * 100.0))
    dim = X.shape[1]
    bounds = list(range(0, dim, MAX_DIM)) + [dim]

    out, widths = [], []
    for lo, hi in zip(bounds, bounds[1:]):
        ctx = quant_context(hi - lo, n_bins=QUANT_BINS, scale_max=scale)
        out.append(np.asarray(ctx.batch_encode(
            np.ascontiguousarray(X[:, lo:hi], np.float32))))
        ctx.close()
        widths.append(hi - lo)
    codes = out[0] if len(out) == 1 else np.concatenate(out, axis=1)
    return codes, widths


def top2_margin(logits: np.ndarray) -> np.ndarray:
    """Gap between the best and second-best logit at each position."""
    part = np.partition(logits, -2, axis=1)[:, -2:]
    return part[:, 1] - part[:, 0]


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    The free-running arm has one observation per prompt, so its rate rests on
    tens of trials while token agreement rests on hundreds of steps. Printing
    both as bare percentages implies a precision the prompt count cannot
    support. The normal approximation is unusable near 0 and 1, which is
    exactly where these land, so Wilson rather than Wald.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing token, or len(a) if identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# ---------------------------------------------------------------------------
# Worker / driver
# ---------------------------------------------------------------------------

def _model_revision() -> str:
    """The resolved commit of the weights, or `unknown` when it cannot be had.

    A tag moves, so the name alone does not identify what was loaded. Delegates
    to ari.hub.hub_revision, the one resolver shared with the attestation
    references, so the cache fingerprint and the manifest name one revision.
    """
    from ari.hub import hub_revision
    return hub_revision(MODEL, "model")


def _versions() -> dict:
    """Library versions read from package metadata, without importing torch.

    torch is imported lazily on purpose so the driver process stays light. The
    driver needs the fingerprint, so the versions have to come from metadata.
    """
    out = {}
    for pkg in ("torch", "transformers"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "absent"
    return out


def _identity() -> dict:
    """Everything that changes what a cached condition means.

    The cache was keyed on the condition name alone, while MODEL and DEVICE come
    from the environment. So `reference.npz` written by a CPU run of TinyLlama
    was reused by a GPU run of Qwen and reported under Qwen -- silently, and
    `infra/run_gpu.sh` sets exactly that combination.
    """
    return {
        "model": MODEL,
        "model_revision": _model_revision(),
        "device": DEVICE,
        "n_new_tokens": N_NEW_TOKENS,
        "quant_bins": QUANT_BINS,
        "calibration_percentile": CALIBRATION_PERCENTILE,
        "n_prompts": len(PROMPTS),
        "prompts_sha256": hashlib.sha256("\n".join(PROMPTS).encode()).hexdigest(),
        **_versions(),
    }


def fingerprint() -> str:
    blob = json.dumps(_identity(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def cache_dir() -> Path:
    return CACHE / fingerprint()


def cache_path(name: str) -> Path:
    return cache_dir() / f"{name}.npz"


def write_cache_manifest() -> None:
    """Record what this fingerprint stands for, so the directory is readable.

    Also validates: a manifest that disagrees with the current identity means
    the hash has collided or the file was hand-edited, and reusing the arrays
    under those conditions is the failure this whole change exists to stop.
    """
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "manifest.json"
    identity = _identity()
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise SystemExit(
                f"cache manifest at {path} does not match this run.\n"
                f"  cached : {json.dumps(existing, sort_keys=True)}\n"
                f"  current: {json.dumps(identity, sort_keys=True)}\n"
                f"Delete the directory to re-measure.")
        return
    path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")


def run_worker(name: str, out: Path) -> None:
    cond = next(c for c in CONDITIONS if c.name == name)
    tok, model = build(cond)

    if cond.name == "reference":
        conts = [free_generate(tok, model, p) for p in PROMPTS]
    else:
        ref = np.load(cache_path("reference"), allow_pickle=True)
        conts = [list(c) for c in ref["free_tokens"]]

    # Teacher-forced pass uses the reference continuations for every condition.
    logits = [teacher_forced_logits(tok, model, p, c, cond.batched)
              for p, c in zip(PROMPTS, conts)]
    free = ([list(c) for c in conts] if cond.name == "reference"
            else [free_generate(tok, model, p) for p in PROMPTS])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        logits=np.concatenate(logits, axis=0).astype(np.float32),
        lengths=np.array([len(c) for c in conts]),
        ref_tokens=np.array(np.concatenate([np.array(c) for c in conts])),
        free_tokens=np.array([np.array(c) for c in free], dtype=object),
    )


def compare(ref: dict, cur: dict, ref_codes: np.ndarray,
            widths: list[int]) -> dict:
    r_log, c_log = ref["logits"], cur["logits"]
    ref_tok = ref["ref_tokens"]

    # Would this condition have emitted the same token, given identical history?
    same_token = (c_log.argmax(1) == r_log.argmax(1))
    codes, _ = semq_codes(c_log, scale_source=r_log)
    # HER is exact-match, so it saturates at 0 for any real precision change
    # and cannot rank severity. A coordinate change rate can.
    diff = chunked_code_diff(ref_codes, codes, n_bins=QUANT_BINS, widths=widths)
    same_code = diff.codes_equal

    margins = top2_margin(r_log)
    # Steps where the token survived but SEMQ still read a change.
    early = same_token & ~same_code

    # Does a flipped token sit at a smaller reference margin than a surviving
    # one? That is the "probabilities move before the argmax does" claim, and
    # it is testable directly rather than by analogy.
    flipped = ~same_token
    margin_flipped = (float(np.median(margins[flipped]))
                      if flipped.any() else float("nan"))
    margin_survived = (float(np.median(margins[same_token]))
                       if same_token.any() else float("nan"))

    free_ref = list(ref["free_tokens"])
    free_cur = list(cur["free_tokens"])
    exact = [list(a) == list(b) for a, b in zip(free_ref, free_cur)]
    n_exact = int(sum(exact))
    exact_lo, exact_hi = wilson_ci(n_exact, len(exact))
    divs = [first_divergence(list(a), list(b))
            for a, b in zip(free_ref, free_cur)]

    return {
        "steps": int(len(same_token)),
        "token_agreement": float(same_token.mean()),
        "semq_her": float(same_code.mean()),
        "semq_coord_change": float(diff.coordinate_change_rate.mean()),
        "semq_byte_change_legacy": float(diff.byte_change_rate.mean()),
        "early_warning_steps": float(early.mean()),
        "logit_max_abs_delta": float(np.abs(c_log - r_log).max()),
        "median_top2_margin": float(np.median(margins)),
        "median_margin_flipped": margin_flipped,
        "median_margin_survived": margin_survived,
        "free_exact_match": float(np.mean(exact)),
        "free_exact_count": n_exact,
        "free_exact_ci95": [exact_lo, exact_hi],
        "free_first_divergence_mean": float(np.mean(divs)),
        "free_first_divergence_min": int(np.min(divs)),
        "n_prompts": len(free_ref),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker, args.out)
        return

    on_gpu = DEVICE != "cpu"
    active = [c for c in CONDITIONS
              if (c.gpu_only and on_gpu) or (not c.gpu_only
                                             and not (c.quantize and on_gpu))]

    if args.list:
        for c in CONDITIONS:
            mark = "run" if c in active else "skip"
            print(f"{c.name:<12} {c.axis:<6} {mark:<5} {c.note}")
        return

    print(f"model {MODEL}   device {DEVICE}")
    print(f"revision {_model_revision()}")
    print(f"{len(PROMPTS)} prompts x {N_NEW_TOKENS} new tokens, greedy")
    print(f"{len(active)} conditions active")
    # Print the cache namespace. A run that reuses arrays should say whose.
    write_cache_manifest()
    print(f"cache {cache_dir().relative_to(HERE)}\n")

    failed: dict[str, str] = {}
    for c in active:
        p = cache_path(c.name)
        if p.exists():
            print(f"  {c.name:<12} cached")
            continue
        print(f"  {c.name:<12} running...", flush=True)
        r = subprocess.run([sys.executable, str(HERE / "run_matrix.py"),
                            "--worker", c.name, "--out", str(p)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            lines = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
            failed[c.name] = lines[-1] if lines else f"exit {r.returncode}"
            print(f"  {c.name:<12} FAILED — {failed[c.name]}")

    usable = [c for c in active if c.name not in failed]
    ref = dict(np.load(cache_path("reference"), allow_pickle=True))
    ref_codes, code_widths = semq_codes(ref["logits"],
                                        scale_source=ref["logits"])

    rows = []
    for c in usable:
        cur = dict(np.load(cache_path(c.name), allow_pickle=True))
        m = compare(ref, cur, ref_codes, code_widths)
        m["condition"], m["axis"], m["note"] = c.name, c.axis, c.note
        rows.append(m)

    n_steps = rows[0]["steps"] if rows else 0
    n_prompts = rows[0]["n_prompts"] if rows else 0
    print(f"\nAll columns compare against the fp32 reference on the same input.")
    print(f"HER  = share of steps whose {'chunked ' if False else ''}code is "
          f"bit-identical to the reference     (n = {n_steps} steps)")
    print(f"coord = share of coordinates whose code symbol differs from the "
          f"reference (n = {n_steps} steps)")
    print(f"tok-same/code-diff = share of ALL steps with an identical token but "
          f"a changed code")
    print(f"free exact = whole generations identical, as a count"
          f"                    (n = {n_prompts} prompts)")
    hdr = (f"{'condition':<12} {'axis':<6} {'token agree':>12} {'SEMQ HER':>9} "
           f"{'SEMQ coord':>11} {'tok-same/':>11} {'margin flip':>12} "
           f"{'free exact':>12} {'95% CI':>16} {'1st div':>8}")
    print("\nTeacher-forced (identical context at every position) + free-running")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        mf = r["median_margin_flipped"]
        lo, hi = r["free_exact_ci95"]
        # A control that never diverges has no flipped tokens, so a median
        # margin over them does not exist. Print nothing rather than a zero.
        print(f"{r['condition']:<12} {r['axis']:<6} {r['token_agreement']:>11.2%} "
              f"{r['semq_her']:>9.4f} {r['semq_coord_change']:>11.2e} "
              f"{r['early_warning_steps']:>10.2%} "
              f"{(f'{mf:.3f}' if mf == mf else '-'):>12} "
              f"{str(r['free_exact_count']) + '/' + str(r['n_prompts']):>12} "
              f"{f'[{lo:.0%}, {hi:.0%}]':>16} "
              f"{r['free_first_divergence_mean']:>8.1f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "decoding_matrix.json").write_text(json.dumps(
        {"model": MODEL, "device": DEVICE, "n_new_tokens": N_NEW_TOKENS,
         "quant_bins": QUANT_BINS,
         "prompts": list(PROMPTS), "failed": failed, "rows": rows},
        indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'decoding_matrix.json'}")

    from ari.attest import (sign_if_configured, build_references, model_ref,
                            config_ref, code_ref)
    # _model_revision() resolves the loaded commit; it is also what the cache
    # fingerprint keys on, so the attestation and the cache name one revision.
    references = build_references(
        models=[model_ref(MODEL, _model_revision(), dtype="float32")],
        config=config_ref({
            "model": MODEL, "n_new_tokens": N_NEW_TOKENS,
            "quant_bins": QUANT_BINS,
            "calibration_percentile": CALIBRATION_PERCENTILE,
            "prompts": list(PROMPTS),
            "conditions": [c.name for c in CONDITIONS],
        }),
        code=code_ref(packages=["semq", "torch", "transformers", "numpy"]),
    )
    sign_if_configured(metric="ARI-D", report_path=RESULTS / "decoding_matrix.json",
                       references=references,
                       extra={"model": MODEL, "device": DEVICE,
                              "n_new_tokens": N_NEW_TOKENS})
    print("\n'early warn' = steps where the token was identical but the SEMQ")
    print("logit code changed. That is the signal token comparison cannot give.")


if __name__ == "__main__":
    main()
