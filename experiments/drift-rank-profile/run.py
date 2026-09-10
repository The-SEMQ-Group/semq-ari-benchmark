# Copyright (c) 2026 The SEMQ Group.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.

"""Where does drift from a realistic serving change actually land, by rank?

SEMQ's only surviving advantage at the decoding layer is coverage. It reads
the whole logit vector, so it sees changes below the top ranks that a top-2
margin statistic is structurally blind to. BASELINES.md showed that with
synthetic tail noise: SEMQ did not move while the margin read exactly zero.

That shows SEMQ *can* see what the margin cannot. It does not show that any
real serving change produces drift down there. This asks the question the
other way round, and it is pre-registered so the answer counts either way:

    Pre-registered question. For a serving change a provider might actually
    make, how is the change in the logit vector distributed across token
    ranks?

    Supports coverage. Drift mass sits below the top few ranks, the top-2
    margin barely moves, and SEMQ reads the change.

    Refutes coverage. Drift concentrates in the top ranks. The margin sees it
    for 8 bytes a step, and SEMQ's extra 16,000 bytes buy nothing. Then the
    coverage argument is dead and this file records that.

**Why the rank profile rather than a pass/fail.** Asking only "did the margin
notice" is binary and hides the mechanism. Bucketing the change by the token's
rank in the reference distribution shows where the mass is, so a reader can
see whether the answer is "the tail" or "everywhere" rather than trusting a
verdict.

**The circularity this avoids.** A bias applied to tokens chosen *by rank*
would settle the question by construction: put the bias at rank 500 and the
margin cannot see it, which proves nothing. Every condition here selects
tokens by meaning or by mechanism, never by rank, and where they land is the
measurement.

Conditions, in the order they are worth trusting:

  adapter   A LoRA adapter merged into the base model. The most realistic
            silent change a provider makes, and the one most likely to refute
            the hypothesis: an adapter is trained to change behaviour, so the
            top ranks are exactly where it should show up. Two public adapters
            run by default, at rank 16 and rank 32, so adapter strength is an
            axis rather than a single point. Both were published by third
            parties for their own purposes, which is the point: neither was
            built to make this experiment come out any particular way.
  bias      A logit bias over a fixed, semantically chosen token set. Where
            those tokens sit in the reference ranking is not chosen.
  bf16      Precision, carried over as the known-isotropic control. Its drift
            should be spread across all ranks, which is what makes it the
            baseline the other two are read against.

A tokenizer change is deliberately absent. Two tokenizers do not share
vocabulary indices, so there is no common space to measure a per-token delta
in, and the input segmentation changes as well, so the trajectory diverges
rather than drifting. That needs its own method, not another row here.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

from ari.code_metrics import chunk_widths, chunked_code_diff

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

MODEL = os.environ.get("DRIFT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
DEVICE = os.environ.get("DRIFT_DEVICE", "cpu")

# Public LoRA adapters per base model. None declares `modules_to_save`, so none
# resizes the embedding or the output head: the adapted model scores over the
# same vocabulary indices as the reference, which is what makes a per-token
# delta meaningful. The script verifies that rather than trusting it.
#
# The 7B set is not a clean factorial design, and the write-up should not claim
# one. What it does contain:
#
#   * a rank pair at matched scaling: namanadep (r16) and SeeFlock (r32), both
#     alpha/r = 2.0 over the same seven projections;
#   * a breadth contrast at matched rank and scaling: SeeFlock adapts all seven
#     projections, zjudai adapts only q_proj and v_proj;
#   * Hatim2221 at r64, which is the only working rank-64 adapter published for
#     this base model, but at alpha/r = 0.25 rather than 2.0. It is a fourth
#     data point, NOT the top of the rank ladder, and confounds rank with
#     scaling. Read it on its own.
#
# `robertou2/task-12-Qwen-Qwen2.5-7B-Instruct` was in this list and is not any
# more: it publishes a zero-byte adapter_model.safetensors, which cost a GPU run
# on 2026-08-08. `adapter_metadata()` below now refuses such repos up front.
ADAPTERS_BY_MODEL = {
    "Qwen/Qwen2.5-1.5B-Instruct": (
        "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16",   # r16 a32, 7 mods
        "renezander030/qwen-2.5-1.5b-de-pii-redactor",       # r32 a64, 7 mods
    ),
    "Qwen/Qwen2.5-7B-Instruct": (
        "namanadep/Qwen2.5-7B-Manus-Distill",                 # r16 a32, 7 mods
        "SeeFlock/task-12-Qwen-Qwen2.5-7B-Instruct",          # r32 a64, 7 mods
        "zjudai/flowertune-medical-lora-qwen2.5-7b-instruct",  # r32 a64, 2 mods
        "Hatim2221/qwen2.5-7b-arabic-math-cot",               # r64 a16, 7 mods
    ),
}
ADAPTERS = tuple(
    a.strip() for a in os.environ.get(
        "DRIFT_ADAPTER",
        ",".join(ADAPTERS_BY_MODEL.get(MODEL, ()))).split(",") if a.strip())
if not ADAPTERS:
    raise SystemExit(
        f"No adapters known for {MODEL}. Set DRIFT_ADAPTER to a comma-separated "
        "list, or add an entry to ADAPTERS_BY_MODEL. The adapter arm is the one "
        "most able to refute this experiment, so it is not optional.")

N_NEW_TOKENS = int(os.environ.get("DRIFT_NEW_TOKENS", "48"))
# Number of prompts to use, counted from the front of the shared corpus. The
# default uses all of them. The recorded 1.5B run used 12, so it stays
# reproducible after the corpus grew to 48.
N_PROMPTS = int(os.environ.get("DRIFT_N_PROMPTS", "0"))
BIAS_STRENGTH = float(os.environ.get("DRIFT_BIAS", "-5.0"))

# Rank buckets. The first two matter most: a top-2 margin statistic can only
# see a change that reaches them.
BUCKETS = ((1, 2), (3, 10), (11, 100), (101, 1000), (1001, 10**9))

def _ari_d_prompts() -> tuple[str, ...]:
    """The 48 prompts ARI-D reports on, loaded from the file that owns them.

    Sharing the corpus makes the two experiments directly comparable: a rank
    profile measured here refers to the same generations whose token agreement
    and HER appear in the ARI-D tables. Copying the list would let the two
    drift apart silently.
    """
    import importlib.util
    import sys

    src = HERE.parent / "decoding-reproducibility" / "run_matrix.py"
    spec = importlib.util.spec_from_file_location("_ari_d_run_matrix", src)
    mod = importlib.util.module_from_spec(spec)
    # `@dataclass` in that module resolves its own class through sys.modules,
    # so registering the module has to happen before it executes.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return tuple(mod.PROMPTS)


PROMPTS = _ari_d_prompts()
if N_PROMPTS:
    PROMPTS = PROMPTS[:N_PROMPTS]

# A refusal and hedging vocabulary, the kind of set a provider suppresses or
# promotes when it changes a safety layer. Chosen for meaning. Where these
# tokens sit in the reference ranking is measured, not selected.
BIAS_WORDS = (
    "sorry", "Sorry", "cannot", "Cannot", "unable", "unfortunately",
    "Unfortunately", "however", "However", "although", "Although",
    "might", "maybe", "perhaps", "possibly", "generally", "typically",
    "assist", "apolog", "regret", "advise", "recommend", "caution",
)


def load_model(adapter: str = ""):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {"dtype": torch.float32}
    if DEVICE != "cpu":
        kw |= {"device_map": DEVICE, "low_cpu_mem_usage": True}
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
    if DEVICE == "cpu":
        model = model.to("cpu")
    if adapter:
        from peft import PeftModel
        # Merging puts the adapter into the weights, which is what a provider
        # serving a fine-tune actually ships. An unmerged adapter would add a
        # runtime branch instead and would not be the same measurement.
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


def chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def reference_run(tok, model):
    """Greedy continuations and the logits that produced them."""
    import torch

    conts, logits = [], []
    for p in PROMPTS:
        ids = tok(chat(tok, p), return_tensors="pt").input_ids.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=N_NEW_TOKENS,
                                 do_sample=False, num_beams=1,
                                 pad_token_id=tok.pad_token_id)
        cont = out[0, ids.shape[1]:].tolist()
        conts.append(cont)
        logits.append(score(tok, model, p, cont))
    return conts, np.concatenate(logits, axis=0)


def score(tok, model, prompt: str, cont: list[int]) -> np.ndarray:
    """Logits at each continuation position under a fixed context."""
    import torch

    p_ids = tok(chat(tok, prompt), return_tensors="pt").input_ids[0].tolist()
    full = torch.tensor([p_ids + cont]).to(DEVICE)
    with torch.no_grad():
        out = model(input_ids=full).logits[0]
    start = len(p_ids) - 1
    return out[start:start + len(cont)].float().cpu().numpy()


def bias_token_ids(tok) -> list[int]:
    ids = set()
    for w in BIAS_WORDS:
        for form in (w, " " + w):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
    return sorted(ids)


def rank_profile(ref: np.ndarray, cur: np.ndarray) -> dict:
    """Share of the total absolute change that falls in each rank bucket.

    Rank is taken from the *reference* distribution, so the buckets mean the
    same thing across conditions.
    """
    n_steps, vocab = ref.shape
    # Ranking is done in row blocks. A full argsort of 2,304 x 151,936 costs
    # 2.8 GB as int64, and two of them plus the deltas will not fit alongside a
    # 7B model's activations. Blocks keep the peak to a few hundred megabytes
    # and the sums are exact either way.
    block = max(1, 2 ** 24 // vocab)
    sums = {b: 0.0 for b in BUCKETS}
    total = 0.0
    for start in range(0, n_steps, block):
        r = ref[start:start + block]
        d = np.abs(cur[start:start + block] - r)
        # float64 accumulation, so the totals do not depend on the block size.
        # In float32 the block and whole-array sums disagree in the ninth digit,
        # which would make a reported share a function of the vocabulary.
        total += float(d.sum(dtype=np.float64))
        order = np.argsort(-r, axis=1)
        ranks = np.empty(order.shape, dtype=np.int32)
        rows = np.arange(r.shape[0])[:, None]
        ranks[rows, order] = np.arange(vocab, dtype=np.int32)[None, :] + 1
        for lo, hi in BUCKETS:
            sums[(lo, hi)] += float(
                d[(ranks >= lo) & (ranks <= hi)].sum(dtype=np.float64))

    out = {}
    for lo, hi in BUCKETS:
        share = sums[(lo, hi)] / total if total > 0 else 0.0
        # The raw share is confounded by how many tokens a bucket holds. Ranks
        # 1001+ are 99.3% of a 152k vocabulary, so a perfectly isotropic change
        # puts 99.3% of its mass there and looks tail-confined. Enrichment
        # divides that out: 1.0 means "no more than chance", and only a value
        # well above 1.0 is a claim about where drift concentrates.
        size = float(min(hi, vocab) - lo + 1) / vocab
        out[f"{lo}-{hi if hi < 10**9 else 'end'}"] = {
            "share": share,
            "bucket_share_of_vocab": size,
            "enrichment": share / size if size > 0 else 0.0,
        }
    return out


def statistics(ref: np.ndarray, cur: np.ndarray) -> dict:
    def top2_margin(x):
        p = np.partition(x, -2, axis=1)[:, -2:]
        return p[:, 1] - p[:, 0]

    from ari.semq_compat import MAX_DIM, quant_context

    scale = float(np.percentile(np.abs(ref), 99.0))

    def code(X):
        dim = X.shape[1]
        bounds = list(range(0, dim, MAX_DIM)) + [dim]
        parts = []
        for lo, hi in zip(bounds, bounds[1:]):
            ctx = quant_context(hi - lo, n_bins=8, scale_max=scale)
            parts.append(np.asarray(ctx.batch_encode(
                np.ascontiguousarray(X[:, lo:hi], np.float32))))
            ctx.close()
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)

    cr, cc = code(ref), code(cur)
    diff = chunked_code_diff(cr, cc, n_bins=8,
                             widths=chunk_widths(ref.shape[1], MAX_DIM))

    # Top-20 set change, in row blocks. A whole-array argpartition allocates an
    # int64 index the size of the logits, which is 2.8 GB at 48 prompts on a
    # 152k vocabulary, and this needs two of them plus a negated copy.
    k = 20
    block = max(1, 2 ** 24 // ref.shape[1])
    jac_parts = []
    for start in range(0, ref.shape[0], block):
        tr = np.argpartition(-ref[start:start + block], k, axis=1)[:, :k]
        tc = np.argpartition(-cur[start:start + block], k, axis=1)[:, :k]
        jac_parts.append(np.array(
            [1.0 - len(set(a.tolist()) & set(b.tolist()))
             / len(set(a.tolist()) | set(b.tolist()))
             for a, b in zip(tr, tc)]))
    jac = np.concatenate(jac_parts)

    return {
        "token_agreement": float((ref.argmax(1) == cur.argmax(1)).mean()),
        "top2_margin_delta": float(np.abs(top2_margin(cur) - top2_margin(ref)).mean()),
        "top20_set_change": float(jac.mean()),
        "semq_coord_change": float(diff.coordinate_change_rate.mean()),
        "semq_byte_change_legacy": float(diff.byte_change_rate.mean()),
        "semq_her": float(diff.codes_equal.mean()),
        "max_abs_dlogit": float(np.abs(cur - ref).max()),
    }


def main() -> None:
    print(f"model {MODEL}   device {DEVICE}")
    for a in ADAPTERS:
        print(f"adapter {a}")
    tok, model = load_model()
    conts, ref = reference_run(tok, model)
    print(f"reference: {len(PROMPTS)} prompts, {ref.shape[0]} steps, "
          f"vocab {ref.shape[1]}\n")

    rows = []

    # --- bias over a semantically chosen set -------------------------------
    ids = bias_token_ids(tok)
    biased = ref.copy()
    biased[:, ids] += BIAS_STRENGTH
    ref_ranks = (-ref).argsort(1).argsort(1) + 1
    hit_ranks = ref_ranks[:, ids]
    rows.append({
        "condition": "logit_bias_refusal_set",
        "detail": f"{len(ids)} tokens, bias {BIAS_STRENGTH}",
        "biased_token_median_rank": float(np.median(hit_ranks)),
        "biased_token_min_rank": int(hit_ranks.min()),
        **statistics(ref, biased),
        "rank_profile": rank_profile(ref, biased),
    })

    # --- precision, the isotropic control ----------------------------------
    # `to()` casts in place, so this leaves `model` holding bf16-rounded values
    # even after the cast back. The reference is already measured and every
    # later arm loads its own weights, so nothing downstream reads it.
    import torch
    m16 = model.to(torch.bfloat16)
    cur = np.concatenate([score(tok, m16, p, c)
                          for p, c in zip(PROMPTS, conts)], axis=0)
    model.to(torch.float32)
    rows.append({
        "condition": "bf16",
        "detail": "precision control, expected isotropic",
        **statistics(ref, cur),
        "rank_profile": rank_profile(ref, cur),
    })

    # --- adapters ----------------------------------------------------------
    # Each arm loads its own fp32 copy of the base model. At 7B that is ~28 GB
    # against a 44 GB card, so exactly one can be resident: the previous copy
    # must be released *and* the caching allocator's blocks returned before the
    # next load, or transformers' warmup allocation fails.
    #
    # The release is written out at each site on purpose. Passing the model to
    # a helper that does `del` frees only the helper's own parameter, leaving
    # the caller's name bound and the memory held -- which is precisely how
    # this failed the first time.
    def reclaim() -> None:
        gc.collect()
        if DEVICE != "cpu":
            torch.cuda.empty_cache()

    del model, m16
    reclaim()

    for adapter in ADAPTERS:
        print(f"loading {adapter}", flush=True)
        _, adapted = load_model(adapter)
        cur = np.concatenate([score(tok, adapted, p, c)
                              for p, c in zip(PROMPTS, conts)], axis=0)
        del adapted
        reclaim()
        # An adapter that resized the output head would score over a different
        # vocabulary, and the per-token delta below would be comparing two
        # different index spaces. Refuse rather than report a number.
        if cur.shape != ref.shape:
            raise SystemExit(
                f"{adapter} scores {cur.shape[1]} logits against the "
                f"reference's {ref.shape[1]}. The vocabulary is not shared, so "
                "a per-token delta has no meaning here.")
        stats = statistics(ref, cur)
        # A merge that silently did nothing produces no drift at all, which
        # this experiment would read as the strongest possible support for
        # coverage. It is the one failure mode that flatters the hypothesis,
        # so it fails loudly instead.
        if stats["max_abs_dlogit"] == 0.0:
            raise SystemExit(
                f"{adapter} changed no logit anywhere. The merge did not "
                "apply. A silent no-op here would look like a clean result.")
        rows.append({
            "condition": "adapter_merged",
            "detail": adapter,
            **stats,
            "rank_profile": rank_profile(ref, cur),
        })

    def bucket_label(lo: int, hi: int) -> str:
        return f"r{lo}-{'end' if hi >= 10**9 else hi}"

    def row_label(r: dict) -> str:
        """Both adapter rows share a condition, so name them by adapter."""
        if r["condition"] != "adapter_merged":
            return r["condition"]
        return "adapter " + r["detail"].split("/")[-1][:24]

    hdr = (f"{'condition':<34}{'tok agree':>10}{'margin d':>10}{'SEMQ coord':>11}"
           + "".join(f"{bucket_label(lo, hi):>10}" for lo, hi in BUCKETS))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        prof = r["rank_profile"]
        print(f"{row_label(r):<34}{r['token_agreement']:>9.2%}"
              f"{r['top2_margin_delta']:>10.4f}{r['semq_coord_change']:>11.4f}"
              + "".join(f"{b['enrichment']:>9.2f}x" for b in prof.values()))

    print("\nRank columns are enrichment: the share of total |delta logit| in")
    print("that band, divided by the share of the vocabulary the band holds.")
    print("1.00x is exactly chance, so an isotropic change reads 1.00x across")
    print("the row and says nothing about where drift lives. Only a value well")
    print("above 1.00x locates it.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "drift_rank_profile.json").write_text(json.dumps(
        {"model": MODEL, "adapters": list(ADAPTERS), "device": DEVICE,
         "n_prompts": len(PROMPTS), "n_new_tokens": N_NEW_TOKENS,
         "n_steps": int(ref.shape[0]),
         "vocab": int(ref.shape[1]), "bias_strength": BIAS_STRENGTH,
         "buckets": [list(b) for b in BUCKETS], "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {RESULTS / 'drift_rank_profile.json'}")

    from ari.attest import sign_if_configured
    sign_if_configured(metric="drift-rank-profile",
                       report_path=RESULTS / "drift_rank_profile.json")


if __name__ == "__main__":
    main()
