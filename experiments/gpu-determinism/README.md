# GPU Determinism — does self-hosted inference drift on GPU?

**Status: run on A10G (Ampere) + T4 (Turing).** The cross-process drift is attributed to
**TF32** (isolation 2×2): TF32 off → 1.000, TF32 on → 0.65–0.88, determinism has no effect;
and it is *not* the framework default (PyTorch 2.3 has TF32 off). H2/H3 confirmed, H4 refuted
(cross-GPU is portable; the gap is CPU↔GPU), H5 refuted (determinism doesn't help). Reconciles
with ReproRAG. Results: [`RESULTS.md`](RESULTS.md). This document fixed the design *before* the
run. The
panel measured self-hosted models on CPU and found HER = 1.000 on the comparable core. CPU
float ops are bit-deterministic, so that result may be a property of the *test environment*,
not of real deployment — which is **GPU**. This experiment tests whether GPU inference is
bit-reproducible, and if not, exactly which knob breaks it.

## The question

Is the self-hosted `proc` = 1.000 an artifact of CPU determinism? On GPU, three sources of
non-determinism that CPU lacks come into play:

- **CUDA atomics** — `atomicAdd` in reductions/scatter accumulates in nondeterministic order.
- **Kernel autotuning** — cuBLAS/cuDNN pick different kernels per run; flash-attention fuses
  reductions nondeterministically.
- **TF32** — on Ampere+ (A10G, A100, L4) `torch.float32` matmul silently runs in **TF32** by
  default, dropping mantissa bits. So "fp32 on GPU" ≠ "fp32 on CPU".

## Hypotheses (pre-registered, with confirm/refute criteria)

| # | hypothesis | confirmed if | refuted if |
| --- | --- | --- | --- |
| **H1** | GPU drifts cross-process at fixed precision (kernels/atomics) | `proc` HER < 1.0 on GPU (default flags) | `proc` HER = 1.0 on GPU |
| **H2** | The precision cliff persists on GPU | `prec` (fp16/bf16 vs fp32) HER ≈ 0 | HER stays ≈ 1.0 |
| **H3** | **TF32 is a hidden cliff** — GPU-fp32(TF32 on) ≠ CPU-fp32 | `mach`(GPU-fp32 vs CPU-fp32) HER < 1.0, and it recovers with TF32 **off** | no change from TF32 |
| **H4** | Cross-GPU drift — different SM architecture → different codes | `mach`(A10G vs T4) HER < 1.0 at fixed precision + deterministic | HER = 1.0 across GPUs |
| **H5** | Forcing determinism fixes H1 | `proc` deterministic = 1.0 while `proc` default < 1.0 | deterministic still < 1.0 |

**Headline the experiment can support if H1/H3/H5 confirm:** *self-hosted GPU serving is
bit-reproducible only if you disable TF32, fix precision, and force deterministic algorithms —
and by default it is not.* An actionable governance finding, invisible to cosine/retrieval.

## Models

- **The 8 panel encoders** (bge-large, bge-m3, multilingual-e5, mpnet, MiniLM, nomic, mxbai,
  arctic) — directly comparable to their CPU 1.000; all run **fp32 on GPU** (each < 2 GB), so
  the full TF32 / `mach` matrix applies.
- **One 7B decoder embedder** (`intfloat/e5-mistral-7b-instruct`, dim 4096) — uses
  flash-attention, the highest non-determinism risk, and the best test of H1. It already has a
  measured `(b, κ)` from the drift-sensitivity sweep, so this gives it its **first ARI +
  a completed fingerprint**. At 7B it runs **fp16/bf16 on GPU** (fp32 OOMs a 24 GB card), so
  its fp32 baseline for `prec`/`mach` is computed on CPU.

## Hardware

| instance | GPU | arch | TF32 | role |
| --- | --- | --- | --- | --- |
| `g5.xlarge` | A10G, 24 GB | Ampere | yes | primary; TF32 on/off contrast |
| `g4dn.xlarge` | T4, 16 GB | Turing | no | the `mach` GPU↔GPU contrast (different SM arch) |

CPU-fp32 baselines are the reference (recomputed deterministically; QBIN is calibrated on the
frozen ARI-Bench-v0.1, so `s` is fixed and codes are comparable across machines).

## Condition matrix (per model)

| cell | config | isolates | realises |
| --- | --- | --- | --- |
| `same` default | GPU, TF32 on, nondeterministic kernels, re-encode in-process | per-call kernel/atomic nondeterminism | H1 (within-process) |
| `proc` default | GPU, fresh process | cross-process autotune/atomics | **H1** |
| `proc` deterministic | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` | does forcing determinism fix it | **H5** |
| `prec` TF32 | fp32 TF32-on vs fp32 TF32-off (encoders only) | the hidden TF32 cliff | **H3** |
| `prec` half | bf16/fp16 vs fp32 | precision cliff | H2 |
| `mach` GPU↔CPU | GPU-fp32 codes vs CPU-fp32 baseline | hardware + TF32 change | H3/H4 |
| `mach` GPU↔GPU | A10G codes vs T4 codes (fixed precision, deterministic) | SM-architecture change | **H4** |

Each cell reports HER, mean Hamming `H̄`, and a 95% block bootstrap CI over inputs, at
n = 1000 (frozen ARI-Bench-v0.1); the 7B at n ≥ 200. Per-condition SHA-256 audit digest as usual.

## Protocol & tooling

- A GPU-capable capture tool encodes each (model, device, dtype, tf32, deterministic) cell and
  **persists the full QBIN code matrix + a meta record** (`scale_s`, `content_hash`, device,
  `sm_arch`, dtype, tf32, deterministic) to S3 — extending the capture/compare pattern already
  in `ari/tools/capture_time_baseline.py`. `mach` is then a compare across two machines'
  stored code matrices (rebuild the probe from the pinned `scale_s`, diff codes) — exactly the
  time-condition machinery, one axis over.
- `same`/`proc` (single machine) are computed in-place as in the panel.
- Determinism cell sets the flags **before** any CUDA context is created.
- The 8 encoders slot into the leaderboard as **`mach`/`prec` diagnostics** (headline ARI
  stays the comparable core `{proc, conc, time}`); the 7B becomes a **new self-hosted row**.

## Where results go

- Per-model GPU condition results → `results/` here (CSV) + a `RESULTS.md`.
- Diagnostics fold into the 8 panel submissions (`mach`, `prec`-TF32); the 7B gets a new
  submission and a completed `(s, b, κ)` registry entry.
- The finding (whatever it is) updates the self-hosted findings and the leaderboard
  status.

## Cost & ops

~3–5 GPU-hours across the two instances → **~$15–25** on a standard cloud GPU host. Ops notes:
install the `semq` SDK and `sentence-transformers` into a fresh venv, keep credentials out of
any instance logs, terminate deterministically by instance id, use `trust_remote_code` for
nomic / e5-mistral, and budget for the 7B model download (~15 GB).
