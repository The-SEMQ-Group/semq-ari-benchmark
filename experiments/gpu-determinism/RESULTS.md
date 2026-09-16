# GPU Determinism — Results (A10G + T4, v0.1-preview)

**The self-hosted 1.000 breaks on GPU — but the cause is specifically TF32, not GPU
nondeterminism in general, and it is *not* the framework default.** On modern PyTorch (2.3,
where `matmul.allow_tf32` defaults to **False**), the 8 encoders are bit-reproducible
cross-process on GPU too — consistent with prior work (ReproRAG, see [reconciliation](#reconciliation-with-prior-work-reprorag)).
But the moment **TF32 is enabled** — the pre-1.12 PyTorch default, and a setting many
serving stacks turn on for speed — cross-process reproducibility breaks (proc HER 0.65–0.88),
and forcing deterministic algorithms does **not** recover it; only disabling TF32 does.
Separately, TF32 breaks equivalence with a fp32/CPU baseline entirely (the audit gap).

Machine-readable data: [`results/a10g.csv`](results/a10g.csv),
[`results/isolation_2x2.csv`](results/isolation_2x2.csv),
[`results/mach_cross_gpu.csv`](results/mach_cross_gpu.csv). Scope: 8 encoders + one 7B, n=512,
on A10G (Ampere, sm_86) and T4 (Turing, sm_75). Fixed QBIN scale from the registry, so codes
are comparable across CPU/GPU.

## Cross-process reproducibility (`proc`) — attribution: it is TF32, not determinism

The isolation 2×2 (TF32 on/off × determinism on/off, at fp32, cross-process; mean over the 8
encoders, per-model in [`results/isolation_2x2.csv`](results/isolation_2x2.csv)):

| | TF32 **off** / nondet | TF32 **off** / det | TF32 **on** / nondet | TF32 **on** / det |
| --- | --- | --- | --- | --- |
| **mean proc HER** | **1.000** | **1.000** | **0.749** | **0.749** |

- **TF32 is the only variable that matters.** TF32 off → 1.000; TF32 on → 0.65–0.88. The two
  determinism columns are **bit-identical** (e.g. bge-large 0.8301 in both), so
  `use_deterministic_algorithms` has **zero effect** on these transformers' cross-process
  reproducibility. *(An earlier draft of this doc misattributed the fix to determinism — that
  cell had TF32 off* and *determinism on; the 2×2 isolates it. The fix is TF32 off.)*
- **It is not the framework default.** PyTorch 2.3 defaults `matmul.allow_tf32` to **False**,
  so a vanilla deployment is cross-process reproducible. The risk is TF32-on: the pre-1.12
  default, and common in speed-optimised serving (TensorRT, explicit `allow_tf32=True`).
- **Mechanism.** `same` (in-process) is always 1.000. Across a fresh process, cuBLAS can pick a
  different kernel (per-process autotuning). At true fp32 that difference is ~1e-7 — below the
  quantizer floor → HER 1.0. TF32 truncates to ~19 bits, so the *same* cross-process kernel
  difference becomes ~1e-3 → crosses the floor → the code flips. **TF32 amplifies cross-process
  kernel variation from invisible to code-flipping** — which is exactly what the high-pass
  instrument is built to expose.

## `mach` (GPU vs CPU) — the TF32 hidden cliff — H3, H2

Comparing each model's GPU codes against its CPU-fp32 baseline (same fixed scale, same inputs):

| encoder | GPU fp32 **TF32 on** vs CPU | GPU fp32 **TF32 off+det** vs CPU | GPU **bf16** vs CPU (prec) |
| --- | --- | --- | --- |
| bge-large-en-v1.5 | 0.830 | 1.000 | 0.004 |
| bge-m3 | 0.641 | 1.000 | 0.000 |
| multilingual-e5-large | 0.670 | 1.000 | 0.000 |
| all-mpnet-base-v2 | 0.793 | 0.996 | 0.016 |
| all-MiniLM-L6-v2 | 0.875 | 0.998 | 0.131 |
| nomic-embed-text-v1.5 | 0.731 | 1.000 | 0.018 |
| **mxbai-embed-large-v1** | **0.541** | **0.537** | 0.004 |
| snowflake-arctic-embed-l | 0.660 | 0.998 | 0.004 |

- **H3 confirmed — TF32 is a hidden cliff.** "fp32 on GPU" runs matmul in **TF32** by default
  on Ampere, so it disagrees with a CPU (true-fp32) baseline on **12–46%** of inputs. Turning
  TF32 off recovers ≈ 1.000 for 7 of 8. A model audited at fp32/CPU and served at fp32/GPU is
  **not** the same system — silently, and invisibly to cosine/retrieval.
- **H2 confirmed.** The precision cliff persists on GPU: bf16 vs fp32 → HER ≈ 0, as on CPU.

## Cross-GPU (`mach`, A10G Ampere ↔ T4 Turing) — H4

Both GPUs at fp32, TF32 off, deterministic, same fixed scale:

| encoder | A10G ↔ T4 | T4 ↔ CPU | A10G ↔ CPU |
| --- | --- | --- | --- |
| bge-large · bge-m3 · e5-large · nomic | 1.000 | 1.000 | 1.000 |
| all-mpnet-base-v2 | 0.996 | 1.000 | 0.996 |
| all-MiniLM-L6-v2 | 0.998 | 1.000 | 0.998 |
| snowflake-arctic-embed-l | 0.998 | 1.000 | 0.998 |
| **mxbai-embed-large-v1** | **1.000** | **0.537** | **0.537** |
| **mean A10G↔T4** | **0.999** | | |

**H4 refuted (mostly).** With TF32 off + deterministic, two different GPU architectures produce
**essentially identical codes** (mean 0.999; residual of a few inputs on 3 models). GPU→GPU is
portable once determinism is forced. **The dangerous axis is CPU↔GPU, not GPU↔GPU-arch.**

## Honest outlier: `mxbai-embed-large-v1` — now precisely characterised

mxbai's two GPUs **agree with each other (1.000)** but **both disagree with CPU (0.537)**. So it
is not a cross-GPU-architecture effect — it is a **CPU↔GPU kernel divergence**: some op computes
differently on the CUDA backend than on CPU, consistently across GPUs. This is the irreducible
"audit on CPU, serve on GPU" gap made concrete for one model — reported, not hidden.

## 7B decoder embedder — `intfloat/e5-mistral-7b-instruct`

At **bf16** (fp32 OOMs a 24 GB card), TF32 off, n=128: `same` = 1.000, **`proc` = 1.000**
(two-capture cross-process). Its first ARI. bf16 does not route through TF32, so a bf16-served
model sidesteps *that* cliff cross-process (while still paying the bf16-vs-fp32 precision
cliff). `s = 0.0439` (bf16-calibrated, dim 4096) — a bf16 fingerprint entry, pending an fp32
`s` on a larger card.

## Reconciliation with prior work (ReproRAG)

[ReproRAG](https://arxiv.org/abs/2509.18869) benchmarks embedding reproducibility across
fp32/fp16/bf16/tf32 × determinism on BGE/E5/Qwen and reports the embeddings **bit-identical**
(L2 = 0.0) — concluding embeddings are reproducible. **We reconcile, we do not contradict:**
they measured *same-GPU, same-config, at framework defaults* (TF32 off) — where we also get
1.000. Two things they did not do, which is where this experiment adds signal: (1) **CPU↔GPU
and TF32-on cross-process**, the axes where drift actually appears; (2) a **discrete-attractor
metric** — where their raw L2 reads "5.74e-4, reproducible", the quantizer reads a flipped
code. Their cudnn-only determinism toggle is also a no-op for transformer matmuls (see the 2×2).

## Verdict

| # | hypothesis | result |
| --- | --- | --- |
| H1 | GPU drifts cross-process at fixed precision | **partly** — only with TF32 on (0.65–0.88); at true fp32 it is 1.000 |
| H2 | precision cliff persists on GPU | **confirmed** — bf16 HER ≈ 0 |
| H3 | TF32 is a hidden cliff (GPU-fp32 ≠ CPU-fp32) | **confirmed** — 0.54–0.88, recovers with TF32 off |
| H5 | forcing determinism fixes the cross-process drift | **refuted** — determinism has zero effect; only TF32 off fixes it |
| H4 | cross-GPU (different SM arch) drift | **refuted** — A10G↔T4 ≈ 1.000; the real gap is CPU↔GPU (mxbai) |

**Headline (calibrated):** on modern PyTorch, TF32 is off by default and GPU embeddings are
cross-process reproducible — consistent with prior work. But **enabling TF32 (common in
speed-optimised serving) silently breaks cross-process reproducibility (HER 0.65–0.88), a
determinism flag does not fix it, and TF32 also breaks equivalence with the fp32/CPU baseline**
— all invisible to cosine/retrieval, quantified per-model only by a SEMQ-class instrument. This
experiment is best read as an **instrument validation on a known phenomenon**, not a new
discovery; the novel data is the black-box **API panel** (§ Deployed-Agent Panel).

## Method & caveats

- Probe: SEMQ QBIN n=2, fixed registry scale `s` (codes comparable across CPU/GPU).
- `proc` measured in-process (fresh subprocess) on the GPU; `mach` by comparing stored GPU code
  matrices against local CPU-fp32 captures. Determinism cell: TF32 off +
  `use_deterministic_algorithms(warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- Two GPUs — A10G (Ampere, sm_86) and T4 (Turing, sm_75) — n=512, 8 encoders + one 7B. Data:
  [`results/a10g.csv`](results/a10g.csv), [`results/mach_cross_gpu.csv`](results/mach_cross_gpu.csv).
- **Pending:** larger n, an fp32 `s` for the 7B on a bigger card, and the `conc`/`time` axes.
