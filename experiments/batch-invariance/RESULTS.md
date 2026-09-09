# Batch Invariance — results

**Run 2026-09-08 on `p5.48xlarge` (8x H100 80GB, Hopper), torch 2.5.1+cu121, semq 1.5.0.**
1,000 frozen inputs, fixed QBIN scale from `spec/fingerprints-v0.1.csv`, one process per
capture, batch 32 as the reference inside each cell.

Seven encoders. `BAAI/bge-m3` is excluded: it ships no safetensors, and transformers 5.x
refuses to `torch.load` a `.bin` under torch < 2.6 (CVE-2025-32434). Upgrading torch for one
model would change the kernels for every other model in the table, and the kernels are what
this measures.

## Cell A — fp32, TF32 off, deterministic on

| model | dim | b1 | b8 | b128 | b512 |
| --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 384 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bge-large-en-v1.5 | 1024 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| all-mpnet-base-v2 | 768 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| multilingual-e5-large | 1024 | **0.9990** | 1.0000 | 1.0000 | 1.0000 |
| nomic-embed-text-v1.5 | 768 | **0.9990** | 1.0000 | 1.0000 | 1.0000 |
| snowflake-arctic-embed-l | 1024 | **0.9990** | **0.9990** | 1.0000 | 1.0000 |
| mxbai-embed-large-v1 | 1024 | **0.5560** | 1.0000 | 1.0000 | 1.0000 |

## Cell B — fp32, TF32 off, deterministic off

Bit-identical to cell A for all seven models, at every batch size. The deterministic flag
changes nothing here.

## Cell C — fp32, TF32 on, deterministic off

| model | b1 | b8 | b128 | b512 |
| --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.9470 | 1.0000 | 1.0000 | 1.0000 |
| bge-large-en-v1.5 | 0.9990 | 1.0000 | 1.0000 | 1.0000 |
| multilingual-e5-large | 0.9980 | 1.0000 | 1.0000 | 1.0000 |
| nomic-embed-text-v1.5 | 0.8670 | 0.9970 | 0.9990 | 0.9990 |
| snowflake-arctic-embed-l | 1.0000 | 0.9980 | 1.0000 | 1.0000 |
| mxbai-embed-large-v1 | 0.5560 | 1.0000 | 1.0000 | 1.0000 |
| all-mpnet-base-v2 | 0.8560 | 0.8530 | 0.8820 | 0.8490 |

## Batch boundary (two encoders, TF32 on)

| model | b1 | b2 | b4 | b8 | b16 | b64 |
| --- | --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | 0.9470 | 0.9990 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bge-large-en-v1.5 | 0.9990 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Verdicts

| # | hypothesis | verdict |
| --- | --- | --- |
| H1 | deterministic on, TF32 off: batch does not change codes | **refuted**, 4 of 7 models disagree at batch 1 |
| H2 | deterministic off: batch does change codes | **refuted**, cells A and B are identical everywhere |
| H3 | any batch effect is weaker than the precision effect | **confirmed**, worst batch reading 0.5560 against 0.0000 under bf16 |
| H4 | TF32 raises batch sensitivity | **mixed**, see below |

H4 does not resolve cleanly. TF32 lowers agreement on MiniLM (1.0000 to 0.9470) and nomic
(1.0000 to 0.8670). It leaves mxbai unchanged at 0.5560. On arctic the batch 1 reading goes
*up*, from 0.9990 to 1.0000.

## Reading

The review comment said batch size should not change a forward pass, because BatchNorm is off
and layer norm runs per sample. The mathematics is right and the measurement mostly agrees,
but not universally.

Batch sensitivity is real, and it is concentrated at batch 1. Every model except arctic reads
1.0000 from batch 8 upward in cells A and B, and the two-encoder boundary sweep shows the
effect gone by batch 4. A batch of one is a matrix-vector product, so cuBLAS selects a
different kernel for that shape and the partial sums combine in a different order. Batch size
selects a kernel rather than acting as a source of variation itself.

The deterministic flag is inert. Cells A and B agree bit for bit on all seven models, which
repeats the GPU determinism result where the same setting also did not help.

Two models need separate treatment.

**all-mpnet-base-v2 under TF32 is not a batch result.** It disagrees at every batch size,
0.8490 to 0.8820, with no ordering by batch. Two captures of this model under TF32 disagree
whatever their shape, so the cell C row measures capture-to-capture variation and not batch
sensitivity. It should not be read alongside the others.

**mxbai-embed-large-v1 reads exactly 0.5560 at batch 1 in all three cells.** The same value
with TF32 on and off makes it precision-independent, which none of the mechanisms above
explain. This model is also the outlier in `results/mach_cross_gpu.csv` from the GPU
determinism experiment, at 0.5371 for `mach_t4_vs_cpu`. Something is specific to this model
and this run does not identify it.

## Scope

Seven encoders, one GPU architecture, one process per capture, one run per cell. Cells are
never compared across rows, since they differ in more than batch size. The `mach` comparison
against the earlier A10G captures is still open, but the code matrices from this run are
retained in S3, so it does not need a repeat.
