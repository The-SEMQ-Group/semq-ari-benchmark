# Batch Invariance — results

**Run 2026-09-08 on `p5.48xlarge` (8x H100 80GB, Hopper), torch 2.5.1+cu121, semq 1.5.0.**
1,000 frozen inputs, fixed QBIN scale from `spec/fingerprints-v0.1.csv`, one process per
capture, batch 32 as the reference inside each cell.

## Result

| model | cell | TF32 | deterministic | b1 | b8 | b128 | b512 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all-MiniLM-L6-v2 | A | off | on | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| all-MiniLM-L6-v2 | B | off | off | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| all-MiniLM-L6-v2 | C | **on** | off | **0.9470** | 1.0000 | 1.0000 | 1.0000 |
| bge-large-en-v1.5 | A | off | on | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bge-large-en-v1.5 | B | off | off | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| bge-large-en-v1.5 | C | **on** | off | **0.9990** | 1.0000 | 1.0000 | 1.0000 |

Both models give the same shape of answer. At true fp32 the encoders are batch invariant. The
only disagreement appears with TF32 on, and only at batch 1.

## Verdicts

| # | hypothesis | verdict |
| --- | --- | --- |
| H1 | deterministic on, TF32 off: batch does not change codes | **confirmed**, HER 1.0000 across both models |
| H2 | deterministic off: batch does change codes | **refuted**, HER 1.0000 across both models |
| H3 | any batch effect is weaker than the precision effect | **confirmed**, 0.947 against 0.0000 under bf16 |
| H4 | TF32 raises batch sensitivity | **confirmed**, and it is the only cell that moves |

## Reading

The review comment said batch size should not change a forward pass, because BatchNorm is off
and layer norm runs per sample. On this hardware, at true fp32, that is what we measure. Cells
A and B read 1.0000 everywhere.

The deterministic flag changes nothing on its own. Cell B removes it and the codes still agree,
so kernel and algorithm selection at fp32 is not what moves these encoders. This repeats the
GPU determinism result, where the deterministic setting also did not help.

TF32 is the whole effect. It is also the smaller-batch effect, which fits how the disagreement
arises: at batch 1 the matrix multiply is a matrix-vector product, cuBLAS picks a different
reduced-precision kernel for that shape, and the partial sums combine in a different order.
Larger batches share one kernel with the batch 32 reference and agree with it exactly. So batch
size is not itself a source of variation. It is a selector, and it only selects something
different once the arithmetic has mantissa bits to lose.

The two models differ in how much they lose. MiniLM at 384 dimensions drops to 0.9470 and
bge-large at 1024 drops to 0.9990. A wider vector needs every one of its coordinates to survive
for the code to match, so the higher agreement on the wider model is not what a naive reading
predicts. Neither is it explained by this run. Treat the two numbers as one result, not as a
ranking.

## Scope

Two encoders, one GPU architecture, one process per capture. Cells are never compared across
rows, since they differ in more than batch size. The `mach` comparison against the earlier A10G
captures is not included here and is still open.
