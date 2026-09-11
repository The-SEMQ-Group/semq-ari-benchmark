# Batch Invariance — does batch size change the codes on GPU?

**Status: run on H100 (Hopper), seven encoders, with a same-batch control.** Every cell's
control reads 1.0000, so capture-to-capture variation is zero and every lower reading is a
batch effect. Batch size changes codes at true fp32 on four of seven encoders, concentrated at
batch 1 and gone by batch 8 on all but one. The deterministic flag makes no difference
anywhere. H1 and H2 refuted, H3 confirmed, H4 mixed. Results: [`RESULTS.md`](RESULTS.md). This
document fixed the design *before* the run.

An earlier two-encoder run read 1.0000 across cells A and B and looked like clean batch
invariance. Both of those encoders turned out to be among the three that are invariant. The
seven-encoder table is the one to read.

## The question

Review comment on the ARI paper:

> we should try to understand why batch size affects embedding API/forward().
> in theory it should not. BatchNorm is turned off. layer norm is per sample/token.

The theory is correct. No operation in these encoders couples one row of a batch to another.
BatchNorm is off at inference and layer norm normalizes per sample. A pure function of one
input should return one answer whatever else travels with it in the batch.

The SciFact table agrees on CPU. Batch 8 and batch 128 both read HER 1.0000 against a batch 32
reference (Appendix F of the paper). So the question is only open on GPU.

On a GPU the batch shape is not neutral. It selects the kernel. cuBLAS picks a tile size and a
split-k factor from the problem shape, and split-k decides how many partial sums combine at the
end. Floating-point addition is not associative, so a different partition of the same sum gives
a different result. If batch size changes the codes, the cause is the reduction path, not the
model.

This experiment separates those two things: whether batch size changes the answer, and whether
the cause is reduction order or kernel choice.

## Hypotheses (pre-registered, with confirm/refute criteria)

| # | hypothesis | confirmed if | refuted if |
| --- | --- | --- | --- |
| **H1** | With deterministic algorithms on and TF32 off, batch size does not change codes | HER = 1.0000 for every batch pair | any pair reads HER < 1.0000 |
| **H2** | With deterministic algorithms off, batch size does change codes | at least one pair reads HER < 1.0000 | every pair reads HER = 1.0000 |
| **H3** | Any batch effect is weaker than the precision effect | min HER over batch pairs > HER under bf16 (0.0000, panel) | batch HER falls to 0 |
| **H4** | TF32 raises batch sensitivity | min HER with TF32 on < min HER with TF32 off | TF32 leaves the batch spread unchanged |

H1 and H2 together are the attribution. If both confirm, the batch effect exists and the
deterministic flag removes it, which places the cause in kernel and algorithm selection. If H1
confirms and H2 refutes, batch size is neutral on this hardware and the paper should say so.

## Design

Three cells, five batch sizes each. Every capture uses the same 1,000 frozen inputs, the same
fixed QBIN scale, and one process per capture.

| cell | dtype | TF32 | deterministic | batch sizes |
| --- | --- | --- | --- | --- |
| A | fp32 | off | on | 1, 8, 32, 128, 512 |
| B | fp32 | off | off | 1, 8, 32, 128, 512 |
| C | fp32 | on | off | 1, 8, 32, 128, 512 |

Batch 32 is the reference in each cell, because the published panel uses it. Each other batch
size is compared against that reference inside its own cell. Cells are never compared across
rows for the batch claim, since they differ in more than batch size.

Each cell also captures batch 32 a second time, in its own process, and compares the two. That
is the same-batch control, and it sets the floor every other reading in the cell is read
against. Without it a reading of 0.9990 cannot be told apart from ordinary capture-to-capture
variation, and at least one model varies between captures at a fixed batch. A batch reading at
or above the control says nothing about batch size, whatever its distance from 1.0000.

The fixed scale matters. A scale calibrated per capture would move the bin edges with the data
and hide or invent disagreement. Every capture takes the canonical scale from
`spec/fingerprints-v0.1.csv`, as the GPU determinism experiment does.

## What this adds to the paper

- Answers the review comment with a measurement rather than a citation.
- Fills tier T1 of the reproducibility ladder, where machine, kernel, hardware and model all
  stay fixed and only the batch shape moves.
- Adds Hopper to the GPU coverage. The limitations appendix currently says two architectures,
  Ampere and Turing.
- Gives one real `mach` reading against the existing A10G captures, which the limitations
  appendix records as unmeasured.

## Run it

```bash
# 1. the non-associativity measurement (CPU or GPU, seconds)
python nonassociativity.py --out results/nonassociativity.json

# 2. the batch sweep on the H100 box
python run_batch_sweep.py --label h100 --n 1000 --out ~/batch_out \
    --publish-s3 s3://semq-agent-memory-benchmark/batch-invariance/results

# 3. read the summary
cat ~/batch_out/summary.json
```

`--publish-s3` uploads each cell as it finishes. Use it on any instance that terminates on a
timer. A run that keeps its only copy on an ephemeral volume is one shutdown away from nothing.

## Cost

Fifteen captures of 1,000 inputs on one encoder. Each capture is a forward pass over 1,000
short texts, so the sweep is minutes of GPU time, not hours. The instance, not the compute, is
the expense.
