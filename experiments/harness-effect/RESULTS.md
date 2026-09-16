# Harness-effect results

This experiment compares SWE-agent and OpenHands on two models in the Open-SWE-Traces dataset.
Data: [harness_effect.json](results/harness_effect.json). Implementation: [run.py](run.py).

The analysis uses `resolved == 1` as a successful outcome.
It includes cases with at least two graded rollouts on each side of a comparison.
The dataset contains up to three rollouts per cell.

## Comparisons

Both comparisons use the same agreement estimator.

| contrast | what changes | what stays fixed |
| --- | --- | --- |
| **harness effect** | the scaffold | the model, the task, the grader |
| **model effect** | the model | the scaffold, the task, the grader |

## Measured effects

Intervals are bootstrap intervals over cases. Each reported interval excludes zero.

| contrast | cases | self-consistency | cross agreement | **ARI-E effect** | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, model = Qwen3.5-122B | 10,788 | 0.874 | 0.785 | **+0.089** | [+0.083, +0.094] |
| scaffold, model = Minimax-M2.5 | 12,195 | 0.899 | 0.849 | **+0.050** | [+0.044, +0.052] |
| model, scaffold = SWE-agent | 10,674 | 0.880 | 0.839 | **+0.041** | [+0.039, +0.046] |
| model, scaffold = OpenHands | 11,933 | 0.892 | 0.833 | **+0.059** | [+0.054, +0.062] |

## Adjustment for self-disagreement

ARI-E subtracts cross-harness agreement from mean self-consistency. It is not a pass-rate difference.

| contrast | apparent disagreement | agent's own noise | ARI-E effect | share removed |
| --- | ---: | ---: | ---: | ---: |
| scaffold, Qwen | 0.215 | 0.126 | +0.089 | **59%** |
| scaffold, Minimax | 0.151 | 0.101 | +0.050 | **67%** |
| model, SWE-agent | 0.161 | 0.120 | +0.041 | **74%** |
| model, OpenHands | 0.167 | 0.108 | +0.059 | **65%** |

## Interpretation and limits

The mean scaffold effect was +0.069; the mean model effect was +0.050.
Their ratio was approximately 1.4 for these pairs and cases.
The adjustment removed 59–74 percent of the unadjusted disagreement.
Qwen's scaffold effect was +0.089; Minimax's was +0.050.
These differences show that the effect depends on the model-harness pair.

Approximately 22 percent of rows had `resolved = -1` and were excluded.
If grading failure depends on task difficulty, this exclusion can bias the result.
The dataset was not collected as a controlled scaffold experiment. Independence between rollouts is not established.
Tool names differ between harnesses, so this report compares outcomes rather than normalized action sequences.

## Reproduce

From the repository root, install the data dependencies:

```bash
python -m pip install -e ".[data]"
python experiments/harness-effect/extract.py
python experiments/harness-effect/run.py
```

The extraction downloads [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces).
Check `extract.py --help` for the row limit before reproducing the full analysis.
Outputs are written under `experiments/harness-effect/results/`.
Match the captured dataset revision and case counts before comparing new results with this table.
