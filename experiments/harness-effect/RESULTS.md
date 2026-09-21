# Harness-effect results

**Status: rerun on 2026-09-16 with the current estimator and a pinned dataset revision.**
The tables below supersede the output of 2026-08-07, which is kept in the [superseded section](#superseded-output-of-2026-08-07).
The estimator, the eligibility rule, and the pin are defined in [ARI-E-Bench v0.1](../../spec/ari-e-bench-v0.1.md).

This experiment compares SWE-agent and OpenHands on two models in the Open-SWE-Traces dataset.
Data: [harness_effect.json](results/harness_effect.json) with its [manifest](results/harness_effect.manifest.json).
Implementation: [fetch_outcomes.py](fetch_outcomes.py) and [run.py](run.py).

## Input

The dataset, the revision `f967cba3312573981a47fd7a7b80029b53909b5f`, the files, the columns, and the outcome table are in [ARI-E-Bench v0.1, section 5](../../spec/ari-e-bench-v0.1.md#5-frozen-dataset-reference).
At that revision the four cells hold 63 Parquet files (13.0 GB). The fetch read 16.7 MB of footers and column chunks.
The outcome table is not committed. `python experiments/harness-effect/fetch_outcomes.py --revision f967cba3312573981a47fd7a7b80029b53909b5f` writes
`results/outcomes.f967cba33125.csv.gz`, 151,219 rows, 4,324,658 bytes, SHA-256 `e8b1d239a22f5fa8...`. The full digest and the per-file digests are in its [manifest](results/outcomes.f967cba33125.manifest.json), which `run.py` checks before it scores.
Eligibility follows [section 1.3](../../spec/ari-e-bench-v0.1.md#13-eligibility). The dataset has at most three rollouts per cell.

| cell | rows | ungraded | graded |
| --- | ---: | ---: | ---: |
| SWE-agent, Qwen3.5-122B | 20,334 | 5,236 | 15,098 |
| SWE-agent, Minimax-M2.5 | 46,819 | 10,864 | 35,955 |
| OpenHands, Qwen3.5-122B | 40,463 | 9,506 | 30,957 |
| OpenHands, Minimax-M2.5 | 43,603 | 9,730 | 33,873 |

The ungraded share is 23.4 percent of all rows. The row counts equal the upstream `data_distribution.md` at this revision.

## Comparisons

Both comparisons use the same estimator.

| contrast | what changes | what stays fixed |
| --- | --- | --- |
| **scaffold effect** | the scaffold | the model, the task, the grader |
| **model effect** | the model | the scaffold, the task, the grader |

## Measured effects

Self-consistency is the mean of the two within-condition agreements. Intervals are case-bootstrap intervals with 2,000 resamples and seed 0.
Each interval excludes zero.

| contrast | cases | self-consistency | cross agreement | ARI-E effect | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, model = Qwen3.5-122B | 3,779 | 0.872 | 0.774 | **+0.098** | [+0.088, +0.107] |
| scaffold, model = Minimax-M2.5 | 9,597 | 0.902 | 0.850 | **+0.052** | [+0.047, +0.057] |
| model, scaffold = SWE-agent | 4,038 | 0.886 | 0.847 | **+0.039** | [+0.032, +0.045] |
| model, scaffold = OpenHands | 8,203 | 0.890 | 0.823 | **+0.067** | [+0.061, +0.072] |

The same rows in LaTeX are in [arie_table.tex](results/arie_table.tex).
Pass rates on the eligible cases: SWE-agent 0.515 and OpenHands 0.358 with Qwen3.5-122B; SWE-agent 0.507 and OpenHands 0.455 with Minimax-M2.5; Qwen3.5-122B 0.531 and Minimax-M2.5 0.512 in SWE-agent; Qwen3.5-122B 0.354 and Minimax-M2.5 0.445 in OpenHands.

## Adjustment for self-disagreement

ARI-E subtracts cross-condition agreement from mean self-consistency. It is not a pass-rate difference.
Apparent disagreement is one minus cross agreement. Own noise is one minus self-consistency.

| contrast | apparent disagreement | own noise | ARI-E effect | share removed |
| --- | ---: | ---: | ---: | ---: |
| scaffold, model = Qwen3.5-122B | 0.226 | 0.128 | +0.098 | 57% |
| scaffold, model = Minimax-M2.5 | 0.150 | 0.098 | +0.052 | 65% |
| model, scaffold = SWE-agent | 0.153 | 0.114 | +0.039 | 74% |
| model, scaffold = OpenHands | 0.177 | 0.110 | +0.067 | 62% |

## Repeat coverage

The two sides of a contrast often have different rollout counts on a case. The columns give the number of eligible cases with each (side A / side B) count.
Side A is SWE-agent in the scaffold contrasts and Qwen3.5-122B in the model contrasts.
The last columns give the effect on the cases where both sides have the same count.

| contrast | 2/2 | 2/3 | 3/2 | 3/3 | equal-repeat cases | effect on equal repeats | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| scaffold, model = Qwen3.5-122B | 549 | 1,103 | 604 | 1,523 | 2,072 | +0.103 | [+0.090, +0.116] |
| scaffold, model = Minimax-M2.5 | 1,040 | 1,469 | 1,957 | 5,131 | 6,171 | +0.055 | [+0.049, +0.061] |
| model, scaffold = SWE-agent | 418 | 1,358 | 381 | 1,881 | 2,299 | +0.039 | [+0.031, +0.048] |
| model, scaffold = OpenHands | 1,199 | 2,207 | 1,382 | 3,415 | 4,614 | +0.064 | [+0.057, +0.071] |

The equal-repeat effects lie inside the full-set intervals. Unequal counts do not move the estimate at this scale.

## Calibration checks

Permutation null, as in [section 7](../../spec/ari-e-bench-v0.1.md#7-calibration-checks): 50 permutations per contrast, each recomputed with 1,000 resamples.
About 5 percent of the null intervals should exclude zero.

| contrast | cases | null mean | null SD | null min | null max | intervals excluding zero |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scaffold, model = Qwen3.5-122B | 3,779 | +0.0004 | 0.0030 | -0.0076 | +0.0066 | 1 of 50 |
| scaffold, model = Minimax-M2.5 | 9,597 | +0.0001 | 0.0016 | -0.0049 | +0.0035 | 4 of 50 |
| model, scaffold = SWE-agent | 4,038 | -0.0003 | 0.0021 | -0.0064 | +0.0045 | 1 of 50 |
| model, scaffold = OpenHands | 8,203 | -0.0005 | 0.0018 | -0.0043 | +0.0031 | 4 of 50 |

The null means lie within 0.0006 of zero. In total 10 of 200 null intervals exclude zero, which is 5.0 percent.
The null standard deviations match the null intervals' half-widths divided by 1.96 to within 0.001.
Every measured effect is more than 12 null standard deviations from zero.
On the archived revision the same check gives 10 of 200 exclusions and null means within 0.0003 of zero.

Split-half. No cell has cases with four or more graded rollouts, so the split-half check between two halves of one condition has no cases at this revision. The check stays in `run.py` for datasets with more repeats.

The simulated false-positive rate at zero gap is in the [power simulation](../harness-power/RESULTS.md).

## Same data, current estimator

The archived output of 2026-08-07 was computed at revision `ad4805a5aa7de70d99cab0bb8f99b15304c76de0` (2026-08-03), the revision current on that date.
The same three columns were read at that revision (84 files, 18.3 GB in the cells, 22.6 MB read). The current estimator was run on them.
Neither the table nor the result of that run is committed. `python experiments/harness-effect/fetch_outcomes.py --revision ad4805a5aa7de70d99cab0bb8f99b15304c76de0`
writes a table of 207,489 rows, 5,881,882 bytes, SHA-256 `4148ba9c11a7e273...`, from 84 files and 22.6 MB of reads. `run.py --outcomes` on that table
reproduces the rerun column below. The eligible case counts equal the archived counts exactly.

| contrast | archived self | rerun self | archived cross | rerun cross | archived effect | rerun effect | archived CI | rerun CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| scaffold, model = Qwen3.5-122B (10,788 cases) | 0.874 | 0.873 | 0.785 | 0.785 | +0.089 | +0.088 | [+0.083, +0.094] | [+0.083, +0.093] |
| scaffold, model = Minimax-M2.5 (12,195 cases) | 0.899 | 0.897 | 0.849 | 0.848 | +0.050 | +0.049 | [+0.044, +0.052] | [+0.045, +0.053] |
| model, scaffold = SWE-agent (10,674 cases) | 0.880 | 0.879 | 0.839 | 0.838 | +0.041 | +0.042 | [+0.039, +0.046] | [+0.038, +0.046] |
| model, scaffold = OpenHands (11,933 cases) | 0.892 | 0.892 | 0.833 | 0.833 | +0.059 | +0.058 | [+0.054, +0.062] | [+0.054, +0.063] |

The archived estimator weighted cases by their number of run pairs and took the interval from one run pairing. The current estimator weights each case equally and takes the interval from the same per-case vector as the estimate.
On the same data the two differ by at most 0.002 in every column. The archived intervals were not centered on their estimates; the rerun intervals are.

The difference between the two revisions is larger than the difference between the two estimators.
The upstream repository removed 56,270 trajectories from these four cells on 2026-08-26.
That removal cut the eligible cases by 65 percent for the Qwen3.5-122B scaffold contrast and by 21 to 62 percent for the other contrasts, and raised the pass rates.
The effects at the pinned revision lie within 0.010 of the effects at the earlier revision, with wider intervals.

## Interpretation and limits

At the pinned revision the mean scaffold effect is +0.075 and the mean model effect is +0.053.
The scaffold effect for Qwen3.5-122B (+0.098) is larger than for Minimax-M2.5 (+0.052). The effect depends on the model-scaffold pair.
The adjustment removed 57 to 74 percent of the unadjusted disagreement.

The effect is an agreement gap on the eligible cases. Its population value under independent Bernoulli outcomes is in [section 6.1](../../spec/ari-e-bench-v0.1.md#61-population-value).
It does not identify a cause and does not say which condition is better.

About 23 percent of rows were ungraded and excluded. If grading failure depends on task difficulty, the eligible set is easier than the full set.
The upstream removal of trajectories with git-hacking behavior was a selection on trajectory content. It changed the eligible case set and the pass rates. The pinned revision is the population these numbers describe.
The dataset was not collected as a controlled experiment. Independence between rollouts is not established.
Tool names differ between harnesses, so this report compares outcomes and not action sequences.

## Archived numbers

The paper appendix (`app:arie`, `tab:arie`) carries the archived numbers. Their comparison with the rerun on the same data and with the pinned revision is in `docs/paper/CLAIM_TO_EVIDENCE.md`.
Every archived number reproduces. The archived effects and agreements are recovered to within 0.002 on the archived revision.
The paper table must cite one revision. The rows for the pinned revision are in [arie_table.tex](results/arie_table.tex); the rows for the archived revision are in [arie_table.ad4805a5aa7d.tex](results/arie_table.ad4805a5aa7d.tex).

## Superseded output of 2026-08-07

The files are not in the working tree. They are in git history at tag `v0.1-preview`, under `experiments/harness-effect/results/` and
`experiments/harness-power/results/`; `git show v0.1-preview:experiments/harness-effect/results/harness_effect.json` reads one.
The signed attestation there still verifies against its own inputs. The numbers below are kept for the record.
They do not describe the current estimator or the pinned revision.

| contrast | cases | self-consistency | cross agreement | ARI-E effect | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, model = Qwen3.5-122B | 10,788 | 0.874 | 0.785 | +0.089 | [+0.083, +0.094] |
| scaffold, model = Minimax-M2.5 | 12,195 | 0.899 | 0.849 | +0.050 | [+0.044, +0.052] |
| model, scaffold = SWE-agent | 10,674 | 0.880 | 0.839 | +0.041 | [+0.039, +0.046] |
| model, scaffold = OpenHands | 11,933 | 0.892 | 0.833 | +0.059 | [+0.054, +0.062] |

Three defects made a rerun necessary.
The estimator weighted cases by run-pair count and took intervals from a single run pairing.
The dataset revision was not recorded; it was identified afterwards from the attestation date and the upstream commit history.
The bound `trajectories.jsonl` was a 4,000-row-per-cell streaming sample with one rollout per case, so it could not reconstruct any contrast.

## Reproduce

Follow [section 8](../../spec/ari-e-bench-v0.1.md#8-reproduce). Compare the SHA-256 values in the manifests before comparing numbers.
`extract.py` reads the dataset layout used before 2026-08-21. It is retained for the superseded analysis only.
