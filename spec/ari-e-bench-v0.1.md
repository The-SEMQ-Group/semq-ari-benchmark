# ARI-E-Bench v0.1 — outcome-layer reproducibility for agent harnesses

Status: v0.1-preview.
This specification defines the ARI-E estimator, its eligibility rule, its uncertainty procedure, and the pinned trace dataset.
The reference implementation is [ari/harness.py](../ari/harness.py).
The measured results are in [experiments/harness-effect/RESULTS.md](../experiments/harness-effect/RESULTS.md).

ARI-E measures the agreement of binary outcomes between repeated runs of an agent.
It compares two conditions on the same cases. A condition is a harness or a model.
It subtracts the agreement inside each condition from the agreement across the two conditions.
The result is an agreement gap. It is not a success rate, and it is not a causal effect.

## 1. Inputs

### 1.1 Definitions

| Term | Definition |
| --- | --- |
| Case | One task instance. In the pinned dataset, one `instance_id`. |
| Condition | One harness or one model. The other factor stays fixed inside a contrast. |
| Run | One rollout of one condition on one case. In the pinned dataset, one `trajectory_id`. |
| Outcome | The binary verdict of a checker that the agent did not see. `true` is a pass. |
| Contrast | Two conditions compared on their shared eligible cases. |
| Graded run | A run with a recorded outcome. Runs with `resolved = -1` are ungraded. |

An outcome must come from a checker that is independent of the agent.
In the pinned dataset, the checker is the repository test suite that graded the agent's patch.
A verdict from a language-model judge measures the judge and is not a valid outcome.

### 1.2 Trajectory record

The reference implementation reads JSON Lines. One object per run:

```json
{"case_id": "owner__repo-123", "harness": "sweagent", "run": 0, "outcome": true, "tool_calls": ["bash", "str_replace_editor"]}
```

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `case_id` | string | yes | The case. |
| `harness` | string | yes | The condition label. |
| `run` | integer | yes | The run index inside `(harness, case_id)`. The numbering does not change any reported value. |
| `outcome` | boolean | yes | The verdict. |
| `tool_calls` | list of strings | no | Ordered action labels. Used only for trajectory agreement. |

`(harness, case_id, run)` must be unique.

### 1.3 Eligibility

A case is eligible for a contrast when both conditions have at least two graded runs on it.
One run gives no within-condition agreement, so the control cannot be measured.
A case with runs in only one condition is dropped from both sides.
Ungraded runs are removed before the eligibility count. Their number is reported for each condition.
The eligible case set is the same for every value reported in one contrast.

## 2. Estimator

For one eligible case $c$, condition $A$ has runs $a_1, \dots, a_m$ and condition $B$ has runs $b_1, \dots, b_n$, with $m, n \ge 2$.
$\mathbf{1}[x = y]$ is 1 when the two outcomes are equal and 0 otherwise.

Within-condition agreement uses every unordered pair of runs:

$$S_A(c) = \binom{m}{2}^{-1} \sum_{i<j} \mathbf{1}[a_i = a_j], \qquad S_B(c) = \binom{n}{2}^{-1} \sum_{i<j} \mathbf{1}[b_i = b_j].$$

Cross-condition agreement uses every pair of one run from each condition:

$$X(c) = \frac{1}{mn} \sum_{i=1}^{m} \sum_{j=1}^{n} \mathbf{1}[a_i = b_j].$$

The per-case gap is the mean of the two within-condition agreements minus the cross-condition agreement:

$$E(c) = \tfrac{1}{2}\left(S_A(c) + S_B(c)\right) - X(c).$$

The ARI-E effect for the contrast is the mean over the $N$ eligible cases:

$$E = \frac{1}{N} \sum_{c} E(c).$$

The reported self-consistency of $A$ is $S_A = \frac{1}{N}\sum_c S_A(c)$, and likewise $S_B$. The reported cross agreement is $X = \frac{1}{N}\sum_c X(c)$.
Because all three use the same case set, $E = \tfrac{1}{2}(S_A + S_B) - X$ holds exactly.

Weighting rules:

- Each eligible case has weight $1/N$. A case with more runs does not get more weight.
- Each condition has weight $1/2$ in the control. A condition with more runs does not get more weight.
- Every pair of runs inside a case has equal weight inside that case.

Under these rules, relabeling the runs of a case does not change any reported value.

## 3. Uncertainty

Intervals are percentile bootstrap intervals over cases.

1. Form the vector of per-case values, for example $E(c)$ for $c = 1, \dots, N$.
2. Draw $N$ cases with replacement. Compute the mean of the drawn values. Repeat $R$ times.
3. Report the 2.5th and 97.5th percentiles of the $R$ means as the 95% interval.

The random generator is `numpy.random.default_rng(seed)`. The reference implementation uses `seed = 0`.
$R$ is `n_resamples`. The default is 1000. The pinned run uses 2000.
The point estimate and the interval come from the same per-case vector, so the estimate lies inside its interval.
$S_A$, $S_B$, and $X$ get intervals from their own per-case vectors by the same procedure.

A contrast is marked `detected` when its interval for $E$ does not contain zero.
Runs within one case are not independent, so resampling runs would give an invalid interval. Only cases are resampled.

## 4. Reported values

A report for one contrast contains:

| Value | Definition |
| --- | --- |
| `n_cases` | $N$, the number of eligible cases. |
| `self_consistency` | $S_A$ and $S_B$, each with a 95% interval. |
| `cross_agreement` | $X$ with a 95% interval. |
| `effect`, `effect_ci` | $E$ with its 95% interval. |
| `detected` | Whether the interval for $E$ excludes zero. |
| `pass_rate` | The share of passing runs in each condition on the eligible cases. |
| `repeat_shapes` | The count of eligible cases for each $(m, n)$ pair of run counts. |
| `permutation_null` | The mean, spread, and detection count of $E$ after shuffling condition labels within each case. |
| ungraded counts | The number of ungraded runs dropped from each condition. |

Trajectory agreement over `tool_calls` is reported only when both conditions record actions under one label rule.
The two harnesses in the pinned dataset use different tool names, so v0.1 reports outcome agreement only.

A report states the dataset name, the dataset revision, the SHA-256 of the outcome table, the seed, `n_resamples`, and the version of the estimator code.
The paper table has the columns contrast, cases, self-consistency, cross agreement, $E$, and 95% interval.
In that table, self-consistency is $\tfrac{1}{2}(S_A + S_B)$.
The table rows are generated to [arie_table.tex](../experiments/harness-effect/results/arie_table.tex).

## 5. Frozen dataset reference

| Field | Value |
| --- | --- |
| Dataset | [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) |
| Revision | `f967cba3312573981a47fd7a7b80029b53909b5f` |
| Revision date | 2026-09-15 |
| Upstream version | v1.0 |
| Files | `data/{sweagent,openhands}/{qwen35_122b,minimax_m25}/swe-rebench-v2/*.parquet`, 63 files |
| Columns read | `instance_id`, `trajectory_id`, `resolved` |
| Rows | 151,219, of which 35,336 are ungraded |
| Outcome table | `outcomes.f967cba33125.csv.gz`, written by `fetch_outcomes.py` and verified against its committed [manifest](../experiments/harness-effect/results/outcomes.f967cba33125.manifest.json) |
| License | CC BY 4.0 |

The mapping to this specification is: `instance_id` is the case, `trajectory_id` is the run, `resolved = 1` is a pass, `resolved = 0` is a fail, `resolved = -1` is ungraded.
The harness is the first path element under `data/`. The model is the second.
The four contrasts are the two scaffold contrasts, one for each model, and the two model contrasts, one for each scaffold.

The revision is a Hugging Face commit hash. A branch name is not a pin.
The upstream repository removed 56,270 trajectories from v1.0 on 2026-08-26.
The revision current before that change, `ad4805a5aa7de70d99cab0bb8f99b15304c76de0`, had 207,489 rows in the four cells and a different file layout.
Results from different revisions are not comparable. Compare the revision before comparing numbers.

[fetch_outcomes.py](../experiments/harness-effect/fetch_outcomes.py) reads only the three columns from the Parquet files at the pinned revision.
It records the file list, the LFS digests, the bytes read, and the SHA-256 of the output table in the manifest.
[run.py](../experiments/harness-effect/run.py) refuses an outcome table that does not match its manifest.

## 6. Limitations

### 6.1 Population value

Assume the runs of a case are independent Bernoulli draws with pass probabilities $p_A$ and $p_B$.
Then $\mathbb{E}[S_A(c)] = p_A^2 + (1-p_A)^2$, $\mathbb{E}[S_B(c)] = p_B^2 + (1-p_B)^2$, and $\mathbb{E}[X(c)] = p_A p_B + (1-p_A)(1-p_B)$.
The population value of the per-case gap is

$$\mathbb{E}[E(c)] = (p_A - p_B)^2.$$

The population effect is the mean of $(p_A - p_B)^2$ over cases.
It is non-negative. It is a squared quantity, so a pass-probability difference of 0.10 on every case gives an effect of 0.01.
It does not show which condition passes more often.
Two conditions with equal pass probability on every case have a population effect of zero, even when their behavior differs in other ways.

### 6.2 Finite samples

$E(c)$ is unbiased for $(p_A - p_B)^2$ under the assumption in 6.1, but a single $E(c)$ can be negative.
With two runs per side, $S_A(c)$ and $S_B(c)$ take only the values 0 and 1.
The mean $E$ can be negative when the true value is near zero.
A negative estimate is a sampling result, not a negative effect.

The percentile bootstrap interval is anti-conservative at small case counts.
The [power simulation](../experiments/harness-power/RESULTS.md) reports the false-positive rate at zero true gap for each case count and repeat pattern.
Read a detection from a small case set against that table.

### 6.3 Unequal repeat counts

The two conditions can have different numbers of runs on a case, and the count can differ between cases.
The estimator accepts this. The control still weights the two conditions equally, and each case still has equal weight.
The variance of $E(c)$ is larger for cases with fewer runs, and the bootstrap interval is conditional on the recorded repeat counts.
Reports must include `repeat_shapes`. The pinned run also reports the effect on the cases with equal counts.

### 6.4 Ungraded runs

Ungraded runs are dropped. The share is about 23 percent in the pinned dataset.
If grading fails more often on hard cases, the eligible set is easier than the full set and the effect is measured on that easier set.
Reports must state the ungraded count for each condition.

### 6.5 Dependence and design

The estimator assumes that the runs inside a condition are exchangeable.
The pinned dataset was collected for model training, not as a controlled experiment.
Independence between rollouts is not established. Shared caches, shared time windows, or retries can make runs more similar than independent draws.
The dataset does not record the order or the time of rollouts.

### 6.6 Interpretation

The effect describes outcome agreement on the eligible cases. It does not identify a cause.
An interval that contains zero does not establish that two conditions are interchangeable.
An effect of zero does not mean identical patches or identical trajectories.
The effect is not a quality score. It does not say which condition is better.

## 7. Calibration checks

Two checks accompany the pinned run.

- Permutation null. For each case, the condition labels of the pooled runs are shuffled, keeping the per-side counts. The effect is recomputed. Under exchangeability the expected value is zero. The pinned run reports 50 permutations for each contrast.
- Split-half. Inside one condition, the runs of cases with at least four graded runs are split at random into two halves of at least two runs each. The effect between the halves has expected value zero. The pinned dataset has at most three rollouts per cell, so this check has no cases there. It stays in the protocol for datasets with more repeats.

The simulated false-positive rate at zero gap is in the [power simulation](../experiments/harness-power/RESULTS.md).

## 8. Reproduce

From the repository root, with `numpy`, `pyarrow`, and `huggingface_hub` installed:

```bash
python experiments/harness-effect/fetch_outcomes.py --revision f967cba3312573981a47fd7a7b80029b53909b5f
python experiments/harness-effect/run.py
```

The first command reads about 17 MB from the Hub and writes the outcome table and its manifest.
The second command reads the table, checks its SHA-256 against the manifest, and writes `harness_effect.json`, `harness_effect.manifest.json`, and `arie_table.tex` under `experiments/harness-effect/results/`.
Compare the output SHA-256 in the manifest before comparing numbers.

## 9. Future versions

A change to the estimator, the eligibility rule, the bootstrap procedure, or the dataset revision requires a new version.
A cross-harness trajectory agreement needs a shared action label rule. That rule is not defined in v0.1.
A dataset with four or more rollouts per cell would make the split-half check informative.
