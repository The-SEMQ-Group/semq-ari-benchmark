# ARI-E on real harnesses: does the scaffold or the model decide the outcome?

**Status: run.** Script: [`run.py`](run.py). Machine-readable:
[`results/harness_effect.json`](results/harness_effect.json).

**Headline: the scaffold decides more of the outcome than the model does, and a naive
comparison overstates that by two to four times.** Removing the agent's own noise cuts the
apparent scaffold effect from 21.5 points to 8.9.

This is the first ARI-E measurement, and it uses harnesses nobody here built.

## Setup

[`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) ran two
agent scaffolds, **SWE-agent** and **OpenHands**, with two models, **Qwen3.5-122B** and
**Minimax-M2.5**, over about 18,000 SWE-bench-style instances. Each cell holds up to three
rollouts of each instance. That crossing is what makes this possible:

| contrast | what changes | what stays fixed |
| --- | --- | --- |
| **harness effect** | the scaffold | the model, the task, the grader |
| **model effect** | the model | the scaffold, the task, the grader |

Both use the same estimator on the same instances, so they can sit side by side.

An outcome is `resolved == 1`, which comes from running the repository's own tests against
the agent's patch. The agent never saw that test. A benchmark whose verdict came from a
judge would measure the judge.

Only instances with at least two rollouts on both sides are used, because one rollout
gives no self-consistency control.

## Results

| contrast | cases | self-consistency | cross agreement | **ARI-E effect** | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, model = Qwen3.5-122B | 10,788 | 0.874 | 0.785 | **+0.089** | [+0.083, +0.094] |
| scaffold, model = Minimax-M2.5 | 12,195 | 0.899 | 0.849 | **+0.050** | [+0.044, +0.052] |
| model, scaffold = SWE-agent | 10,674 | 0.880 | 0.839 | **+0.041** | [+0.039, +0.046] |
| model, scaffold = OpenHands | 11,933 | 0.892 | 0.833 | **+0.059** | [+0.054, +0.062] |

**Mean scaffold effect +0.069. Mean model effect +0.050.** Every interval excludes zero.

## Reading

**1. Agents disagree with themselves 10 to 13 percent of the time.** Self-consistency runs
between 0.873 and 0.907. Two rollouts of one scaffold, one model and one instance reach
different verdicts about one time in nine. Any comparison that ignores this attributes that
noise to whatever it happens to be comparing.

**2. The control removes most of the apparent effect.** This is the result that justifies
the metric:

| contrast | apparent disagreement | agent's own noise | ARI-E effect | share removed |
| --- | ---: | ---: | ---: | ---: |
| scaffold, Qwen | 0.215 | 0.126 | +0.089 | **59%** |
| scaffold, Minimax | 0.151 | 0.101 | +0.050 | **67%** |
| model, SWE-agent | 0.161 | 0.120 | +0.041 | **74%** |
| model, OpenHands | 0.167 | 0.108 | +0.059 | **65%** |

A report that diffs two scaffolds and stops would say the scaffold changed 21.5% of
outcomes on Qwen. The attributable figure is 8.9%. **The naive number is 2.4 times too
large**, and on the model contrasts it is nearly 4 times too large.

**3. The scaffold matters more than the model, but not overwhelmingly.** The mean scaffold
effect is 1.4 times the mean model effect. That supports the direction of the published
claim without supporting its strongest form. On this dataset the two are the same order of
magnitude.

**4. Scaffold and model interact.** The scaffold effect is +0.089 with Qwen3.5 and +0.050
with Minimax. Pass rates show why: Qwen3.5 scores 46.8% under SWE-agent and 33.4% under
OpenHands, a 13.5-point spread, while Minimax scores 46.9% and 42.1%, a 4.9-point spread.
One model is far more sensitive to the scaffold than the other. Reporting a single
"harness effect" for a benchmark, with no model named, would hide this.

**5. A pass-rate gap is not the effect.** Qwen under the two scaffolds differs by 13.5
points of pass rate but produces an ARI-E effect of 8.9 points. The effect is a difference
of agreement rates, and that compresses. Do not read ARI-E numbers as score gaps.

## Why this design was chosen

An earlier plan ran two agents against a self-hosted model on 25 tasks. The power analysis
in [`../harness-power/`](../harness-power/) showed that design detects a 20-point pass-rate
gap 8% of the time. This dataset supplies about 11,000 cases per contrast instead of 25,
which is why every interval here is narrow.

It also answers a question the earlier plan could not. If we write both harnesses, the
measured effect is a property of our design choices. SWE-agent and OpenHands are widely
used and were built by other people.

## Caveats

- **Three rollouts per cell is the dataset's limit.** More repeats sharpen the control.
  The intervals are narrow because there are many cases, not because the control is
  strong.
- **Roughly 22% of rows carry `resolved = -1`** and were dropped, because an ungraded run
  is not a failed one. If grading failure correlates with task difficulty, the surviving
  set is easier than the whole. This is the largest threat to the numbers above.
- **The dataset was built to train models, not to test scaffolds.** The rollouts were not
  produced as a controlled experiment, and nothing guarantees the three rollouts of an
  instance are independent draws.
- **Outcome agreement only.** The two scaffolds do not share a tool vocabulary: SWE-agent
  calls `bash`, `str_replace_editor` and `submit`, while OpenHands calls `execute_bash`,
  `str_replace_editor`, `think`, `finish` and `fetch`. Trajectory agreement across them
  needs a mapping between those names first, and `think` has no SWE-agent counterpart at
  all, so a mapping would hide a real capability difference.
- **Two scaffolds and two models.** The interaction in point 4 says the numbers move with
  the pair, so treat these as two measurements rather than a constant.
