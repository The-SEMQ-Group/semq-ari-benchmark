# Engineering status, August 2026

For an engineer joining this work. It covers what is open, what was measured, what was
retracted, and what will break if you are not warned.

Read [key-results.md](key-results.md) first if you want the charts. This document is the
detail behind them.

---

## 1. The frame

ARI was one index. It measured embedding reproducibility and was named for agent
reproducibility, which is a wider claim than the measurement supports. The current
proposal splits it into three
([docs/proposals/ari-decomposition.md](proposals/ari-decomposition.md)):

| index | question | status |
| --- | --- | --- |
| **ARI-R** | Does the same input give the same internal representation? | Measured on CPU and GPU |
| **ARI-D** | Given the same context, does the same token come out? | Built this cycle. Measured on CPU and GPU |
| **ARI-E** | Holding the model fixed, how much does the harness decide the outcome? | Measured on SWE-agent and OpenHands, ~11,000 cases per contrast |

The proposal is not adopted. It changes no specification yet.

---

## 2. Open pull requests

| repo | PR | title | note |
| --- | --- | --- | --- |
| semq-ari-benchmark | **#9** | regime discrimination (ARI-R) and a working ARI-D probe | **Stacked on #8.** 36 files, 13 commits. Most of this cycle's work |
| semq-ari-benchmark | **#8** | retract the probe-necessity claim in section 3.1 | Merge this first |
| semq-ari-benchmark | #7 | complete conversion to plain english | Predates this work. Conflicted with #8 and #9 on the since-removed whitepaper |
| semq | **#19** | cross-arch invariance of SEMQ vs a frozen-codebook VQ probe | Carries the test that #8 cites |
| semq | #20 | remove the experimental KV-cache module (semq.kv) | Safe to merge. ARI-E no longer depends on it. See section 7 |
| semq-research | #9 | probe-purity reframe as seed-stability | Companion to #8 |
| semq-research | #8 | FFN activation compression | Unrelated to this cycle |

Merge order that works: semq #19, then benchmark #8, then benchmark #9. PR #7 needs a
rebase against whichever of #8 and #9 lands first.

---

## 3. ARI-R, representation

**Question.** Do retrieval metrics detect a serving change, or only SEMQ? The
`gpu-determinism` write-up asserted twice that these events are "invisible to
cosine/retrieval" and never measured a retrieval metric. This supplies that half.

**Setup.** BEIR SciFact, 5,183 documents, 300 queries with real relevance judgements,
`all-MiniLM-L6-v2`. Reference is fp32 on CPU. CIs are paired bootstrap over queries,
10,000 resamples.

| condition | host | R@10 | ΔR@10 | 95% CI | top-10 lists same | SEMQ HER |
| --- | --- | ---: | ---: | :--- | ---: | ---: |
| reference | both | 0.7833 | - | - | 100.00% | 1.0000 |
| new process | CPU | 0.7833 | +0.0000 | [0.0000, 0.0000] | 100.00% | 1.0000 |
| 1 thread | CPU | 0.7833 | +0.0000 | [0.0000, 0.0000] | 100.00% | 1.0000 |
| batch 8 / 128 | CPU | 0.7833 | +0.0000 | [0.0000, 0.0000] | 100.00% | 1.0000 |
| bf16 | CPU | 0.7867 | +0.0033 | [+0.0000, +0.0100] | 45.00% | 0.0000 |
| int8 | CPU | 0.7772 | −0.0061 | [−0.0278, +0.0156] | 0.00% | 0.0000 |
| fp16 | GPU | 0.7833 | +0.0000 | [0.0000, 0.0000] | 92.67% | 0.0000 |
| **TF32 off** | GPU | 0.7833 | +0.0000 | [0.0000, 0.0000] | 100.00% | **0.9992** |
| **TF32 on** | GPU | 0.7833 | **+0.0000** | **[0.0000, 0.0000]** | 96.33% | **0.5267** |
| bf16 | GPU | 0.7800 | −0.0033 | [−0.0100, +0.0000] | 43.00% | 0.0002 |

**The TF32 pair is the headline.** A model audited at fp32 on a CPU and served at fp32 on
a GPU gets a different code for 47% of documents, and Recall@10 does not move at all. The
TF32-off row reading 0.9992 against the same reference is the control. It attributes the
effect to TF32 rather than to the GPU.

**One claim needed narrowing.** "Invisible to retrieval" holds for quality metrics and
fails for result lists. int8 changes the top-10 list for every query. The honest advantage
is that HER needs no stored per-query reference and no query set at all.

Data: `experiments/regime-discrimination/results/regime_matrix.json`. Each row carries a
`host` field, because the CPU and GPU runs share the fp32 CPU reference.

---

## 4. ARI-D, decoding

**Question.** Does a serving change move the logits before it moves the tokens? If yes, a
SEMQ trace over logits gives a signal that comparing output cannot.

**Method.** Two arms. *Teacher-forced* scores every condition on the reference token
sequence in one forward pass, so position `i` sees an identical context in every
condition. *Free-running* lets each condition generate on its own. Greedy throughout, so
every difference comes from infrastructure and not from sampling.

### TinyLlama-1.1B, CPU, 539 steps

| condition | token agree | H̄ | early warning | margin flipped | margin survived | free exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| controls | 100.00% | 0.0000 | 0.00% | - | 1.706 | 100.0% |
| bf16 | 99.07% | 0.0343 | 99.07% | **0.043** | **1.757** | 75.0% |
| int8 | 74.21% | 0.8614 | 74.21% | 0.588 | 2.681 | 0.0% |

The margin columns confirm the mechanism directly. Tokens that flipped sat at a 41 times
smaller top-2 margin than tokens that survived, so flips concentrate where the decision
was already near a tie while the distribution moved everywhere.

### Qwen2.5-3B-Instruct, L40S, 536 steps

This run exercises the chunked vocabulary path. 151,936 tokens do not fit one context.

| condition | token agree | HER | H̄ | early warning | free exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| new process, 1 thread | 100.00% | 1.0000 | 0.00000 | 0.00% | 100.0% |
| **TF32 off** | 100.00% | **1.0000** | 0.00000 | 0.00% | 100.0% |
| **TF32 on** | **100.00%** | 0.0000 | 0.00326 | **100.00%** | **100.0%** |
| **batched** | 100.00% | **0.4683** | 0.00001 | **53.17%** | 100.0% |
| fp16 | 100.00% | 0.0000 | 0.00739 | 100.00% | 91.7% |
| bf16 | 99.25% | 0.0000 | 0.05900 | 99.25% | 75.0% |

Three things to take from this table:

1. **TF32 is the cleanest early-warning case anywhere in this work.** Every logit code
   changes, all 12 completions come out character-identical, and the TF32-off control
   reads 1.0000.
2. **Batching is not a null result.** The TinyLlama run reported 0% and its write-up said
   the test was probably too small to show the effect. That reading was right. On a real
   3B model, padded batching changes 53.17% of steps while output stays identical.
3. **fp16 compounds.** Teacher-forced token agreement is 100%, yet one completion in
   twelve still diverged when generating freely.

Two design facts came out of this work:

- **H̄ is the ARI-D metric, not HER.** HER is exact match over a 32,000-dimension code and
  saturates at 0.0000 for bf16 and int8 alike. H̄ separates them 0.0343 against 0.8614.
- **`Context` caps `max_dim` at 65,536.** TinyLlama fits. Llama-3, Qwen2.5 and Gemma do
  not. `semq_codes()` chunks a longer vocabulary and passes one scale to every chunk.
  Calibrating each chunk separately would make the concatenated code meaningless.

Data: `results/decoding_matrix.json` (TinyLlama) and
`results/decoding_matrix_qwen3b_gpu.json` (Qwen). They are separate files on purpose.
Different model, and one uses chunks.

---

## 5. Does SEMQ beat the cheap alternatives?

This is the most important section for anyone deciding what to build on.

**Mostly no.** Full detail in
[experiments/decoding-reproducibility/BASELINES.md](../experiments/decoding-reproducibility/BASELINES.md).

| statistic | response at σ=1e-3 | bytes of reference state per step | bit-reproducible |
| --- | ---: | ---: | :---: |
| SEMQ H̄ | 1.33e-3 | 16,000 | yes |
| **top-2 margin** | **1.15e-3** | **8** | **yes** |
| KL | 2.18e-7 | 128,000 | no, 4.6% drift |
| JS | 5.39e-8 | 128,000 | no, 363% drift |

- **KL is out.** It is second order in the perturbation. At σ=1e-4 it reads −5.1e-10,
  which is zero with the wrong sign. It is also not reproducible, so a published KL value
  is not a number a third party can recompute.
- **The top-2 margin wins on cost.** It gets 86% of the signal for 1/2000 of the storage
  and is equally reproducible. For detecting a precision change, ship the margin.

**What survives is coverage, not sensitivity.** Under noise applied only below rank 20,
SEMQ does not move (0.1289 against 0.1288 for a full perturbation) while the margin and
the token statistic read exactly zero. They are structurally blind to the tail.

Whether that matters is **open**. A precision change is close to isotropic, and the margin
catches it. A token-suppression filter, an adapter, or a tokenizer change need not be. No
one has shown that a real serving change produces tail-confined drift.

Recommended posture is both. Margin as an 8-byte tripwire, SEMQ for coverage and for
attestation.

---

## 6. What was retracted

Do not re-derive these. They are settled and the reasoning is recorded.

1. **"SEMQ is necessary for ARI".** Retracted twice ([ledger](retractions.md)). The first version
   read a seed-stability result as a class result. The second claimed a frozen vector
   quantizer is "not verifiable" because near-ties let two algebraically identical
   distance formulas disagree, citing 39% of symbols. Those numbers came from a
   constructed codebook with centroids 1e-6 apart. On real k-means codebooks,
   `encode(reconstruct(c)) = c` holds at every configuration swept, including one at 91%
   near-tie density. Two centroids that close do not survive a k-means fit. What remains
   is an argument about verification cost.
   See [experiments/probe-verifiability/RESULTS.md](../experiments/probe-verifiability/RESULTS.md).
2. **The "1.6% of assignments near a tie at a typical configuration" figure.** Synthetic,
   and it described an atypical split. The standard configuration gives 0.045%.
3. **"N times more sensitive than retrieval."** Retracted earlier, recorded in the ledger
   section 6.

One number is still weak and is flagged in the retraction ledger. The ~14x LSH sensitivity
gap comes from one synthetic run and has not been replicated.

---

## 7. Known problems

**`semq.kv` is out, and ARI-E no longer needs it.** PR semq#20 removes the module, and
`qwen-harness-trace` imported it at `checkpoint_store.py:28`. That coupling is gone.
`ari/harness.py` never touched semq at all, and it now reads runs from a JSON Lines file:

```json
{"case_id": "go/gin-1", "harness": "octomind", "run": 0,
 "outcome": true, "tool_calls": ["read", "edit", "test"]}
```

Any runner that emits those fields can feed the metric. KV snapshots were added to support
replay from a decision point, and they never supported it: the format is lossy and a
restored cache diverges within a few tokens. `qwen-harness-trace` is not deleted, but
nothing in ARI-E depends on it now. Treat it as unused until someone wants the model
server again.

**The published SDK and the experiments had drifted.** CodeArtifact ships `semq 1.4.1`,
which exposes `SEMQ_OP_QBIN` and `qbin_n_bins`. Local builds renamed these to
`SEMQ_OP_QUANT` and `quant_n_bins`. The experiments were written against a local build and
raised `ImportError` on the instance after the encoding had already finished.
`ari/semq_compat.py` now resolves either name to the same operator. **Test against the
published wheel, not your working copy.**

**`g5.xlarge` cannot hold a 3B model in fp32.** `from_pretrained` materializes the whole
model in CPU RAM before any `.to(device)`. That is 12.4 GiB against 16 GiB of host RAM,
and the box thrashed until SSH stopped answering. The fix passes `device_map` on a GPU
host. Use `g6e.xlarge` for 3B and above. It has 32 GiB of RAM and an L40S with 45.7 GB.

**A GPU host is the wrong place to measure a CPU precision condition.** The instance CPU
has AVX2 and no AVX-512, so torch runs bf16 through an emulation path. One condition took
over 13 minutes and measured the instance. `ARI_R_CONDITIONS` restricts a run to a named
set.

**The write-ups do not yet carry the GPU numbers.** `key-results.md` and the three
`RESULTS.md` files describe the CPU runs only. The GPU results are in the JSON and in this
document. Figure 1 has no TF32 columns.

---

## 7b. What `qwen-harness-trace` is

A private companion repository, about 3,500 lines of Python plus two forked
agents. **Nothing in ARI depends on it any more.** This section exists so that a
colleague without access knows what it holds and why it is set aside.

**Why it was built.** An article about a long autonomous Qwen coding run drew a comment
that a long run's meaning depends on what checked the work and what made it stop, not on
duration. The commenter cited `octobench`: two harnesses on the *same model and same
endpoint* scored 24/25 against 19/25. This repository was the attempt to reproduce that
comparison under controlled conditions.

**What is in it.**

| part | what it does |
| --- | --- |
| `server.py` | An OpenAI-compatible endpoint over a local `transformers` model. Both harnesses hit it unmodified. It diffs the prompt prefix per session to reuse the KV cache, because the harnesses resend the whole history every call |
| `model_runtime.py` | Session-keyed generation and cache handling across three `transformers` cache layouts |
| `case_runner.py` | Runs one task: `setup.sh`, then the agent, then `quality.sh` and `validate.sh`. Each case gets its own virtualenv, because a task's `pip install -e .` once corrupted the runner's interpreter |
| `checkpoint_store.py` | Snapshots state at each agent decision, content-addresses it, signs it, and chains the manifests so the *sequence* is tamper-evident and not only each artifact |
| `forks/oh-my-cli`, `forks/octomind` | The two agents, patched with a sink that posts a checkpoint at each decision boundary. One is TypeScript, the other Rust |
| `cases/` | 48 tasks with `case.yaml`, `setup.sh`, `quality.sh`, `validate.sh`. `validate.sh` fetches the gold test from upstream at verify time, so the agent never sees it |
| `verifier/verify.py` | Standalone checker. Stdlib and `cryptography` only, no SEMQ import, so a third party can confirm the artifacts without trusting our code |

**Why it is set aside.** The checkpoints stored transformer KV caches, to support reloading
a run at a decision point and continuing it under the other harness. That never worked. The
format is lossy, the bin count caps at 64, and a restored cache diverges from the original
within about ten greedy tokens. `semq.kv` is being removed from the SDK for the same reason
(PR semq#20).

**What survives.** ARI-E needs a verdict and an ordered list of actions. Neither needs model
state. The metric now reads JSON Lines, so the case corpus, the graders and the two patched
agents are still reusable if someone wants to produce that file. The model server and the
checkpoint chain are not needed for it.

**Not deleted.** The repository is local-only with no remote. Treat it as parked.

---

## 8. Infrastructure

Everything lives in [`infra/`](../infra/README.md). AWS account `127348475353`, region
`us-east-2`, profile `ilona-admin-access`.

| script | does |
| --- | --- |
| `budget.sh` | Monthly budget and e-mail alerts. Run once |
| `launch.sh` | Provisions the instance. Has `--dry-run` |
| `bootstrap.sh` | User-data. Arms the shutdown timer first, then installs |
| `run_gpu.sh` | Runs both experiments on the instance |
| `fetch_results.sh` | Pulls JSON back. Skips the multi-gigabyte caches |

Three cost guards, in decreasing order of how much to trust them: a `shutdown -h` armed in
user-data before anything that can hang, `instance-initiated-shutdown-behavior=terminate`,
and a budget alarm. AWS cost data lags by hours, so the alarm is a backstop and not
protection. The G-instance quota on this account is 4 vCPUs, which is exactly one
`g5.xlarge` or one `g6e.xlarge`.

Two GPU sessions cost about **$3.20** in total.

`bootstrap.sh` clones over HTTPS and fails on a private repository. Use `rsync` instead,
and install `semq` from a CodeArtifact wheel. The README has both commands.

---

## 9. ARI-E, and what to do next

**ARI-E now has a result from real harnesses.** `experiments/harness-effect/` runs the
metric over `nvidia/Open-SWE-Traces`, which crossed two scaffolds that nobody here built,
SWE-agent and OpenHands, with two models over about 18,000 SWE-bench-style instances at up
to three rollouts each.

| contrast | cases | self-consistency | cross agreement | ARI-E effect | 95% CI |
| --- | ---: | ---: | ---: | ---: | :--- |
| scaffold, Qwen3.5-122B | 10,788 | 0.874 | 0.785 | **+0.089** | [+0.083, +0.094] |
| scaffold, Minimax-M2.5 | 12,195 | 0.899 | 0.849 | **+0.050** | [+0.044, +0.052] |
| model, SWE-agent | 10,674 | 0.880 | 0.839 | **+0.041** | [+0.039, +0.046] |
| model, OpenHands | 11,933 | 0.892 | 0.833 | **+0.059** | [+0.054, +0.062] |

**The scaffold decides more than the model, by a factor of 1.4.** Every interval excludes
zero.

**The control is worth more than the comparison.** Agents disagree with themselves 10 to
13% of the time. A report that diffs two scaffolds and stops would say the scaffold changed
21.5% of outcomes on Qwen. The attributable figure is 8.9%, so the naive number is 2.4
times too large. Across the four contrasts the control removes 59 to 74% of the apparent
effect.

The largest threat to these numbers is that about 22% of rows carry `resolved = -1` and
were dropped. An ungraded run is not a failed one, but if grading failure tracks task
difficulty then the surviving set is easier than the whole.

**What has been measured is whether such a run could work at all**, and the answer is
mostly no at the sample size that was planned. `experiments/harness-power/` simulates runs
with a known harness effect and counts how often the interval excludes zero:

| pass-rate gap | true effect | repeats | 25 cases | 100 cases | 200 cases |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.0000 | 2 | 8% | 6% | 6% |
| 0.20 | 0.0400 | 2 | **8%** | 11% | 37% |
| 0.20 | 0.0400 | 5 | 23% | 74% | 96% |
| 0.40 | 0.1599 | 2 | **39%** | 94% | 100% |
| 0.40 | 0.1599 | 5 | 99% | 100% | 100% |

Three things follow.

**The planned run was underpowered.** octobench has 25 cases. At two repeats it detects a
20-point pass-rate gap 8% of the time. The octobench datapoint that motivated this work,
24/25 against 19/25, is a 20-point gap. This design would miss it in more than nine runs
out of ten.

**Repeats buy more than cases.** At 25 cases, going from 2 repeats to 5 lifts power on a
40-point gap from 39% to 99%. Repeats sharpen the self-consistency control, and the control
is what the effect is measured against.

**The effect is a much smaller number than the pass-rate difference.** A 40-point gap in
pass probability produces a true harness effect of 0.16, and a 20-point gap produces 0.04.
The effect is a difference between two agreement rates, and that compresses. Do not expect
ARI-E numbers to look like benchmark score gaps.

One caution: at 10 cases the false-positive rate reaches 18% rather than 5%. The bootstrap
is anti-conservative on very few cases. Do not run ARI-E below about 50.

The design point worth preserving: **cross-harness disagreement means nothing without a
self-consistency control.** Agents are stochastic, so the metric measures how often one
harness disagrees with itself and subtracts that:

```
E(h1, h2) = mean self-consistency - cross-harness agreement
```

When a harness has one run per case, self-consistency cannot be measured. The code refuses
to compute an effect and records why, rather than assuming 1.0. Assuming it would charge
all agent stochasticity to the harness, which is the error the metric exists to avoid.

Next steps, in the order I would take them:

1. **Produce a trajectory file.** ARI-E reads JSON Lines and needs nothing else. The
   parked repository already has 48 graded tasks and two patched agents that can emit it,
   and a fresh runner would also work.
2. **Run ARI-E at a size that can answer the question.** At least 50 cases and 5 repeats,
   and 100 cases if the expected gap is near 20 points. octobench's 25 cases at 2 repeats
   will not do.
3. **Update the write-ups with the GPU numbers.** `key-results.md` and the three
   `RESULTS.md` files still describe the CPU runs only, and figure 1 has no TF32 column.
4. **Re-derive the 14x LSH figure** before it goes in front of anyone.
5. **Answer whether real drift is tail-confined.** This decides whether SEMQ's coverage
   argument is commercially real or only true.

`semq.kv` is settled. PR semq#20 can merge whenever its author wants it to.
