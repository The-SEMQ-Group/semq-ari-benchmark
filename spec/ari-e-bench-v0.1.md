# ARI-E-Bench v0.1 — environment/harness reproducibility for coding agents

**Status:** DRAFT, not adopted · September 2026 · productizes the outcome-layer
measurement of the ARI paper (Appendix C, "Outcome-layer reproducibility (ARI-E)")
into gated, attested leaderboard rows. v0.1 measures **outcome agreement only** — the
same quantity and estimator the paper reports, `Δ = D_cross − D_self` — so it runs on
any model, including hosted APIs, and needs no self-hosted infrastructure. The
KV-cache checkpoint attribution from the original decomposition proposal is a **v0.2
extension** (§10), deliberately outside v0.1. This document changes no code by itself.

**Summary.** The ARI paper measures three layers; the environment layer (ARI-E) is
the least developed of them — a self-consistency-controlled agreement gap over agent
*outcomes* on a public dataset (`nvidia/Open-SWE-Traces`), reported **without
quantization** and marked **descriptive, not causal**. That reading is real but
ungated: it reuses rollouts we did not generate, so it cannot be attested, and it
carries a ~22% ungraded-drop bias. This spec turns that measurement into leaderboard
rows that pass the same scorer, pin and attestation gate as ARI-R and ARI-D: a frozen
task set, a controlled run we grade ourselves, and signed reports a third party
re-checks. **It changes the provenance and governance of the paper's ARI-E, not its
metric.**

---

## 1. Why this layer, why now

ARI is named for *agent* reproducibility, and the environment layer is where the
public evidence says most of the variance lives: holding the model and the endpoint
fixed, the scaffold around the model — the harness — moves outcomes as much as the
model choice does. Two preprints in one quarter name the problem ("Stop Comparing LLM
Agents Without Disclosing the Harness"; "Harness-Bench"), a standardized-evaluation
effort (HAL) spent ~$40k of compute to separate models from scaffolds, and the
datapoint that started the thread — `octobench`, 24/25 vs 19/25 on the *same model and
endpoint* under two harnesses — is a reproducibility failure, not a capability one.

The paper established that this is measurable: on `nvidia/Open-SWE-Traces`, the
agreement gap between two scaffolds runs to `Δ = +0.089` on one model, net of the
agent's own self-inconsistency. What the paper did **not** do is put that behind the
board's gate — pinned inputs, a grader we run, a signed report. This bench does that.
It takes no side on which harness is "better": it reports the *size of an agreement
gap*, not a verdict on any project.

## 2. What v0.1 adds to the paper's Appendix C

The paper's ARI-E is one reading on a dataset built to train models, not to test
scaffolds. v0.1 changes three things about it, and **none is the metric**:

1. **A frozen task set and a grader we operate.** The public rollouts gave no
   guarantee that repeats of an instance were independent draws, and ~22% of runs
   carried `resolved = -1` and were dropped (the single largest threat to those
   numbers). A controlled run fixes both: independent draws by construction (fresh
   seed/process per rollout), and a grader we operate, so an ungraded run is *re-run*
   rather than silently dropped.
2. **Gated, attested rows.** The reading becomes a report that passes the scorer,
   pins the frozen task set by content hash, and carries an Ed25519 signature a third
   party re-checks — the same gate ARI-R and ARI-D rows already pass. The paper's
   public reading cannot be attested to a protocol we did not run; a v0.1 row can.
3. **Any model, including hosted APIs.** Outcome agreement needs only the two
   harnesses driving one model endpoint — no self-hosting. The paper measured
   Qwen3.5-122B and Minimax-M2.5; a hosted-API model (a GPT, Claude, or hosted-Qwen
   endpoint) reads on the same axis. The model and endpoint are held fixed across the
   two harnesses; only the scaffold changes.

The paper is explicit that ARI-E is measured **without quantization**: the outcome is
a plain held-out test verdict, pass or fail, and v0.1 keeps it that way — no probe,
no code. The one place SEMQ's symbolic machinery re-enters the environment layer,
KV-cache checkpoint attribution, is a **v0.2 extension (§10)**, deliberately outside
v0.1.

## 3. Subjects — what a row is

A row is a **contrast**: `(model fixed, harness_A vs harness_B)` measured on the
frozen task set. The harness set in v0.1 is a pair, so a row is exactly one contrast,
exactly as the paper's Table 4 reports it. The model is named on every row and is not
optional: the paper found the gap *interacts* with the model (Qwen3.5-122B `+0.089`
vs Minimax-M2.5 `+0.050` for the same scaffold pair), so a single number with no model
named would hide the one thing it is conditioned on. A row therefore reads as "the
scaffold-pair agreement gap for this model on these tasks," never "the harness
effect."

A richer shape — a row = a single `(model, harness)` config with its own
self-consistency, the gap derived pairwise across configs — is deferred to v0.2. It
needs ≥3 configs to be worth the extra machinery, and the v0.1 pair does not.

## 4. The frozen task set

`ARI-E-Bench-v0.1`: a frozen 200-task slice of **SWE-bench Verified**, pinned by hash
exactly as ARI-Bench and ARI-D-Bench are — SHA-256 over the ordered records
(`content_hash = d44f4975…a91b26`, over `input_id\0text\n`), every prefix hash valid so
a reduced run can prove it measured the first N (`prefix:N`, marked not comparable). The
slice is repo-capped (≤25 instances per repo, so django's 231 do not dominate: 12 repos,
none above 25), which also keeps any single repo's slow test setup from dominating the
run; selection bias is harmless here (see below). Each instance carries its own held-out
test suite (FAIL_TO_PASS / PASS_TO_PASS) as the grader, pinned by the dataset revision
`SWE-bench/SWE-bench_Verified@78f471bf…` rather than restated in the file; the agent never sees that test, so the
outcome is a verdict the agent cannot game — a benchmark graded by a judge would measure
the judge. This is the same task family the paper's Open-SWE reading draws from, so v0.1
rows sit on the same axis as it.

The task is a stimulus, not a capability probe: the bench measures a model against
itself across harnesses, so any task-selection bias applies to both harnesses equally
and cancels in the self-consistency-controlled gap.

**Size is set by power, not taste (§8).** The harness-power simulation
(`experiments/harness-power/`) is explicit about how many tasks a *controlled*
(small-n) run needs, and it is more than the first instinct: a 25-case, 2-repeat suite
detects a 20-point pass-rate gap only **7.5%** of the time; 50 cases at k=5 still
reaches only **0.46** power; **100 cases → 0.74, 200 cases → 0.965** (all at a 20-point
gap, k=5). At ~10 cases the false-positive rate reaches 0.18. So v0.1 targets the
high end — **~200 tasks**, never fewer than 100 — at **k≥5 rollouts** per (task,
harness). The exact count is fixed at freeze time against the dry-run's observed
self-consistency, but the floor is 100.

## 5. Generation protocol

Per task, per harness, `k≥5` independent rollouts. Held fixed across the whole row:
the **model** and its **endpoint** (one API or one served instance, one arithmetic),
and the **grader** (the task's own test suite). Varied: only the **harness**. Each
rollout runs in a fresh process with a fresh seed so repeats are independent draws.

Every rollout is captured whole and bound: the full trajectory (tool calls,
observations, the final patch), the grader's verdict (`resolved ∈ {0,1}`; an ungraded
run is **re-run**, never dropped), and the pinned harness revision. The manifest binds
the task set and every transcript by digest, so any detector defined later runs
against the same evidence.

**Harness versions are pinned like a `lib` condition.** A harness is software; its
version moves outcomes exactly as a library version does at the representation layer.
Each row records the exact harness revision (a git SHA), and a version change
re-freezes the row the way a re-measured API row does.

## 6. Conditions

The environment layer's varying axis is the **harness**, so the condition set is a
frozen pair of harnesses rather than the {proc, conc, time} of the other layers:

| condition | meaning at this layer | repeats |
| --- | --- | --- |
| `same` | k rollouts of one harness, one model, one task — the agent's own outcome floor (the self-consistency control) | k≥5 |
| `harness` | the same tasks under the other harness in the frozen pair, same model and endpoint | k≥5 |

`same` is the control every reading is taken against — the direct analog of `same` at
the other layers, and load-bearing here in a way it is not elsewhere: an agent
disagrees with *itself* 10–13% of the time (self-consistency 0.87–0.90 in the paper),
and any comparison that ignores that noise attributes it to whatever it happens to be
comparing. A `harness` reading is only meaningful net of `same`.

The frozen pair for v0.1 is **SWE-agent** vs **OpenHands** — the two widely-used coding
harnesses the ARI-E literature is about, and the pair the paper's reading uses.
`time`- and `conc`-style axes (does a harness's outcome drift across a >24 h gap, or
under concurrent load?) are real ARI-E questions deferred to v0.2.

## 7. The metric

The estimator is the paper's, unchanged. For an instance measured under two harnesses
with at least two rollouts on each side:

- **`D_self`** = the rate at which two rollouts of *one* harness on that instance reach
  *different* verdicts, averaged over both harnesses (the disagreement a fixed
  condition produces against itself — the `same` floor).
- **`D_cross`** = the rate at which a rollout under harness A and a rollout under
  harness B reach different verdicts on the same instance.
- **`Δ = D_cross − D_self`**: the disagreement between the two harnesses *in excess of*
  what either produces against itself. Equivalently `S − X` where `S = 1 − D_self` is
  self-consistency and `X = 1 − D_cross` is cross-agreement — the report stores the
  agreement form (§9), the headline is the gap.

`Δ` is reported over instances with ≥2 rollouts per side (one rollout gives no
self-consistency control); the row's `Δ` is the mean over qualifying instances, with a
95% bootstrap CI over instances.

**`Δ` is descriptive, not causal — and the spec says so at the row.** It assumes
rollouts within a condition are exchangeable, so that `D_self` estimates the
disagreement a fixed condition generates against itself; it is defined only where both
sides have ≥2 rollouts; and **it does not estimate a scaffold's effect on task
success.** It answers a narrower question: *how much of the disagreement between two
harnesses exceeds what either produces against itself.* A row also reports `S`, `X`,
the case count, and each harness's raw pass rate — the pass-rate gap is **not** the
metric (in the paper's Qwen3.5-122B contrast a ~13-point pass-rate spread compressed to
a 0.089 agreement gap; agreement compresses), and showing both stops a reader from
misreading `Δ` as a score gap.

**Headline: `ARI-E = Δ`.** A row with any core condition `pending` (a first run before
the second harness is measured) publishes as a draft with no value and no rank,
exactly as ARI-D drafts do.

## 8. Statistics, before any spend

The harness-power analysis (`experiments/harness-power/`) sizes the task set and,
just as importantly, bounds the claim. At 200 cases × k=5 the design detects a 20-point
pass-rate gap 96.5% of the time — so a **controlled v0.1 row resolves a large
agreement gap cleanly**.

What it does **not** resolve is a small one. A 10-point pass-rate gap is out of reach
at every size simulated (max 0.175 power at 200×5), and `Δ` compresses relative to the
pass-rate gap it comes from. So a controlled row cannot certify a *small* `Δ` — the
scale that pins `Δ~0.05` with a tight interval is the ~11,000 cases per contrast the
paper's Open-SWE reading had, which a controlled run cannot afford. This is the honest
division of labor, and it holds together because both readings are on the **same
harness pair** (SWE-agent vs OpenHands): the paper's reading is the high-n,
small-effect estimate on that pair — on two large models we did not host, so it cannot
be attested — while a v0.1 row is the large-gap reading on *our* pinned model, on
independent draws, with our own grader, which we *can* attest. Same axis and same
estimator, different provenance and different model.

**So the value a controlled v0.1 run adds over the paper's reading is provenance, not
sensitivity:** independent draws by construction, no ~22% silent drop, and a signed
report. It does not out-resolve a dataset a thousand times its size, and does not
claim to. Before the first paid run, a simulation over `(n_tasks × k)` grids
re-confirms the detectable gap against the dry-run's observed self-consistency (the
base rate in the sim was 0.75; the real one may differ). Intervals are bootstrap over
instances; the classification is CI-overlap against `same`, as everywhere on the board.

## 9. Schema, scorer, verification

**ARI-E rows pass the same gate as Representation and Decoding, in the same report
shape.** The design principle is reuse, not a parallel structure: an ARI-E report is
the same envelope every ARI report carries, with `same`/`harness` as its two
conditions and one generic detector each. It is additive to `report-schema.json` and
touches no R/D shape.

**The shared envelope** (identical required fields to R and D): `agent_id` (the
model), `ari_version`, `input_set: "ARI-E-Bench-v0.1"` (joins `INPUT_SETS` in the
scorer), `input_content_hash` (the frozen task set's pin), `environment`,
`results_per_condition`, `ARI`, `audit_hashes`, `layer: "environment"`,
`measured_at`. `probe_calibration` is `"outcome-level-v0.1"` — the ARI-D convention
for a layer that runs no probe (the paper's *without quantization* point, encoded).

**The two conditions** reuse `results_per_condition` and the existing
`genericDetectorResult` shape (the one ARI-D's sequence detectors use — a bounded
`value` in `[0,1]`, `value_ci`, `criterion_type`, `bytes_of_reference_state`, `unit`):

- `same` — an `outcome_agreement` detector whose `value` is `S = 1 − D_self`: the
  probability two rollouts of one harness on one instance reach the same held-out
  verdict, averaged over both harnesses.
- `harness` — an `outcome_agreement` detector whose `value` is `X = 1 − D_cross`: the
  probability a rollout under one harness and a rollout under the other reach the same
  verdict on the same instance.

Both carry `criterion_type: "interval"`, `bytes_of_reference_state: 0`,
`unit: "instance_pair"`.

**The headline** is `ARI = same.value − harness.value = S − X = Δ = D_cross − D_self`,
in the report's existing `ARI` field. The scorer's per-layer branch — the same
mechanism that already reads `detectors.exact_generation` for decoding — reads, for
`layer: "environment"`, `ARI == same − harness` within tolerance, with the CI a
bootstrap over the pinned instances. Pin, attestation and signer-registry checks are
byte-for-byte the R/D ones.

**ARI-E-specific metadata** lives in `environment` (free-form by schema, as in R/D):
`endpoint` (the model API or served instance), the two `harnesses` (name + git SHA
each), the per-harness `pass_rate` diagnostics, and `cases` / `k`. Nothing needs a new
top-level field.

A compact example (values illustrative, from the paper's Qwen3.5-122B contrast):

```json
{
  "agent_id": "qwen/qwen3.5-122b",
  "ari_version": "0.1", "layer": "environment",
  "probe_calibration": "outcome-level-v0.1",
  "input_set": "ARI-E-Bench-v0.1", "input_content_hash": "…",
  "environment": {
    "endpoint": "…", "harnesses": {"swe-agent": "swe-agent@<sha>", "openhands": "openhands@<sha>"},
    "pass_rate": {"swe-agent": 0.468, "openhands": 0.334}, "cases": 200, "k": 5
  },
  "results_per_condition": {
    "same":    {"detectors": {"outcome_agreement": {"value": 0.874, "value_ci": [0.85, 0.89],
                 "criterion_type": "interval", "bytes_of_reference_state": 0, "unit": "instance_pair"}}},
    "harness": {"detectors": {"outcome_agreement": {"value": 0.785, "value_ci": [0.76, 0.81],
                 "criterion_type": "interval", "bytes_of_reference_state": 0, "unit": "instance_pair"}}}
  },
  "ARI": 0.089,
  "audit_hashes": {"same": "…", "harness": "…"},
  "measured_at": "…"
}
```

Reports are signed with the project key (`alias/semq-ari-attestation`), the manifest
binding the task set, the endpoint identity, both harness versions, and every rollout
transcript. They verify with the existing standalone verifier; rows classify
self-reported/attested exactly as the other layers. The scorer's row builder
(`to_environment_row`, alongside `to_decoding_row`) emits the leaderboard row; the
Environment tab reads gated rows from a new `environment_entries` list in
`leaderboard.json`, and shows the paper's Open-SWE reading beside them as a labeled
reference (§8).

**One polarity note.** `ARI = Δ` reads *opposite* to R and D: there, higher is more
reproducible; here a higher `Δ` means a larger cross-harness disagreement gap. This
follows the paper and the decomposition's own framing (the ARI decomposition proposal,
§4: "a zero is meaningful only in R" — E is a gap, not a `1.0 = perfect` rate). The
board publishes `Δ` and labels the E column as a *gap*, not a reproducibility rate;
`1 − Δ` was considered and rejected as relabeling a gap as a rate. The stored `S`/`X`
are unaffected either way.

## 10. Model, cost, and the v0.2 checkpoint extension

**v0.1 needs no GPUs of ours.** Outcome agreement runs the two harnesses against one
model **endpoint**; if that endpoint is a hosted API, there is no self-hosting and no
GPU block — cost is API tokens for the ~2,000 rollouts (~200 tasks × k≥5 × 2
harnesses), borne per run, not per-token board spend. A hosted model is the default
v0.1 subject; a self-hosted one is equally valid but is not required by the metric.
This is the property that lets ARI-E scale the way ARI-R and ARI-D do — a third party
can run the two harnesses on *their own* endpoint and submit an outcome-agreement
report, without our infrastructure.

**The v0.2 extension: KV-cache checkpoint attribution.** The original decomposition
proposed more than an outcome gap: *checkpoint-level attribution of where two harnesses
diverge*. At each harness decision point (test run, retry/give-up, stop) the model's
KV-cache state is snapshotted via `semq.smkv` into a content-addressed, notarized
artifact, reloadable under the *other* harness to see where the outcome was decided.
This is the SEMQ-specific differentiator no public dataset can carry, and its mechanism
is validated (the SMKV↔Qwen3 round-trip works). It is **not in v0.1**, and for three
honest reasons: it forces self-hosting (an 8×H100-class node for a large model), it is
`semq.smkv`-Qwen3-dense-only (so it cannot attribute the API models people most care
about), and it does not scale by submission. v0.1 ships the scalable outcome-agreement
substrate; the checkpoint layer is a curated deep-dive on a small set of self-hosted
reference rows, specced separately in v0.2.

## 11. Harness notification and fairness

ARI-E names software projects, not just companies, and most target harnesses are open
source. The bench owes them the same process ARI-D owes providers, adapted:

- A row that reads a harness as materially widening the agreement gap is shared with
  that project's maintainers before it first publishes, with the complete receipt
  (report, signed manifest, the one-line re-verify command).
- The reading is never a quality verdict. A large `Δ` says the two harnesses disagree
  on outcomes beyond the model's own noise — it does **not** say either harness is
  worse. The published copy states this at the row, not in a footnote.
- Version-pinning is the fairness mechanism: a maintainer who fixed a nondeterminism
  source can point to the SHA the row measured, and a re-measure at the new SHA is a
  checkable claim, not a shrug.

## 12. What v0.1 does not claim, and open questions

- **A causal effect of the scaffold.** `Δ` is a descriptive agreement gap, not an
  estimate of how much a harness changes task success. The board labels it as a gap.
  This is the paper's own careful line, kept verbatim.
- **A small gap, by a controlled row.** At the ~200-task scale a controlled run
  affords, the design resolves a large gap (a 20-point pass-rate gap, 0.965 power) but
  not a small one (a 10-point gap tops out at 0.175 power). A controlled row that reads
  near zero does **not** certify the harnesses agree — it says no gap was resolvable at
  this n. The high-n, small-gap estimate lives in the paper's Open-SWE reference
  reading, which we cannot attest but can cite (§8).
- **Outcome-only, not trajectory.** Two harnesses reaching the same verdict by
  different paths read as agreement here. Trajectory-level attribution needs a
  tool-vocabulary mapping the harnesses do not share; v0.1 does not attempt it, and the
  paper measures the outcome *without quantization* for the same reason.
- **The gap moves with the model.** The paper showed a 1.8× swing between two models. A
  v0.1 row is one (model, harness-pair) point, and says so: "the gap for this model on
  these tasks," never "the harness effect."
- **A grading bias survives if the grader is systematically blind.** Re-running
  ungraded rollouts removes the ~22% silent-drop bias of the public dataset, but if a
  task's held-out tests are flaky, that flakiness enters `S`. The dry run measures
  per-task grader stability and drops tasks whose own tests are not reproducible before
  the freeze.
- **Two harnesses and one model is a measurement, not a constant.** Breadth (more
  harnesses, more models, the checkpoint attribution, the `time`/`conc` axes) is the
  v0.2 roadmap, not a v0.1 claim.

## 13. Rollout

1. **Pick the v0.1 model + endpoint** (a hosted API is the default; the paper's models
   or a hosted Qwen are natural choices for continuity). No GPU provisioning for v0.1.
2. **Populate the harness runner:** pin SWE-agent and OpenHands to fixed revisions,
   configure both to drive the one endpoint (both speak the OpenAI chat API), and wire
   them to run the task set at matrix scale.
3. **Power simulation** over the `(n_tasks, k)` grid against the dry-run's observed
   self-consistency; fix the frozen task count and k.
4. **Freeze `data/ari-e-bench-v0.1.jsonl`** (the pinned SWE-bench Verified subset +
   graders), publish its content hash. Dry run one harness pair on a handful of tasks:
   settles grader stability per task, the self-consistency floor, and whether the set
   discriminates (if every task saturates `same` at 1.000, it cannot separate a gap
   from noise and gets enriched before the run).
5. **Run the controlled matrix**, grade, sign the report.
6. **First gated rows** land in the Environment tab through the standard scorer gate,
   beside the paper's Open-SWE reference reading. The tab stops being a single research
   appendix the same way Representation and Decoding did: measured, pinned, attested.

## 14. Provenance of this design

This document productizes prior work rather than inventing. The metric —
self-consistency-controlled outcome agreement, `Δ = D_cross − D_self`, descriptive not
causal — and its first values are the ARI paper's Appendix C (its Table 4, the
interaction finding, and the "pass-rate is not the gap" caveat), on
`nvidia/Open-SWE-Traces` (~18,000 SWE-bench-style instances; ~10,000–12,000 graded
cases per contrast after dropping ~22% ungraded). The sizing comes from the
harness-power simulation (`experiments/harness-power/`). The layer's place in the
`(ARI-R, ARI-D, ARI-E)` decomposition comes from the ARI decomposition proposal, which
also proposed the checkpoint attribution now scoped to v0.2. The pin, the signing key,
the independent verifier and the scorer-gate discipline are the leaderboard's existing
machinery. What is new here is the *governance*: turning the paper's outcome-agreement
appendix into a frozen task set, a controlled and self-graded run, and gated, attested
rows a third party re-checks. The checkpoint-attribution differentiator, and the
`qwen-harness-trace` infrastructure that carries it, are the subject of ARI-E v0.2.
