# ARI-D dry run — interim results

**Status: vendor subject complete (2026-08-31), platform subject measured and
rehosted subject measured with the burst sweep (2026-09-01); platform and
rehosted `time` pairs measured (2026-09-02). **The dry run's protocol
cells are complete for every measured subject.** Evidence: `results/manifest.json` (transcript
digests) + `results/analysis.json` + `results/door_compare.json` (regenerate
with `analyze_dry.py` / `door_compare.py`); raw transcripts retained, gzipped,
bound by digest in the manifest, and mirrored at
`s3://semq-ari-baselines-127348475353/arid-dry-run/transcripts/`.

**What these numbers are.** Dry-run measurements over the frozen set — the
harness shakedown and calibrations the spec's rollout item 3 calls for. They
are **not published bench rows**: rows enter the Matrix through the scorer
gate, with the provider-notification process the spec defines (§11), when the
bench launches. Every figure below is recomputable from a digest-bound
transcript; nothing is quoted from an interactive session without retained
evidence.

## Vendor subject: gpt-4o-mini via api.openai.com, 100 prompts, frozen set

| condition | k | exact_generation | 95% CI | first_divergence | vs same |
| --- | --- | --- | --- | --- | --- |
| `same` | 8 | **0.470** | [0.401, 0.538] | 0.699 | — |
| `proc` | 8 | 0.443 | [0.370, 0.515] | 0.676 | ≈ same |
| `conc` (64) | 12 | 0.479 | [0.407, 0.550] | 0.688 | ≈ same |
| `time` (>24 h) | 8×8 cross | 0.469 | [0.402, 0.537] | 0.690 | ≈ same |

The `time` cell (`time_pair.py` → `results/time_pair.json`) pairs the two
`same` batches across a 24.5 h gap, k×k per prompt. Fleet continuity is
recorded per §6: 43 of day one's 45 `system_fingerprint`s reappear on day
two (47 seen) — stability measured under **no** silent backend change,
which is what makes the ≈ reading meaningful. With all three core
conditions in, the line's draft headline is
**ARI-D = mean(proc, conc, time) = 0.464**.

Reading: **the provider's own floor is ~0.47** — at temperature 0, with a
seed, over one reused connection, more than half of back-to-back repeat pairs
are not byte-identical. Nothing moves that floor: not a fresh connection per
call, not a sustained 64-in-flight burst, not a day passing (all three ≈
`same` under the CI-overlap rule). The nondeterminism is **per-call**, not
connection-, load- or time-sensitive — the same shape the representation
panel found for its APIs, now reproduced at the decoding layer. One `same` batch touched **45 distinct `system_fingerprint`
values**, which documents the mechanism (a heterogeneous serving fleet behind
one endpoint) and vindicates recording fingerprints per call.
`topk_logprob_overlap` reads 0.961: the distributions agree almost entirely
until the flip — drift lives below the argmax and surfaces where margins are
thin, consistent with the research base.

## Rehosted subject: Llama-3.3-70B-Instruct-Turbo via Together, raw completions

Official Llama-3 template applied by the harness, logprobs at the endpoint's
live maximum (top-5), `max_tokens` honored (unlike the platform door).

| condition | k | exact_generation | 95% CI | first_divergence | vs same |
| --- | --- | --- | --- | --- | --- |
| `same` | 8 | **0.452** | [0.382, 0.523] | 0.672 | — |
| `proc` | 8 | 0.431 | [0.365, 0.497] | 0.676 | ≈ same |
| `conc` (16) | 12 | 0.378 | [0.317, 0.441] | 0.613 | ≈ same |
| `conc` (64) | 12 | 0.448 | [0.393, 0.506] | 0.681 | ≈ same |
| `conc` (128) | 12 | 0.399 | [0.340, 0.461] | 0.627 | ≈ same |
| `time` (>24 h) | 8×8 cross | 0.435 | [0.373, 0.498] | 0.663 | ≈ same |

Reading: a rehosted fp8 "turbo" endpoint sits at the same self-consistency
floor as the vendor door (0.452 vs 0.470), per-call and load-insensitive.
Whether it serves faithful published weights is a different question — the
reference axis — and waits for the fp32 reference pass. The declared
precision (fp8) plus the reference pair is exactly the two-axis reading §7
defines for this class.

## Second and third vendor subjects: the floor is a choice

**mistral-medium-2604** (via api.mistral.ai; chat, `random_seed`, no logprobs;
`conc` at burst 16 — the tier's documented 50-requests/minute cap, throttling
recorded per call):

| condition | k | exact_generation | 95% CI | first_divergence | vs same |
| --- | --- | --- | --- | --- | --- |
| `same` | 8 | **0.525** | [0.478, 0.576] | 0.707 | — |
| `proc` | 8 | 0.550 | [0.496, 0.606] | 0.725 | ≈ same |
| `conc` (16) | 12 | 0.432 | [0.375, 0.490] | 0.652 | ≈ same |
| `time` | — | pending (>24 h re-run) | | | |

**gemini-3.6-flash** (via generativelanguage.googleapis.com; no logprobs for
this model; thinking held at `MINIMAL` so the full token budget is visible
output):

| condition | k | exact_generation | 95% CI | first_divergence | vs same |
| --- | --- | --- | --- | --- | --- |
| `same` | 8 | **1.0000** | [1.0, 1.0] | 1.0000 | — |
| `proc` | 8 | 1.0000 | [1.0, 1.0] | 1.0000 | ≈ same |
| `conc` (64) | 12 | 1.0000 | [1.0, 1.0] | 1.0000 | ≈ same |
| `time` | — | pending (>24 h re-run) | | | |

Reading: with three vendors measured, the decoding floor splits the industry
in two. OpenAI (0.470) and Mistral (0.525) sit in one neighborhood —
per-call nondeterminism that nothing external moves. Gemini serves **2,800
completions without a single divergent byte**, including under a sustained
64-in-flight burst — and it is the same provider that read bit-exact on
every axis of the representation panel. One provider doing it at scale, on
both layers, makes the ~0.5 floor read as **an engineering choice, not
physics** — which is precisely the kind of comparison this bench exists to
put a number on. (Both lines await their `time` cells before any headline
is computed.)

## Calibration verdicts (spec §13, item 3)

1. **Burst sweep** — **settled**: {16, 64, 128} on the rehosted row, k=12
   per arm, achieved in-flight peak equal to the requested burst in every
   arm, zero reconnects. No arm separates from the row's own `same` under
   the CI-overlap rule, so burst size does not change what `conc` detects
   on this service class; **the panel-wide burst freezes at 64** (continuous
   with the representation-layer evidence base). Recorded in spec §6.
2. **Length confounder** — **passes for this subject**: Spearman ρ between
   pair byte length and normalized `first_divergence` is weak in every bucket
   (|ρ| ≤ 0.18; sign inconsistent). Length does not dominate the signal; no
   redesign. Re-check on the rehosted subject before freezing the verdict.
3. **Set discrimination** — **passes**: the `same` floor spans 0.041
   (open-ended prose) to 0.759 (instruction-following) across buckets, the
   exact inverse of the top-2 margin distribution (median 2.0 vs 11.6 nats) —
   flips concentrate where margins are thin, and no bucket saturates. The §4
   decision-dense amendment is not needed for v0.1.

## Platform subject: the same model through the Salesforce Models API

Exploratory `platform` class (not a v0.1 panel row): the Einstein Trust Layer
serving `sfdc_ai__DefaultOpenAIGPT4OmniMini` — which resolves, per the
response's own passthrough metadata, to **the same `gpt-4o-mini-2024-07-18`
snapshot the vendor subject measures directly**, on the same fleet: the
OpenAI `system_fingerprint` passes through the trust layer, and 6 of the
platform run's 12 fingerprints also appear in the direct run's 45
(`results/door_compare.json`). Same model, same backends, same frozen
prompts, two doors. Reduced protocol (rate-capped developer org):
`same` and `proc` at k=8, paced; no `conc`.

| condition | k | exact_generation | 95% CI | first_divergence | vs same |
| --- | --- | --- | --- | --- | --- |
| `same` | 8 | **0.107** | [0.058, 0.163] | 0.231 | — |
| `proc` | 8 | 0.115 | [0.061, 0.173] | 0.228 | ≈ same |
| `time` (>24 h) | 8×8 cross | 0.109 | [0.058, 0.166] | 0.228 | ≈ same |

**Door-to-door, confounders controlled.** The raw gap (0.470 vs 0.107) has
two candidate confounders, both now measured away:

1. *Length* — the platform ignores `max_tokens` (below), so its completions
   run longer (mean 1050 vs 440 bytes) and mechanically have more room to
   diverge. Controlled by comparing byte-identical **prefix** rates at fixed
   horizons, computable because transcripts are retained:

   | horizon | direct (seed) | direct (no seed) | platform |
   | --- | --- | --- | --- |
   | 64 B | 0.924 | 0.928 | 0.471 |
   | 128 B | 0.813 | 0.807 | 0.245 |
   | 256 B | 0.644 | 0.612 | 0.130 |
   | 512 B | 0.494 | 0.459 | 0.107 |
   | full | 0.470 | 0.441 | 0.107 |

2. *Seed* — the direct run sent one; the platform cannot accept one. The
   dedicated no-seed arm shows the seed explains essentially nothing
   (0.441 vs 0.470 full, prefix rates within noise).

The platform `time` cell (`time_pair.py --batch-a salesforce_same.jsonl.gz --batch-b salesforce_same_t2.jsonl.gz`) reads at the platform's own floor across a >24 h gap, with all 12 of day one's fingerprints reappearing — per-call noise, stable in time, no silent backend change on either door.

The residual is the finding: **through the platform door, the same model on
the same fleet is roughly 2× less byte-stable at the first 64 bytes and 4×
less over full completions, and the cause is not attributable from outside**
— their effective sampling parameters, prompt handling and trust-layer
processing are all invisible to the caller. That non-attributability is the
platform-class reading, not a gap in this analysis.

**Control surface findings** (verified live, both endpoints):

- The `parameters` block is **silently ignored**: `temperature` and
  `maxTokens` accepted and discarded (`maxTokens: 8` against a long-essay
  prompt returned 250 completion tokens with `finish_reason: stop`;
  non-numeric garbage in either field returns 200). A caller who believes
  they pinned temperature 0 through this door did not, and nothing tells
  them so. Evidence: `probe_control_surface.py` →
  `salesforce_control_probes.jsonl.gz`, digest-bound in the manifest.
- No seed, no logprobs, no model-version pinning. In exchange, the response
  exposes the upstream model snapshot and passes `system_fingerprint`
  through — more version transparency than the door's own controls suggest.
- Operational: the `api.salesforce.com` gateway requires JWT-format access
  tokens; a classic opaque token gets a bare 404 with no error body, which
  reads as a provisioning failure and is not one.

## Harness fixes the dry run produced (its actual job)

- Prefix smoke transcripts get their own filename; a smoke no longer
  overwrites a full run's digest-bound evidence.
- 401-refresh auth path (Salesforce client-credentials tokens expire
  mid-run at paced rates).
- `--pace` for rate-capped orgs; `--label` so a re-run (the `time` pair)
  never overwrites the first pass; `--no-seed` for confounder arms.
- sfgen response parsing corrected against the live schema (provider
  passthrough sits at the top-level `parameters`, not under `generation`).

## Next

- Platform subject `conc` (needs a production-tier org, or the org access
  this measurement suggests asking the platform vendor for directly).
- fp32 references (rollout item 4): a 70B model in fp32 is ~282 GB of
  weights, so the reference pass needs a multi-GPU node with more than
  that in aggregate VRAM (e.g. 8×L40S, 384 GB); bf16 (~141 GB) fits on
  half of one.
