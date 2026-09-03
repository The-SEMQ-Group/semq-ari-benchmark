# Does SEMQ beat the cheap alternatives at the decoding layer?

**Status: run on CPU.** Script: [`baselines.py`](baselines.py). Machine-readable:
[`results/baselines.json`](results/baselines.json).

**Headline: for the precision changes measured so far, no. The top-2 margin gets 86% of
SEMQ's signal for 1/2000th of the storage. SEMQ's real advantage is elsewhere, and it is
narrower than the ARI-D result on its own implied.**

This was worth measuring because the ARI-D result is only interesting if a simpler
statistic does not do the same job.

## Axis 1. Sensitivity

Response to Gaussian logit noise of magnitude σ, mean over steps, 5 trials:

| statistic | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1.0 | order |
| --- | ---: | ---: | ---: | ---: | ---: | :--- |
| **SEMQ H̄** | 1.33e-4 | 1.33e-3 | 1.33e-2 | 0.129 | 0.813 | **linear** |
| top-2 margin Δ | 1.14e-4 | 1.15e-3 | 1.14e-2 | 0.112 | 0.873 | **linear** |
| max \|Δlogit\| | 4.29e-4 | 4.28e-3 | 4.28e-2 | 0.429 | 4.28 | linear |
| top-20 set change | 1.77e-4 | 7.07e-4 | 5.62e-3 | 0.051 | 0.346 | sub-linear |
| KL | −5.1e-10 | 2.18e-7 | 2.23e-5 | 2.21e-3 | 0.226 | **quadratic** |
| JS | −1.9e-10 | 5.39e-8 | 5.57e-6 | 5.50e-4 | 0.051 | quadratic |
| token flip | 0 | 1.86e-3 | 2.97e-3 | 0.029 | 0.236 | threshold |

**KL is the wrong instrument for small perturbations, and the reason is structural.** KL
is second order in the perturbation, so a 10× rise in σ gives a 100× rise in KL. At
σ = 1e-4 it reads −5.1e-10. That value is zero to numerical precision, with the *wrong sign*.
SEMQ reads 1.33e-4 at the same point. So the "can I get 80% from KL" question has a clean
answer for the small-σ regime: no, KL is not in the running there. JS inherits the same
problem.

**The top-2 margin is a different story, and it is the one that matters.** It is linear in
σ, tracks SEMQ within 14%, and needs 8 bytes of reference state per step against SEMQ's
16,000. That is a serious result against SEMQ and it should not be softened.

## Axis 2. Reference state a monitor must retain, per decoding step

| statistic | bytes |
| --- | ---: |
| token flip | 4 |
| **top-2 margin Δ** | **8** |
| top-20 set change | 80 |
| **SEMQ H̄** | **16,000** |
| KL / JS / max\|Δlogit\| / L2 | 128,000 |

At 32,000 vocabulary. SEMQ is 8× cheaper than keeping fp32 logits, and 2,000× more
expensive than keeping two floats.

## Axis 3. Is the statistic reproducible bit-exactly?

The script computes each statistic two algebraically equivalent ways. For KL,
`Σp(log p − log q)` against `Σp log p − Σp log q`; for the others, reversed summation
order:

| statistic | bit-identical | max relative difference |
| --- | :---: | ---: |
| **SEMQ H̄** | **yes** | 0 |
| top-2 margin Δ | yes | 0 |
| max \|Δlogit\| | yes | 0 |
| L2 \|Δlogit\| | no | 1.4e-7 |
| **KL** | **no** | **4.6%** |
| **JS** | **no** | **363%** |
| top-20 set change | no | 100% |
| token flip | no | 100% |

**KL is not reproducible.** Regrouping the arithmetic changes it by up to 4.6%, and JS by
more than 100%. A published KL value is not a number a third party can recompute and
confirm. A SEMQ code is. That matters for *attestation* and not for *detection*. For
detection you would set a threshold on KL, and the last bits would not matter.

But note the top-2 margin is bit-stable too. This axis does not separate SEMQ from the
cheap baseline either.

## The one axis that does separate them

Every statistic above was measured under **isotropic** noise, which perturbs the top of the
distribution and the tail alike. That cannot distinguish a statistic reading the whole
vector from one reading two entries. So: perturb *only* the tail, leaving the top ranks
untouched by construction (σ = 0.1):

| statistic | full vector | below rank 20 | below rank 100 |
| --- | ---: | ---: | ---: |
| **SEMQ H̄** | 0.1288 | **0.1289** | **0.1285** |
| max \|Δlogit\| | 0.4289 | 0.4283 | 0.4281 |
| KL | 2.21e-3 | 2.19e-4 | 6.12e-5 |
| top-20 set change | 0.0509 | 0.0335 | **0** |
| **top-2 margin Δ** | 0.1115 | **0** | **0** |
| **token flip** | 0.0289 | **0** | **0** |

**SEMQ's response does not change: 0.1289 against 0.1288. The margin and the token
statistic go to exactly zero.** They are not less sensitive. They are structurally
blind, because they read only the top of the distribution.

## What this means for ARI-D

**The claim "SEMQ is a more sensitive decoding-layer probe" does not survive.** For the
precision changes measured in the main ARI-D run, the top-2 margin is 86% as responsive,
equally bit-stable, and 2,000× cheaper. If detecting bf16-vs-fp32 is the goal, ship the
margin statistic.

**The defensible claim is coverage, not sensitivity.** SEMQ reads the entire distribution,
so it detects drift confined below the top ranks that a margin statistic cannot see at all.
Whether that matters depends on whether real drift is isotropic or structured:

- **Isotropic**. A precision or quantization change scales all logits roughly
  proportionally. The margin catches it. SEMQ adds little.
- **Structured**. A token-suppression or safety filter, a LoRA adapter, a tokenizer or
  vocabulary change, speculative decoding. These can move the tail while leaving the top-2
  intact. The margin reads exactly zero. SEMQ reads the change at full strength.

"A provider silently added an output filter" is a realistic instance of the second case,
and it is precisely the scenario the margin cannot cover.

**Recommended posture: both, not either.** The margin statistic is the right always-on
tripwire at 8 bytes per step. SEMQ is the full-coverage, bit-reproducible record for the
cases the tripwire cannot see and for attestation, where KL's 4.6% irreproducibility rules
it out. Presenting SEMQ as a *replacement* for the cheap statistic would not survive
scrutiny from anyone who runs this comparison.

## Caveats

- One model, one reference set, 539 steps, CPU only.
- The structured-perturbation test uses synthetic tail noise. It shows SEMQ *can* see what
  the margin cannot. It does **not** show that any real serving change makes
  tail-confined drift. That question is open, and it decides whether the coverage
  argument is commercially real or only true.
- Isotropic Gaussian noise is itself a model of drift, not drift. The GPU conditions
  (TF32, fp16) supply real structured changes to test against.
- Storage figures are the reference state per step, not the cost of computing the
  statistic.
