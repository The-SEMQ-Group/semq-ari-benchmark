# Probe sweep. Does a larger bin count buy resolution worth its storage?

**Status: exploratory, development data only.** Selects a configuration to carry
forward. It establishes no comparative advantage and supports no deployment
claim. Grid in [`PREREGISTRATION.md`](PREREGISTRATION.md), scripts
[`run_sweep.py`](run_sweep.py) and [`decide.py`](decide.py).

**Headline: more bins buy upper bounds and cost exact-code agreement. They buy
no lower bound at all for bf16, at any bin count tested. Against ordinary
uniform scalar quantization at the same bit width, QUANT has no consistent
advantage — the difference reverses sign between three and four bits.**

## Setup

3,111 development documents (`doc_index % 5 < 3`) of the 5,183-document BEIR
SciFact corpus, encoded by `all-MiniLM-L6-v2` at dim 384, from the cache in
[`../regime-discrimination/`](../regime-discrimination/). Six real serving
conditions against one reference: `proc`, `threads1`, `batch8`, `batch128`
declared near-null; `bf16` and `int8` declared interventions.

Nine QUANT cells (`n_bins` 2, 4, 8 × percentile 0.99, 0.995, 0.999) and three
uniform scalar cells at matched bit width. The reserved 40% is untouched, and
is not an independent confirmatory sample: it comes from the same collection
run, process, machine and session as the development half.

## Saturation, storage and the encoder boundary

| `n_bins` | pct | bytes | saturation | predicted | nominal edge | encoder edge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.99 | 96 | 0.1963 | 0.198 | 0.06592 | 0.06592 |
| 4 | 0.99 | 144 | 0.0533 | 0.053 | 0.09888 | 0.09888 |
| 8 | 0.99 | 192 | 0.0243 | 0.024 | 0.11536 | 0.11536 |
| 2 | 0.999 | 96 | 0.0992 | — | 0.08428 | 0.08428 |
| 4 | 0.999 | 144 | 0.0136 | — | 0.12642 | 0.12642 |
| 8 | 0.999 | 192 | 0.0040 | — | 0.14749 | 0.14749 |

The Gaussian prediction holds on real embeddings: 19.6% / 5.3% / 2.4% measured
against 19.8% / 5.3% / 2.4% predicted at p99. The nominal outermost edge from
the calibration agreed with the encoder-derived boundary in all nine cells.

Storage is not proportional to bin count. Four times the regions costs twice
the bytes, because the payload is `ceil(dim * ceil(log2(2 * n_bins)) / 8)`.

## Upper and lower bounds respond differently

At percentile 0.99, over changed coordinates:

| `n_bins` | condition | HER | changed rate | multi-region / changed | unbounded / changed |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2 | bf16 | 0.1016 | 0.005922 | 0.00000 | 0.50502 |
| 2 | int8 | 0.0000 | 0.205971 | 0.00027 | 0.46676 |
| 4 | bf16 | 0.0077 | 0.012203 | 0.00000 | 0.09219 |
| 4 | int8 | 0.0000 | 0.403046 | 0.03909 | 0.08476 |
| 8 | bf16 | 0.0000 | 0.024760 | 0.00000 | 0.02566 |
| 8 | int8 | 0.0000 | 0.636098 | 0.27404 | 0.02779 |

`proc`, `threads1`, `batch8` and `batch128` hold HER at exactly 1.0000 in every
one of the nine cells. No near-null condition disturbed a single code.

**Upper bounds improve with bin count.** The unbounded share of changed
coordinates falls from ~50% to ~2.6%.

**Lower bounds do not, for bf16.** Adjacent regions touch, so a positive lower
bound requires a move of two or more regions. bf16 produces zero multi-region
moves at every bin count tested, so it gains no positive lower bound anywhere
in the grid. Only int8 — a far coarser change — gains them.

**The cost is HER saturation.** bf16's exact-code agreement falls 0.1016 →
0.0077 → 0.0000. At eight bins the metric no longer separates bf16 from int8.

## Against uniform scalar quantization

Paired over the same documents, `int8`, multi-region share of changed
coordinates, QUANT minus uniform, 95% percentile bootstrap over documents:

| bits | QUANT | uniform | difference | 95% CI |
| ---: | ---: | ---: | ---: | :--- |
| 2 | 0.00027 | 0.00004 | +0.02 pp | [+0.02, +0.03] |
| 3 | 0.03909 | 0.03453 | +0.46 pp | [+0.43, +0.49] |
| 4 | 0.27404 | 0.29059 | −1.65 pp | [−1.72, −1.59] |

Every interval excludes zero and the sign reverses: QUANT resolves more at two
and three bits, uniform more at four. The effects are small, and the intervals
cover corpus sampling within one run, not run-to-run variation. **The case for
a configuration is not a case for the operator.**

At the same bit width bf16 yields no multi-region moves under either operator,
so the difference there is exactly zero and carries no information.

## The decision rule, and where it lands

The rule in [`decide.py`](decide.py) requires near-null codes intact, at least
1% of changed coordinates moving two or more regions, and a bootstrap lower
bound above zero.

- **Under the "any intervention" reading: `n_bins = 4` at percentile 0.999.**
- **Under the "all interventions" reading: nothing qualifies.** bf16 never
  reaches the threshold at any cell, so no configuration resolves both
  declared interventions.

The two readings disagree, which is why both are reported. Choosing between
them now would be selecting on the outcome.

### The tie-break picked the least sensitive qualifying cell

`PREREGISTRATION.md` records the flaw in advance; this run is what it looks
like. Ties break on `n_bins`, then storage, then saturation. All three 4-bin
cells qualify at identical storage, so saturation decides, and lower
saturation means a wider scale. The rule therefore selected percentile 0.999 —
the widest scale and the least sensitive of the qualifying cells, whose
multi-region share (0.01210) sits just above the 1% threshold, against 0.03909
at percentile 0.99.

**This does not agree with the configuration the sweep was expected to
select.** A confirmatory rule has to break ties on something that is not in
tension with the quantity being secured. Freezing a configuration on this
run's output would be freezing the tie-break defect with it.

### The original prose rule

The first version of the rule read "a non-zero share of multi-region moves".
On this data that is satisfied by `n_bins = 2` at percentile 0.99, where
**66 coordinates out of 246,058 changed** move more than one region — a share
of 0.00027. The rule is kept in `PREREGISTRATION.md` as written, and the
defect recorded, rather than retrofitted.

## What this does not establish

No claim about index incompatibility, retrieval degradation or functional
harm. Documents are not episodes: every interval here covers sampling of one
corpus collected in one run, on one machine, in one session. A comparative
claim needs the freshly collected episodes of SEM-49.

## Reproduction

```bash
python experiments/probe-sweep/run_sweep.py \
  --cache experiments/regime-discrimination/results/cache \
  --out experiments/probe-sweep/results/sweep.json
python experiments/probe-sweep/decide.py \
  --sweep experiments/probe-sweep/results/sweep.json \
  --out experiments/probe-sweep/results/decision.json
```

Run of 2026-09-11 against the scripts as committed here, on the cache written
by `regime-discrimination`. 12 cells over 3,111 documents, about 5 seconds.

| artifact | sha256 (first 16) | size |
| --- | --- | ---: |
| `results/sweep.per-document.npz` | `67d63705475ad1d6` | 170.2 KiB |
| `results/decision.json` | `9335873c2ed78130` | 14.0 KiB |
| `results/sweep.json` | varies, see below | 40.7 KiB |

`sweep.json` records `encode_seconds` per cell, so it does not hash
reproducibly across runs. The two artifacts above do, and `decision.json`
carries every number this document reports. Check those.

| | |
| --- | --- |
| python | 3.11.12 |
| numpy | 2.4.6 |
| platform | macOS-26.5.1-arm64 |
| semq | `0.0.0+unknown` |

**The `semq` version is a local development build and does not identify a
release.** Every number above is reproducible only against that build. Before
any of this is cited, re-run it against a versioned SDK and replace this table.
