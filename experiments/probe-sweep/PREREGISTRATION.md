# Probe sweep — grid and decision rule

> **Status: exploratory, relabelled from "preregistration".**
>
> The grid was fixed before any cell was scored, and that part stands. The
> decision rule was not: it named no threshold, referred to a set of conditions
> that does not exist, and was applied by eye. Read literally, "a non-zero
> share" is satisfied by 66 coordinates out of 246,058.
>
> The rule now lives in [`decide.py`](decide.py) with a declared threshold,
> declared condition roles and document-resampled intervals. It was written
> after the first results were seen, so it is not preregistered either and the
> study is exploratory throughout. A rule that earns the label belongs with the
> fresh episodes in SEM-48 and SEM-49.
>
> The original rule is kept at the bottom, unedited.

Development data only. This selects a configuration to carry forward; it does
not establish a comparative advantage, and no confirmatory claim follows.

## Question

The shipped embedding probe uses two magnitude bins. Two bins give four
regions, two of them outermost and therefore unbounded, and adjacent regions
touch. A positive lower bound on movement requires a move of two or more
regions, so at two bins a single-region move — the only kind realistic drift
produces — carries no magnitude information at all.

Does a larger bin count buy resolution worth its storage, and at what cost in
saturation?

## Data

`experiments/regime-discrimination/results/cache/` — 5,183 BEIR SciFact
documents encoded by `sentence-transformers/all-MiniLM-L6-v2` (dim 384) under
a reference condition and six real serving conditions: `proc`, `threads1`,
`batch8`, `batch128`, `bf16`, `int8`.

**Development / reserved split.** Documents are split by a frozen index rule:
`doc_index % 5 < 3` is development (60%), the rest is reserved. Selection uses
development only.

The reserved documents are **not independent confirmatory episodes**. They
come from the same collection run as the development half and share its
process, machine and session. They guard against overfitting the document
sample, and nothing more. A confirmatory comparison needs freshly collected
episodes (SEM-49).

## Grid

| axis | values |
| --- | --- |
| `n_bins` | 2, 4, 8 |
| calibration | percentile 0.99 (current), 0.995, 0.999 |
| dimension | 384 (real); 1024 reported separately if a real encoder run is available |

Every cell is scored on the same documents and the same conditions.

## Storage accounting

Code payload is `ceil(dim * ceil(log2(2 * n_bins)) / 8)` bytes, padding
included. At dim 384 that is 96, 144 and 192 bytes for 2, 4 and 8 bins:
**storage is not proportional to bin count**. Calibration metadata is counted
separately — one float32 scale for QUANT, two for a uniform grid — and never
folded into the payload.

## Hypotheses, stated as hypotheses

Under an approximately Gaussian coordinate distribution with a p99-calibrated
scale, the outermost bin opens at a nominal magnitude of
`scale_max * (1 - 1/n_bins)`, which predicts occupancy near **19.8%**, **5.3%**
and **2.4%** for 2, 4 and 8 bins. These are predictions of a model, not
measurements. The sweep reports measured occupancy against the actual
encoder-derived boundary, which is not the nominal one.

## Measured per cell

Saturation occupancy; changed-coordinate rate; region-move distribution;
positive lower-bound coverage; unbounded fraction conditional on having
changed; finite interval widths; localization across coordinate blocks;
encode latency; storage. Generic bounds and norm-aware bounds are reported
separately.

## Comparator

Ordinary uniform scalar quantization at the same bit width, given the same
calibration opportunity over the same development data, with its own metadata
bytes counted.

## Decision rule, as implemented

In [`decide.py`](decide.py), so it is evaluated rather than interpreted.

**Condition roles are declared and used.** `proc`, `threads1`, `batch8` and
`batch128` are near-null: they hold the nominal configuration fixed, and a
configuration that disturbs their codes is rejected whatever else it buys.
`bf16` and `int8` are the declared interventions a configuration has to
resolve.

**The effect threshold is named.** At least **1% of changed coordinates** must
move two or more regions, that being the precondition for any positive lower
bound, and the interval's lower end must clear zero.

**The threshold is a screen, not an estimate.** The 1% is applied to the point
estimate while the interval is only required to exclude zero, so a qualifying
cell has a share that is real and looks large enough to carry forward. It has
**not** been shown to have a true share at or above 1%; that needs the
interval's lower end to clear the threshold, which `decide.py` reports
separately as `qualifies_strict`. Both are reported because a development
screen and a confirmatory bound are different questions, and only the screen
is what this study is entitled to ask.

**Uncertainty is over documents.** Coordinates inside one document are not
independent draws, so every interval is a 95% percentile bootstrap resampling
documents, 2,000 draws, seed 47. Documents are **not** episodes: they come
from one collection run and share its process, machine and session, so these
intervals cover corpus sampling and say nothing about run-to-run variation.

**The rule is reported under both readings** of what to do when the declared
interventions disagree — whether a configuration must qualify on *every*
intervention or on *any* — because on this data they disagree, and choosing
between them after seeing that would be selecting on the outcome.

### Known flaw in the tie-break

Ties break on storage and then on saturation. Saturation and resolution pull
against each other: a wider scale saturates less and resolves less. The
tie-break therefore drives selection toward the widest scale, which is the
least sensitive of the qualifying cells. A confirmatory rule has to break ties
on something that is not in tension with the quantity being secured.

---

## Appendix: the original rule, as written

Kept unedited as the record.

> Prefer the smallest `n_bins` that produces a **non-zero share of
> multi-region moves on the near-null conditions' real interventions**, since
> that is the precondition for any positive lower bound. Break ties on storage,
> then on saturation. Report the tradeoff and any null result; do not select on
> a metric computed over reserved documents.

## Out of scope

No claim about index incompatibility, retrieval degradation, or functional
harm. Dither and group-specific calibration belong to SEM-40 and SEM-41 and
are not duplicated here.
