# Retrieval outcomes study

**Status: protocol draft. No episode collected.**

- Protocol: [PROTOCOL.md](PROTOCOL.md), version v0.1-draft, not yet frozen.
- Pilot plan: [PILOT.md](PILOT.md).
- Linear task: SEM-50.

## What the study is

The study measures whether ARI diagnostics predict two operational events on a
fixed retrieval system built from BEIR SciFact and `all-MiniLM-L6-v2`:

1. A practically meaningful loss in Recall@10 or nDCG@10 against real
   relevance judgements.
2. A restoration of quality after the index is rebuilt.

It compares ARI quantities (HER, coordinate change rate, bit Hamming distance,
displacement bounds) with matched-budget baselines at the same achieved alarm
rate. It separates three conclusions: the representation changed, retrieval
degraded, rebuilding helped. Each has its own evidence and its own permitted
wording (PROTOCOL.md section 10).

The study reuses frozen definitions from three places:

| Definition | Source |
| --- | --- |
| Corpus, queries, judgements, encoder, cache manifest, retrieval metrics, paired bootstrap | [regime-discrimination](../regime-discrimination/RESULTS.md) |
| Alarm unit, one-sided bound, degenerate null, matched budgets, episode split | [matched-budget-detectors/PROTOCOL.md](../matched-budget-detectors/PROTOCOL.md) |
| Scorer definitions and denominators | `ari/code_metrics.py`, `ari/metrics.py`, `ari/change_profile.py`, `ari/bound_quality.py` |

## What the study is not

- It is not a measurement of hosted API encoders. The encoder runs locally.
- It does not measure the `time` or `lib` axes of
  [the condition set](../../spec/condition-set.md).
- It does not establish that any real serving change harms retrieval. The prior
  experiment found no such loss on SciFact. A repeat of that null is an
  expected outcome and is reported as such.
- It does not compare probes across families. Results at `n_bins = 8` and
  `n_bins = 2` are reported separately.
- Synthetic transformations (noise, rotations) are diagnostics of what the
  scorers measure. They are labelled `synthetic` in every table and are never
  summed with real interventions.

## Relation to SEM-39, SEM-40 and SEM-41

SEM-39, SEM-40 and SEM-41 restrict the scope of claims in existing documents.
This study is separate from those tasks. It does not remove or relax any of
those restrictions, and its protocol does not depend on them. If the
confirmatory run produces a supported operational claim in the wording of
PROTOCOL.md section 10, any change to the restricted documents is a separate
change with its own review. Until then the restrictions stand as written.

## Order of work

1. Read PROTOCOL.md and resolve the open decisions in its section 13.
2. Run the pilot in PILOT.md. Record timings and stop-rule outcomes.
3. Freeze PROTOCOL.md as v1.0.
4. Collect confirmatory episodes. Write RESULTS.md in this directory.

No experiment code exists yet for this study. Code is written after the
protocol is frozen and lives in this directory.
