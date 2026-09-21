# Retrieval outcomes study

**Status: protocol draft. No episode collected.**

- Protocol: [PROTOCOL.md](PROTOCOL.md), version v0.1-draft, not yet frozen.
- Pilot plan: [PILOT.md](PILOT.md).
- Linear task: SEM-50.

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

## Order of work

1. Read PROTOCOL.md and resolve the open decisions in its section 13.
2. Run the pilot in PILOT.md. Record timings and stop-rule outcomes.
3. Freeze PROTOCOL.md as v1.0.
4. Collect confirmatory episodes. Write RESULTS.md in this directory.

No experiment code exists yet for this study. Code is written after the
protocol is frozen and lives in this directory.
