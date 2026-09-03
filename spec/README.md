# The ARI standard (spec)

The **normative** definition of the Agent Reproducibility Index. Everything here is versioned;
a report is comparable to another only if both target the same versioned artifacts.

| file | what it fixes |
| --- | --- |
| [`ari-canonical-v0.1.md`](ari-canonical-v0.1.md) | The canonical **probe** — SEMQ QBIN n=2, 99th-pct calibration — and why (design rationale). |
| [`ari-bench-v0.1.md`](ari-bench-v0.1.md) | The canonical **input set** — a frozen, hash-pinned slice of real BEIR text ([`../data/`](../data/)). |
| [`condition-set.md`](condition-set.md) | The **environmental conditions** ARI averages over (`same`/`proc`/`mach`/`prec`/`lib`/`conc`/`time`). |
| [`report-schema.json`](report-schema.json) | The JSON **schema** every ARI report must satisfy (validated by the scorer + CI). |
| [`fingerprints-v0.1.csv`](fingerprints-v0.1.csv) | The published **(s, b, κ) fingerprint registry** — one row per model, with coverage status. |

**Status:** v0.1-preview **frozen** — `ARI-Bench-v0.1` (input set) and `ARI-Canonical-v0.1`
(probe + calibration + reference model) are fixed. The fingerprint registry is published and
grows as per-model sweeps run (adding a `κ` does not change the frozen probe). Breaking
changes bump the version.

Licensing: the spec is **CC-BY-4.0** (see the [Licensing](../README.md#licensing) section).
