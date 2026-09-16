# Condition effects relative to repeated calls

This analysis covers the original 13 embedding submissions, before the SFR addition.
It does not redefine ARI or recalculate the published reports.
The source values are the historical [leaderboard submissions](https://github.com/The-SEMQ-Group/ari-leaderboard/tree/main/submissions).

For each condition k, `delta(k) = HER(k) − HER(same)`.
The tables retain the original descriptive classification labels.
“Apparent improvement” means a higher point estimate. “Real effect” means a lower estimate with non-overlapping intervals.
“Indistinguishable” means a lower or equal estimate with overlapping intervals.
These labels are not causal conclusions or formal hypothesis tests.
Higher agreement can reflect a real configuration difference or sampling variation; it is not necessarily measurement error.

## API results

The comparable core is proc, conc, and time. Same is a separate repeated-call measurement.

| agent | same (CI) | proc (CI) → Δ, class | conc (CI) → Δ, class | time (CI) → Δ, class | published ARI |
|---|---|---|---|---|---|
| cohere/embed-v4.0 | 0.133 (0.111–0.154) | 0.146 (0.124–0.169) → +0.013, **higher estimate** (CI overlaps) | 0.130 (0.110–0.152) → −0.003, **intervals overlap** | 0.232 (0.206–0.262) → +0.099, **higher estimate** (CI does NOT overlap) | 0.169333 |
| openai/text-embedding-3-large | 0.859 (0.837–0.881) | 0.862 (0.840–0.883) → +0.003, **higher estimate** (CI overlaps) | 0.851 (0.830–0.872) → −0.008, **intervals overlap** | 0.822 (0.797–0.844) → −0.037, **intervals overlap** | 0.845000 |
| mistral/mistral-embed | 0.734 (0.708–0.759) | 0.753 (0.727–0.778) → +0.019, **higher estimate** (CI overlaps) | 0.791 (0.765–0.816) → +0.057, **higher estimate** (CI does NOT overlap) | 0.554 (0.523–0.587) → −0.180, **lower; intervals disjoint** | 0.699333 |
| voyage/voyage-4-large | 0.571 (0.541–0.600) | 0.615 (0.585–0.644) → +0.044, **higher estimate** (CI overlaps) | 0.209 (0.185–0.233) → −0.362, **lower; intervals disjoint** | 0.676 (0.645–0.706) → +0.105, **higher estimate** (CI does NOT overlap) | 0.500000 |
| gemini/gemini-embedding-001 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) → 0.000, **intervals overlap** | 1.000 (1.000–1.000) → 0.000, **intervals overlap** | 1.000 (1.000–1.000) → 0.000, **intervals overlap** | 1.000000 |

## Local model results

Core agreement was measured as one for these runs. The probe identity does not guarantee agreement across deployment changes.

| agent | same/proc/conc/time | prec HER (CI) → Δ vs. same, class | published ARI |
|---|---|---|---|
| BAAI/bge-large-en-v1.5 | 1.000 (all) | 0.003906 (0.000–0.011719) → −0.996094, **lower; intervals disjoint** | 1.000000 |
| BAAI/bge-m3 | 1.000 (all) | 0.000000 (0.000–0.000000) → −1.000000, **lower; intervals disjoint** | 1.000000 |
| Snowflake/snowflake-arctic-embed-l | 1.000 (all) | 0.003906 (0.000–0.011719) → −0.996094, **lower; intervals disjoint** | 1.000000 |
| intfloat/multilingual-e5-large | 1.000 (all) | 0.003906 (0.000–0.011719) → −0.996094, **lower; intervals disjoint** | 1.000000 |
| mixedbread-ai/mxbai-embed-large-v1 | 1.000 (all) | 0.007812 (0.000–0.019531) → −0.992188, **lower; intervals disjoint** | 1.000000 |
| nomic-ai/nomic-embed-text-v1.5 | 1.000 (all) | 0.007812 (0.000–0.019531) → −0.992188, **lower; intervals disjoint** | 1.000000 |
| sentence-transformers/all-mpnet-base-v2 | 1.000 (all) | 0.035156 (0.015625–0.058594) → −0.964844, **lower; intervals disjoint** | 1.000000 |
| sentence-transformers/all-MiniLM-L6-v2 | 1.000 (all) | 0.093750 (0.058594–0.128906) → −0.906250, **lower; intervals disjoint** | 1.000000 |

## Precision diagnostic

Higher HER means greater code agreement. MiniLM has the highest point estimate in this table, not the lowest.

| rank | agent | prec HER | prec CI |
|---|---|---|---|
| 1 | sentence-transformers/all-MiniLM-L6-v2 | 0.093750 | 0.058594–0.128906 |
| 2 | sentence-transformers/all-mpnet-base-v2 | 0.035156 | 0.015625–0.058594 |
| 3= | mixedbread-ai/mxbai-embed-large-v1 | 0.007812 | 0.000000–0.019531 |
| 3= | nomic-ai/nomic-embed-text-v1.5 | 0.007812 | 0.000000–0.019531 |
| 5= | BAAI/bge-large-en-v1.5 | 0.003906 | 0.000000–0.011719 |
| 5= | Snowflake/snowflake-arctic-embed-l | 0.003906 | 0.000000–0.011719 |
| 5= | intfloat/multilingual-e5-large | 0.003906 | 0.000000–0.011719 |
| 8 | BAAI/bge-m3 | 0.000000 | 0.000000–0.000000 |

## Illustrative aggregate with precision

This noncanonical calculation includes prec. It must not replace the published three-condition ARI.

| rank | agent | ARI-with-prec |
|---|---|---|
| 1 | sentence-transformers/all-MiniLM-L6-v2 | 0.773438 |
| 2 | sentence-transformers/all-mpnet-base-v2 | 0.758789 |
| 3= | mixedbread-ai/mxbai-embed-large-v1 | 0.751953 |
| 3= | nomic-ai/nomic-embed-text-v1.5 | 0.751953 |
| 5= | BAAI/bge-large-en-v1.5 | 0.750977 |
| 5= | Snowflake/snowflake-arctic-embed-l | 0.750977 |
| 5= | intfloat/multilingual-e5-large | 0.750977 |
| 8 | BAAI/bge-m3 | 0.750000 |

## Interpretation

Voyage's concurrency interval and Mistral's time interval are below their respective repeated-call intervals without overlap.
All three OpenAI core intervals overlap its repeated-call interval.
Cohere's time interval is higher than its repeated-call interval without overlap.
These observations distinguish directions and uncertainty; they do not identify backend causes.

Most local-model precision intervals overlap. MiniLM and mpnet have greater agreement under this diagnostic than the lower-valued models.
Some interval boundaries touch at the reported precision, so strict separation should not be inferred from rounded endpoints.
No historical submission in this analysis reports `mach` or `lib`.
An absent condition is unmeasured, not zero drift.

## Use in reports

Show absolute HER, its interval, and the difference from `same` together.
Keep diagnostics separate from the canonical score.
Do not equate overlapping intervals with proof of no effect.
The overlap rule has no multiple-comparison adjustment and is not a paired difference test.
A paired analysis requires per-input evidence beyond these aggregate intervals.
