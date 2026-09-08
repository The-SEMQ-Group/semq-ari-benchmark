# ARI-Bench-v0.1

Status: frozen, v0.1-preview.
This specification defines the embedding input set.
The inputs measure reproducibility and do not constitute a capability benchmark.

## Composition

Frozen file: [ari-bench-v0.1.jsonl](../data/ari-bench-v0.1.jsonl).

| Source | Domain | Items |
| --- | --- | ---: |
| BEIR NFCorpus | Medical | 334 |
| BEIR SciFact | Scientific | 333 |
| BEIR FiQA-2018 | Financial | 333 |

There are 1,000 records. Each text combines title and body, truncated to 512 characters.
The selection uses seed `20260701` and corpora `[nfcorpus, scifact, fiqa]`.
The source documents retain their original licenses; see [data provenance](../data/README.md).

## Content identity

The content hash is:

```text
e9ec8b01c62635dee9fbbbdf8127cde5cac094aba66368ee8de306acc3afbe1d
```

Compute SHA-256 over the ordered UTF-8 records `input_id\0text\n`.
A comparable report must use the same inputs and order.
See the [verification procedure](../data/README.md#verify-the-content).

## Reference model

When the agent exposes embeddings, probe those embeddings directly.
Otherwise, place the probe after the final pre-pooling layer of the published reference model.
The v0.1 reference is `BAAI/bge-large-en-v1.5`.
See [ARI-Canonical-v0.1](ari-canonical-v0.1.md) for calibration.

## Reconstruction

The [builder](../ari/tools/build_ari_bench.py) sorts candidate identifiers and samples without replacement using the fixed seed.
It then sorts the selected records globally by identifier.
Exact reconstruction also requires matching source data and implementation behavior.
Compare the output content hash before accepting a rebuilt selection.
Use the [reconstruction procedure](../data/README.md#rebuild-the-embedding-selection).

## Future versions

A larger set or selection based on distance to quantizer boundaries requires another version and content hash.
At 1,000 independent Bernoulli observations, the maximum standard error of HER is approximately 0.016.
Increasing the set to 5,000 would reduce that bound to approximately 0.007.
These calculations do not include dependence between inputs or deployment conditions.
