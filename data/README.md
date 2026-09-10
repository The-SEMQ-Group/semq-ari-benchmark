# Frozen input sets

Run commands from the repository root after the [development setup](../README.md#build-and-test).

| File | Records | Format | Source |
| --- | ---: | --- | --- |
| [ari-bench-v0.1.jsonl](ari-bench-v0.1.jsonl) | 1,000 | `input_id`, `source_id`, `text` | BEIR NFCorpus, SciFact, and FiQA. |
| [arid-bench-v0.1.jsonl](arid-bench-v0.1.jsonl) | 100 | `input_id`, `bucket`, `text` | Prompts authored for this benchmark. |

The embedding set uses 334 NFCorpus, 333 SciFact, and 333 FiQA records.
Each text combines the title and body, truncated to 512 characters.
The decoding set contains 20 prompts in each of five categories.
These categories are factual questions, code generation, multistep reasoning, instruction following, and open-ended prose.

## Verify the content

Each content hash covers ordered `input_id\0text\n` records encoded as UTF-8.
It is not the SHA-256 digest of the JSONL file bytes.

```bash
python - <<'PYTHON'
from pathlib import Path
from ari.inputs import load_ari_bench
for path in sorted(Path("data").glob("*-bench-v0.1.jsonl")):
    inputs = load_ari_bench(path)
    print(path, len(inputs), inputs.content_hash)
PYTHON
```

Expected values:

| Set | Content hash |
| --- | --- |
| ARI-Bench-v0.1 | `e9ec8b01c62635dee9fbbbdf8127cde5cac094aba66368ee8de306acc3afbe1d` |
| ARI-D-Bench-v0.1 | `af5b1a2bb35b9fa5f8d4e8f05b85a98f495a3edeb93e95da9c9ced6d783e0d6f` |

## Rebuild the embedding selection

```bash
python -m pip install -e ".[data]"
python ari/tools/build_ari_bench.py --n 1000 --out /tmp/ari-bench-rebuilt.jsonl
```

The command downloads the source corpora. Compare the rebuilt content hash with the frozen value before using the output.
Do not overwrite the frozen file to resolve a mismatch.
The decoding file is authored source data and has no dataset reconstruction step.

## Provenance and licenses

Embedding source text comes from [BEIR](https://github.com/beir-cellar/beir).
The selection and ordering use CC BY 4.0. Underlying documents retain their source licenses.
The authored decoding prompts use CC BY 4.0.
See the [embedding specification](../spec/ari-bench-v0.1.md) and [decoding specification](../spec/arid-bench-v0.1.md) for the frozen contracts.
