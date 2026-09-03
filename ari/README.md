# ARI Harness

Runs the ARI protocol on an agent under the canonical condition set and emits a signed
[ARI report](../spec/report-schema.json) that the leaderboard scorer (in the
[ari-leaderboard](https://github.com/The-SEMQ-Group/ari-leaderboard) repo) validates and appends.

> **Status: implemented.** Orchestration, metrics, and report assembly run **end-to-end** on
> the mock agent (no SDK/credentials) *and* on the real **`semq` probe** (binding verified
> against `semq==1.2.0`: QBIN n=2 calibrate → bit-packed codes, discrete-attractor `same`=1.0
> confirmed). The API agents and capture tools ([`tools/`](tools/)) produced all 13 leaderboard
> rows. `ari/run.py` is the mock end-to-end demo; the real captures use the tools in `tools/`.

## Architecture — two stages, so the panel fans out across instances

```
stage 1  encode      inputs ──(agent)──> embeddings ──(SEMQ QBIN probe)──> codes
         (per env)   run wherever the environment lives: a given instance / precision /
                     library / region / API session. Emits a code bundle per condition.

stage 2  aggregate   diff each condition's codes vs the baseline (`same`) ──> HER, H̄,
                     bootstrap CIs ──> assemble the signed ARI report JSON.
```

This split is what makes the real multi-instance panel work: `mach` means "run stage 1 on a
second instance", `prec` means "run stage 1 in fp16", etc. — then stage 2 collects the
bundles and diffs them. The seams are `run.encode_condition` (stage 1) and
`run.aggregate_report` (stage 2).

## Modules

| module | role |
| --- | --- |
| `probe.py` | The canonical SEMQ QBIN probe. Binds to the `semq` SDK; falls back to a deterministic reference mock when the SDK is absent. |
| `agents.py` | `Agent` backends — `MockAgent` (deterministic, for tests), `SentenceTransformerAgent` (self-hosted), and `OpenAIAgent`/`VoyageAgent`/`CohereAgent` (API stubs). |
| `inputs.py` | ARI-Bench loader + content-hash pin. |
| `metrics.py` | per-input HE / Hamming → HER / H̄ with bootstrap CIs. |
| `report.py` | assemble the report dict (ARI over the *present* averaged conditions), compute audit hashes, write JSON. |
| `run.py` | orchestrator + CLI; `run_mock_panel()` is the end-to-end mock path. |

## Run it

```bash
# end-to-end on the mock agent, then validate with the leaderboard scorer
python -m ari.run --out report.json --validate

# with the frozen ARI-Bench slice (JSONL of {input_id, text})
python -m ari.run --inputs data/ari-bench-v0.1.jsonl --out report.json --validate
```

Install the probe from the SEMQ SDK to use `--probe-backend semq` (private package index
today; public PyPI once the SDK ships).

```bash
python ../tests/test_end_to_end.py     # smoke test: mock report is schema-valid + scorer-passing
```

## What's stubbed (integration TODOs)

- ~~**`semq` probe binding**~~ — **done** (`probe._SemqProbe`, verified against `semq==1.2.0`).
- ~~**OpenAI client**~~ — **wired** (`agents.OpenAIAgent`: pinned model/dimensions,
  one-input-per-request, `encode_fresh` for the `proc` condition; reads `OPENAI_API_KEY`).
  **Voyage / Cohere** clients still to wire.
- **API `proc` realisation** — fresh-context resampling implemented as a pilot
  (`tools/proc_pilot.py`), validated offline; run it against OpenAI on AWS to freeze K /
  window and confirm the levers (see [deployed-agent-panel](../experiments/deployed-agent-panel/README.md)).
- **Real condition runners** — stage-1 execution on distinct AWS instances / precisions /
  library envs (the mock simulates these).
