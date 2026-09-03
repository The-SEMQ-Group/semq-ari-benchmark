# ARI — Agent Reproducibility Index

**The missing instrument for measuring reproducibility in AI agents.**

Regulatory frameworks (EU AI Act Art. 15, NIST AI RMF Manage 4.1, US Treasury AI RMF)
name reproducibility among the required properties of AI systems — none define how to
measure it. Capability leaderboards (MTEB, HELM, AgentBench) measure *what a model can do* —
none measure *whether it does the same thing twice*. That gap is what ARI fills.

**ARI is that measurement.** It is a third-party-verifiable index of how stably an
agent's internal representation survives the operational noise of real deployment —
BLAS non-determinism, fp16/fp32 mixing, library bumps, hardware swaps, cross-region
serving, cross-process replay. It is built on **SEMQ**, a deterministic quantizer whose
invariance follows from the form of the operator rather than from how it was trained —
which is what lets a third party verify a reading without trusting the instrument
([probe choice and validation](docs/analysis/probe-validation.md)).

> This repository is the instrument, its specification, its evidence base, and a public
> leaderboard — everything needed to measure agent reproducibility and to reproduce the
> measurement independently.

---

## What's in here

| Path | What it is |
| --- | --- |
| [`spec/`](spec/) | The **normative standard** — canonical probe, condition set, report JSON schema (versioned). |
| [`data/`](data/) | The **frozen input set** — ARI-Bench-v0.1 (1,000 real BEIR items, hash-pinned). |
| [`ari-leaderboard`](https://github.com/The-SEMQ-Group/ari-leaderboard) | The **leaderboard** — board data, scoring gate, submission flow, and the static Space — in its own repo, which pins this one as a submodule. |
| [`experiments/`](experiments/) | The **evidence** — instrument-validation experiments + the deployed-agent panel, with data. |
| [`ari/`](ari/) | The **reference implementation** — probe binding, agents, metrics, report builder, tools. |
| [`docs/`](docs/) | The **findings**, **methodology**, the [retraction ledger](docs/retractions.md) and the analysis records. The peer-reviewed reference is the ARI workshop paper (NeurIPS 2026, Machine Learning for Systems). |

---

## The one-paragraph summary

An AI agent is a deep stack of floating-point operations whose reproducibility is
*empirically unmeasured at the system level*. FP32 + cosine retrieval acts as a low-pass
filter: micro-perturbations (10⁻⁷–10⁻⁴ per component) are absorbed before they change a
top-K ranking, so genuine non-determinism stays invisible. **SEMQ is a high-pass
instrument** — it turns those same perturbations into a deterministic, monotone Hamming
signal. ARI aggregates that signal into a single number per agent, per environment,
auditable by hash without access to the vendor's infrastructure.

## The first real numbers (ARI leaderboard, v0.1-preview)

We ran ARI on **14 embedding models** — five widely-used commercial APIs and nine
widely-used open models from Hugging Face — on 1,000 real BEIR items. Full table + method:
[`experiments/deployed-agent-panel/RESULTS.md`](experiments/deployed-agent-panel/RESULTS.md).

| agent | ARI | 95% CI |
| --- | --- | --- |
| `gemini/gemini-embedding-001` (API) | **1.000** | [1.000, 1.000] |
| 9 self-hosted open models (bge, e5, MiniLM, nomic, SFR, …) | **1.000** (core) | — |
| `openai/text-embedding-3-large` | 0.845 | [0.822, 0.866] |
| `mistral/mistral-embed` | 0.699 | [0.672, 0.727] |
| `voyage/voyage-4-large` (Anthropic's recommended) | 0.500 | [0.472, 0.528] |
| `cohere/embed-v4.0` | 0.169 | [0.147, 0.194] |

**The central result is the commercial-API panel** — the first *standardized, comparable,
third-party-verifiable* reproducibility measurement across five providers. Reproducibility is a
real, provider-specific axis: the paid APIs span ARI 0.15 → 1.00. Gemini is bit-reproducible
(verified not a caching artifact); Cohere `embed-v4` drifts ~83% of the time, `voyage-4-large`
50%, OpenAI ~15%. That these APIs are non-deterministic is *known* — OpenAI's especially, documented
by the community [since 2023](https://github.com/openai/openai-python/issues/868). What's new is
making it a comparable number: they are **black boxes** (you can't set their precision or
determinism), so a discrete-attractor instrument on a frozen benchmark is the only way to get a
decision-grade, cross-provider figure. It's an MTEB-style contribution — not discovering the
phenomenon, but making it the reference measurement.

Self-hosted models are a useful contrast (and an instrument validation): perfect on CPU, but
reproducibility is **config-dependent** — bf16 vs fp32 is a precision cliff, and enabling **TF32**
on GPU breaks cross-process reproducibility (HER 0.65–0.88, invisible to cosine). That GPU/TF32
mechanism is *known* (and off by default in modern PyTorch); we measure it precisely, reconciling
with prior work ([ReproRAG](https://arxiv.org/abs/2509.18869)) — see
[GPU determinism results](experiments/gpu-determinism/RESULTS.md). The novelty is the instrument
and the API numbers, not the GPU phenomenon.

## The core result, stated honestly

SEMQ and retrieval recall are **different classes of instrument**. Recall@10 is a binary
threshold detector that reads 1.0 across the entire regime where production drift lives.
SEMQ is a continuous-resolution instrument with a universal power law:

```
Hamming ≈ κ · √(2·dim/π) · σ        slope b = 0.979 ± 0.028 across 13 models
```

The advantage is not "N× more sensitive" — a slope ratio is ill-posed when the baseline is
pinned at 1.0. It is **a different category of measurement**: a deterministic probe whose behavior
follows from a declared rule plus one calibration scalar — cheap for a third party to
verify ([probe-validation](docs/analysis/probe-validation.md) weighs the alternatives:
independently seeded quantizers fail the `same` control outright; a frozen-codebook
probe can work, but must be audited codebook by codebook) — and a universal response
law with a per-model fingerprint `(s, b, κ)`. See the
[Drift Sensitivity Benchmark](experiments/drift-sensitivity/) and [`docs/analysis/probe-validation.md`](docs/analysis/probe-validation.md).

---

## Status

- **Evidence (instrument):** the drift-sensitivity, probe-purity, and cross-process
  experiments are done and consolidated with reproduce guides and machine-readable data.
- **Leaderboard:** **14 representation rows** on the frozen ARI-Bench-v0.1 (5 APIs + 9
  self-hosted) plus the first decoding row, each backed by a report with audit digests and
  pinned to its frozen input set by content hash. **Every row is attested**: signed with
  the registered Ed25519 key (AWS KMS custody, [`spec/signers.json`](spec/signers.json))
  and re-verifiable by a third party with `ari/verify_report.py`. See
  [`experiments/deployed-agent-panel/RESULTS.md`](experiments/deployed-agent-panel/RESULTS.md).
- **Inputs:** ARI-Bench-v0.1 **frozen** — 1,000 real BEIR items, hash-pinned and committed.
- **Spec:** ARI-Canonical-v0.1 **frozen** (probe + calibration + reference model) and the
  **`(s, b, κ)` fingerprint registry** published — full `(s, b, κ)` for 11 of the 13 panel
  models (+2 sweep-only decoders), `s` for all 13 original panel models; 2 (mpnet,
  MiniLM) are `floor_limited`.
  See [`spec/`](spec/).
- **Conditions:** the **comparable core `{proc, conc, time}` is complete for every model** —
  APIs and self-hosted alike. (`conc` surfaced a real finding: Voyage drifts ~3× under
  concurrent load, dropping it to 0.500; Gemini stays 1.000 even under a 64-way burst.) The
  Hugging Face Space app is built (local), not yet published.
- **In flight:** the `semq` SDK to public PyPI; publishing the HF Space.
- **Not published yet.** This repo is written to be public-ready and iterated in the open.

The API headline ARI averages the full comparable core `{proc, conc, time}`. See RESULTS.md for
the honest caveats.

---

## Quickstart

```bash
pip install -e ".[scoring]"          # the reference package + the scorer's deps

# produce a report for an agent (needs the SEMQ probe + the agent's SDK/key)
python ari/tools/run_report.py --agent openai --inputs data/ari-bench-v0.1.jsonl --submit
```

The SEMQ QBIN probe (`semq`) is the SDK that computes the codes — public on PyPI once the SDK
ships. The scorer and the board live in the
[ari-leaderboard](https://github.com/The-SEMQ-Group/ari-leaderboard) repo; to **submit** an
agent, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Licensing

- **Code** (`ari/`) — [Apache-2.0](LICENSE).
- **Spec** (`spec/`) and **input set** (`data/`) — CC-BY-4.0. The ARI-Bench text derives from
  BEIR corpora, which retain their original licenses (see [`data/README.md`](data/README.md)).

Cite ARI via [`CITATION.cff`](CITATION.cff).
