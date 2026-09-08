# Frozen input sets

## ARI-Bench (embeddings)

[`ari-bench-v0.1.jsonl`](ari-bench-v0.1.jsonl) — the **frozen** input set every ARI report runs
on. One JSON object per line: `{input_id, source_id, text}`, in fixed order.

- **Size:** 1,000 items. **Composition:** real BEIR text, domain-diverse — NFCorpus (medical),
  SciFact (scientific), FiQA (financial), ~1/3 each. Each item is title+body truncated to
  512 characters.
- **The pin:** `content_hash = e9ec8b01c62635dee9fbbbdf8127cde5cac094aba66368ee8de306acc3afbe1d`
  (SHA-256 over the ordered `input_id\0text\n` records). A submitter confirms they ran the exact
  same inputs by matching this hash. See [`../spec/ari-bench-v0.1.md`](../spec/ari-bench-v0.1.md).
- **Reproduce:** `python ari/tools/build_ari_bench.py --n 1000 --out data/ari-bench-v0.1.jsonl`
  (deterministic — same corpora/seed → identical file and hash).

**Provenance & licensing.** The text is derived from [BEIR](https://github.com/beir-cellar/beir)
corpora (publicly available; see their terms). This repository redistributes the selection,
ordering, and truncation as the frozen input set. The selection/ordering is **CC-BY-4.0**; the
underlying documents retain their original BEIR/source licenses.

## ARI-D-Bench (decoding)

[`arid-bench-v0.1.jsonl`](arid-bench-v0.1.jsonl) — the **frozen** prompt set for the
decoding-layer bench ([`../spec/arid-bench-v0.1.md`](../spec/arid-bench-v0.1.md),
§4). One JSON object per line: `{input_id, bucket, text}`, in fixed order.

- **Size:** 100 prompts, 20 per bucket, in this order: `factual_qa`, `code_generation`,
  `multi_step_reasoning`, `instruction_following` (over a supplied passage), `open_ended_prose`.
  Single-turn, no system prompt, English, 20–78 tokens each (cl100k_base).
- **The pin:** `content_hash = af5b1a2bb35b9fa5f8d4e8f05b85a98f495a3edeb93e95da9c9ced6d783e0d6f`
  (SHA-256 over the ordered `input_id\0text\n` records — same scheme as ARI-Bench, so every
  prefix hash is also valid and a reduced run can prove it measured the first N as `prefix:N`).
- **Verify:**
  ```
  python -c "import json,hashlib;h=hashlib.sha256();[h.update(i['input_id'].encode()+b'\0'+i['text'].encode()+b'\n') for i in map(json.loads,open('data/arid-bench-v0.1.jsonl'))];print(h.hexdigest())"
  ```

**Provenance & licensing.** The prompts are authored for this bench — there is no upstream
corpus and nothing to rebuild from; the file itself is the source, and the hash is the freeze.
The set is **CC-BY-4.0**. The bucket mix is face-validity only (the proposal says so
explicitly); enriching it toward decision-dense prompts is a v0.2 question. The mix is not
sacred; what is sacred is that it freezes.

## ARI-E-Bench (environment / harness)

[`ari-e-bench-v0.1.jsonl`](ari-e-bench-v0.1.jsonl) — the **frozen** task set for the
environment-layer bench ([`../spec/ari-e-bench-v0.1.md`](../spec/ari-e-bench-v0.1.md),
§4). One JSON object per line: `{input_id, instance_id, repo, base_commit, version,
environment_setup_commit, text}`, in fixed order. `text` is the SWE-bench `problem_statement`
— what the agent sees; the other fields identify the instance for the grader.

- **Size:** 200 tasks, a repo-capped slice of **SWE-bench Verified** (500 instances). Each
  repo is capped at 25 before sampling, so django's 231 instances do not dominate: 12 repos,
  no repo above 25. Each instance carries its own held-out test suite (FAIL_TO_PASS /
  PASS_TO_PASS) as the grader; the agent never sees it.
- **The pin:** `content_hash = d44f4975279798ce7271ee8971acafa9920199010d77811a9da4fbfdfaa91b26`
  (SHA-256 over the ordered `input_id\0text\n` records — same scheme as ARI-Bench, so every
  prefix hash is also valid and a reduced run can prove it measured the first N as `prefix:N`).
  The held-out tests are pinned separately, by the dataset revision below, not restated here.
- **Grader source:** `SWE-bench/SWE-bench_Verified` at revision
  `78f471bf655a3137b2e8a75af1501690ec009ec3` — the canonical current dataset, carrying the
  prebuilt eval `image` and `eval_script` the swebench 5.x harness runs (problem statements and
  base/env commits are byte-identical to the older `princeton-nlp` mirror, verified 500/500).
  The grading harness loads each instance and its tests from this exact revision; bumping it
  re-freezes the bench.
- **Reproduce:** `python ari/tools/build_ari_e_bench.py --out data/ari-e-bench-v0.1.jsonl`
  (deterministic — same revision/cap/n/seed → identical file and hash).
- **Verify:**
  ```
  python -c "import json,hashlib;h=hashlib.sha256();[h.update(i['input_id'].encode()+b'\0'+i['text'].encode()+b'\n') for i in map(json.loads,open('data/ari-e-bench-v0.1.jsonl'))];print(h.hexdigest())"
  ```

**Provenance & licensing.** The tasks are derived from
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
(human-validated SWE-bench; see its terms). This repository redistributes the selection and
ordering as the frozen task set; the selection/ordering is **CC-BY-4.0**, the underlying
repositories and problem statements retain their original licenses.
