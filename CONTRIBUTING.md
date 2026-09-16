# Contributing

## Develop and test

Start with the [root setup procedure](README.md#build-and-test).
Use a separate branch for each change. Keep unrelated generated files out of the commit.

1. Change the implementation and its documentation together.
2. Add regression tests for changed behavior.
3. Run `python -m pytest -q -rs` from the repository root.
4. With the SDK installed, run the mock-agent report command in the root README.
5. Run `python -m build` when packaging changes.
6. Describe the behavior change and verification results in the pull request.

Report skipped tests. Public CI can skip SDK-dependent tests because `semq` requires separate access.
Use the [SDK installation procedure](ari/README.md#install-the-canonical-probe) to test those paths when you have access.

## Use the SEMQ operators through the SDK

The SEMQ operators, including QUANT, are proprietary.
The SEMQ Group Inc. distributes them only in the `semq` package.
That package is subject to a commercial license owned by The SEMQ Group Inc. and is patent pending.
This repository does not contain them and must not contain them.

- Do not add an implementation of a SEMQ operator to this repository.
  This includes ports, translations, clean-room rewrites, and numerically equivalent approximations.
- Do not copy or paraphrase code from the `semq` package into this repository.
- Call the operators through the `semq` package. Use [semq_compat.py](ari/semq_compat.py) for version differences.
- This repository contains no substitute for the operators. Code that needs them requires the SDK.
  Tests that need them skip without it.

Pull requests that add an operator implementation are closed without review.
The Apache-2.0 license covers the benchmark harness in this repository.
It does not grant any right to the SEMQ operators or to the `semq` package.

## Write documentation

Use ASD-STE100 Simplified Technical English for explanatory text and procedures.
Use the [official specification](https://www.asd-ste100.org/) for vocabulary and writing rules.

- Use one term for each concept. Define project-specific technical terms at first use.
- Limit procedural sentences to 20 words and descriptive sentences to 25 words.
- Write one instruction per sentence. Use numbered steps for procedures.
- State the working directory, prerequisites, command, output, and success check.
- Use literal descriptions. Remove promotional claims, metaphors, and commentary about the writing process.
- Keep each fact in one authoritative document. Link to it from other documents.
- Keep methods and measured results beside the experiment that produced them.
- Preserve evidence, uncertainty, units, and limitations when shortening text.
- Put temporary tasks in issues or pull requests. Do not add session transcripts or duplicate status summaries.
- Update links when moving or removing a document.

Code identifiers, equations, citations, and required schema terms retain their exact forms.
Technical names include ARI, HER, Hamming distance, SEMQ, QUANT, calibration, logits, and bootstrap confidence interval.
Define additional technical names where they occur. A sentence-length check alone does not establish STE compliance.

## Change a specification

Preserve frozen inputs, hashes, probe parameters, and scoring rules within a specification version.
Version incompatible changes. Separate editorial changes from measurement changes in the pull request.
Record released behavior changes in [CHANGELOG.md](CHANGELOG.md).

## Submit a report

1. Run the [canonical protocol](ari/README.md#capture-a-real-model) with the frozen inputs.
2. Record each condition's environment and audit digest.
3. Check the report against [report-schema.json](spec/report-schema.json).
4. Follow the [leaderboard submission instructions](https://github.com/The-SEMQ-Group/ari-leaderboard/blob/main/submissions/SUBMISSION_FORMAT.md).

Schema validation checks document structure. It does not establish measurement accuracy or signature authenticity.
Report the actual conditions measured. A partial capture cannot represent the complete comparable core.

Contributions use the repository's [license terms](README.md#license-and-citation).
