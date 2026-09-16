# ARI-D dry run (rollout item 3)

One vendor subject (`openai`, chat endpoint — fullest metadata surface: seed,
`system_fingerprint`, top-20 logprobs), one rehosted subject (`together`,
raw `/v1/completions`, official Llama-3 template applied by us), and one
**exploratory `platform` subject** (`salesforce`, Models API — the Einstein
Trust Layer in front of the same gpt-4o-mini our vendor subject measures
directly, so direct-vs-platform is one comparison on one frozen set; not a
v0.1 panel row). The dry run's job is to fix the harness and settle the three
calibrations of proposal §13 item 3 before any paid panel run:

1. **burst sweep** — `conc` at {16, 64, 128} on the rehosted row; freezes the
   panel-wide burst.
2. **length confounder** — Spearman ρ between pair byte length and normalized
   `first_divergence`, per bucket; if length dominates, the detector is
   redesigned.
3. **set discrimination** — `same` floor per bucket + top-2 logprob margin
   distribution; saturation triggers the §4 decision-dense amendment
   (re-freeze, new hash) before the panel.

## Run

```
export OPENAI_API_KEY=...   TOGETHER_API_KEY=...
python run_dry.py --subject openai   --condition same
python run_dry.py --subject openai   --condition proc
python run_dry.py --subject openai   --condition conc --burst 64
python run_dry.py --subject together --condition same
python run_dry.py --subject together --condition proc
python run_dry.py --subject together --condition sweep   # conc at 16/64/128
python analyze_dry.py
```

`--limit 10` smoke-tests on the first ten prompts (`prefix:10`, recorded in the
manifest, not comparable). The prompt set is verified against the frozen pin
before any call. `time` is not runnable in a dry run by definition (>24 h gap);
re-running `same` on a later day and pairing the two transcripts is the
follow-up, not part of this pass.

### Salesforce (platform class)

```
export SF_MY_DOMAIN=https://{org}.my.salesforce.com
export SF_CONSUMER_KEY=...  SF_CONSUMER_SECRET=...
python run_dry.py --subject salesforce --condition same --limit 3   # schema smoke
python run_dry.py --subject salesforce --condition same --pace 8
python run_dry.py --subject salesforce --condition proc --pace 8
```

Needs an External Client App with OAuth scopes `api`, `refresh_token
offline_access`, `sfap_api` and the client-credentials flow enabled; the
harness fetches the bearer token and refreshes it on 401. Developer/trial
orgs are rate-capped in the hundreds-per-hour class, hence `--pace` (seconds
between sequential calls) and no `conc` for this subject until a production
org exists. No seed, no logprobs, no fingerprint through this door, and the
model snapshot cannot be pinned — the reduced surface is recorded as part of
the platform-class reading, not worked around.

## Protocol notes

- **Direct HTTPS, stdlib only.** No SDKs, no router: `same` vs `proc` is a
  claim about connection lifecycle, so the harness owns the socket
  (`http.client.HTTPSConnection`), reuses it for `same`, builds a fresh one per
  call for `proc`, and gives each `conc` worker its own. Reconnects are
  counted in the manifest.
- **Evidence first.** Every call's raw response body is retained in the
  gzipped transcript (`results/transcripts/`, local — bound by digest in
  `results/manifest.json`, committed). Analysis is pure over transcripts; any
  detector defined later runs against the same evidence.
- Keys via env vars for the dry run; the automated panel moves them to
  Secrets Manager. Measured calls never traverse the LiteLLM router.
