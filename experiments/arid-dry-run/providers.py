"""Direct-HTTPS subjects for the ARI-D dry run (proposal §13, rollout item 3).

No SDKs and no router on purpose: the measured call must be the call we
describe. `same` vs `proc` is a statement about connection lifecycle, so the
harness owns the connection object (`http.client.HTTPSConnection`) instead of
delegating it to a pooled client that reuses sockets on its own schedule.

Three subjects (§3 classes, plus one exploratory):

- ``openai``     — vendor row, chat endpoint, seed + system_fingerprint +
                   top-20 logprobs: the fullest metadata surface, so the dry
                   run exercises the most harness machinery.
- ``together``   — rehosted row, raw ``/v1/completions`` with the model's
                   official prompt template applied by us (§7: reference
                   detectors only run against raw endpoints; the dry run
                   produces transcripts the fp32 reference pass can reuse).
- ``salesforce`` — exploratory ``platform`` class (not in the v0.1 panel): a
                   proxy door in front of vendor models (Einstein Trust
                   Layer). Serves the same underlying model as our vendor
                   subject (gpt-4o-mini), so direct-vs-platform is measurable
                   on the frozen set. No seed, no logprobs, no fingerprint —
                   sequence detectors only, and the model snapshot cannot be
                   pinned through this door; that reduced surface is itself
                   part of the platform-class reading.

Keys come from ``OPENAI_API_KEY`` / ``TOGETHER_API_KEY`` env vars for the dry
run; the automated panel moves them to Secrets Manager. Salesforce needs
``SF_MY_DOMAIN`` (https://....my.salesforce.com), ``SF_CONSUMER_KEY`` and
``SF_CONSUMER_SECRET`` from an External Client App with the client-credentials
flow enabled; the bearer token is fetched and refreshed by the harness.
"""
from __future__ import annotations

import gzip
import http.client
import json
import time
import urllib.parse
from dataclasses import dataclass, field

MAX_TOKENS = 128
SEED = 20260831  # recorded in every request body that accepts it (§5)
RETRY_STATUSES = {500, 502, 503, 504}
MAX_ATTEMPTS = 5
MAX_429_ATTEMPTS = 25   # throttling budget: a rate-capped tier waits, it does not fail

# The official Llama-3 instruct template, applied by us (§7): single-turn,
# no system prompt, exactly as the reference generation will apply it.
LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


@dataclass
class Subject:
    name: str
    klass: str            # 'vendor' | 'rehosted' | 'platform'
    host: str
    path: str
    model: str
    key_env: str
    endpoint_kind: str    # 'chat' | 'completions' | 'sfgen'
    extra_headers: dict = field(default_factory=dict)
    supports_seed: bool = True
    seed_param: str = "seed"          # Mistral calls it random_seed
    supports_logprobs: bool = True    # Mistral's chat API exposes none
    auth_style: str = "bearer"        # Google wants x-goog-api-key instead

    def build_body(self, prompt_text: str) -> dict:
        if self.endpoint_kind == "sfgen":
            # Models API generations request; verified live in the smoke run
            # (the public reference does not render the schema server-side)
            return {
                "prompt": prompt_text,
                "parameters": {"temperature": 0, "maxTokens": MAX_TOKENS},
            }
        if self.endpoint_kind == "gemini":
            cfg = {"temperature": 0, "topP": 1, "maxOutputTokens": MAX_TOKENS,
                   # visible tokens only: default thinking spends the shared
                   # budget on hidden reasoning and starves the completion
                   "thinkingConfig": {"thinkingLevel": "MINIMAL"}}
            if self.supports_seed:
                cfg["seed"] = SEED
            if self.supports_logprobs:
                cfg["responseLogprobs"] = True
                cfg["logprobs"] = 5
            return {"contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
                    "generationConfig": cfg}
        if self.endpoint_kind == "chat":
            body = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": 0,
                "top_p": 1,
                "max_tokens": MAX_TOKENS,
            }
            if self.supports_logprobs:
                body["logprobs"] = True
                body["top_logprobs"] = 20
            if self.supports_seed:
                body[self.seed_param] = SEED
            return body
        return {
            "model": self.model,
            "prompt": LLAMA3_TEMPLATE.format(prompt=prompt_text),
            "temperature": 0,
            "top_p": 1,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "logprobs": 5,  # Together's live maximum; the §10 top-20 claim was wrong
            "echo": False,
            "stop": ["<|eot_id|>"],
        }

    def parse(self, body: dict) -> dict:
        """Pull the comparison surface out of a response; the raw body is
        retained separately as the evidence (§5: captured whole)."""
        if self.endpoint_kind == "sfgen":
            # verified live 2026-08-31: finish_reason sits under generation.parameters;
            # the provider passthrough (system_fingerprint, underlying model snapshot)
            # sits under the TOP-LEVEL parameters block
            gen = body.get("generation") or {}
            gparams = gen.get("parameters") or {}
            tparams = body.get("parameters") or {}
            return {
                "text": gen.get("generatedText") or "",
                "finish_reason": gparams.get("finish_reason") or gparams.get("finishReason"),
                "system_fingerprint": tparams.get("system_fingerprint"),
                "upstream_model": tparams.get("model"),
                "logprobs": [],
            }
        if self.endpoint_kind == "gemini":
            cand = (body.get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            text = "".join(p.get("text", "") for p in parts)
            steps = (cand.get("logprobsResult") or {}).get("topCandidates") or []
            chosen = (cand.get("logprobsResult") or {}).get("chosenCandidates") or []
            logprobs = []
            for i, st in enumerate(steps):
                top = [[c.get("token"), c.get("logProbability")]
                       for c in (st.get("candidates") or [])]
                tok = chosen[i].get("token") if i < len(chosen) else (top[0][0] if top else None)
                lp = chosen[i].get("logProbability") if i < len(chosen) else None
                logprobs.append({"token": tok, "logprob": lp, "top": top})
            return {
                "text": text,
                "finish_reason": cand.get("finishReason"),
                "system_fingerprint": None,
                "upstream_model": body.get("modelVersion"),
                "logprobs": logprobs,
            }
        choice = body["choices"][0]
        if self.endpoint_kind == "chat":
            text = choice["message"]["content"] or ""
            steps = (choice.get("logprobs") or {}).get("content") or []
            logprobs = [
                {
                    "token": s["token"],
                    "logprob": s["logprob"],
                    "top": [[t["token"], t["logprob"]] for t in s.get("top_logprobs", [])],
                }
                for s in steps
            ]
        else:
            text = choice.get("text") or ""
            lp = choice.get("logprobs") or {}
            tokens = lp.get("tokens") or []
            tlps = lp.get("token_logprobs") or []
            tops = lp.get("top_logprobs") or []
            logprobs = [
                {
                    "token": tokens[i],
                    "logprob": tlps[i] if i < len(tlps) else None,
                    "top": sorted(
                        ([t, v] for t, v in (tops[i] or {}).items()),
                        key=lambda x: -x[1],
                    ) if i < len(tops) and tops[i] else [],
                }
                for i in range(len(tokens))
            ]
        return {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "system_fingerprint": body.get("system_fingerprint"),
            "logprobs": logprobs,
        }


SUBJECTS = {
    "openai": Subject(
        name="openai", klass="vendor",
        host="api.openai.com", path="/v1/chat/completions",
        model="gpt-4o-mini", key_env="OPENAI_API_KEY", endpoint_kind="chat",
    ),
    "together": Subject(
        name="together", klass="rehosted",
        host="api.together.xyz", path="/v1/completions",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        key_env="TOGETHER_API_KEY", endpoint_kind="completions",
    ),
    "mistral": Subject(
        name="mistral", klass="vendor",
        host="api.mistral.ai", path="/v1/chat/completions",
        model="mistral-medium-2604", key_env="MISTRAL_API_KEY",
        endpoint_kind="chat", seed_param="random_seed", supports_logprobs=False,
    ),
    "gemini": Subject(
        name="gemini", klass="vendor",
        host="generativelanguage.googleapis.com",
        path="/v1beta/models/{model}:generateContent",
        model="gemini-3.6-flash", key_env="GEMINI_API_KEY",
        endpoint_kind="gemini", auth_style="goog", supports_logprobs=False,
    ),
    "salesforce": Subject(
        name="salesforce", klass="platform",
        host="api.salesforce.com",
        path="/einstein/platform/v1/models/{model}/generations",
        model="sfdc_ai__DefaultOpenAIGPT4OmniMini",
        key_env="SF_CONSUMER_KEY",  # two-step auth; see SalesforceAuth
        endpoint_kind="sfgen",
        extra_headers={
            "x-sfdc-app-context": "EinsteinGPT",
            "x-client-feature-id": "ai-platform-models-connected-app",
        },
        supports_seed=False,
    ),
}


class StaticToken:
    """A plain API key: never expires, never refreshes."""

    def __init__(self, token: str):
        self._token = token

    def token(self) -> str:
        return self._token

    def refresh(self) -> bool:
        return False


class SalesforceAuth:
    """Client-credentials bearer token from the org's OAuth endpoint.

    The token expires mid-run at our pacing, so the Caller invalidates and
    re-fetches on 401 instead of treating it as a failure.
    """

    def __init__(self, my_domain: str, consumer_key: str, consumer_secret: str):
        self._host = my_domain.replace("https://", "").rstrip("/")
        self._key = consumer_key
        self._secret = consumer_secret
        self._token: str | None = None

    def token(self) -> str:
        if self._token is None:
            body = urllib.parse.urlencode({
                "grant_type": "client_credentials",
                "client_id": self._key,
                "client_secret": self._secret,
            }).encode()
            conn = http.client.HTTPSConnection(self._host, timeout=60)
            try:
                conn.request("POST", "/services/oauth2/token", body=body,
                             headers={"Content-Type": "application/x-www-form-urlencoded"})
                resp = conn.getresponse()
                raw = resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"salesforce oauth: HTTP {resp.status}: {raw[:300]!r}")
                self._token = json.loads(raw)["access_token"]
            finally:
                conn.close()
        return self._token

    def refresh(self) -> bool:
        self._token = None
        return True


@dataclass
class Caller:
    """One HTTPS connection with explicit lifecycle.

    ``same`` reuses a single Caller sequentially; ``proc`` builds a fresh one
    per call; ``conc`` gives each worker thread its own. Reconnects (server
    closed a kept-alive socket) are counted, not hidden. ``auth`` is either a
    StaticToken (plain API key) or a refreshing provider (Salesforce).
    """
    subject: Subject
    auth: object
    reconnects: int = 0
    _conn: http.client.HTTPSConnection | None = field(default=None, repr=False)

    def _connect(self) -> http.client.HTTPSConnection:
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(self.subject.host, timeout=180)
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def call(self, prompt_text: str) -> dict:
        payload = json.dumps(self.subject.build_body(prompt_text)).encode()
        path = self.subject.path.format(model=self.subject.model)
        last_err = None
        attempts_429 = 0
        attempt = 0
        while attempt < MAX_ATTEMPTS:
            attempt += 1
            auth_hdr = ({"x-goog-api-key": self.auth.token()}
                        if self.subject.auth_style == "goog"
                        else {"Authorization": f"Bearer {self.auth.token()}"})
            headers = {
                **auth_hdr,
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip",
                **self.subject.extra_headers,
            }
            t0 = time.monotonic()
            try:
                conn = self._connect()
                conn.request("POST", path, body=payload, headers=headers)
                resp = conn.getresponse()
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                status = resp.status
                resp_headers = {k.lower(): v for k, v in resp.getheaders()
                                if k.lower() not in ("set-cookie",)}
            except (http.client.HTTPException, OSError) as e:
                self.close()
                self.reconnects += 1
                last_err = repr(e)
                time.sleep(min(2 ** attempt, 30))
                continue
            if status == 401 and self.auth.refresh():
                last_err = "HTTP 401 (token expired, refreshed)"
                continue
            if status == 429:
                # Throttling is expected behavior at rate-capped tiers, not
                # failure: it gets its own, much larger patience budget so a
                # 50-RPM tier can still complete a concurrent condition (the
                # achieved in-flight and wall time record the throttling).
                attempts_429 += 1
                if attempts_429 > MAX_429_ATTEMPTS:
                    raise RuntimeError(f"{self.subject.name}: still rate-limited after "
                                       f"{MAX_429_ATTEMPTS} waits")
                retry_after = resp_headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after
                           else min(2 ** min(attempts_429, 6), 60))
                attempt -= 1
                continue
            if status in RETRY_STATUSES:
                last_err = f"HTTP {status}"
                retry_after = resp_headers.get("retry-after")
                time.sleep(float(retry_after) if retry_after else min(2 ** attempt, 30))
                continue
            if status != 200:
                raise RuntimeError(f"{self.subject.name}: HTTP {status}: {raw[:400]!r}")
            body = json.loads(raw)
            return {
                "ok": True,
                "attempts": attempt,
                "waits_429": attempts_429,
                "latency_s": round(time.monotonic() - t0, 3),
                "status": status,
                "headers": resp_headers,
                "parsed": self.subject.parse(body),
                "raw": body,
            }
        raise RuntimeError(f"{self.subject.name}: gave up after {MAX_ATTEMPTS} attempts ({last_err})")
