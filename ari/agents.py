"""Agents under test — the embedding backends the ARI harness probes.

An Agent maps a list of input texts to an (n, d) float embedding matrix. Two classes:

- **Self-hosted** (we control the environment): `SentenceTransformerAgent`. The condition
  set (proc/mach/prec/lib/...) is realised *externally* by running the encode step on
  different instances / precisions / library versions, then aggregating the code bundles.
- **API** (black-box): `OpenAIAgent`, `VoyageAgent`, `CohereAgent`. We cannot control the
  provider's process/machine, so `proc` is realised as fresh-context resampling and the
  score is a lower bound on provider drift (see ../../benchmark/deployed-agent-panel).

`MockAgent` is a deterministic stand-in for end-to-end testing: it derives embeddings from a
seeded hash of each text and can simulate per-condition drift, so the pipeline runs without
credentials or the SDK.
"""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np


class Agent(Protocol):
    agent_id: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class MockAgent:
    """Deterministic embeddings from a seeded per-text RNG. Optional `drift` maps a
    condition name to a σ that simulates the drift a real environment change would cause
    (used only for testing the harness end-to-end)."""

    def __init__(self, dim: int = 256, drift: dict[str, float] | None = None):
        self.agent_id = "mock/deterministic-agent"
        self.dim = dim
        self._drift = drift or {}

    def _base(self, texts: list[str]) -> np.ndarray:
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "big")
            v = np.random.default_rng(seed).standard_normal(self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._base(texts)

    def encode_batched(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        return self._base(texts)  # mock: batch composition changes nothing → batch HER = 1.0

    def encode_under(self, texts: list[str], condition: str) -> np.ndarray:
        """Simulate encoding under a condition: base embedding + calibrated noise.
        `same` gets zero noise (discrete-attractor control ⇒ HER = 1.0 exactly)."""
        X = self._base(texts)
        sigma = 0.0 if condition == "same" else self._drift.get(condition, 0.0)
        if sigma == 0.0:
            return X
        # deterministic per-condition noise so runs are reproducible
        seed = int.from_bytes(hashlib.sha256(condition.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return X + rng.normal(0.0, sigma, size=X.shape)


class SentenceTransformerAgent:
    """Self-hosted model via sentence-transformers. Real encode; condition realisation is
    external (run on the target instance/precision/library and collect the code bundle)."""

    def __init__(self, model_id: str, device: str | None = None):
        self.agent_id = model_id
        self._model_id = model_id
        self._device = device
        self._model = None  # lazy

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id, device=self._device, trust_remote_code=True)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        return np.asarray(model.encode(texts, normalize_embeddings=True))


class _APIAgent:
    """Base for black-box embedding APIs. Subclasses implement `_embed`. All
    determinism-affecting params must be pinned by the caller (dimensions, encoding
    format, truncation, snapshot) so any drift is provider-internal."""

    provider = "abstract"

    def __init__(self, model_id: str, dim: int, snapshot: str | None = None, max_workers: int = 16):
        self.agent_id = model_id
        self.dim = dim
        self.snapshot = snapshot  # pinned snapshot id, or None → 'floating'
        self.max_workers = max_workers  # concurrent requests; still one input per request

    def _retry(self, one_fn, t, attempts: int = 7):
        """Retry a single request on rate-limit (429) with exponential backoff — needed for
        free/trial keys (Cohere 100/min, Gemini/Mistral free tiers)."""
        for i in range(attempts):
            try:
                return one_fn(t)
            except Exception as e:  # noqa: BLE001 - SDK exception types differ per provider
                msg = str(e).lower()
                rate_limited = any(k in msg for k in ("429", "too many", "rate limit",
                                                      "resource_exhausted", "quota"))
                if rate_limited and i < attempts - 1:
                    time.sleep(min(2 ** i, 30))
                    continue
                raise

    def _run(self, one_fn, texts: list[str]) -> np.ndarray:
        """Map `one_fn` over texts, order-preserving, with per-request 429 retry. Concurrent
        when max_workers > 1. One input per request throughout (no batch-composition confound)."""
        call = lambda t: self._retry(one_fn, t)
        if self.max_workers <= 1:
            rows = [call(t) for t in texts]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                rows = list(ex.map(call, texts))
        return np.asarray(rows, dtype=np.float64)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embed(texts), dtype=np.float64)

    def encode_batched(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """The `batch` condition — the same texts, but **many inputs per request** (batches),
        vs `encode()`'s one-per-request. A provider whose embedding of an input depends on its
        batch-mates diverges here (breaks bit-exact caching). Providers without a batched path
        (e.g. Gemini) raise NotImplementedError → the caller skips the axis for them."""
        rows: list = []
        for i in range(0, len(texts), batch_size):
            rows.extend(self._retry(self._chunk_call, texts[i:i + batch_size]))
        return np.asarray(rows, dtype=np.float64)

    def _chunk_call(self, chunk: list[str]):  # many inputs, one request
        raise NotImplementedError(f"{self.provider} has no batched-encode path")

    def _embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - integration
        raise NotImplementedError(
            f"{self.provider} client not wired yet — implement _embed() with the pinned "
            "snapshot + one-input-per-request protocol (see deployed-agent-panel/README.md)."
        )


class OpenAIAgent(_APIAgent):
    """OpenAI embeddings. The key is read from ``OPENAI_API_KEY`` (env var). Determinism-
    affecting params are pinned: fixed model id (OpenAI embedding models are not dated
    snapshots, so the id *is* the pin → 'pinned'), fixed `dimensions`, `encoding_format=float`,
    one input per request (no batch-composition confound). `fresh=True` uses a brand-new
    client (fresh connection pool) — the `proc` fresh-context resampling."""

    provider = "openai"

    def __init__(self, model_id: str = "text-embedding-3-large", dim: int = 3072,
                 dimensions: int | None = None, max_workers: int = 16):
        super().__init__(model_id, dim, snapshot="pinned", max_workers=max_workers)  # id is the pin
        self.dimensions = dimensions or dim
        self._cached = None

    def _client(self, fresh: bool):
        from openai import OpenAI

        if fresh:
            return OpenAI(max_retries=5)  # brand-new client + pool (the `proc` condition)
        if self._cached is None:
            self._cached = OpenAI(max_retries=5)  # persistent client (the `same` control)
        return self._cached

    def _embed(self, texts: list[str], fresh: bool = False) -> np.ndarray:
        client = self._client(fresh)

        def one(t):  # one input per request — no batch-composition confound
            resp = client.embeddings.create(
                model=self.agent_id, input=[t],
                dimensions=self.dimensions, encoding_format="float",
            )
            return resp.data[0].embedding

        return self._run(one, texts)

    def encode_fresh(self, texts: list[str]) -> np.ndarray:
        """Encode on a fresh client/connection — the `proc` fresh-context resample."""
        return np.asarray(self._embed(texts, fresh=True), dtype=np.float64)

    def _chunk_call(self, chunk):
        resp = self._client(False).embeddings.create(
            model=self.agent_id, input=chunk, dimensions=self.dimensions, encoding_format="float")
        return [d.embedding for d in resp.data]


class VoyageAgent(_APIAgent):
    """Voyage embeddings. Key from ``VOYAGE_API_KEY``. Fixed model id (the pin),
    one input per request; persistent client for `same`, fresh client for `proc`."""

    provider = "voyage"

    def __init__(self, model_id: str = "voyage-4-large", dim: int = 1024,
                 input_type: str | None = None, max_workers: int = 16):
        super().__init__(model_id, dim, snapshot="pinned", max_workers=max_workers)
        self.input_type = input_type
        self._cached = None

    def _client(self, fresh: bool):
        import voyageai

        return voyageai.Client() if fresh else (self._cached or self._cache())

    def _cache(self):
        import voyageai

        self._cached = voyageai.Client()
        return self._cached

    def _embed(self, texts: list[str], fresh: bool = False) -> np.ndarray:
        client = self._client(fresh)

        def one(t):
            return client.embed([t], model=self.agent_id, input_type=self.input_type).embeddings[0]

        return self._run(one, texts)

    def encode_fresh(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embed(texts, fresh=True), dtype=np.float64)

    def _chunk_call(self, chunk):
        return self._client(False).embed(chunk, model=self.agent_id,
                                          input_type=self.input_type).embeddings


class CohereAgent(_APIAgent):
    """Cohere embeddings. Key from ``CO_API_KEY``. Fixed model id + input_type (the v3 API
    requires one), one input per request; persistent client for `same`, fresh for `proc`."""

    provider = "cohere"

    def __init__(self, model_id: str = "embed-v4.0", dim: int = 1024,
                 input_type: str = "search_document", max_workers: int = 16):
        super().__init__(model_id, dim, snapshot="pinned", max_workers=max_workers)
        self.input_type = input_type
        self._cached = None

    def _client(self, fresh: bool):
        import cohere

        if fresh:
            return cohere.Client()
        if self._cached is None:
            self._cached = cohere.Client()
        return self._cached

    def _embed(self, texts: list[str], fresh: bool = False) -> np.ndarray:
        client = self._client(fresh)

        def one(t):
            r = client.embed(texts=[t], model=self.agent_id, input_type=self.input_type,
                             embedding_types=["float"])
            emb = getattr(r.embeddings, "float", None) or getattr(r.embeddings, "float_", None)
            return (emb or r.embeddings)[0]

        return self._run(one, texts)

    def encode_fresh(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embed(texts, fresh=True), dtype=np.float64)

    def _chunk_call(self, chunk):
        r = self._client(False).embed(texts=chunk, model=self.agent_id, input_type=self.input_type,
                                      embedding_types=["float"])
        emb = getattr(r.embeddings, "float", None) or getattr(r.embeddings, "float_", None)
        return emb or r.embeddings


class MistralAgent(_APIAgent):
    """Mistral embeddings via their OpenAI-compatible endpoint. Key from ``MISTRAL_API_KEY``."""

    provider = "mistral"
    BASE_URL = "https://api.mistral.ai/v1"

    def __init__(self, model_id: str = "mistral-embed", dim: int = 1024, max_workers: int = 8):
        super().__init__(model_id, dim, snapshot="pinned", max_workers=max_workers)
        self._cached = None

    def _client(self, fresh: bool):
        import os
        from openai import OpenAI

        mk = lambda: OpenAI(api_key=os.environ["MISTRAL_API_KEY"], base_url=self.BASE_URL, max_retries=5)
        if fresh:
            return mk()
        if self._cached is None:
            self._cached = mk()
        return self._cached

    def _embed(self, texts: list[str], fresh: bool = False) -> np.ndarray:
        client = self._client(fresh)

        def one(t):
            return client.embeddings.create(model=self.agent_id, input=[t]).data[0].embedding

        return self._run(one, texts)

    def encode_fresh(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embed(texts, fresh=True), dtype=np.float64)

    def _chunk_call(self, chunk):
        return [d.embedding for d in
                self._client(False).embeddings.create(model=self.agent_id, input=chunk).data]


class GeminiAgent(_APIAgent):
    """Google Gemini embeddings via google-genai. Key from ``GEMINI_API_KEY``."""

    provider = "gemini"

    def __init__(self, model_id: str = "gemini-embedding-001", dim: int = 3072, max_workers: int = 4):
        super().__init__(model_id, dim, snapshot="pinned", max_workers=max_workers)
        self._cached = None

    def _client(self, fresh: bool):
        import os
        from google import genai

        if fresh:
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        if self._cached is None:
            self._cached = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._cached

    def _embed(self, texts: list[str], fresh: bool = False) -> np.ndarray:
        client = self._client(fresh)

        def one(t):
            return client.models.embed_content(model=self.agent_id, contents=[t]).embeddings[0].values

        return self._run(one, texts)

    def encode_fresh(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._embed(texts, fresh=True), dtype=np.float64)


# The five commercial embedding APIs on the panel: agent key → (class, default model id).
# Single source of truth for the capture tools (run_report / conc_probe / capture_time_baseline /
# drift_sweep), so the roster and default models can't drift between them.
DEFAULT_API_MODELS = {
    "openai": (OpenAIAgent, "text-embedding-3-large"),
    "voyage": (VoyageAgent, "voyage-4-large"),
    "cohere": (CohereAgent, "embed-v4.0"),
    "mistral": (MistralAgent, "mistral-embed"),
    "gemini": (GeminiAgent, "gemini-embedding-001"),
}
