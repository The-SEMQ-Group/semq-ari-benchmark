# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""Load a SentenceTransformer in the precision the caller asked for, and check it.

transformers 5 loads a checkpoint in the dtype its config declares unless told
otherwise. ``mixedbread-ai/mxbai-embed-large-v1`` declares float16, so a load that
names no dtype runs that model in fp16 even in an "fp32" condition, on CPU or GPU.
That silently turned its fp32 captures into fp16 ones. Every capture path loads
through here, so the dtype is always explicit and verified after loading.
"""

from __future__ import annotations

_NAMES = {"fp32": "float32", "float32": "float32",
          "fp16": "float16", "float16": "float16",
          "bf16": "bfloat16", "bfloat16": "bfloat16"}


def load_sentence_transformer(model_id: str, *, device: str = "cpu", dtype: str = "fp32",
                              **kwargs):
    """A SentenceTransformer whose parameters are in ``dtype`` (fp32, fp16 or bf16)."""
    import torch
    from sentence_transformers import SentenceTransformer

    if dtype not in _NAMES:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(set(_NAMES))}")
    want = getattr(torch, _NAMES[dtype])
    model_kwargs = dict(kwargs.pop("model_kwargs", None) or {})
    model_kwargs["torch_dtype"] = want
    model = SentenceTransformer(model_id, device=device, model_kwargs=model_kwargs, **kwargs)
    got = next(model.parameters()).dtype
    if got != want:
        raise RuntimeError(f"{model_id} loaded as {got}, not the requested {want}")
    return model
