# Copyright (c) 2026 The SEMQ Group Inc.
# Licensed under the Apache License, Version 2.0. See LICENSE for terms.
"""The `proc` worker runs under the parent's numerics.

A fresh process starts with PyTorch's defaults, TF32 off for matmul. If the worker did not
apply the parent's TF32 and determinism flags, a `proc` capture with --tf32 on would measure
the TF32 switch, not cross-process repeatability. These tests run the parent's flag setter and
the worker's source against a stub torch that records the flags; the worker stops at the
model load.
"""
from __future__ import annotations

import json
import sys
import types
from argparse import Namespace

import pytest

from ari.tools import gpu_capture


class _Stop(Exception):
    pass


def _stub_torch():
    torch = types.ModuleType("torch")
    torch.backends = types.SimpleNamespace(
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        cudnn=types.SimpleNamespace(allow_tf32=False),
    )
    torch.deterministic = False
    torch.use_deterministic_algorithms = lambda on, warn_only=False: setattr(
        torch, "deterministic", on
    )
    torch.float32, torch.float16, torch.bfloat16 = "fp32", "fp16", "bf16"
    return torch


def _flags(torch):
    b = torch.backends
    return b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32, torch.deterministic


@pytest.mark.parametrize("tf32", ["on", "off"])
@pytest.mark.parametrize("deterministic", [True, False])
def test_worker_applies_the_parents_flags(tmp_path, monkeypatch, tf32, deterministic):
    args = Namespace(model="m", device="cuda", dtype="fp32", tf32=tf32,
                     deterministic=deterministic)

    parent = _stub_torch()
    monkeypatch.setitem(sys.modules, "torch", parent)
    gpu_capture._apply_torch_flags(args.tf32 == "on", args.deterministic)

    worker = _stub_torch()
    st = types.ModuleType("sentence_transformers")

    def _model(*a, **k):
        raise _Stop

    st.SentenceTransformer = _model
    monkeypatch.setitem(sys.modules, "torch", worker)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    pj = tmp_path / "p.json"
    pj.write_text(json.dumps(gpu_capture._worker_payload(args, ["t"], "v.npy", 8)))
    monkeypatch.setattr(sys, "argv", ["w.py", str(pj)])
    with pytest.raises(_Stop):
        exec(compile(gpu_capture._worker_src(), "w.py", "exec"), {"__name__": "__main__"})

    assert _flags(worker) == _flags(parent) == (tf32 == "on", tf32 == "on", deterministic)
