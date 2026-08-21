# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch Jet-Hopfield twin (09-13)."""

from __future__ import annotations

import torch
from omnibias.core.jet_hopfield import JetHopfieldConfig
from omnibias.torch.architectures.jet_hopfield import jet_hopfield_retrieve, worked_example


def test_g1() -> None:
    torch.set_default_dtype(torch.float64)
    ex = worked_example()
    assert ex["value_err"] < 1e-6
    cfg = JetHopfieldConfig(lam=1.0, beta=10.0)
    q = torch.tensor([1.0, 0.01])
    mem = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    germ = jet_hopfield_retrieve(q, mem, config=cfg)
    assert abs(float(germ[0]) - 1.0) < 1e-6
