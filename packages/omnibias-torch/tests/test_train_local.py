# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Depth-causal local jet, PyTorch (theory 08-03)."""

from __future__ import annotations

import math

import pytest
import torch
from omnibias.core.local_jet import LocalJetConfig, LocalJetForbidden
from omnibias.torch.train_local import local_jet_step


def _section5_layers() -> list[tuple[torch.Tensor, None, str | None]]:
    w = torch.tensor([[0.5]])
    v = torch.tensor([[0.2]])
    return [(w, None, "tanh"), (v, None, None)]


def test_section5_readout_then_hidden() -> None:
    torch.set_default_dtype(torch.float64)
    cfg = LocalJetConfig(n_directions=1, damping=0.0, variant="readout")
    layers, report = local_jet_step(
        _section5_layers(),
        torch.tensor([1.0]),
        config=cfg,
        target=torch.tensor([1.0]),
    )
    v = layers[1][0].reshape(-1)
    w = layers[0][0].reshape(-1)
    h = torch.tanh(w * 1.0)
    yhat = (v * h).reshape(-1)
    assert abs(float(yhat[0]) - 1.0) <= 1e-12
    assert report.greedy_only_claimed_optimal is False
    assert report.n_params == 2


def test_g1_forbid_flood() -> None:
    torch.set_default_dtype(torch.float64)
    with pytest.raises(LocalJetForbidden, match="allow_full"):
        local_jet_step(
            _section5_layers(),
            torch.tensor([1.0]),
            config=LocalJetConfig(n_directions=2, damping=0.0),
            target=torch.tensor([1.0]),
        )


def test_invert_and_match_tanh() -> None:
    torch.set_default_dtype(torch.float64)
    z_star = 0.3
    target = torch.tensor([math.tanh(z_star)])
    w0 = torch.tensor([[0.5]])
    layers, _report = local_jet_step(
        [(w0, None, "tanh")],
        torch.tensor([1.0]),
        config=LocalJetConfig(
            n_directions=1, damping=0.0, variant="invert", allow_full=True
        ),
        target=target,
    )
    u = float(layers[0][0].reshape(-1)[0])
    assert abs(u - z_star) <= 1e-12


def test_invert_gelu_raises() -> None:
    torch.set_default_dtype(torch.float64)
    with pytest.raises(ValueError, match="strictly monotone"):
        local_jet_step(
            [(torch.tensor([[0.5]]), None, "gelu")],
            torch.tensor([1.0]),
            config=LocalJetConfig(
                n_directions=1, damping=0.0, variant="invert", allow_full=True
            ),
            target=torch.tensor([0.1]),
        )


def test_predcode_linear_to_eps() -> None:
    torch.set_default_dtype(torch.float64)
    w0 = torch.tensor([[0.0]])
    layers, report = local_jet_step(
        [(w0, None, None)],
        torch.tensor([1.0]),
        config=LocalJetConfig(
            n_directions=1, damping=0.0, variant="predcode", allow_full=True
        ),
        target=torch.tensor([2.0]),
    )
    yhat = float(layers[0][0].reshape(-1)[0])
    assert abs(yhat - 2.0) <= 1e-12
    assert report.residual_norms[0] <= 1e-12
