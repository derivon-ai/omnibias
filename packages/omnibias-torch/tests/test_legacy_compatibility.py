# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
from __future__ import annotations

import pytest
import torch

from omnibias.core.local_jet import LocalJetConfig
from omnibias.torch.information import fisher_information
from omnibias.torch.train_local import local_jet_step


def test_torch_information_module_retains_fisher_metric_primitive() -> None:
    value = fisher_information(torch.tensor(0.0, dtype=torch.float64))

    torch.testing.assert_close(value, torch.tensor(0.25, dtype=torch.float64))


def test_legacy_local_training_module_runs_one_small_sweep() -> None:
    layers = [(torch.tensor([[0.2]]), torch.tensor([0.1]), "tanh")]
    updated, report = local_jet_step(
        layers,
        torch.tensor([0.3]),
        target=torch.tensor([0.0]),
        config=LocalJetConfig(n_directions=1, jet_order=1, damping=1e-2),
    )

    assert len(updated) == 1
    assert report.n_params == 2
    assert report.residual_norms[0] >= 0.0
    assert report.greedy_only_claimed_optimal is False


def test_legacy_fisher_metric_is_differentiable() -> None:
    theta = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    fisher_information(theta).backward()

    assert theta.grad is not None
    assert torch.isfinite(theta.grad)
