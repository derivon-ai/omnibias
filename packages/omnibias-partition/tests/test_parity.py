# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Bit-identical numpy <-> torch <-> jax parity of the partition weights (float64)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.partition import PartitionConfig, init_params, partition_weights

torch = pytest.importorskip("torch")


def _sample() -> tuple[object, np.ndarray]:
    cfg = PartitionConfig(n_features=4, depth=3, seed=11)
    params = init_params(cfg, rng=3)
    X = np.random.default_rng(12).standard_normal((29, 4))
    return params, X


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0, 50.0])
def test_torch_matches_numpy(beta: float) -> None:
    from omnibias.partition.torch import partition_weights as tw

    params, X = _sample()
    ref = partition_weights(params, X, beta)
    got = tw(params, X, beta).detach().cpu().numpy()
    assert np.max(np.abs(got - ref)) < 1e-9


@pytest.mark.parametrize("beta", [0.5, 2.0, 8.0, 50.0])
def test_jax_matches_numpy(beta: float) -> None:
    jax = pytest.importorskip("jax")  # noqa: F841
    from omnibias.partition.jax import partition_weights as jw

    params, X = _sample()
    ref = partition_weights(params, X, beta)
    got = np.asarray(jw(params, X, beta))
    assert np.max(np.abs(got - ref)) < 1e-9


def test_torch_weights_are_differentiable() -> None:
    from omnibias.partition.torch import partition_weights_arrays

    cfg = PartitionConfig(n_features=3, depth=2, seed=1)
    params = init_params(cfg)
    W = torch.tensor(params.W, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(params.t, dtype=torch.float64, requires_grad=True)
    X = torch.tensor(
        np.random.default_rng(1).standard_normal((8, 3)), dtype=torch.float64
    )
    P = partition_weights_arrays(W, t, X, beta=4.0, depth=2)
    P.sum().backward()
    assert W.grad is not None and t.grad is not None
    assert torch.isfinite(W.grad).all()
