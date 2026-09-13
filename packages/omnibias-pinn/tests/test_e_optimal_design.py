# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent E-design eigenvalue derivatives and repeated-spectrum refusal."""

import numpy as np
import pytest
from omnibias.pinn.inverse.design import design_gradient, design_score, differentiable_design_score


def test_e_optimal_design_gradient_and_backend_live_graphs():
    import jax
    import jax.numpy as jnp
    import torch

    jax.config.update("jax_enable_x64", True)
    # Diagonal information gives an independent scalar eigenvalue derivative.
    blocks = np.array([[[2.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 4.0]]])
    prior = np.diag([1.0, 5.0])
    weights = np.array([0.3, 0.8])
    expected = np.array([2.0, 0.5])
    np.testing.assert_allclose(design_gradient(blocks, weights, prior, criterion="E"), expected)
    assert design_score(blocks, weights, prior, criterion="E") == pytest.approx(2.0)

    def f(w):
        return differentiable_design_score(
            jnp.asarray(blocks), w, jnp.asarray(prior), backend="jax", criterion="E"
        )

    np.testing.assert_allclose(jax.jit(jax.grad(f))(jnp.asarray(weights)), expected, atol=1e-12)
    w = torch.tensor(weights, dtype=torch.float64, requires_grad=True)
    value = differentiable_design_score(
        torch.tensor(blocks), w, torch.tensor(prior), backend="torch", criterion="E"
    )
    gradient = torch.autograd.grad(value, w, create_graph=True)[0]
    np.testing.assert_allclose(gradient.detach().numpy(), expected, atol=1e-12)
    assert gradient.requires_grad


def test_e_gradient_refuses_multiple_smallest_eigenvalue():
    blocks = np.array([np.diag([1.0, 0.0]), np.diag([0.0, 1.0])])
    with pytest.raises(ValueError, match="simple smallest"):
        design_gradient(blocks, np.zeros(2), np.eye(2), criterion="E")
    # The value is still defined at the nondifferentiable repeated eigenvalue.
    assert design_score(blocks, np.zeros(2), np.eye(2), criterion="E") == 1.0
