# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend x64 parity for domain boundary jets and fields."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.pinn.domain import Sphere, boundary_factor_jet
from omnibias.pinn.domain.jax import boundary_factor_jet_at as jax_boundary_jet
from omnibias.pinn.domain.torch import boundary_factor_jet_at as torch_boundary_jet

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sphere():
    return Sphere(center=(0.0, 0.0), radius=1.0)


def test_boundary_jet_torch_jax_parity(sphere) -> None:
    x0 = np.array([0.2, -0.3], dtype=np.float64)
    ref = boundary_factor_jet(sphere, x0, order=2, normalize=False)
    t_jet = torch_boundary_jet(
        sphere, torch.tensor(x0, dtype=torch.float64), order=2, normalize=False
    )
    j_jet = jax_boundary_jet(
        sphere, jnp.asarray(x0, dtype=jnp.float64), order=2, normalize=False
    )
    assert torch.allclose(t_jet.cpu(), torch.tensor(ref), atol=1e-12, rtol=1e-12)
    assert np.allclose(np.asarray(j_jet), ref, atol=1e-12, rtol=1e-12)


def test_boundary_jet_normalized_sphere(sphere) -> None:
    x0 = np.array([1.05, 0.0], dtype=np.float64)
    ref = boundary_factor_jet(sphere, x0, order=1, normalize=True)
    t_jet = torch_boundary_jet(
        sphere, torch.tensor(x0, dtype=torch.float64), order=1, normalize=True
    )
    assert torch.allclose(t_jet.cpu(), torch.tensor(ref), atol=1e-10, rtol=1e-10)
