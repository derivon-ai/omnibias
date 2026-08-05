# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend bit-parity for the molecular local-energy surface.

Feeds identical numpy weights / coordinates through the torch and jax
``molecular`` twins and asserts the Coulomb potential, the closed-form jet
``(grad, lap) log|psi|``, and the assembled local energy agree to
``rtol=1e-9, atol=1e-12``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

jax.config.update("jax_enable_x64", True)

from omnibias.qpinn.jax import molecular as MJ
from omnibias.qpinn.torch import molecular as MT

from .conftest import _allclose


@pytest.mark.parametrize("seed", [0, 5, 42])
def test_coulomb_potential_parity(seed):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=6)
    r = rng.normal(size=9)
    charges = np.array([1.0, 6.0])
    ct = MT.coulomb_potential(
        torch.from_numpy(R), torch.from_numpy(r), torch.from_numpy(charges), n_e=3
    )
    cj = MJ.coulomb_potential(
        jnp.asarray(R), jnp.asarray(r), jnp.asarray(charges), n_e=3
    )
    assert _allclose(ct, cj)


@pytest.mark.parametrize("seed", [1, 8, 99])
def test_log_psi_derivatives_parity(seed):
    rng = np.random.default_rng(seed)
    D, hidden = 6, 5
    W1 = rng.normal(size=(hidden, D))
    b1 = rng.normal(size=hidden)
    W2 = rng.normal(size=(1, hidden))
    b2 = rng.normal(size=1)
    x0 = rng.normal(size=D)

    t_layers = [
        (torch.from_numpy(W1), torch.from_numpy(b1), "tanh"),
        (torch.from_numpy(W2), torch.from_numpy(b2), None),
    ]
    j_layers = [
        (jnp.asarray(W1), jnp.asarray(b1), "tanh"),
        (jnp.asarray(W2), jnp.asarray(b2), None),
    ]
    tg, tl = MT.log_psi_derivatives(torch.from_numpy(x0), t_layers, order=2)
    jg, jl = MJ.log_psi_derivatives(jnp.asarray(x0), j_layers, order=2)
    assert _allclose(tg, jg)
    assert _allclose(tl, jl)


@pytest.mark.parametrize("seed", [2, 17])
def test_molecular_local_energy_parity(seed):
    rng = np.random.default_rng(seed)
    D, hidden = 3, 6
    W1 = rng.normal(size=(hidden, D))
    b1 = rng.normal(size=hidden)
    W2 = rng.normal(size=(1, hidden))
    b2 = rng.normal(size=1)
    x0 = rng.normal(size=D)  # single electron, D = 3
    charges = np.array([2.0])
    R = np.zeros(3)

    t_layers = [
        (torch.from_numpy(W1), torch.from_numpy(b1), "tanh"),
        (torch.from_numpy(W2), torch.from_numpy(b2), None),
    ]
    j_layers = [
        (jnp.asarray(W1), jnp.asarray(b1), "tanh"),
        (jnp.asarray(W2), jnp.asarray(b2), None),
    ]
    tg, tl = MT.log_psi_derivatives(torch.from_numpy(x0), t_layers, order=2)
    jg, jl = MJ.log_psi_derivatives(jnp.asarray(x0), j_layers, order=2)

    et = MT.molecular_local_energy(
        tg, tl, torch.from_numpy(R), torch.from_numpy(x0),
        torch.from_numpy(charges), n_e=1,
    )
    ej = MJ.molecular_local_energy(
        jg, jl, jnp.asarray(R), jnp.asarray(x0), jnp.asarray(charges), n_e=1
    )
    assert _allclose(et, ej)
