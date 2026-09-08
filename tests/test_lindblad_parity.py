# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX bit-parity for theory 09-32 Lindblad dynamics (float64).

Import named functions from the leaf modules. Both backend packages
re-export the same names at package level, which would shadow the
submodules the way occupancy does.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

import jax.numpy as jnp  # noqa: E402  (after importorskip / config update)
from omnibias.jax.lindblad import apply_lindblad as jax_apply_lindblad
from omnibias.jax.lindblad import derivative_tower as jax_derivative_tower
from omnibias.jax.lindblad import evolve as jax_evolve
from omnibias.jax.lindblad import liouvillian as jax_liouvillian
from omnibias.jax.lindblad import propagator as jax_propagator
from omnibias.jax.lindblad import steady_state as jax_steady_state
from omnibias.jax.lindblad import thermal_population as jax_thermal_population
from omnibias.torch.lindblad import apply_lindblad as torch_apply_lindblad
from omnibias.torch.lindblad import derivative_tower as torch_derivative_tower
from omnibias.torch.lindblad import evolve as torch_evolve
from omnibias.torch.lindblad import liouvillian as torch_liouvillian
from omnibias.torch.lindblad import propagator as torch_propagator
from omnibias.torch.lindblad import steady_state as torch_steady_state
from omnibias.torch.lindblad import thermal_population as torch_thermal_population


def _qubit_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ham = np.array([[0.0, 0.0], [0.0, 1.1]], dtype=np.complex128)
    jumps = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
        ],
        dtype=np.complex128,
    )
    rates = np.array([0.8, 0.8 * np.exp(-0.7 * 1.1)], dtype=np.float64)
    rho0 = np.array([[0.2, 0.1 - 0.05j], [0.1 + 0.05j, 0.8]], dtype=np.complex128)
    return ham, jumps, rates, rho0


def test_liouvillian_parity() -> None:
    ham, jumps, rates, _ = _qubit_payload()
    t = torch_liouvillian(torch.tensor(ham), torch.tensor(jumps), torch.tensor(rates))
    j = jax_liouvillian(jnp.asarray(ham), jnp.asarray(jumps), jnp.asarray(rates))
    np.testing.assert_allclose(t.detach().cpu().numpy(), np.asarray(j), rtol=1e-13, atol=1e-13)


def test_evolve_parity() -> None:
    ham, jumps, rates, rho0 = _qubit_payload()
    t = torch_evolve(
        torch.tensor(rho0), torch.tensor(ham), torch.tensor(jumps), torch.tensor(rates), 0.4
    )
    j = jax_evolve(jnp.asarray(rho0), jnp.asarray(ham), jnp.asarray(jumps), jnp.asarray(rates), 0.4)
    np.testing.assert_allclose(t.detach().cpu().numpy(), np.asarray(j), rtol=1e-13, atol=1e-12)


def test_apply_and_propagator_parity() -> None:
    ham, jumps, rates, rho0 = _qubit_payload()
    t_l = torch_apply_lindblad(
        torch.tensor(rho0), torch.tensor(ham), torch.tensor(jumps), torch.tensor(rates)
    )
    j_l = jax_apply_lindblad(
        jnp.asarray(rho0), jnp.asarray(ham), jnp.asarray(jumps), jnp.asarray(rates)
    )
    np.testing.assert_allclose(t_l.detach().cpu().numpy(), np.asarray(j_l), rtol=1e-13, atol=1e-13)
    t_p = torch_propagator(torch.tensor(ham), torch.tensor(jumps), torch.tensor(rates), 0.25)
    j_p = jax_propagator(jnp.asarray(ham), jnp.asarray(jumps), jnp.asarray(rates), 0.25)
    np.testing.assert_allclose(t_p.detach().cpu().numpy(), np.asarray(j_p), rtol=1e-13, atol=1e-12)


def test_derivative_tower_and_steady_state_parity() -> None:
    ham, jumps, rates, rho0 = _qubit_payload()
    t = torch_derivative_tower(
        torch.tensor(rho0),
        torch.tensor(ham),
        torch.tensor(jumps),
        torch.tensor(rates),
        0.3,
        order=3,
    )
    j = jax_derivative_tower(
        jnp.asarray(rho0),
        jnp.asarray(ham),
        jnp.asarray(jumps),
        jnp.asarray(rates),
        0.3,
        order=3,
    )
    np.testing.assert_allclose(t.detach().cpu().numpy(), np.asarray(j), rtol=1e-13, atol=1e-12)
    t_ss = torch_steady_state(torch.tensor(ham), torch.tensor(jumps), torch.tensor(rates))
    j_ss = jax_steady_state(jnp.asarray(ham), jnp.asarray(jumps), jnp.asarray(rates))
    np.testing.assert_allclose(t_ss.detach().cpu().numpy(), np.asarray(j_ss), rtol=1e-13, atol=1e-12)


def test_thermal_population_parity() -> None:
    t = torch_thermal_population(torch.tensor(1.3), torch.tensor(0.7))
    j = jax_thermal_population(jnp.asarray(1.3), jnp.asarray(0.7))
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-15)


def test_jax_jit_vmap_grad_smokes() -> None:
    ham, jumps, rates, rho0 = _qubit_payload()
    h_j, j_j, r_j, rho_j = (
        jnp.asarray(ham),
        jnp.asarray(jumps),
        jnp.asarray(rates),
        jnp.asarray(rho0),
    )
    evolved = jax.jit(lambda t: jax_evolve(rho_j, h_j, j_j, r_j, t))(0.2)
    ref = jax_evolve(rho_j, h_j, j_j, r_j, 0.2)
    np.testing.assert_allclose(np.asarray(evolved), np.asarray(ref), rtol=1e-13)
    tower = jax.jit(
        lambda rho: jax_derivative_tower(rho, h_j, j_j, r_j, 0.2, order=2)
    )(rho_j)
    assert tower.shape[0] == 3
    times = jnp.array([0.0, 0.1, 0.2])
    batched = jax.vmap(lambda t: jax_evolve(rho_j, h_j, j_j, r_j, t))(times)
    assert batched.shape == (3, 2, 2)
    grad_t = jax.grad(lambda t: jnp.real(jax_evolve(rho_j, h_j, j_j, r_j, t)[1, 1]))(0.3)
    assert np.isfinite(float(grad_t))


def test_torch_autograd_smoke() -> None:
    ham, jumps, rates, rho0 = _qubit_payload()
    rate_t = torch.tensor(rates, requires_grad=True)
    out = torch_evolve(torch.tensor(rho0), torch.tensor(ham), torch.tensor(jumps), rate_t, 0.4)
    out[1, 1].real.backward()
    assert rate_t.grad is not None
    assert torch.isfinite(rate_t.grad).all()
