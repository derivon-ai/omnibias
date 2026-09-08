# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch / JAX bit-parity for theory 04-03 Fermi occupancy (float64).

Both backends share the exact same Eulerian-number coefficients from
:mod:`omnibias.core.polynomials`; given the same ``sigmoid`` sample the
towers are bit-identical, so ``rtol=1e-13`` is a tight, meaningful bound
(not merely "close").
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

jax.config.update("jax_enable_x64", True)
torch.set_default_dtype(torch.float64)

import jax.numpy as jnp  # noqa: E402  (after importorskip / config update)

# NB: both ``omnibias.torch`` and ``omnibias.jax`` re-export the
# ``occupancy`` *function* at package level (theory 04-03's own public
# surface). By ordinary Python import semantics, once the package
# ``__init__`` has run, that assignment shadows the ``occupancy``
# *submodule* attribute on the package -- so ``import
# omnibias.torch.occupancy as ot`` (and even ``from omnibias.torch import
# occupancy as ot``) silently rebind ``ot`` to the *function*, not the
# module. Importing the individual names directly with a ``from
# omnibias.<backend>.occupancy import ...`` fromlist (as below) resolves
# straight off ``sys.modules`` for the fully-qualified leaf module and is
# unaffected by that shadow.
from omnibias.jax.occupancy import entropy_per_state as jax_entropy_per_state
from omnibias.jax.occupancy import grand_potential_density as jax_grand_potential_density
from omnibias.jax.occupancy import occupancy as jax_occupancy
from omnibias.jax.occupancy import occupancy_derivative as jax_occupancy_derivative
from omnibias.jax.occupancy import occupancy_mu_derivative as jax_occupancy_mu_derivative
from omnibias.jax.occupancy import occupancy_window as jax_occupancy_window
from omnibias.jax.occupancy import reduced_argument as jax_reduced_argument
from omnibias.jax.occupancy import thermal_broadening as jax_thermal_broadening
from omnibias.torch.occupancy import entropy_per_state as torch_entropy_per_state
from omnibias.torch.occupancy import (
    grand_potential_density as torch_grand_potential_density,
)
from omnibias.torch.occupancy import occupancy as torch_occupancy
from omnibias.torch.occupancy import occupancy_derivative as torch_occupancy_derivative
from omnibias.torch.occupancy import (
    occupancy_mu_derivative as torch_occupancy_mu_derivative,
)
from omnibias.torch.occupancy import occupancy_window as torch_occupancy_window
from omnibias.torch.occupancy import reduced_argument as torch_reduced_argument
from omnibias.torch.occupancy import thermal_broadening as torch_thermal_broadening


class _Namespace:
    """Tiny attribute-namespace so the test bodies below can keep using
    the ``ot.foo`` / ``oj.foo`` spelling without triggering the package-
    level shadow described above."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


ot = _Namespace(
    occupancy=torch_occupancy,
    occupancy_derivative=torch_occupancy_derivative,
    occupancy_mu_derivative=torch_occupancy_mu_derivative,
    occupancy_window=torch_occupancy_window,
    entropy_per_state=torch_entropy_per_state,
    grand_potential_density=torch_grand_potential_density,
    reduced_argument=torch_reduced_argument,
    thermal_broadening=torch_thermal_broadening,
)
oj = _Namespace(
    occupancy=jax_occupancy,
    occupancy_derivative=jax_occupancy_derivative,
    occupancy_mu_derivative=jax_occupancy_mu_derivative,
    occupancy_window=jax_occupancy_window,
    entropy_per_state=jax_entropy_per_state,
    grand_potential_density=jax_grand_potential_density,
    reduced_argument=jax_reduced_argument,
    thermal_broadening=jax_thermal_broadening,
)

_CASES = (
    (2.0, 0.5, 0.3),
    (0.7, -1.0, 2.5),
    (5.0, 0.0, 0.0),
    (10.0, 2.0, 1.9),
    (0.3, -3.0, -2.5),
    (1.0, 0.0, 40.0),  # far tail (f -> 0)
    (1.0, 0.0, -40.0),  # far tail (f -> 1)
)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_occupancy_parity(beta: float, mu: float, energy: float) -> None:
    t = ot.occupancy(torch.tensor(energy), beta, mu)
    j = oj.occupancy(jnp.array(energy), beta, mu)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4, 6])
def test_occupancy_derivative_parity(
    beta: float, mu: float, energy: float, order: int
) -> None:
    t = ot.occupancy_derivative(torch.tensor(energy), beta, mu, order=order)
    j = oj.occupancy_derivative(jnp.array(energy), beta, mu, order=order)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
@pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
def test_occupancy_mu_derivative_parity(
    beta: float, mu: float, energy: float, order: int
) -> None:
    t = ot.occupancy_mu_derivative(torch.tensor(energy), beta, mu, order=order)
    j = oj.occupancy_mu_derivative(jnp.array(energy), beta, mu, order=order)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-12)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_entropy_per_state_parity(beta: float, mu: float, energy: float) -> None:
    t = ot.entropy_per_state(torch.tensor(energy), beta, mu)
    j = oj.entropy_per_state(jnp.array(energy), beta, mu)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_grand_potential_density_parity(beta: float, mu: float, energy: float) -> None:
    t = ot.grand_potential_density(torch.tensor(energy), beta, mu)
    j = oj.grand_potential_density(jnp.array(energy), beta, mu)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("beta,mu,energy", _CASES)
def test_thermal_broadening_parity(beta: float, mu: float, energy: float) -> None:
    t = ot.thermal_broadening(torch.tensor(energy), beta, mu)
    j = oj.thermal_broadening(jnp.array(energy), beta, mu)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-300)


@pytest.mark.parametrize("beta,mu", [(2.0, 0.5), (0.6, -1.5), (4.0, 0.2)])
def test_occupancy_window_parity(beta: float, mu: float) -> None:
    e_lo, e_hi = -3.0, 3.0
    t = ot.occupancy_window(torch.tensor(e_lo), torch.tensor(e_hi), beta, mu)
    j = oj.occupancy_window(jnp.array(e_lo), jnp.array(e_hi), beta, mu)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-300)


def test_reduced_argument_parity() -> None:
    beta, mu, energy = 2.0, 0.5, 0.3
    t = ot.reduced_argument(torch.tensor(energy), beta, mu)
    j = oj.reduced_argument(jnp.array(energy), beta, mu)
    np.testing.assert_allclose(float(t), float(j), rtol=1e-13, atol=1e-300)


# ----- jax jit / vmap smoke -------------------------------------------------


def test_jax_jit_smoke() -> None:
    # ``order`` is a static Python int (see the module docstring); it is
    # closed over rather than passed as a traced ``jit`` argument.
    f = jax.jit(lambda e, b, m: oj.occupancy_derivative(e, b, m, order=2))
    got = f(jnp.array(0.3), 2.0, 0.5)
    ref = oj.occupancy_derivative(jnp.array(0.3), 2.0, 0.5, order=2)
    np.testing.assert_allclose(float(got), float(ref), rtol=1e-13)


def test_jax_vmap_smoke() -> None:
    energies = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
    got = jax.vmap(lambda e: oj.occupancy(e, 2.0, 0.5))(energies)
    ref = np.array([float(oj.occupancy(e, 2.0, 0.5)) for e in energies])
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-13)


def test_jax_grad_smoke() -> None:
    grad_beta = jax.grad(
        lambda b: oj.occupancy_derivative(jnp.array(0.3), b, 0.5, order=2)
    )(2.0)
    assert np.isfinite(float(grad_beta))


def test_torch_autograd_smoke() -> None:
    beta_t = torch.tensor(2.0, requires_grad=True)
    mu_t = torch.tensor(0.5, requires_grad=True)
    e_t = torch.tensor(0.3, requires_grad=True)
    out = ot.occupancy_derivative(e_t, beta_t, mu_t, order=2)
    out.backward()
    assert torch.isfinite(beta_t.grad)
    assert torch.isfinite(mu_t.grad)
    assert torch.isfinite(e_t.grad)


def test_occupancy_derivative_rejects_negative_order() -> None:
    with pytest.raises(ValueError, match="order must be >= 0"):
        ot.occupancy_derivative(torch.tensor(0.0), 1.0, 0.0, order=-1)
    with pytest.raises(ValueError, match="order must be >= 0"):
        oj.occupancy_derivative(jnp.array(0.0), 1.0, 0.0, order=-1)
