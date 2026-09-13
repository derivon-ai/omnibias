# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Polynomial collision coordinates checked against independent polynomial calculus."""

from math import factorial

import numpy as np
import pytest
from numpy.polynomial import Polynomial

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

torch = pytest.importorskip("torch")
from omnibias.jax.confluence import polynomial_pair as jax_pair
from omnibias.torch.confluence import polynomial_pair as torch_pair


def oracle(coefficients, z, rho, eta):
    """Shift by differentiating a NumPy Polynomial, independently of binomial code."""
    polynomial = Polynomial(coefficients)
    a, b, da, db = (np.zeros_like(z) for _ in range(4))
    for order in range(len(coefficients)):
        factor = polynomial.deriv(order)(z) * eta**order / factorial(order)
        power = order // 2
        if order % 2:
            b += factor * rho**power
            if power:
                db += power * factor * rho ** (power - 1)
        else:
            a += factor * rho**power
            if power:
                da += power * factor * rho ** (power - 1)
    return a, b, da, db


@pytest.mark.parametrize("rho_value", [0.0, 1e-24, 0.125])
@pytest.mark.parametrize(
    "coefficients", [[0.75], [0.5, -1.25], [0.5, -1.25, 0.75, 0.125, -0.25, 0.5, -0.125]]
)
def test_polynomial_pair_values_live_coefficients_and_spread_derivatives(rho_value, coefficients):
    z = np.array([-0.75, -0.25, 0.0, 0.5])
    eta = np.array([0.5, -0.75, 0.0, 1.25])
    coefficients = np.asarray(coefficients)
    expected_a, expected_b, da, db = oracle(coefficients, z, rho_value, eta)
    coeff = torch.tensor(coefficients, requires_grad=True)
    rho = torch.tensor(rho_value, dtype=torch.float64, requires_grad=True)
    ta, tb = torch_pair(torch.tensor(z), rho, torch.tensor(eta), coeff)
    loss = (ta + 0.25 * tb).sum()
    dc, drho = torch.autograd.grad(loss, (coeff, rho))

    def jloss(c, r):
        a, b = jax_pair(jnp.asarray(z), r, jnp.asarray(eta), c)
        return (a + 0.25 * b).sum()

    ja, jb = jax_pair(
        jnp.asarray(z), jnp.asarray(rho_value), jnp.asarray(eta), jnp.asarray(coefficients)
    )
    jdc, jdrho = jax.jit(jax.grad(jloss, argnums=(0, 1)))(
        jnp.asarray(coefficients), jnp.asarray(rho_value)
    )
    ca, cb = jax.jit(jax_pair)(
        jnp.asarray(z), jnp.asarray(rho_value), jnp.asarray(eta), jnp.asarray(coefficients)
    )
    for actual, expected in (
        (ta.detach(), expected_a),
        (tb.detach(), expected_b),
        (ja, expected_a),
        (jb, expected_b),
        (ca, expected_a),
        (cb, expected_b),
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
    np.testing.assert_array_equal(ta.detach(), ja)
    np.testing.assert_array_equal(tb.detach(), jb)
    expected_coeff = []
    for basis in np.eye(len(coefficients)):
        a, b, _, _ = oracle(basis, z, rho_value, eta)
        expected_coeff.append((a + 0.25 * b).sum())
    for actual in (dc, jdc):
        np.testing.assert_allclose(actual, expected_coeff, atol=2e-15, rtol=2e-15)
    for actual in (drho, jdrho):
        np.testing.assert_allclose(actual, (da + 0.25 * db).sum(), atol=2e-15, rtol=2e-15)


def test_finite_polynomial_pair_matches_defining_pair_without_truncation():
    coefficients = np.array([0.125, -0.5, 0.25, 0.75, -0.125, 0.5, 0.25, -0.125])
    polynomial = Polynomial(coefficients)
    z = np.array([-0.5, 0.0, 0.25])
    eta = np.array([-0.75, 1.0, 0.5])
    h = 0.25
    plus, minus = polynomial(z + h * eta), polynomial(z - h * eta)
    a, b = torch_pair(
        torch.tensor(z), torch.tensor(h * h), torch.tensor(eta), torch.tensor(coefficients)
    )
    np.testing.assert_allclose(a, (plus + minus) / 2, atol=1e-15)
    np.testing.assert_allclose(b, (plus - minus) / (2 * h), atol=1e-15)


def test_polynomial_pair_rejects_invalid_coefficient_shape():
    for coefficients in (np.empty(0), np.ones((2, 2))):
        with pytest.raises(ValueError, match="coefficient"):
            torch_pair(
                torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0), torch.tensor(coefficients)
            )
        with pytest.raises(ValueError, match="coefficient"):
            jax_pair(
                jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0), jnp.asarray(coefficients)
            )
