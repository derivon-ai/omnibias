# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Kantorovich-accepted Newton jax twin (theory 08-04)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from omnibias.jax.optim_kantorovich import (
    approximate_inverse_jacobian,
    kantorovich_accept_step,
    kantorovich_gated_gauss_newton_step,
    polynomial_sqrt2_maps,
    select_accepted_params,
)

jax.config.update("jax_enable_x64", True)

SQRT2 = math.sqrt(2.0)


def _sqrt2_residual(theta: jnp.ndarray) -> jnp.ndarray:
    x = jnp.reshape(theta, (-1,))[0]
    return jnp.reshape(x * x - 2.0, (1,))


def test_accept_reject_on_named_points() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    good = kantorovich_accept_step(
        func, jac, jnp.array([[1.0 / 3.0]]), jnp.array([1.5]),
        lipschitz_df=lip, r_max=0.2,
    )
    far = kantorovich_accept_step(
        func, jac, jnp.array([[1.0 / 3.0]]), jnp.array([3.0]),
        lipschitz_df=lip, r_max=0.2,
    )
    assert good.accepted is True
    assert far.accepted is False
    assert good.certificate is not None
    assert abs(SQRT2 - 1.5) <= good.certificate.radius


def test_gated_gn_near_root_accepts() -> None:
    func, jac, lip = polynomial_sqrt2_maps()
    params = jnp.array([1.5])
    new, decision = kantorovich_gated_gauss_newton_step(
        _sqrt2_residual, func, jac, params, lipschitz_df=lip, r_max=0.2
    )
    assert decision.accepted is True
    assert abs(float(new[0]) - SQRT2) < abs(1.5 - SQRT2) + 1e-12


def test_select_and_inverse_shapes() -> None:
    params = jnp.array([1.5])
    a = approximate_inverse_jacobian(_sqrt2_residual, params)
    assert len(a) == 1 and len(a[0]) == 1
    assert abs(a[0][0] - 1.0 / 3.0) < 1e-12
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, a, params, lipschitz_df=lip, r_max=0.2
    )
    kept = select_accepted_params(jnp.array([0.0]), params, decision)
    assert float(kept[0]) == 1.5
