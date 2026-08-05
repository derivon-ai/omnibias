# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard-constraint boundary/initial-condition ansatz ``u = g + b N`` (JAX).

Mirrors the torch suite: the Dirichlet / box / initial conditions hold exactly by
construction, the closed-form jet derivatives of the wrapped field match ``jax``
autograd, the field is a jit-compatible pytree whose gradients flow only to the
network, and a Gauss-Newton PINN solves 1-D Poisson with no boundary-loss term.
Cross-backend bit-parity vs torch lives in ``tests/test_hardbc_parity.py``.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.flatten_util import ravel_pytree  # noqa: E402
from omnibias.jax import jet_multiply  # noqa: E402
from omnibias.jax.architectures import make_jet_mlp  # noqa: E402
from omnibias.jax.architectures.hardbc import (  # noqa: E402
    AffineFactor,
    AffineLift,
    BoundaryMask,
    HardConstraintField,
    dirichlet_interval,
    homogeneous_box,
    initial_value,
)
from omnibias.jax.jet_mv import identity_jet, jet_gradient, jet_hessian  # noqa: E402
from omnibias.jax.optim import gauss_newton_minimize, make_residual_fn  # noqa: E402


def _scalar_value(field: HardConstraintField):
    def u(xi: jnp.ndarray) -> jnp.ndarray:
        return field.value(xi[None, :])[0, 0]

    return u


# --- jet_multiply -------------------------------------------------------------


def test_jet_multiply_is_polynomial_product() -> None:
    x0 = jnp.array([0.3, -0.7])
    idj = identity_jet(x0, 2)
    prod = jet_multiply(idj[:, 0], idj[:, 1], 2, 2)
    assert jnp.allclose(jet_gradient(prod, 2, 1), jnp.array([-0.7, 0.3]))
    assert jnp.allclose(jet_hessian(prod, 2, 2), jnp.array([[0.0, 1.0], [1.0, 0.0]]))


def test_jet_multiply_rejects_wrong_rows() -> None:
    x0 = jnp.array([0.1, 0.2])
    idj = identity_jet(x0, 2)
    with pytest.raises(ValueError, match="rows"):
        jet_multiply(idj[:, 0], jnp.zeros(2), 2, 2)


# --- Dirichlet interval -------------------------------------------------------


def test_dirichlet_interval_endpoints_exact() -> None:
    net = make_jet_mlp(1, 16, 1, depth=3, seed=1)
    field = dirichlet_interval(net, 0.0, 2.0, lower_value=1.0, upper_value=-3.0)
    u = field.value(jnp.array([[0.0], [2.0]]))
    assert jnp.allclose(u[:, 0], jnp.array([1.0, -3.0]), atol=1e-12)


def test_dirichlet_homogeneous_has_no_lift() -> None:
    net = make_jet_mlp(1, 8, 1, depth=2, seed=2)
    field = dirichlet_interval(net, -1.0, 1.0)
    assert field.lift is None
    assert jnp.allclose(field.value(jnp.array([[-1.0], [1.0]])), jnp.zeros((2, 1)), atol=1e-13)


def test_dirichlet_derivatives_match_autograd() -> None:
    net = make_jet_mlp(1, 16, 1, depth=3, seed=3)
    field = dirichlet_interval(net, 0.0, 1.0, lower_value=0.5, upper_value=2.0)
    x = jnp.array([[0.13], [0.42], [0.86]])
    u = _scalar_value(field)
    g_ad = jnp.stack([jax.grad(u)(xi) for xi in x])  # (B,1)
    h_ad = jnp.stack([jax.hessian(u)(xi) for xi in x])  # (B,1,1)
    assert jnp.allclose(field.gradient(x)[:, :, 0], g_ad, atol=1e-11)
    assert jnp.allclose(field.hessian(x)[:, :, :, 0], h_ad, atol=1e-11)


# --- Homogeneous box ----------------------------------------------------------


def test_homogeneous_box_boundary_is_zero() -> None:
    net = make_jet_mlp(2, 12, 1, depth=2, seed=4)
    field = homogeneous_box(net, [0.0, 0.0], [1.0, 1.0])
    boundary = jnp.array([[0.0, 0.3], [1.0, 0.5], [0.4, 0.0], [0.6, 1.0]])
    assert jnp.allclose(field.value(boundary), jnp.zeros((4, 1)), atol=1e-13)
    assert abs(float(field.value(jnp.array([[0.5, 0.5]]))[0, 0])) > 0.0


def test_box_derivatives_match_autograd() -> None:
    net = make_jet_mlp(2, 10, 1, depth=2, seed=5)
    field = homogeneous_box(net, [-1.0, 0.0], [1.0, 2.0])
    x = jnp.array([[0.1, 0.7], [-0.4, 1.3]])
    u = _scalar_value(field)
    g_ad = jnp.stack([jax.grad(u)(xi) for xi in x])  # (B,2)
    h_ad = jnp.stack([jax.hessian(u)(xi) for xi in x])  # (B,2,2)
    assert jnp.allclose(field.gradient(x)[:, :, 0], g_ad, atol=1e-10)
    assert jnp.allclose(field.hessian(x)[:, :, :, 0], h_ad, atol=1e-10)


# --- Initial condition --------------------------------------------------------


def test_initial_value_exact_on_slice() -> None:
    net = make_jet_mlp(2, 10, 1, depth=2, seed=6)
    field = initial_value(net, t_axis=1, t0=0.0, value=2.0)
    ic = jnp.array([[0.1, 0.0], [0.9, 0.0], [0.5, 0.0]])
    assert jnp.allclose(field.value(ic), 2.0 * jnp.ones((3, 1)), atol=1e-13)
    assert abs(float(field.value(jnp.array([[0.5, 0.7]]))[0, 0]) - 2.0) > 0.0


# --- consistency / arbitrary order / jit / pytree -----------------------------


def test_value_grad_hessian_consistent() -> None:
    net = make_jet_mlp(2, 10, 1, depth=2, seed=7)
    field = homogeneous_box(net, [0.0, 0.0], [1.0, 1.0])
    x = jax.random.uniform(jax.random.PRNGKey(0), (5, 2), dtype=jnp.float64)
    v, g, h = field.value_grad_hessian(x)
    assert jnp.allclose(v, field.value(x), atol=1e-12)
    assert jnp.allclose(g, field.gradient(x), atol=1e-12)
    assert jnp.allclose(h, field.hessian(x), atol=1e-12)


def test_partials_third_order_match_autograd() -> None:
    net = make_jet_mlp(1, 14, 1, depth=2, seed=8)
    field = dirichlet_interval(net, 0.0, 1.0, lower_value=1.0, upper_value=0.0)
    x = jnp.array([[0.37]])
    u = _scalar_value(field)
    d3 = jax.grad(lambda xi: jax.grad(lambda xj: jax.grad(u)(xj)[0])(xi)[0])(x[0])
    assert jnp.allclose(field.partials(x, 3)[(3,)][:, 0], d3, atol=1e-9)


def test_field_is_jit_compatible_pytree() -> None:
    net = make_jet_mlp(1, 10, 1, depth=2, seed=9)
    field = dirichlet_interval(net, 0.0, 1.0, lower_value=1.0, upper_value=2.0)
    x = jnp.array([[0.25], [0.75]])
    jitted = jax.jit(lambda fld, xx: fld.value(xx))
    assert jnp.allclose(jitted(field, x), field.value(x), atol=1e-12)


def test_gradients_flow_only_to_network() -> None:
    net = make_jet_mlp(1, 10, 1, depth=2, seed=10)
    field = dirichlet_interval(net, 0.0, 1.0)
    x = jnp.linspace(0, 1, 12)[:, None]

    def loss(fld: HardConstraintField) -> jnp.ndarray:
        return jnp.mean(fld.hessian(x)[:, 0, 0, 0] ** 2)

    grad_field = jax.grad(loss)(field)
    wnorm = sum(float(jnp.sum(w**2)) for w in grad_field.net.weights)
    assert wnorm > 0.0
    # mask / lift are static aux -> not differentiated, structure preserved
    assert grad_field.mask == field.mask


def test_gauss_newton_solves_poisson_without_boundary_loss() -> None:
    net = make_jet_mlp(1, 24, 1, depth=2, seed=0)
    field = dirichlet_interval(net, 0.0, 1.0)
    x = jnp.linspace(0, 1, 40)[:, None]
    f = -(math.pi**2) * jnp.sin(math.pi * x[:, 0])

    def build_residual(fld: HardConstraintField) -> jnp.ndarray:
        return fld.hessian(x)[:, 0, 0, 0] - f

    flat0, rfn = make_residual_fn(build_residual, field)
    _, unravel = ravel_pytree(field)
    init_loss = 0.5 * float((rfn(flat0) ** 2).mean())
    state, history = gauss_newton_minimize(rfn, flat0, steps=40, damping=1e-2)
    assert history[-1] < init_loss * 1e-4

    field_opt = unravel(state.params)
    xx = jnp.linspace(0, 1, 101)[:, None]
    u = field_opt.value(xx)[:, 0]
    u_star = jnp.sin(math.pi * xx[:, 0])
    assert float(jnp.abs(u - u_star).max()) < 1e-3
    bc = field_opt.value(jnp.array([[0.0], [1.0]]))
    assert float(jnp.abs(bc).max()) < 1e-12


# --- validation ---------------------------------------------------------------


def test_dirichlet_interval_rejects_bad_bounds() -> None:
    net = make_jet_mlp(1, 8, 1, depth=2, seed=11)
    with pytest.raises(ValueError, match="must exceed"):
        dirichlet_interval(net, 1.0, 0.0)


def test_homogeneous_box_rejects_wrong_length() -> None:
    net = make_jet_mlp(2, 8, 1, depth=2, seed=12)
    with pytest.raises(ValueError, match="length in_dim"):
        homogeneous_box(net, [0.0], [1.0])


def test_mask_factor_axis_out_of_range() -> None:
    net = make_jet_mlp(1, 8, 1, depth=2, seed=13)
    with pytest.raises(ValueError, match="out of range"):
        HardConstraintField(net=net, mask=BoundaryMask((AffineFactor(3, 1.0, 0.0),)))


def test_lift_dim_mismatch_rejected() -> None:
    net = make_jet_mlp(2, 8, 1, depth=2, seed=14)
    mask = BoundaryMask((AffineFactor(0, 1.0, 0.0),))
    bad_lift = AffineLift(((0.0,), (0.0,)), (0.0, 0.0))
    with pytest.raises(ValueError, match="out_dim|in_dim"):
        HardConstraintField(net=net, mask=mask, lift=bad_lift)


def test_empty_mask_and_zero_scale_rejected() -> None:
    with pytest.raises(ValueError, match="at least one factor"):
        BoundaryMask(())
    with pytest.raises(ValueError, match="non-zero"):
        AffineFactor(0, 0.0, 1.0)
