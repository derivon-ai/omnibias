# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract tests for :mod:`omnibias.jax.folx_compat`.

We verify three things:

1. :func:`omnibias.jax.forward_laplacian` returns an object with the
   folx-compatible attributes (``x``, ``dense_jacobian``, ``jacobian``,
   ``laplacian``) and matches a ``jax.hessian``-derived reference.

2. :func:`omnibias.jax.laplacian_factory` returns a ``(lap, grad)``
   tuple matching the DeepQMC ``LaplacianFactory`` protocol.

3. :func:`omnibias.jax.closed_form_forward_laplacian` matches
   :func:`omnibias.jax.forward_laplacian` to float64 precision on
   an omnibias one-layer scalar field (the "happy path" the
   integration plans describe).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.folx_compat import (  # noqa: E402
    closed_form_forward_laplacian,
    forward_laplacian,
    laplacian_factory,
)
from omnibias.jax import neural_field_value  # noqa: E402

D = 12
H = 16
RNG = np.random.default_rng(7)
W_NP = RNG.normal(size=(H, D)) * 0.3
BETA_NP = RNG.normal(size=(H,)) * 0.2
C_NP = RNG.normal(size=(H,)) * 0.4
B_NP = 0.1
X_NP = RNG.normal(size=(D,)) * 1.0


def _to_jax(*xs):
    return tuple(jnp.asarray(x, dtype=jnp.float64) for x in xs)


def _scalar_field_tanh(x: jnp.ndarray) -> jnp.ndarray:
    W, beta, c = _to_jax(W_NP, BETA_NP, C_NP)
    return neural_field_value(x, W, beta, c, B_NP, "tanh")


def test_forward_laplacian_matches_jax_hessian():
    (x,) = _to_jax(X_NP)
    out = forward_laplacian(_scalar_field_tanh)(x)

    val_ref = _scalar_field_tanh(x)
    grad_ref = jax.grad(_scalar_field_tanh)(x)
    lap_ref = jnp.trace(jax.hessian(_scalar_field_tanh)(x))

    np.testing.assert_allclose(out.x, val_ref, atol=1e-12)
    np.testing.assert_allclose(out.dense_jacobian, grad_ref, atol=1e-10)
    np.testing.assert_allclose(out.laplacian, lap_ref, atol=1e-9)
    # `jacobian` is an alias of `dense_jacobian` (no sparse representation today)
    np.testing.assert_array_equal(out.dense_jacobian, out.jacobian)


def test_laplacian_factory_matches_protocol():
    (x,) = _to_jax(X_NP)
    lap_fn = laplacian_factory(_scalar_field_tanh)
    lap, grad = lap_fn(x)

    grad_ref = jax.grad(_scalar_field_tanh)(x)
    lap_ref = jnp.trace(jax.hessian(_scalar_field_tanh)(x))

    np.testing.assert_allclose(lap, lap_ref, atol=1e-9)
    np.testing.assert_allclose(grad, grad_ref, atol=1e-10)


def test_closed_form_path_matches_autograd_path():
    x, W, beta, c = _to_jax(X_NP, W_NP, BETA_NP, C_NP)
    closed = closed_form_forward_laplacian(x, W, beta, c, B_NP, "tanh")
    autograd = forward_laplacian(_scalar_field_tanh)(x)

    # On the omnibias one-layer field, the analytic Laplacian and the
    # autograd-derived Laplacian must agree to float64 precision.
    np.testing.assert_allclose(closed.x, autograd.x, atol=1e-13)
    np.testing.assert_allclose(closed.dense_jacobian, autograd.dense_jacobian, atol=1e-11)
    np.testing.assert_allclose(closed.laplacian, autograd.laplacian, atol=1e-9)


def test_closed_form_path_requires_fastpath_activation():
    """An activation with no registered fastpath fails loudly."""
    x, W, beta, c = _to_jax(X_NP, W_NP, BETA_NP, C_NP)
    with pytest.raises(
        (KeyError, ValueError, NotImplementedError), match="[Uu]nknown activation"
    ):
        closed_form_forward_laplacian(x, W, beta, c, B_NP, "not_a_registered_activation")


def test_closed_form_path_relu_uses_ae_regular_part_convention():
    """relu ships an almost-everywhere fastpath, so it does not raise.

    relu is in the piecewise "regular-part all-orders" tower: ``sigma^(n>=2)``
    drops the singular delta at the kink (see ``omnibias.jax.activations``). For
    a one-layer field the affine pre-activation makes the closed-form Laplacian
    the finite regular part (exactly ``0``), *not* a raised error. The smooth
    twin ``soft_relu`` carries the tempered higher derivatives when the singular
    part matters.
    """
    x, W, beta, c = _to_jax(X_NP, W_NP, BETA_NP, C_NP)
    out = closed_form_forward_laplacian(x, W, beta, c, B_NP, "relu")
    lap = np.asarray(out.laplacian)
    assert np.isfinite(lap).all()
    np.testing.assert_allclose(lap, 0.0, atol=1e-12)


def test_forward_laplacian_closed_form_flag_not_yet_supported():
    """closed_form=True is reserved for the upcoming JAX interpreter."""
    with pytest.raises(NotImplementedError, match="Tier-3"):
        forward_laplacian(_scalar_field_tanh, closed_form=True)
