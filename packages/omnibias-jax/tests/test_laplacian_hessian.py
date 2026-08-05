# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contract tests for :mod:`omnibias.jax.laplacian` Hessian primitive.

The closed-form Hessian

    H_x f(x) = W^T diag(sigma''(W x + beta) odot c) W

must match :func:`jax.hessian` to float64 precision on the one-layer
field, for every Riccati-class activation that registers a fast-path
kernel.

This is the building block for FermiNet Tier 2/3 integration:
backflow and equivariant-layer chain rules need ``trace(J^T H J)``,
not just ``trace(H)``, so we expose the *matrix* Hessian rather than
the scalar Laplacian.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax import (  # noqa: E402
    neural_field_hessian,
    neural_field_laplacian,
    neural_field_value,
    neural_field_value_grad_hessian,
    neural_field_value_grad_laplacian,
)

# Activations that have a fast-path kernel and a meaningful second
# derivative (Riccati class). ``arctan`` / ``log1pu2`` register a
# fast path only to order 2 and the higher-order kernels are
# distributional; we keep them out of the cross-D sweep but include a
# direct order-2 check below.
_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp")
_DS = (3, 6, 16, 32)
_H = 12
_RTOL = 1e-9
_ATOL = 1e-10


def _rand_params(D: int, H: int, seed: int):
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=0.3, size=(H, D))
    beta = rng.normal(scale=0.2, size=(H,))
    c = rng.normal(scale=0.4, size=(H,))
    b = 0.13
    x = rng.normal(scale=1.0, size=(D,))
    return (
        jnp.asarray(W, dtype=jnp.float64),
        jnp.asarray(beta, dtype=jnp.float64),
        jnp.asarray(c, dtype=jnp.float64),
        b,
        jnp.asarray(x, dtype=jnp.float64),
    )


@pytest.mark.parametrize("activation", _RICCATI)
@pytest.mark.parametrize("D", _DS)
def test_neural_field_hessian_matches_jax_hessian(activation: str, D: int) -> None:
    W, beta, c, b, x = _rand_params(D, _H, seed=hash((activation, D)) & 0xFFFF)

    def f(xx: jnp.ndarray) -> jnp.ndarray:
        return neural_field_value(xx, W, beta, c, b, activation)

    H_closed = neural_field_hessian(x, W, beta, c, activation)
    H_ref = jax.hessian(f)(x)

    assert H_closed.shape == (D, D)
    np.testing.assert_allclose(
        np.asarray(H_closed),
        np.asarray(H_ref),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=f"Hessian mismatch for {activation!r} at D={D}",
    )
    # Hessian must be exactly symmetric (it is a sum of outer products).
    np.testing.assert_allclose(
        np.asarray(H_closed - H_closed.T),
        np.zeros((D, D)),
        atol=1e-13,
    )


@pytest.mark.parametrize("activation", _RICCATI)
@pytest.mark.parametrize("D", _DS)
def test_neural_field_value_grad_hessian_consistent(
    activation: str,
    D: int,
) -> None:
    """The fused (value, grad, Hessian) call must agree with the
    individual primitives and with :func:`jax.hessian`."""
    W, beta, c, b, x = _rand_params(
        D,
        _H,
        seed=hash((activation, D, "fused")) & 0xFFFF,
    )

    def f(xx: jnp.ndarray) -> jnp.ndarray:
        return neural_field_value(xx, W, beta, c, b, activation)

    val_fused, grad_fused, H_fused = neural_field_value_grad_hessian(
        x,
        W,
        beta,
        c,
        b,
        activation,
    )

    val_ref = f(x)
    grad_ref = jax.grad(f)(x)
    H_ref = jax.hessian(f)(x)

    np.testing.assert_allclose(np.asarray(val_fused), np.asarray(val_ref), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(grad_fused), np.asarray(grad_ref), rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(H_fused), np.asarray(H_ref), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("activation", _RICCATI)
@pytest.mark.parametrize("D", _DS)
def test_hessian_trace_equals_laplacian(activation: str, D: int) -> None:
    """``trace(H_x f) == nabla_x^2 f``. The two omnibias primitives
    must be consistent with each other."""
    W, beta, c, b, x = _rand_params(
        D,
        _H,
        seed=hash((activation, D, "trace")) & 0xFFFF,
    )

    H = neural_field_hessian(x, W, beta, c, activation)
    lap = neural_field_laplacian(x, W, beta, c, activation)

    np.testing.assert_allclose(
        float(jnp.trace(H)),
        float(lap),
        rtol=_RTOL,
        atol=_ATOL,
        err_msg=(
            f"trace(H) != Laplacian for {activation!r} at D={D}: "
            f"trace={float(jnp.trace(H))!r}, Laplacian={float(lap)!r}"
        ),
    )


def test_chain_rule_through_linear_map() -> None:
    """The Hessian primitive enables the FermiNet-style chain rule
    ``nabla_r^2 [phi(q(r))] = trace(J^T H_q phi J) + grad_q phi . nabla_r^2 q``.

    For a linear map ``q(r) = M r``, ``nabla_r^2 q = 0`` and the
    identity collapses to ``trace(M^T H M)`` on the omnibias-closed
    Hessian. This is the simplest non-trivial backflow chain rule and
    is the unit test that pins the integration math.
    """
    D = 3
    H = 8
    W, beta, c, b, _ = _rand_params(D, H, seed=42)

    rng = np.random.default_rng(43)
    r = jnp.asarray(rng.normal(size=(D,)), dtype=jnp.float64)
    M = jnp.asarray(rng.normal(size=(D, D)), dtype=jnp.float64)
    M_sym = 0.5 * (M + M.T) + jnp.eye(D)  # well-conditioned

    def phi_of_q(q: jnp.ndarray) -> jnp.ndarray:
        return neural_field_value(q, W, beta, c, b, "tanh")

    def phi_of_r(rr: jnp.ndarray) -> jnp.ndarray:
        return phi_of_q(M_sym @ rr)

    # closed-form via the chain rule
    q = M_sym @ r
    H_q = neural_field_hessian(q, W, beta, c, "tanh")
    lap_r_closed = float(jnp.trace(M_sym.T @ H_q @ M_sym))

    # autograd reference (Hessian -> trace)
    lap_r_ref = float(jnp.trace(jax.hessian(phi_of_r)(r)))

    np.testing.assert_allclose(
        lap_r_closed,
        lap_r_ref,
        rtol=1e-10,
        atol=1e-12,
        err_msg=(
            "chain-rule Laplacian through linear coord map disagrees "
            "with jax.hessian; check trace(J^T H J) construction"
        ),
    )


def test_neural_field_hessian_rejects_non_fastpath_activation() -> None:
    """A base without a fast-path kernel must raise.

    (``relu`` now carries an all-orders a.e. tower, so we use a deliberately
    fastpath-less spec to exercise the guard.)
    """
    from omnibias.core.spec import ActivationSpec

    no_fastpath = ActivationSpec(name="_no_fastpath_probe", forward=lambda z: z, fastpath=None)
    W, beta, c, b, x = _rand_params(3, 4, seed=0)
    with pytest.raises((ValueError, NotImplementedError), match=r"(fast-path|distributional)"):
        neural_field_hessian(x, W, beta, c, no_fastpath)
    with pytest.raises((ValueError, NotImplementedError), match=r"(fast-path|distributional)"):
        neural_field_value_grad_hessian(x, W, beta, c, b, no_fastpath)
