# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX smooth-surrogate towers / jets and a curvature-aware quantizer backward.

These go *beyond* the straight-through estimator: instead of treating the hard
quantizer's backward as the identity, omnibias uses the exact derivative tower of
the smooth ``tanh(beta z)`` surrogate (one ``tanh`` evaluation, any order, via the
Riccati polynomials). The full Taylor jet of the surrogate -- built with the same
:func:`omnibias.jax.jet.compose_jet` machinery as Phase 1 -- carries the local
*curvature* of the surrogate, not just its slope, enabling 2nd-order-corrected
("jet-STE") quantizer gradients.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.binary.jax.ops.quantize import riccati_tanh_derivative
from omnibias.jax.jet import compose_jet

__all__ = [
    "binarize_curvature",
    "curvature_corrected_slope",
    "surrogate_jet",
    "surrogate_tower",
]


def surrogate_tower(z: Array, beta: float | Array, order: int) -> Array:
    r"""Derivative tower ``[s, s', ..., s^(order)]`` of ``s(z) = tanh(beta z)``.

    Row ``k`` is ``d^k/dz^k tanh(beta z) = beta^k * T_k(tanh(beta z))`` where
    ``T_k`` is the Riccati tanh polynomial; a single ``tanh`` evaluation suffices
    regardless of order. Differentiable in both ``z`` and ``beta``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = jnp.asarray(z)
    t = jnp.tanh(beta * z)
    rows = [t]
    for k in range(1, order + 1):
        rows.append((beta**k) * riccati_tanh_derivative(t, order=k))
    return jnp.stack(rows, axis=0)


def surrogate_jet(z: Array, beta: float | Array, order: int) -> Array:
    r"""Taylor jet of ``u -> tanh(beta u)`` at ``u = z`` via :func:`compose_jet`.

    Built by composing the exact ``tanh`` derivative tower onto the affine
    pre-activation jet of ``beta * (z + t)``; because the inner map is affine the
    result equals ``tower_to_jet(surrogate_tower(z, beta, order))`` -- a useful
    cross-check exercised by the test-suite. ``jet[1]`` is the surrogate slope
    (the standard backward), ``jet[2]`` its curvature, and so on: this is the
    jet-STE backward signal.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    z = jnp.asarray(z)
    t = jnp.tanh(beta * z)
    # sigma^(k)(u0) for tanh at u0 = beta z (derivative w.r.t. tanh's argument).
    sigma_rows = [t] + [riccati_tanh_derivative(t, order=k) for k in range(1, order + 1)]
    sigma_tower = jnp.stack(sigma_rows, axis=0)
    # Pre-activation jet of u(t) = beta * (z + t): [beta z, beta, 0, ...].
    u_rows = [beta * z, jnp.full_like(z, 1.0) * beta]
    u_rows += [jnp.zeros_like(z) for _ in range(order - 1)]
    u_jet = jnp.stack(u_rows[: order + 1], axis=0)
    return compose_jet(u_jet, sigma_tower)


def curvature_corrected_slope(
    z: Array, beta: float | Array, *, window: float | None = None
) -> Array:
    r"""Windowed-average surrogate slope ``s'(z) + (h^2/6) s'''(z)`` (``h = window``).

    The point slope ``s'(z)`` is replaced by the average of ``s'`` over a
    symmetric window of half-width ``h`` (default ``h = 1/beta``, the surrogate's
    natural smoothing scale):

    .. math:: \frac{1}{2h}\int_{-h}^{h} s'(z + u)\,du = s'(z) + \frac{h^2}{6} s'''(z) + O(h^4).

    Using the exact 3rd-order tower this is a better *effective* gradient through
    the hard step than the point slope alone (it accounts for the surrogate's
    finite transition width). Reduces to ``s'(z)`` as ``h -> 0``.
    """
    h = (1.0 / beta) if window is None else window
    tower = surrogate_tower(z, beta, order=3)
    return tower[1] + (h * h / 6.0) * tower[3]


def _binarize_curv_fwd(z: Array, beta: float) -> tuple[Array, tuple[Array, float]]:
    y = jnp.where(z >= 0, 1.0, -1.0)
    return y, (z, beta)


def _binarize_curv_bwd(res: tuple[Array, float], grad_out: Array) -> tuple[Array, None]:
    z, beta = res
    return grad_out * curvature_corrected_slope(z, beta), None


@jax.custom_vjp
def binarize_curvature(z: Array, beta: float = 10.0) -> Array:
    """Hard ``sign(z)`` with a 2nd-order, curvature-corrected surrogate backward.

    Identical hard forward to :func:`omnibias.binary.jax.ops.binarize`; the
    backward uses :func:`curvature_corrected_slope` (the windowed-average slope)
    instead of the point slope. ``beta`` is a fixed hyperparameter here.
    """
    return jnp.where(z >= 0, 1.0, -1.0)


binarize_curvature.defvjp(_binarize_curv_fwd, _binarize_curv_bwd)
