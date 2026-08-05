# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Asymptotic / removable boundary-condition losses (jax twin).

These helpers turn the differentiable jet-``lim`` operator into trainable PINN
losses. They are built on :func:`omnibias.jax.mlp_jet` (the *exact* directional
Taylor jet of a deep MLP along a ray ``x(t) = x0 + t v``) and
:func:`omnibias.jax.lhopital_ratio` (the differentiable L'Hopital limit), so a
PINN can impose conditions that live *at a limit*:

* **removable regularity** at a singular point -- e.g.
  ``lim_{t->0} N(x0 + t v) / t**p = c``: a finite slope / curvature where a PDE
  coefficient blows up (the classic ``u(r)/r`` regularity at ``r = 0``); and
* **far-field decay** -- pin the field and its directional derivatives toward
  zero at a far base point so the solution flattens at the boundary.

Unlike a quadrature or finite-difference boundary penalty, the limit is taken in
*closed form* from the network's Taylor coefficients: it is exact at the base
point, differentiable (the boundary condition backpropagates into the weights),
and ``jit`` / ``vmap`` friendly. The exponent / target may itself be learnable.

Bit-identical twin of :mod:`omnibias.pinn.torch.losses.asymptotic`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.jax.jet import lhopital_ratio, mlp_jet

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from omnibias.jax.activations import JaxActivationSpec

    JaxLayer = tuple[Array, Array | None, JaxActivationSpec | str | None]


def _power_jet(rate: int, order: int, like: Array) -> Array:
    """Taylor jet of ``t**rate`` truncated at ``order``.

    With the convention ``jet[k] = f^(k)(0)/k!`` the only non-zero coefficient of
    ``t**rate`` sits at index ``rate`` and equals ``1``.
    """
    coeffs = jnp.zeros((order + 1,), dtype=like.dtype)
    return coeffs.at[rate].set(jnp.asarray(1.0, dtype=like.dtype))


def network_ray_jet(
    layers: Sequence[JaxLayer],
    base_point: Array,
    direction: Array,
    *,
    order: int,
    out_index: int = 0,
) -> Array:
    """Scalar Taylor jet of one network output along ``x0 + t v``.

    Thin wrapper over :func:`omnibias.jax.mlp_jet` that selects output channel
    ``out_index``; returns an array of shape ``(order + 1,)`` whose ``k``-th
    entry is ``(d^k/dt^k N(x0 + t v))|_{t=0} / k!``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    jet = mlp_jet(base_point, direction, layers, order)
    return jet[:, out_index]


def asymptotic_ratio(
    layers: Sequence[JaxLayer],
    base_point: Array,
    direction: Array,
    *,
    rate: int = 1,
    order: int | None = None,
    out_index: int = 0,
) -> Array:
    r"""Differentiable ``lim_{t->0} N(x0 + t v) / t**rate`` via L'Hopital.

    ``rate = 0`` returns the limiting value ``N(x0)``; ``rate = 1`` the
    directional slope ``(d/dt) N(x0 + t v)|_{0}``; a higher ``rate`` resolves a
    deeper ``0/0`` form (a removable singularity of order ``rate``). The result
    backpropagates into ``layers`` so it can drive a trainable boundary
    condition. ``order`` controls the jet truncation and defaults to ``rate``
    (the minimum needed to read the leading coefficient).
    """
    if rate < 0:
        raise ValueError(f"rate must be >= 0, got {rate}")
    resolved_order = rate if order is None else order
    if resolved_order < rate:
        raise ValueError(
            f"order {resolved_order} must be >= rate {rate} to resolve the limit"
        )
    net_jet = network_ray_jet(
        layers, base_point, direction, order=resolved_order, out_index=out_index
    )
    den = _power_jet(rate, resolved_order, net_jet)
    return lhopital_ratio(net_jet, den, order=rate)


def asymptotic_bc_loss(
    layers: Sequence[JaxLayer],
    base_point: Array,
    direction: Array,
    *,
    target: Array | float = 0.0,
    rate: int = 1,
    order: int | None = None,
    out_index: int = 0,
    weight: float = 1.0,
) -> Array:
    r"""Squared-error asymptotic / removable boundary condition as a loss.

    Returns ``weight * (asymptotic_ratio(...) - target) ** 2``. With ``rate = 0``
    and a far ``base_point`` this pins the far-field *value*; with ``rate >= 1``
    it imposes the removable-regularity condition
    ``lim_{t->0} N(x0 + t v) / t**rate = target``.
    """
    value = asymptotic_ratio(
        layers, base_point, direction, rate=rate, order=order, out_index=out_index
    )
    diff = value - jnp.asarray(target, dtype=value.dtype)
    return jnp.asarray(weight, dtype=value.dtype) * diff * diff


def far_field_decay_loss(
    layers: Sequence[JaxLayer],
    base_point: Array,
    direction: Array,
    *,
    order: int = 1,
    out_index: int = 0,
    weight: float = 1.0,
) -> Array:
    r"""Penalize a network's value and directional derivatives toward zero.

    Drives the Taylor coefficients ``jet[0 .. order]`` of ``N(x0 + t v)`` toward
    ``0`` -- the field flattens (value, slope, curvature, ...) at the base point.
    Evaluated at a far ``base_point`` this is a differentiable far-field decay
    condition. Returns ``weight * mean(jet[:order + 1] ** 2)``; the factorial
    weighting of the Taylor coefficients naturally emphasises the value and slope.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    net_jet = network_ray_jet(
        layers, base_point, direction, order=order, out_index=out_index
    )
    return jnp.asarray(weight, dtype=net_jet.dtype) * jnp.mean(net_jet * net_jet)


__all__ = [
    "asymptotic_bc_loss",
    "asymptotic_ratio",
    "far_field_decay_loss",
    "network_ray_jet",
]
