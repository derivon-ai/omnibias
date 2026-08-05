# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Analytic moment propagation (JAX) -- bit-identical twin of the torch module.

Pushes the moments of a random input through a network in closed form using the
exact multivariate jet (:func:`omnibias.jax.jet_mv.mlp_jet_mv`).  See
:mod:`omnibias.torch.moments` for the mathematical description.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core.moments import (
    delta_method_central_moments,
    gaussian_central_moments,
)
from omnibias.core.spec import ActivationSpec
from omnibias.jax.jet_mv import jet_gradient, jet_hessian, mlp_jet_mv

import jax.numpy as jnp
from jax import Array

Layer = tuple[Array, Array | None, ActivationSpec[Array] | str | None]


def gaussian_moment_propagation(
    layers: Sequence[Layer],
    mean: Array,
    cov: Array,
    *,
    order: int = 2,
) -> tuple[Array, Array]:
    r"""Propagate a Gaussian ``N(mean, cov)`` through an MLP analytically.

    See :func:`omnibias.torch.moments.gaussian_moment_propagation`.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    mean = jnp.asarray(mean)
    cov = jnp.asarray(cov)
    dim = int(mean.shape[-1])
    jet_order = max(order, 2) if order >= 2 else 1
    jet = mlp_jet_mv(mean, layers, jet_order)
    value = jet[0]
    jac = jet_gradient(jet, dim, jet_order)  # (D, C)
    out_cov = jnp.einsum("ic,ij,jd->cd", jac, cov, jac)
    if order >= 2:
        hess = jet_hessian(jet, dim, jet_order)  # (D, D, C)
        correction = 0.5 * jnp.einsum("ijc,ij->c", hess, cov)
        out_mean = value + correction
    else:
        out_mean = value
    return out_mean, out_cov


def delta_method_moments(
    derivatives: Sequence[Array],
    central_in: Sequence[float | Array],
    *,
    order: int,
) -> dict[str, object]:
    r"""Elementwise analytic delta method from a derivative tower + central moments.

    See :func:`omnibias.torch.moments.delta_method_moments`.
    """
    return delta_method_central_moments(list(derivatives), list(central_in), order)


def delta_method_gaussian(
    derivatives: Sequence[Array],
    variance: float | Array,
    *,
    order: int,
) -> dict[str, object]:
    """:func:`delta_method_moments` for a Gaussian input of the given ``variance``."""
    central = gaussian_central_moments(variance, order)
    return delta_method_central_moments(list(derivatives), central, order)


__all__ = [
    "delta_method_gaussian",
    "delta_method_moments",
    "gaussian_moment_propagation",
]
