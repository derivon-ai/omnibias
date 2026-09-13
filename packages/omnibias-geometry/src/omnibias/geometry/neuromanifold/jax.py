# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Live jax realization geometry using the shared parameter jet substrate."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.core.realization import RealizationSpec
from omnibias.jax.realization import evaluate, parameter_jet

from .geometry import Realization


def dense_realization(
    points: Array, spec: RealizationSpec, *, max_coefficients: int = 1_000_000
) -> Realization[Array]:
    """Finite value observations of an explicit dense six-role architecture."""

    def value(theta: Array) -> Array:
        return evaluate(points, theta, spec).reshape(-1)

    def jvp(theta: Array, direction: Array) -> Array:
        return parameter_jet(
            points, theta, spec, direction.reshape(-1, 1), 1, max_coefficients=max_coefficients
        )[1].reshape(-1)

    def vjp(theta: Array, cotangent: Array) -> Array:
        _, pullback = jax.vjp(value, theta)
        return cast(Array, pullback(cotangent)[0])

    def jet(theta: Array, basis: Array, order: int) -> Array:
        result = parameter_jet(points, theta, spec, basis, order, max_coefficients=max_coefficients)
        return result.reshape(result.shape[0], -1)

    return Realization(value, jvp, vjp, jet, spec)


def observation_realization(
    value: Callable[[Array], Array],
    *,
    jet: Callable[[Array, Array, int], Array],
    scope: str = "finite_observations",
    derivative_provenance: str = "supplied_analytic_jets",
) -> Realization[Array]:
    """Callback observations with an explicit jet provider; no ignored model."""

    def jvp(theta: Array, direction: Array) -> Array:
        return jet(theta, direction.reshape(-1, 1), 1)[1]

    def vjp(theta: Array, cotangent: Array) -> Array:
        _, pullback = jax.vjp(value, theta)
        return cast(Array, pullback(cotangent)[0])

    return Realization(
        value, jvp, vjp, jet, observation_scope=scope, derivative_provenance=derivative_provenance
    )


def dense_jacobian(
    realization: Realization[Array], theta: Array, *, max_entries: int = 1_000_000
) -> Array:
    """Explicit, budgeted dense Jacobian; metric_action avoids this allocation."""
    if theta.ndim != 1 or theta.shape[0] * realization.value(theta).shape[0] > max_entries:
        raise ValueError("dense observation Jacobian budget exceeded")
    basis = jnp.eye(theta.shape[0], dtype=theta.dtype)
    return jnp.stack([realization.jvp(theta, basis[i]) for i in range(theta.shape[0])], axis=1)


__all__ = ["dense_jacobian", "dense_realization", "observation_realization"]
