# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Information-geometry bridge (JAX): exponential-family Fisher-Rao metric.

Bit-identical twin of :mod:`omnibias.geometry.torch.ops.fisher`.

The Fisher information of an exponential family is a Riemannian metric, so it
plugs straight into the connection / curvature operators of this package. For
``d`` natural-parameter coordinates sharing the scalar log-partition ``base``
(the GLM activation, e.g. ``softplus`` -> Bernoulli, ``exp`` -> Poisson), the
Fisher-Rao metric in natural coordinates is the diagonal matrix

.. math::

    g_{ij}(\eta) = A''(\eta_i)\,\delta_{ij},

with ``A'' `` the closed-form second cumulant of the omnibias derivative tower
(:func:`omnibias.jax.information.glm_variance`). :func:`exponential_family_fisher_metric`
returns a :class:`~omnibias.geometry._core.manifold.MetricSpec`, after which
:func:`omnibias.geometry.jax.ops.christoffel`,
:func:`~omnibias.geometry.jax.ops.scalar_curvature`, etc. consume it unchanged.

Because a product of one-dimensional metrics is flat, the Levi-Civita curvature
of this metric vanishes (the exponential family is dually flat) while the
Christoffel symbols ``Gamma^k_{kk} = A'''(eta_k) / (2 A''(eta_k))`` are non-zero
-- a closed-form cross-check tying the geometry pipeline back to the tower.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.geometry._core.manifold import ManifoldSpec, MetricSpec
from omnibias.jax.information import glm_variance


def exponential_family_fisher_metric(
    *, base: str = "softplus", dim: int = 2, name: str | None = None
) -> MetricSpec:
    r"""Fisher-Rao :class:`MetricSpec` ``diag(A''(eta_k))`` for ``dim`` coordinates."""

    def g_point(eta: Array) -> Array:
        return jnp.diag(glm_variance(eta, base=base))

    return MetricSpec(g_point=g_point, dim=dim, name=name or f"fisher[{base}]")


def exponential_family_fisher_manifold(
    *, base: str = "softplus", dim: int = 2, name: str | None = None
) -> ManifoldSpec:
    r"""Convenience :class:`ManifoldSpec` wrapping :func:`exponential_family_fisher_metric`."""
    metric = exponential_family_fisher_metric(base=base, dim=dim)
    return ManifoldSpec(
        name=name or f"exp_family_fisher[{base}]", dim=dim, metric=metric
    )


__all__ = [
    "exponential_family_fisher_manifold",
    "exponential_family_fisher_metric",
]
