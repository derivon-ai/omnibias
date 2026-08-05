# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Characteristic classes / Chern-Weil numbers (jax).

The jax twin of :mod:`omnibias.geometry.gauge.torch.ops.topology`; see that module for
the full operator and honesty documentation.
"""

from __future__ import annotations

import math
from typing import Any

import jax.numpy as jnp
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge.jax.ops.hodge import levi_civita

Array = Any

_TWO_PI = 2.0 * math.pi


def topological_charge_density(F: Array) -> Array:
    r"""Charge density ``(1/32 pi^2) eps^{mu nu rho sigma} F_{mu nu}^a F_{rho sigma}^a``."""
    eps = levi_civita(F.shape[1], dtype=F.dtype)
    return kernels.topological_charge_density(jnp, F, eps)


def topological_charge(F: Array, *, weights: Array | None = None) -> Array:
    r"""Integrated topological charge (instanton number) as a scalar."""
    density = topological_charge_density(F)
    if weights is None:
        return jnp.sum(density)
    return jnp.sum(density * weights)


def second_chern_number(F: Array, *, weights: Array | None = None) -> Array:
    """Alias of :func:`topological_charge` (the second Chern number ``c_2``)."""
    return topological_charge(F, weights=weights)


def first_chern_class(field_strength_2form: Array) -> Array:
    r"""Abelian first Chern class 2-form ``c_1 = F / (2 pi)`` for a ``U(1)`` bundle."""
    return field_strength_2form / _TWO_PI


def first_chern_number(
    field_strength_2form: Array,
    *,
    plane: tuple[int, int] = (0, 1),
    weights: Array | None = None,
) -> Array:
    r"""First Chern number ``c_1 = (1/2 pi) \int F_{plane}`` of a ``U(1)`` bundle."""
    i, j = plane
    density = field_strength_2form[:, i, j] / _TWO_PI
    if weights is None:
        return jnp.sum(density)
    return jnp.sum(density * weights)


def euler_number_so2(
    field_strength_2form: Array,
    *,
    plane: tuple[int, int] = (0, 1),
    weights: Array | None = None,
) -> Array:
    r"""Euler number of a rank-2 (``SO(2) \cong U(1)``) bundle ``= c_1``."""
    return first_chern_number(field_strength_2form, plane=plane, weights=weights)


def pontryagin_density(F: Array) -> Array:
    r"""First Pontryagin density ``p_1 = -2 c_2`` of an ``SU(N)`` bundle."""
    return -2.0 * topological_charge_density(F)


def pontryagin_number(F: Array, *, weights: Array | None = None) -> Array:
    r"""First Pontryagin number ``p_1 = -2 c_2`` (``SU(N)``, ``c_1 = 0``)."""
    return -2.0 * topological_charge(F, weights=weights)


def is_quantized(charge: Array | float, *, tol: float = 1e-2) -> bool:
    """Whether ``charge`` is within ``tol`` of an integer."""
    q = float(charge)
    return abs(q - round(q)) <= tol


__all__ = [
    "euler_number_so2",
    "first_chern_class",
    "first_chern_number",
    "is_quantized",
    "pontryagin_density",
    "pontryagin_number",
    "second_chern_number",
    "topological_charge",
    "topological_charge_density",
]
