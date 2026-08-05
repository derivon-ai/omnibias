# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Characteristic classes / Chern-Weil numbers (torch).

The non-abelian topological charge density

.. math::

    q(x) = \frac{1}{32\pi^2}\,\varepsilon^{\mu\nu\rho\sigma}
        F_{\mu\nu}^a F_{\rho\sigma}^a

is metric-independent (it uses the Levi-Civita *symbol*), and its integral is the
integer second Chern number / instanton number ``c_2``. This module also exposes
the abelian first Chern class ``c_1`` (for a ``U(1)`` field strength), the first
Pontryagin number ``p_1`` and the ``SO(2)`` Euler number.

Honesty
-------
The densities are closed-form in the (closed-form) field strength; the
characteristic *numbers* are numerical integrals (a quadrature ``weights``
contraction) whose **integer** value can be certified by an
:class:`omnibias.core.verified.Interval` enclosure. The higher-rank Euler class
via the Pfaffian, and non-perturbative index theorems, are out of thesis.
"""

from __future__ import annotations

import math

import torch
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge.torch.ops.hodge import levi_civita
from torch import Tensor

_TWO_PI = 2.0 * math.pi


def topological_charge_density(F: Tensor) -> Tensor:
    r"""Charge density ``(1/32 pi^2) eps^{mu nu rho sigma} F_{mu nu}^a F_{rho sigma}^a``."""
    eps = levi_civita(F.shape[1], dtype=F.dtype, device=F.device)
    return kernels.topological_charge_density(torch, F, eps)


def topological_charge(F: Tensor, *, weights: Tensor | None = None) -> Tensor:
    r"""Integrated topological charge (instanton number) as a scalar."""
    density = topological_charge_density(F)
    if weights is None:
        return torch.sum(density)
    return torch.sum(density * weights)


def second_chern_number(F: Tensor, *, weights: Tensor | None = None) -> Tensor:
    """Alias of :func:`topological_charge` (the second Chern number ``c_2``)."""
    return topological_charge(F, weights=weights)


def first_chern_class(field_strength_2form: Tensor) -> Tensor:
    r"""Abelian first Chern class 2-form ``c_1 = F / (2 pi)`` for a ``U(1)`` bundle.

    ``field_strength_2form`` is the real (abelian) field strength ``F_{mu nu}`` of
    shape ``(B, d, d)`` (antisymmetric in ``mu, nu``) -- e.g. the electromagnetic
    field-strength tensor. Returns ``F/(2 pi)``, the de Rham representative of
    ``c_1``; integrate it over a closed 2-surface with :func:`first_chern_number`.
    """
    return field_strength_2form / _TWO_PI


def first_chern_number(
    field_strength_2form: Tensor,
    *,
    plane: tuple[int, int] = (0, 1),
    weights: Tensor | None = None,
) -> Tensor:
    r"""First Chern number ``c_1 = (1/2 pi) \int F_{plane}`` of a ``U(1)`` bundle.

    Integrates the ``(plane[0], plane[1])`` component of the abelian field
    strength ``F`` over a closed 2-surface (the quadrature ``weights``). For a
    Dirac monopole of charge ``n`` this is exactly the integer ``n``.
    """
    i, j = plane
    density = field_strength_2form[:, i, j] / _TWO_PI
    if weights is None:
        return torch.sum(density)
    return torch.sum(density * weights)


def euler_number_so2(
    field_strength_2form: Tensor,
    *,
    plane: tuple[int, int] = (0, 1),
    weights: Tensor | None = None,
) -> Tensor:
    r"""Euler number of a rank-2 (``SO(2) \cong U(1)``) bundle ``= c_1``.

    For an oriented rank-2 real bundle the Euler class equals the first Chern
    class of the associated complex line bundle, so this is an alias of
    :func:`first_chern_number`. The Euler *characteristic of a surface* (rather
    than of a bundle) is :func:`omnibias.geometry...gauss_bonnet_euler`.
    """
    return first_chern_number(field_strength_2form, plane=plane, weights=weights)


def pontryagin_density(F: Tensor) -> Tensor:
    r"""First Pontryagin density ``p_1 = -2 c_2`` of an ``SU(N)`` bundle.

    Uses the textbook relation ``p_1 = c_1^2 - 2 c_2`` with ``c_1 = 0`` for an
    ``SU(N)`` bundle, so ``p_1 = -2 c_2``; here ``c_2`` is the closed-form
    :func:`topological_charge_density`.
    """
    return -2.0 * topological_charge_density(F)


def pontryagin_number(F: Tensor, *, weights: Tensor | None = None) -> Tensor:
    r"""First Pontryagin number ``p_1 = -2 c_2`` (``SU(N)``, ``c_1 = 0``)."""
    return -2.0 * topological_charge(F, weights=weights)


def is_quantized(charge: Tensor | float, *, tol: float = 1e-2) -> bool:
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
