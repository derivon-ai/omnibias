# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trigonometric & hyperbolic activations for physics-class wavefunctions.

These activations target physics-class wavefunctions and
periodic / lattice-Bloch neural-VMC. Every spec exposes a closed-form n-th
derivative; the supported orders depend on the activation:

==================  =========================================  ===================
Activation          n-th derivative formula                    Max fastpath order
==================  =========================================  ===================
``sin(z)``          ``sin(z + n * pi/2)``                      every n
``cos(z)``          ``cos(z + n * pi/2)``                      every n
``sinh(z)``         alternating ``sinh / cosh``                every n
``cosh(z)``         alternating ``cosh / sinh``                every n
``tan(z)``          Riccati ``P(t) = 1 + t^2``                 n in {0, 1, 2, 3}
``cot(z)``          Riccati ``P(c) = -(1 + c^2)``              n in {0, 1, 2, 3}
``sech(z)``         ``sech * (1 - 2 sech^2)`` etc.             n in {0, 1, 2}
``coth(z)``         Riccati ``P(c) = 1 - c^2`` (tanh-like)     n in {0, 1, 2, 3}
==================  =========================================  ===================

All four hyperbolic activations are closed-form for arbitrarily many orders;
the support cap on ``tan`` / ``cot`` / ``sech`` is purely a development cost
choice — the recursions are well known and can be extended on demand.

Physical motivation
-------------------

* ``sin`` / ``cos`` — plane-wave embeddings for periodic systems
  (Bloch wavefunctions on a lattice; k-point sampling for crystalline
  Hamiltonians).
* ``sinh`` / ``cosh`` — anti-Bragg / evanescent states; logarithmic
  derivatives of Slater determinants in 1D models with linear backbone
  symmetry.
* ``sech`` — soliton wavefunctions (sine-Gordon, KdV, NLS); Pöschl-Teller
  potential bound states; the canonical ``sech^2`` reflectionless
  potential.
* ``tan`` / ``cot`` — phase-only activations for unitary RNN-style
  amplitude/phase factorisations, and as raw building blocks for the
  envelope-modulator branch of period-aware ansatze.
* ``coth`` — Langevin-function-like saturation; thermal-state amplitude
  in finite-T NQS. Riccati-class (same polynomial shape as tanh).
"""

from __future__ import annotations

import math

from omnibias.torch.activations.registry import ActivationSpec, register_activation
from omnibias.torch.transforms import (
    COS_TRANSFORMS,
    COSH_TRANSFORMS,
    SECH_TRANSFORMS,
    SIN_TRANSFORMS,
    SINH_TRANSFORMS,
)

import torch
from torch import Tensor

_HALF_PI = 0.5 * math.pi


# ---------------------------------------------------------------------------
# Tier 1 — closed-form for every order n
# ---------------------------------------------------------------------------


def _sin_forward(z: Tensor) -> Tensor:
    return torch.sin(z)


def _sin_derivative(z: Tensor) -> Tensor:
    return torch.cos(z)


def _sin_integral(z: Tensor) -> Tensor:
    return -torch.cos(z)


def _sin_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return torch.sin(z + n * _HALF_PI)


SIN = register_activation(
    ActivationSpec(
        name="sin",
        transforms=SIN_TRANSFORMS,
        forward=_sin_forward,
        derivative=_sin_derivative,
        fastpath=_sin_fastpath,
        integral=_sin_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Plane-wave / Bloch embedding; eigenfunction of d^2/dz^2 with "
            "eigenvalue -1; sin(z + n*pi/2) gives every order in closed form."
        ),
    )
)


def _cos_forward(z: Tensor) -> Tensor:
    return torch.cos(z)


def _cos_derivative(z: Tensor) -> Tensor:
    return -torch.sin(z)


def _cos_integral(z: Tensor) -> Tensor:
    return torch.sin(z)


def _cos_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return torch.cos(z + n * _HALF_PI)


COS = register_activation(
    ActivationSpec(
        name="cos",
        transforms=COS_TRANSFORMS,
        forward=_cos_forward,
        derivative=_cos_derivative,
        fastpath=_cos_fastpath,
        integral=_cos_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Plane-wave / Bloch embedding (real part); cos(z + n*pi/2) "
            "gives every order in closed form."
        ),
    )
)


def _sinh_forward(z: Tensor) -> Tensor:
    return torch.sinh(z)


def _sinh_derivative(z: Tensor) -> Tensor:
    return torch.cosh(z)


def _sinh_integral(z: Tensor) -> Tensor:
    return torch.cosh(z)


def _sinh_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return torch.sinh(z) if (n & 1) == 0 else torch.cosh(z)


SINH = register_activation(
    ActivationSpec(
        name="sinh",
        transforms=SINH_TRANSFORMS,
        forward=_sinh_forward,
        derivative=_sinh_derivative,
        fastpath=_sinh_fastpath,
        integral=_sinh_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Hyperbolic odd primitive; eigenfunction of d^2/dz^2 with "
            "eigenvalue +1; alternates with cosh under differentiation."
        ),
    )
)


def _cosh_forward(z: Tensor) -> Tensor:
    return torch.cosh(z)


def _cosh_derivative(z: Tensor) -> Tensor:
    return torch.sinh(z)


def _cosh_integral(z: Tensor) -> Tensor:
    return torch.sinh(z)


def _cosh_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return torch.cosh(z) if (n & 1) == 0 else torch.sinh(z)


COSH = register_activation(
    ActivationSpec(
        name="cosh",
        transforms=COSH_TRANSFORMS,
        forward=_cosh_forward,
        derivative=_cosh_derivative,
        fastpath=_cosh_fastpath,
        integral=_cosh_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Hyperbolic even primitive; eigenfunction of d^2/dz^2 with "
            "eigenvalue +1; alternates with sinh under differentiation."
        ),
    )
)


# ---------------------------------------------------------------------------
# Tier 2 — Riccati polynomials, closed form for n in {0, 1, 2, 3}
# ---------------------------------------------------------------------------


def _tan_forward(z: Tensor) -> Tensor:
    return torch.tan(z)


def _tan_derivative(z: Tensor) -> Tensor:
    t = torch.tan(z)
    return 1.0 + t * t


def _tan_integral(z: Tensor) -> Tensor:
    return -torch.log(torch.abs(torch.cos(z)))


def _tan_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _tan_forward(z)
    t = torch.tan(z)
    s2 = 1.0 + t * t  # sec^2 = 1 + tan^2
    if n == 1:
        return s2
    if n == 2:
        return 2.0 * t * s2
    if n == 3:
        return 2.0 * s2 * (1.0 + 3.0 * t * t)
    raise NotImplementedError(f"tan fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


TAN = register_activation(
    ActivationSpec(
        name="tan",
        forward=_tan_forward,
        derivative=_tan_derivative,
        fastpath=_tan_fastpath,
        integral=_tan_integral,
        riccati_polynomial=(1.0, 0.0, 1.0),  # P(t) = 1 + t^2
        noise_model="none",
        operator_role=(
            "Riccati-class periodic activation; tan'(z) = 1 + tan^2(z); "
            "phase factor in unitary-RNN amplitude/phase factorisations."
        ),
    )
)


def _cot_forward(z: Tensor) -> Tensor:
    return 1.0 / torch.tan(z)


def _cot_derivative(z: Tensor) -> Tensor:
    c = _cot_forward(z)
    return -(1.0 + c * c)


def _cot_integral(z: Tensor) -> Tensor:
    return torch.log(torch.abs(torch.sin(z)))


def _cot_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _cot_forward(z)
    c = _cot_forward(z)
    csc2 = 1.0 + c * c  # csc^2 = 1 + cot^2
    if n == 1:
        return -csc2
    if n == 2:
        return 2.0 * c * csc2
    if n == 3:
        return -2.0 * csc2 * (1.0 + 3.0 * c * c)
    raise NotImplementedError(f"cot fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


COT = register_activation(
    ActivationSpec(
        name="cot",
        forward=_cot_forward,
        derivative=_cot_derivative,
        fastpath=_cot_fastpath,
        integral=_cot_integral,
        riccati_polynomial=(-1.0, 0.0, -1.0),  # P(c) = -(1 + c^2)
        noise_model="none",
        operator_role=(
            "Riccati-class periodic activation; cot'(z) = -(1 + cot^2(z)); "
            "complementary to tan in phase-only ansatze."
        ),
        aliases=("ctg", "ctan"),
    )
)


def _coth_forward(z: Tensor) -> Tensor:
    return 1.0 / torch.tanh(z)


def _coth_derivative(z: Tensor) -> Tensor:
    c = _coth_forward(z)
    return 1.0 - c * c


def _coth_integral(z: Tensor) -> Tensor:
    return torch.log(torch.abs(torch.sinh(z)))


def _coth_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _coth_forward(z)
    c = _coth_forward(z)
    one_minus_c2 = 1.0 - c * c  # = -csch^2(z)
    if n == 1:
        return one_minus_c2
    if n == 2:
        return -2.0 * c * one_minus_c2  # 2 c (c^2 - 1)
    if n == 3:
        # d/dz (-2 c (1 - c^2)) = -2 (1 - c^2) (1 - 3 c^2)
        return -2.0 * one_minus_c2 * (1.0 - 3.0 * c * c)
    raise NotImplementedError(f"coth fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


COTH = register_activation(
    ActivationSpec(
        name="coth",
        forward=_coth_forward,
        derivative=_coth_derivative,
        fastpath=_coth_fastpath,
        integral=_coth_integral,
        riccati_polynomial=(1.0, 0.0, -1.0),  # P(c) = 1 - c^2 (same shape as tanh)
        noise_model="none",
        operator_role=(
            "Hyperbolic Riccati: coth'(z) = 1 - coth^2(z); Langevin-like "
            "saturation, finite-T NQS amplitude."
        ),
    )
)


# ---------------------------------------------------------------------------
# Tier 2b — sech (soliton / Poschl-Teller), closed form for n in {0, 1, 2}
# ---------------------------------------------------------------------------


def _sech_forward(z: Tensor) -> Tensor:
    # Numerically stable: sech(z) = 1 / cosh(z) = 2 / (e^z + e^-z).
    return 1.0 / torch.cosh(z)


def _sech_derivative(z: Tensor) -> Tensor:
    s = _sech_forward(z)
    return -s * torch.tanh(z)


def _sech_integral(z: Tensor) -> Tensor:
    return 2.0 * torch.atan(torch.tanh(0.5 * z))


def _sech_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _sech_forward(z)
    s = _sech_forward(z)
    t = torch.tanh(z)
    if n == 1:
        return -s * t
    if n == 2:
        # d/dz (-s t) = s t^2 - s sech^2 = s (1 - 2 sech^2) using t^2 = 1 - s^2
        return s * (1.0 - 2.0 * s * s)
    if n == 3:
        # d/dz (s (1 - 2 s^2)) = s' (1 - 2 s^2) + s * (-4 s s')
        # s' = -s t   =>   = -s t (1 - 2 s^2) - 4 s^2 (-s t)
        #               = -s t (1 - 2 s^2) + 4 s^3 t
        #               = s t (-1 + 2 s^2 + 4 s^2)
        #               = s t (6 s^2 - 1)
        return s * t * (6.0 * s * s - 1.0)
    raise NotImplementedError(f"sech fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


SECH = register_activation(
    ActivationSpec(
        name="sech",
        transforms=SECH_TRANSFORMS,
        forward=_sech_forward,
        derivative=_sech_derivative,
        fastpath=_sech_fastpath,
        integral=_sech_integral,
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Soliton bound-state amplitude: sech(z) is the Poschl-Teller "
            "ground state and the sine-Gordon / KdV / NLS soliton profile."
        ),
    )
)


__all__ = [
    "COS",
    "COSH",
    "COT",
    "COTH",
    "SECH",
    "SIN",
    "SINH",
    "TAN",
]
