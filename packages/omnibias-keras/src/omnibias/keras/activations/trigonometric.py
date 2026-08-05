# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trigonometric & hyperbolic activations for physics-class wavefunctions.

Mirrors :mod:`omnibias.torch.activations.trigonometric` on ``keras.ops``.
``sin`` / ``cos`` / ``sinh`` / ``cosh`` are closed form for every order;
``tan`` / ``cot`` / ``coth`` for n in {0, 1, 2, 3}; ``sech`` for
n in {0, 1, 2, 3}.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.keras.activations.registry import ActivationSpec, register_activation

from keras import ops

_HALF_PI = 0.5 * math.pi


# --- sin / cos (every order) ----------------------------------------------


def _sin_forward(z: Any) -> Any:
    return ops.sin(z)


def _sin_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return ops.sin(z + n * _HALF_PI)


SIN = register_activation(
    ActivationSpec(
        name="sin",
        forward=_sin_forward,
        derivative=lambda z: ops.cos(z),
        fastpath=_sin_fastpath,
        integral=lambda z: -ops.cos(z),
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Plane-wave / Bloch embedding; eigenfunction of d^2/dz^2 with "
            "eigenvalue -1; sin(z + n*pi/2) gives every order in closed form."
        ),
    )
)


def _cos_forward(z: Any) -> Any:
    return ops.cos(z)


def _cos_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return ops.cos(z + n * _HALF_PI)


COS = register_activation(
    ActivationSpec(
        name="cos",
        forward=_cos_forward,
        derivative=lambda z: -ops.sin(z),
        fastpath=_cos_fastpath,
        integral=lambda z: ops.sin(z),
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Plane-wave / Bloch embedding (real part); cos(z + n*pi/2) "
            "gives every order in closed form."
        ),
    )
)


def _sinh_forward(z: Any) -> Any:
    return ops.sinh(z)


def _sinh_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return ops.sinh(z) if (n & 1) == 0 else ops.cosh(z)


SINH = register_activation(
    ActivationSpec(
        name="sinh",
        forward=_sinh_forward,
        derivative=lambda z: ops.cosh(z),
        fastpath=_sinh_fastpath,
        integral=lambda z: ops.cosh(z),
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Hyperbolic odd primitive; eigenfunction of d^2/dz^2 with "
            "eigenvalue +1; alternates with cosh under differentiation."
        ),
    )
)


def _cosh_forward(z: Any) -> Any:
    return ops.cosh(z)


def _cosh_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    return ops.cosh(z) if (n & 1) == 0 else ops.sinh(z)


COSH = register_activation(
    ActivationSpec(
        name="cosh",
        forward=_cosh_forward,
        derivative=lambda z: ops.sinh(z),
        fastpath=_cosh_fastpath,
        integral=lambda z: ops.sinh(z),
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Hyperbolic even primitive; eigenfunction of d^2/dz^2 with "
            "eigenvalue +1; alternates with sinh under differentiation."
        ),
    )
)


# --- tan / cot / coth (Riccati, n in {0,1,2,3}) ---------------------------


def _tan_forward(z: Any) -> Any:
    return ops.tan(z)


def _tan_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _tan_forward(z)
    t = ops.tan(z)
    s2 = 1.0 + t * t
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
        derivative=lambda z: 1.0 + ops.tan(z) ** 2,
        fastpath=_tan_fastpath,
        integral=lambda z: -ops.log(ops.abs(ops.cos(z))),
        riccati_polynomial=(1.0, 0.0, 1.0),
        noise_model="none",
        operator_role=(
            "Riccati-class periodic activation; tan'(z) = 1 + tan^2(z); "
            "phase factor in unitary-RNN amplitude/phase factorisations."
        ),
    )
)


def _cot_forward(z: Any) -> Any:
    return 1.0 / ops.tan(z)


def _cot_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _cot_forward(z)
    c = _cot_forward(z)
    csc2 = 1.0 + c * c
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
        derivative=lambda z: -(1.0 + (1.0 / ops.tan(z)) ** 2),
        fastpath=_cot_fastpath,
        integral=lambda z: ops.log(ops.abs(ops.sin(z))),
        riccati_polynomial=(-1.0, 0.0, -1.0),
        noise_model="none",
        operator_role=(
            "Riccati-class periodic activation; cot'(z) = -(1 + cot^2(z)); "
            "complementary to tan in phase-only ansatze."
        ),
        aliases=("ctg", "ctan"),
    )
)


def _coth_forward(z: Any) -> Any:
    return 1.0 / ops.tanh(z)


def _coth_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _coth_forward(z)
    c = _coth_forward(z)
    one_minus_c2 = 1.0 - c * c
    if n == 1:
        return one_minus_c2
    if n == 2:
        return -2.0 * c * one_minus_c2
    if n == 3:
        return -2.0 * one_minus_c2 * (1.0 - 3.0 * c * c)
    raise NotImplementedError(f"coth fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


COTH = register_activation(
    ActivationSpec(
        name="coth",
        forward=_coth_forward,
        derivative=lambda z: 1.0 - (1.0 / ops.tanh(z)) ** 2,
        fastpath=_coth_fastpath,
        integral=lambda z: ops.log(ops.abs(ops.sinh(z))),
        riccati_polynomial=(1.0, 0.0, -1.0),
        noise_model="none",
        operator_role=(
            "Hyperbolic Riccati: coth'(z) = 1 - coth^2(z); Langevin-like "
            "saturation, finite-T NQS amplitude."
        ),
    )
)


# --- sech (soliton / Poschl-Teller, n in {0,1,2,3}) -----------------------


def _sech_forward(z: Any) -> Any:
    return 1.0 / ops.cosh(z)


def _sech_fastpath(z: Any, n: int) -> Any:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _sech_forward(z)
    s = _sech_forward(z)
    t = ops.tanh(z)
    if n == 1:
        return -s * t
    if n == 2:
        return s * (1.0 - 2.0 * s * s)
    if n == 3:
        return s * t * (6.0 * s * s - 1.0)
    raise NotImplementedError(f"sech fast path only implements n in {{0, 1, 2, 3}}, got {n}.")


SECH = register_activation(
    ActivationSpec(
        name="sech",
        forward=_sech_forward,
        derivative=lambda z: -(1.0 / ops.cosh(z)) * ops.tanh(z),
        fastpath=_sech_fastpath,
        integral=lambda z: 2.0 * ops.arctan(ops.tanh(0.5 * z)),
        riccati_polynomial=None,
        noise_model="none",
        operator_role=(
            "Soliton bound-state amplitude: sech(z) is the Poschl-Teller "
            "ground state and the sine-Gordon / KdV / NLS soliton profile."
        ),
    )
)


__all__ = ["COS", "COSH", "COT", "COTH", "SECH", "SIN", "SINH", "TAN"]
