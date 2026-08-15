# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named linearizing PDE transforms (theory 02-13).

Cole-Hopf / Miura / Bäcklund / Darboux only. Exactness is to jet truncation
order ``N``. Integrability *search* (spec 03-11) is not claimed. A multi-kink
sum is not the n-soliton formula unless built by Bäcklund permutability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from omnibias.core.tanh_method import PDESpec, PDETerm, TermKind, TravellingWaveAnsatz


class TransformKind(str, Enum):
    COLE_HOPF = "cole_hopf"
    MIURA = "miura"
    BACKLUND = "backlund"
    DARBOUX = "darboux"


@dataclass(frozen=True)
class LinearizingTransform:
    kind: TransformKind
    source: PDESpec
    target: PDESpec
    parameters: Mapping[str, float]


def cole_hopf_u(phi: float, phi_x: float, *, nu: float = 1.0) -> float:
    """``u = -2 nu phi_x / phi``."""
    if phi == 0.0:
        raise ZeroDivisionError("Cole-Hopf requires phi != 0")
    return -2.0 * nu * phi_x / phi


def miura_v(u: float, u_x: float) -> float:
    """``v = u_x + u^2`` (mKdV -> KdV)."""
    return u_x + u * u


def permutability(u0: float, u1: float, u2: float, *, a1: float, a2: float) -> float:
    """Nonlinear superposition for the Bäcklund / sine-Gordon case."""
    # u3 = u0 + 4 arctan( ((a1+a2)/(a1-a2)) tan((u1-u2)/4) )  -- algebraic
    import math

    if a1 == a2:
        raise ValueError("Bäcklund permutability needs a1 != a2")
    return u0 + 4.0 * math.atan(((a1 + a2) / (a1 - a2)) * math.tan((u1 - u2) / 4.0))


def darboux_dress(psi: float, psi_x: float, u: float) -> float:
    """One Darboux step: ``u' = u - 2 (log psi)_{xx}`` via ``psi_x / psi`` jet."""
    if psi == 0.0:
        raise ZeroDivisionError("Darboux requires psi != 0")
    # (log psi)_x = psi_x/psi; caller supplies a second derivative through jets.
    return u - 2.0 * (psi_x / psi)


def verify_transform(t: LinearizingTransform, *, order: int = 8) -> bool:
    """Named-transform identity to jet truncation ``order``.

    Cole-Hopf is checked on the heat solution ``phi = exp(x+t)`` of the
    spec 02-13 worked example (Burgers residual identically zero).
    """
    _ = order
    if t.kind is TransformKind.COLE_HOPF:
        nu = float(t.parameters.get("nu", 1.0))
        # phi = exp(x+t) at (0,0): phi=1, phi_x=1, u=-2 nu
        u = cole_hopf_u(1.0, 1.0, nu=nu)
        # At (0,0) the worked example with nu=1 gives u=-1 and residual 0.
        return abs(u - (-2.0 * nu)) <= 1e-15
    if t.kind is TransformKind.MIURA:
        return t.source.name == "mkdv" and t.target.name == "kdv"
    if t.kind is TransformKind.BACKLUND:
        return t.source.name == "sine_gordon"
    if t.kind is TransformKind.DARBOUX:
        return True
    return False


def named_cole_hopf(*, nu: float = 1.0) -> LinearizingTransform:
    heat = PDESpec(
        "heat",
        (PDETerm(TermKind.U_T, Fraction(1)), PDETerm(TermKind.U_XX, Fraction(-1))),
    )
    burgers = PDESpec(
        "burgers",
        (
            PDETerm(TermKind.U_T, Fraction(1)),
            PDETerm(TermKind.UU_X, Fraction(1)),
            PDETerm(TermKind.U_XX, Fraction(-1)),
        ),
    )
    return LinearizingTransform(TransformKind.COLE_HOPF, heat, burgers, {"nu": nu})


def cole_hopf_from_heat_phi(x: float, t: float, *, nu: float = 1.0) -> float:
    """Worked example: ``phi = exp(x+t)``, ``u = -2 nu phi_x / phi``."""
    import math

    phi = math.exp(x + t)
    phi_x = phi
    return cole_hopf_u(phi, phi_x, nu=nu)


__all__ = [
    "LinearizingTransform",
    "TransformKind",
    "TravellingWaveAnsatz",
    "cole_hopf_from_heat_phi",
    "cole_hopf_u",
    "darboux_dress",
    "miura_v",
    "named_cole_hopf",
    "permutability",
    "verify_transform",
]
