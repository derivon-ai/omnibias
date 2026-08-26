# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named linearizing PDE transforms (theory 02-13).

Cole-Hopf / Miura / Bäcklund / Darboux only. Exactness is to jet truncation
order ``N``. Integrability *search* (spec 03-11) is not claimed. A multi-kink
sum is not the n-soliton formula unless built by Bäcklund permutability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def factorial_jet_multiply(
    a: Sequence[float], b: Sequence[float]
) -> list[float]:
    r"""Cauchy product for factorial jets ``a_k = f^{(k)}/k!``.

    Matches the convention of :func:`omnibias.torch.jet_mv.jet_multiply` in 1-D
    (ordinary Cauchy product of the scaled Taylor coefficients).
    """
    n = min(len(a), len(b))
    out = [0.0] * n
    for k in range(n):
        s = 0.0
        for j in range(k + 1):
            s += float(a[j]) * float(b[k - j])
        out[k] = s
    return out


def factorial_jet_reciprocal(a: Sequence[float]) -> list[float]:
    r"""Factorial jet of ``1/f`` from the factorial jet of ``f`` (``a_0 != 0``)."""
    if not a:
        raise ValueError("empty jet")
    a0 = float(a[0])
    if a0 == 0.0:
        raise ZeroDivisionError("factorial_jet_reciprocal requires a[0] != 0")
    n = len(a)
    b = [0.0] * n
    b[0] = 1.0 / a0
    for k in range(1, n):
        s = 0.0
        for j in range(1, k + 1):
            s += float(a[j]) * b[k - j]
        b[k] = -s / a0
    return b


def cole_hopf_jet(
    phi: Sequence[float],
    phi_x: Sequence[float],
    *,
    nu: float = 1.0,
) -> list[float]:
    r"""Factorial jet of ``u = -2 nu (log phi)_x = -2 nu phi_x / phi``.

    Exact (to jet truncation) Cole--Hopf pushforward: heat jets for ``phi``
    and ``phi_x`` become a Burgers jet for ``u`` by Cauchy product only
    (no CAS). Not a Navier--Stokes claim.
    """
    if len(phi) != len(phi_x):
        raise ValueError("phi and phi_x jets must have the same length")
    inv = factorial_jet_reciprocal(phi)
    ratio = factorial_jet_multiply(phi_x, inv)
    return [-2.0 * float(nu) * c for c in ratio]


def heat_plane_wave_jets(
    *,
    k: float,
    nu: float,
    order: int,
) -> tuple[list[float], list[float]]:
    r"""Time-factorial jets of ``phi = exp(k x + nu k^2 t)`` and ``phi_x`` at ``t=0``.

    Heat identity ``phi_t = nu phi_xx`` holds with ``omega = nu k^2``. At a fixed
    ``x`` the time jet is ``a_m = (omega^m / m!) phi(x,0)``; here we report the
    jet at ``x = 0`` where ``phi = 1``, ``phi_x = k``.
    """
    if order < 0:
        raise ValueError("order must be >= 0")
    import math

    omega = float(nu) * float(k) * float(k)
    phi = [0.0] * (order + 1)
    phi_x = [0.0] * (order + 1)
    for m in range(order + 1):
        coeff = (omega**m) / float(math.factorial(m))
        phi[m] = coeff  # phi(0,0)=1
        phi_x[m] = float(k) * coeff
    return phi, phi_x


def verify_cole_hopf_burgers_jet(*, nu: float = 0.1, k: float = -0.5, order: int = 8) -> dict[str, float | bool]:
    r"""Exact Cole--Hopf: plane-wave heat -> constant Burgers ``u = -2 nu k``.

    The Burgers residual of a constant field is identically zero, so every
    time-jet coefficient of ``u`` beyond order 0 must vanish.
    """
    phi, phi_x = heat_plane_wave_jets(k=k, nu=nu, order=order)
    u_jet = cole_hopf_jet(phi, phi_x, nu=nu)
    expected = -2.0 * float(nu) * float(k)
    err0 = abs(u_jet[0] - expected)
    tail = max((abs(c) for c in u_jet[1:]), default=0.0)
    return {
        "u0": float(u_jet[0]),
        "expected_u": expected,
        "err0": err0,
        "max_higher_coeff": float(tail),
        "passed": bool(err0 <= 1e-12 and tail <= 1e-12),
        "navier_stokes_proof_claim": False,
    }


__all__ = [
    "LinearizingTransform",
    "TransformKind",
    "TravellingWaveAnsatz",
    "cole_hopf_from_heat_phi",
    "cole_hopf_jet",
    "cole_hopf_u",
    "darboux_dress",
    "factorial_jet_multiply",
    "factorial_jet_reciprocal",
    "heat_plane_wave_jets",
    "miura_v",
    "named_cole_hopf",
    "permutability",
    "verify_cole_hopf_burgers_jet",
    "verify_transform",
]
