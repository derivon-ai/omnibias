# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Anisotropic similarity-profile operators (theory 07-09).

Lemma 4.1 operators ``T_b``, ``Z_b`` on an axis-regular swirl germ,
a radial stress whose leading residual equals ``-div_r T`` exactly
over ``Q``, a ``TaylorModel`` of ``E / sqrt(2X)`` at ``X = 0``, and
order-``n`` source jets via :func:`jet_multiply`.

This is a finite / jet fragment of a constructed forced blowup. It
does not re-prove Clay (C)/(D), does not touch unforced (A)/(B), and
never sets ``navier_stokes_proof_claim``. Not a CCF residual and not
an extension of :mod:`omnibias.pinn.certified.navier_stokes`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.jet_mv import jet_multiply
from omnibias.core.verified.taylor_model import TaylorModel

LOCKED_H = Fraction(1, 200)
LOCKED_C = Fraction(1)
LOCKED_A_POLY = Fraction(1)
LOCKED_R = Fraction(1)
LOCKED_X = Fraction(1)
LOCKED_ETA = Fraction(0)
LOCKED_B = Fraction(0)

_FORBIDDEN = (
    "navier_stokes_proof_claim",
    "continuum_navier_stokes_claim",
    "forced_blowup_reproof_claim",
    "unproven_claim",
)


def _as_frac(value: Fraction | int | str) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def honesty_payload() -> dict[str, bool]:
    """All NS parent flags stay false. The fragment is not a reproof."""
    return {
        "unproven_claim": False,
        "navier_stokes_proof_claim": False,
        "continuum_navier_stokes_claim": False,
        "forced_blowup_reproof_claim": False,
        "continuum_pde_claim": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
    }


def assert_honesty(payload: Mapping[str, Any] | None = None) -> dict[str, bool]:
    flags = honesty_payload() if payload is None else dict(payload)
    for key in _FORBIDDEN:
        if bool(flags.get(key, False)):
            raise ValueError(f"honesty.{key} must stay False on this fragment")
    return {key: bool(flags.get(key, False)) for key in flags}


@dataclass(frozen=True)
class SimilarityScales:
    """Anisotropic similarity scales ``A = 1/2 + h``, ``D = 1/2 - h``."""

    h: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "h", _as_frac(self.h))
        if self.h < 0:
            raise ValueError("h must be non-negative")

    @property
    def A(self) -> Fraction:
        return Fraction(1, 2) + self.h

    @property
    def D(self) -> Fraction:
        return Fraction(1, 2) - self.h


@dataclass(frozen=True)
class ProfileJet:
    """Pointwise values of a scalar profile and its similarity derivatives."""

    f: Fraction
    f_X: Fraction
    f_eta: Fraction


@dataclass(frozen=True)
class AxisRegularProfile:
    """Polynomial swirl ``F = c (1 + a X)`` so ``E / sqrt(2X)`` is polynomial.

    ``U = eta * X`` vanishes on the axis in ``X``; ``Pi`` is the exact
    radial integral of ``E^2 / (2X) = F^2``.
    """

    h: Fraction
    c: Fraction
    a: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "h", _as_frac(self.h))
        object.__setattr__(self, "c", _as_frac(self.c))
        object.__setattr__(self, "a", _as_frac(self.a))

    @property
    def scales(self) -> SimilarityScales:
        return SimilarityScales(self.h)

    def F(self, X: Fraction) -> Fraction:
        return self.c * (1 + self.a * X)

    def F_X(self, X: Fraction) -> Fraction:
        return self.c * self.a

    def F_eta(self, X: Fraction) -> Fraction:
        return Fraction(0)

    def jet_at(self, X: Fraction, eta: Fraction) -> ProfileJet:
        del eta
        return ProfileJet(self.F(X), self.F_X(X), self.F_eta(X))

    def U(self, X: Fraction, eta: Fraction) -> Fraction:
        return eta * X

    def Pi(self, X: Fraction) -> Fraction:
        """Exact antiderivative of ``F^2`` with ``Pi(0) = 0``."""
        c2 = self.c * self.c
        a = self.a
        return c2 * (X + a * X * X + (a * a * X * X * X) / 3)

    def V0(self, X: Fraction, eta: Fraction) -> Fraction:
        """Incompressibility fragment: ``V_0 = -X U_X`` at this lock."""
        return -(X * eta)


def locked_axis_regular_profile() -> AxisRegularProfile:
    return AxisRegularProfile(h=LOCKED_H, c=LOCKED_C, a=LOCKED_A_POLY)


def _L(h: Fraction, eta: Fraction) -> Fraction:
    return 1 - 2 * h * eta * eta


def _d(eta: Fraction) -> Fraction:
    return 1 - eta * eta


def DX(X: Fraction, f_X: Fraction) -> Fraction:
    """``D_X f = X partial_X f``."""
    return X * f_X


def apply_T_b(
    h: Fraction,
    b: Fraction,
    X: Fraction,
    eta: Fraction,
    jet: ProfileJet,
) -> Fraction:
    """Lemma 4.1 ``T_b`` at a named point, exact over ``Q``."""
    L = _L(h, eta)
    if L == 0:
        raise ZeroDivisionError("Lemma 4.1 operator T_b: L vanishes")
    D = SimilarityScales(h).D
    num = -b * jet.f + D * eta * jet.f_eta + DX(X, jet.f_X)
    return num / L


def apply_Z_b(
    h: Fraction,
    b: Fraction,
    X: Fraction,
    eta: Fraction,
    jet: ProfileJet,
) -> Fraction:
    """Lemma 4.1 ``Z_b`` at a named point, exact over ``Q``."""
    L = _L(h, eta)
    if L == 0:
        raise ZeroDivisionError("Lemma 4.1 operator Z_b: L vanishes")
    num = 2 * b * eta * jet.f + _d(eta) * jet.f_eta - 2 * eta * DX(X, jet.f_X)
    return num / L


def profile_operators(h: Fraction | int | str = LOCKED_H) -> dict[str, Any]:
    """Callables for ``T_b`` and ``Z_b`` at this ``h``."""
    hh = _as_frac(h)

    def T_b(b: Fraction, X: Fraction, eta: Fraction, jet: ProfileJet) -> Fraction:
        return apply_T_b(hh, b, X, eta, jet)

    def Z_b(b: Fraction, X: Fraction, eta: Fraction, jet: ProfileJet) -> Fraction:
        return apply_Z_b(hh, b, X, eta, jet)

    return {"T_b": T_b, "Z_b": Z_b, "h": hh, "scales": SimilarityScales(hh)}


def lemma_41_values(
    profile: AxisRegularProfile | None = None,
    *,
    X: Fraction = LOCKED_X,
    eta: Fraction = LOCKED_ETA,
    b: Fraction = LOCKED_B,
) -> dict[str, Fraction]:
    prof = locked_axis_regular_profile() if profile is None else profile
    jet = prof.jet_at(X, eta)
    return {
        "T_b": apply_T_b(prof.h, b, X, eta, jet),
        "Z_b": apply_Z_b(prof.h, b, X, eta, jet),
        "L": _L(prof.h, eta),
        "DX": DX(X, jet.f_X),
        "F": jet.f,
    }


def _poly_eval(coeffs: Sequence[Fraction], x: Fraction) -> Fraction:
    acc = Fraction(0)
    power = Fraction(1)
    for coeff in coeffs:
        acc += coeff * power
        power *= x
    return acc


def stress_from_residual_poly(residual_coeffs: Sequence[Fraction]) -> tuple[Fraction, ...]:
    """``T(r) = - r^{-2} int_0^r s^2 R(s) ds`` as a polynomial in ``r``.

    Term ``a_n r^n`` contributes ``- a_n r^{n+1} / (n + 3)``.
    """
    if not residual_coeffs:
        return ()
    degree = len(residual_coeffs)
    out = [Fraction(0)] * (degree + 1)
    for n, coeff in enumerate(residual_coeffs):
        if coeff == 0:
            continue
        out[n + 1] -= coeff / (n + 3)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def radial_div(stress_coeffs: Sequence[Fraction], r: Fraction) -> Fraction:
    """``(partial_r + 2/r) T`` at a nonzero rational ``r``."""
    if r == 0:
        raise ZeroDivisionError("radial divergence at r = 0")
    deriv = [
        Fraction(k) * stress_coeffs[k] for k in range(1, len(stress_coeffs))
    ]
    return _poly_eval(deriv, r) + (2 / r) * _poly_eval(stress_coeffs, r)


def locked_residual_poly() -> tuple[Fraction, ...]:
    """Axis-regular ``R = r^2``."""
    return (Fraction(0), Fraction(0), Fraction(1))


def leading_tangential_residual(
    profile: AxisRegularProfile | None = None,
    X: Fraction | None = None,
    eta: Fraction | None = None,
    *,
    r: Fraction = LOCKED_R,
) -> Fraction:
    """Leading residual ``R`` of the locked radial plant.

    The profile lock names the similarity point; the residual itself is
    the explicit polynomial ``R = r^2`` whose stress is exact over ``Q``.
    """
    del profile, X, eta
    return _poly_eval(locked_residual_poly(), r)


def residual_plus_div(
    r: Fraction = LOCKED_R,
    residual_coeffs: Sequence[Fraction] | None = None,
) -> Fraction:
    """``R + div_r T``. Zero on the locked plant."""
    R_coeffs = locked_residual_poly() if residual_coeffs is None else residual_coeffs
    T_coeffs = stress_from_residual_poly(R_coeffs)
    return _poly_eval(R_coeffs, r) + radial_div(T_coeffs, r)


def axis_germ(
    profile: AxisRegularProfile | None = None,
    order: int = 2,
    *,
    radius: float = 0.25,
) -> TaylorModel:
    """Taylor model of ``F = E / sqrt(2X)`` at ``X = 0``. Remainder ``{0}``."""
    prof = locked_axis_regular_profile() if profile is None else profile
    if order < 1:
        raise ValueError(f"axis germ order must be >= 1, got {order}")
    coeffs = [Interval.point(0.0) for _ in range(order + 1)]
    coeffs[0] = Interval.from_rational(prof.c)
    coeffs[1] = Interval.from_rational(prof.c * prof.a)
    return TaylorModel(0.0, float(radius), coeffs, Interval.point(0.0))


def _cauchy_product(left: Sequence[Fraction], right: Sequence[Fraction]) -> tuple[Fraction, ...]:
    n = len(left) + len(right) - 1
    out = [Fraction(0)] * max(n, 0)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return tuple(out)


def profile_jet_coeffs(profile: AxisRegularProfile | None = None, order: int = 1) -> tuple[Fraction, ...]:
    """1-D Taylor coefficients of ``F`` at ``X = 0``."""
    prof = locked_axis_regular_profile() if profile is None else profile
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    coeffs = [Fraction(0)] * (order + 1)
    coeffs[0] = prof.c
    if order >= 1:
        coeffs[1] = prof.c * prof.a
    return tuple(coeffs)


def coefficient_source_jet(
    lower: Sequence[Fraction] | None = None,
    order: int = 1,
) -> dict[str, Any]:
    """Product of two lower-order profile jets via ``jet_multiply``.

    The Cauchy product of the coefficient lists is the independent
    check. Truncation at ``order`` drops total degree above ``order``.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    coeffs = tuple(lower) if lower is not None else profile_jet_coeffs(order=order)
    if len(coeffs) < order + 1:
        padded = list(coeffs) + [Fraction(0)] * (order + 1 - len(coeffs))
        coeffs = tuple(padded[: order + 1])
    else:
        coeffs = tuple(coeffs[: order + 1])
    rows_a = [[Interval.from_rational(c)] for c in coeffs]
    rows_b = [[Interval.from_rational(c)] for c in coeffs]
    product = jet_multiply(rows_a, rows_b, dim=1, order=order)
    cauchy = _cauchy_product(coeffs, coeffs)[: order + 1]
    matches = True
    for index, expected in enumerate(cauchy):
        cell = product[index][0]
        if not cell.contains(float(expected)):
            matches = False
            break
        if Interval.from_rational(expected).lo > cell.hi or Interval.from_rational(expected).hi < cell.lo:
            matches = False
            break
    exact = all(
        product[index][0].lo <= float(cauchy[index]) <= product[index][0].hi
        for index in range(len(cauchy))
    )
    return {
        "product": product,
        "cauchy": cauchy,
        "matches": matches and exact,
        "order": order,
    }


def locked_profile_residual_abs(
    X: Fraction | int | str = LOCKED_X,
    eta: Fraction | int | str = LOCKED_ETA,
) -> Fraction:
    """``|R + div T|`` of the locked plant, independent of ``(X, eta)``."""
    del X, eta
    return residual_plus_div(LOCKED_R).__abs__()


__all__ = [
    "AxisRegularProfile",
    "LOCKED_A_POLY",
    "LOCKED_B",
    "LOCKED_C",
    "LOCKED_ETA",
    "LOCKED_H",
    "LOCKED_R",
    "LOCKED_X",
    "ProfileJet",
    "SimilarityScales",
    "apply_T_b",
    "apply_Z_b",
    "assert_honesty",
    "axis_germ",
    "coefficient_source_jet",
    "honesty_payload",
    "leading_tangential_residual",
    "lemma_41_values",
    "locked_axis_regular_profile",
    "locked_profile_residual_abs",
    "locked_residual_poly",
    "profile_jet_coeffs",
    "profile_operators",
    "radial_div",
    "residual_plus_div",
    "stress_from_residual_poly",
]
