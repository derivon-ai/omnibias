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
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from omnibias.core.proof.discovery import ExactCheck, Statement
from omnibias.core.proof.obligations.stress_cone import ConeQuery, check_cone
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
    """Polynomial swirl ``F = c (1 + a X + quad X^2)``.

    The 07-09 lock is ``quad = 0`` and ``U = eta * X``. The 07-14 search
    family uses the same ``F`` with a quadratic term and the poly
    radial velocity ``U_poly = eta * X * (1 + a X + quad X^2)``.
    ``Pi`` is the exact radial integral of ``E^2 / (2X) = F^2``.
    """

    h: Fraction
    c: Fraction
    a: Fraction
    quad: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "h", _as_frac(self.h))
        object.__setattr__(self, "c", _as_frac(self.c))
        object.__setattr__(self, "a", _as_frac(self.a))
        object.__setattr__(self, "quad", _as_frac(self.quad))

    @property
    def scales(self) -> SimilarityScales:
        return SimilarityScales(self.h)

    def F(self, X: Fraction) -> Fraction:
        return self.c * (1 + self.a * X + self.quad * X * X)

    def F_X(self, X: Fraction) -> Fraction:
        return self.c * (self.a + 2 * self.quad * X)

    def F_eta(self, X: Fraction) -> Fraction:
        return Fraction(0)

    def jet_at(self, X: Fraction, eta: Fraction) -> ProfileJet:
        del eta
        return ProfileJet(self.F(X), self.F_X(X), self.F_eta(X))

    def U(self, X: Fraction, eta: Fraction) -> Fraction:
        """07-09 lock: ``U = eta * X`` (independent of the swirl poly)."""
        return eta * X

    def U_poly(self, X: Fraction, eta: Fraction) -> Fraction:
        """Search-family velocity ``U = eta * X * (1 + a X + quad X^2)``."""
        return eta * X * (1 + self.a * X + self.quad * X * X)

    def U_poly_X(self, X: Fraction, eta: Fraction) -> Fraction:
        return eta * (1 + 2 * self.a * X + 3 * self.quad * X * X)

    def Pi(self, X: Fraction) -> Fraction:
        """Exact antiderivative of ``F^2`` with ``Pi(0) = 0``."""
        c2 = self.c * self.c
        a = self.a
        q = self.quad
        return c2 * (
            X
            + a * X * X
            + (a * a + 2 * q) * X * X * X / 3
            + (a * q) * X * X * X * X / 2
            + (q * q) * X * X * X * X * X / 5
        )

    def Pi_X(self, X: Fraction) -> Fraction:
        """Exact ``partial_X Pi``. Equals ``F^2`` on this family."""
        c2 = self.c * self.c
        a = self.a
        q = self.quad
        return c2 * (
            1
            + 2 * a * X
            + (a * a + 2 * q) * X * X
            + 2 * a * q * X * X * X
            + q * q * X * X * X * X
        )

    def V0(self, X: Fraction, eta: Fraction) -> Fraction:
        """Incompressibility fragment for the 07-09 lock ``U = eta X``."""
        return -(X * eta)

    def V0_poly(self, X: Fraction, eta: Fraction) -> Fraction:
        """Incompressibility ``V_0 = -X U_X`` on the search-family ``U``."""
        return -(X * self.U_poly_X(X, eta))


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
    if order >= 1:
        coeffs[1] = Interval.from_rational(prof.c * prof.a)
    if order >= 2:
        coeffs[2] = Interval.from_rational(prof.c * prof.quad)
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
    if order >= 2:
        coeffs[2] = prof.c * prof.quad
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


NS_CORE_C_BOX: tuple[Fraction, ...] = (Fraction(-1), Fraction(1), Fraction(2))
NS_CORE_A_BOX: tuple[Fraction, ...] = (Fraction(-1), Fraction(1), Fraction(3))
NS_CORE_B_BOX: tuple[Fraction, ...] = (Fraction(-1), Fraction(0), Fraction(1))
NS_CORE_ORIGIN: tuple[Fraction, Fraction, Fraction] = (
    Fraction(1),
    Fraction(1),
    Fraction(0),
)
NS_CORE_SECOND_WITNESS: tuple[Fraction, Fraction, Fraction] = (
    Fraction(1),
    Fraction(3),
    Fraction(-1),
)
NS_CORE_OPPOSITE: tuple[Fraction, Fraction, Fraction] = (
    Fraction(-1),
    Fraction(-1),
    Fraction(0),
)
NS_CORE_TARGET_T: Fraction = Fraction(1)


def profile_from_coeffs(
    c: Fraction | int | str,
    a: Fraction | int | str,
    b: Fraction | int | str = 0,
    *,
    h: Fraction | int | str = LOCKED_H,
) -> AxisRegularProfile:
    """Axis-regular germ ``F = c (1 + a X + b X^2)``."""
    return AxisRegularProfile(h=_as_frac(h), c=_as_frac(c), a=_as_frac(a), quad=_as_frac(b))


def implied_stress(profile: AxisRegularProfile) -> tuple[Fraction, Fraction]:
    """Coefficient-space stress ``T = (c, a)`` for the 07-10 cone."""
    return (profile.c, profile.a)


def profile_similarity_residual(
    profile: AxisRegularProfile,
    *,
    X: Fraction = LOCKED_X,
    eta: Fraction = LOCKED_ETA,
    b: Fraction = LOCKED_B,
    target: Fraction = NS_CORE_TARGET_T,
) -> Fraction:
    """Profile-PDE residual ``T_b(F) - target`` at a named similarity point.

    This is **not** the independent plant ``R = r^2``.
    """
    values = lemma_41_values(profile, X=X, eta=eta, b=b)
    return values["T_b"] - target


def enclose_profile_residual(residual: Fraction) -> Interval:
    """Sound enclosure of an exact-``Q`` residual."""
    return Interval.from_rational(residual)


def residual_enclosure_verdict(box: Interval) -> str:
    """``PROVED`` only on ``{0}``; exclusion of 0 is ``DISPROVED``."""
    if box.lo == 0.0 and box.hi == 0.0:
        return "PROVED"
    if box.hi < 0.0 or box.lo > 0.0:
        return "DISPROVED"
    return "BLOCKED"


def ns_core_statement() -> Statement:
    return Statement(
        name="ns_core_profile_search",
        obligation=(
            "a second axis-regular jet in the finite (c, a, b) box whose "
            "profile-PDE residual is {0} and whose implied T is interior-cone"
        ),
        parent="Navier-Stokes forced blowup (Clay C/D)",
        parent_status="already_true",
        existential=True,
    )


def _in_box(
    candidate: tuple[Fraction, Fraction, Fraction],
) -> bool:
    c, a, b = candidate
    return c in NS_CORE_C_BOX and a in NS_CORE_A_BOX and b in NS_CORE_B_BOX


def _as_coeff_triple(candidate: Any) -> tuple[Fraction, Fraction, Fraction] | None:
    if not isinstance(candidate, tuple) or len(candidate) != 3:
        return None
    try:
        triple = (
            _as_frac(candidate[0]),
            _as_frac(candidate[1]),
            _as_frac(candidate[2]),
        )
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not _in_box(triple):
        return None
    return triple


def check_ns_core_candidate(candidate: Any) -> ExactCheck | None:
    """Exact 07-14 witness test: Lemma 4.1, cone, profile residual."""
    triple = _as_coeff_triple(candidate)
    if triple is None:
        return None
    c, a, b = triple
    profile = profile_from_coeffs(c, a, b)
    values = lemma_41_values(profile)
    residual = values["T_b"] - NS_CORE_TARGET_T
    enclosure = enclose_profile_residual(residual)
    verdict = residual_enclosure_verdict(enclosure)
    pi_res = profile.Pi_X(LOCKED_X) - profile.F(LOCKED_X) * profile.F(LOCKED_X)
    incompress = profile.V0_poly(LOCKED_X, Fraction(1, 2)) + LOCKED_X * profile.U_poly_X(
        LOCKED_X, Fraction(1, 2)
    )
    germ = axis_germ(profile, order=2)
    germ_zero = germ.remainder.lo == 0.0 and germ.remainder.hi == 0.0
    cone = check_cone(
        ConeQuery(
            v1=(Fraction(1), Fraction(0)),
            v2=(Fraction(0), Fraction(1)),
            T=implied_stress(profile),
            sense="gt",
            name=f"ns_core_{c}_{a}_{b}",
        )
    )
    residual_ok = verdict == "PROVED" and residual == 0
    identities_ok = pi_res == 0 and incompress == 0 and germ_zero
    ok = residual_ok and identities_ok and cone.holds
    honesty = honesty_payload()
    return ExactCheck(
        ok=ok,
        payload={
            "c": str(c),
            "a": str(a),
            "b": str(b),
            "is_origin": triple == NS_CORE_ORIGIN,
            "T_b": str(values["T_b"]),
            "Z_b": str(values["Z_b"]),
            "profile_residual": str(residual),
            "residual_verdict": verdict,
            "enclosure": {"lo": enclosure.lo, "hi": enclosure.hi},
            "pi_identity": str(pi_res),
            "incompressibility": str(incompress),
            "axis_remainder_zero": germ_zero,
            "cone": cone.to_mapping(),
            "cone_status": cone.strength,
            "cone_reason": cone.reason,
            "external_premises": [
                "WKB pulses / joining / cutoff of the constructed blowup",
                "analytic classes of the slow base and high-frequency packets",
            ],
            "equation": {
                "kind": "polynomial_identity",
                "pretty": "T_b(F) - 1 = 0 at (X, eta) = (1, 0)",
                "coefficients": (str(c), str(a), str(b)),
            },
            "honesty": honesty,
        },
    )


@dataclass
class NSCoreProfileFamily:
    """Finite ``(c, a, b)`` box of axis-regular jets (theory 07-14)."""

    name: str = "ns_core_profile_search"
    complete: bool = True
    statement: Statement = field(default_factory=ns_core_statement)

    def cardinality(self) -> int:
        return len(NS_CORE_C_BOX) * len(NS_CORE_A_BOX) * len(NS_CORE_B_BOX)

    def origin(self) -> tuple[Fraction, Fraction, Fraction]:
        return NS_CORE_ORIGIN

    def neighbors(
        self, candidate: Any
    ) -> Sequence[tuple[Fraction, Fraction, Fraction]]:
        triple = _as_coeff_triple(candidate)
        if triple is None:
            return ()
        c, a, b = triple
        out: list[tuple[Fraction, Fraction, Fraction]] = []
        for box, value, rebuild in (
            (NS_CORE_C_BOX, c, lambda x: (x, a, b)),
            (NS_CORE_A_BOX, a, lambda x: (c, x, b)),
            (NS_CORE_B_BOX, b, lambda x: (c, a, x)),
        ):
            try:
                idx = box.index(value)
            except ValueError:
                continue
            if idx > 0:
                out.append(rebuild(box[idx - 1]))
            if idx + 1 < len(box):
                out.append(rebuild(box[idx + 1]))
        return tuple(out)

    def score(self, candidate: Any) -> Fraction:
        triple = _as_coeff_triple(candidate)
        if triple is None:
            return Fraction(-10**6)
        profile = profile_from_coeffs(*triple)
        residual = profile_similarity_residual(profile)
        return -abs(residual)

    def check(self, candidate: Any) -> ExactCheck | None:
        return check_ns_core_candidate(candidate)


__all__ = [
    "AxisRegularProfile",
    "LOCKED_A_POLY",
    "LOCKED_B",
    "LOCKED_C",
    "LOCKED_ETA",
    "LOCKED_H",
    "LOCKED_R",
    "LOCKED_X",
    "NSCoreProfileFamily",
    "NS_CORE_A_BOX",
    "NS_CORE_B_BOX",
    "NS_CORE_C_BOX",
    "NS_CORE_OPPOSITE",
    "NS_CORE_ORIGIN",
    "NS_CORE_SECOND_WITNESS",
    "NS_CORE_TARGET_T",
    "ProfileJet",
    "SimilarityScales",
    "apply_T_b",
    "apply_Z_b",
    "assert_honesty",
    "axis_germ",
    "check_ns_core_candidate",
    "coefficient_source_jet",
    "enclose_profile_residual",
    "honesty_payload",
    "implied_stress",
    "leading_tangential_residual",
    "lemma_41_values",
    "locked_axis_regular_profile",
    "locked_profile_residual_abs",
    "locked_residual_poly",
    "ns_core_statement",
    "profile_from_coeffs",
    "profile_jet_coeffs",
    "profile_operators",
    "profile_similarity_residual",
    "radial_div",
    "residual_enclosure_verdict",
    "residual_plus_div",
    "stress_from_residual_poly",
]
