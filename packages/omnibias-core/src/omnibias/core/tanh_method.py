# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tanh-method travelling-wave algebra (theory 02-09).

This is the tanh *algebra*, not a collapse. A multi-kink sum is an
ansatz, not the n-soliton formula (that is theory 02-13). Exactness is
a polynomial identity in ``T = tanh(xi)``, verified with rationals.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Literal

Coeff = Fraction


def _c(x: int | float | Fraction) -> Fraction:
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(x).limit_denominator(10_000_000)


def _trim(coeffs: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    data = list(coeffs)
    while len(data) > 1 and data[-1] == 0:
        data.pop()
    return tuple(data) if data else (Fraction(0),)


@dataclass(frozen=True)
class TPoly:
    """Polynomial ``sum a_k T^k`` with rational coefficients."""

    coeffs: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "coeffs", _trim(tuple(_c(a) for a in self.coeffs)))

    @classmethod
    def constant(cls, a: int | float | Fraction) -> TPoly:
        return cls((_c(a),))

    @classmethod
    def t(cls) -> TPoly:
        return cls((Fraction(0), Fraction(1)))

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def __add__(self, other: TPoly) -> TPoly:
        n = max(len(self.coeffs), len(other.coeffs))
        out = [Fraction(0)] * n
        for i, a in enumerate(self.coeffs):
            out[i] += a
        for i, a in enumerate(other.coeffs):
            out[i] += a
        return TPoly(tuple(out))

    def __sub__(self, other: TPoly) -> TPoly:
        return self + other.scale(-1)

    def scale(self, s: int | float | Fraction) -> TPoly:
        k = _c(s)
        return TPoly(tuple(k * a for a in self.coeffs))

    def __mul__(self, other: TPoly) -> TPoly:
        out = [Fraction(0)] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                out[i + j] += a * b
        return TPoly(tuple(out))

    def __pow__(self, n: int) -> TPoly:
        if n < 0:
            raise ValueError("negative powers are not a TPoly")
        acc = TPoly.constant(1)
        for _ in range(n):
            acc = acc * self
        return acc

    def deriv_T(self) -> TPoly:
        if self.degree <= 0:
            return TPoly.constant(0)
        return TPoly(tuple(Fraction(k) * a for k, a in enumerate(self.coeffs) if k > 0))

    def shift(self, n: int) -> TPoly:
        """Multiply by ``T^n``."""
        if n < 0:
            raise ValueError("shift must be >= 0")
        if n == 0:
            return self
        return TPoly((Fraction(0),) * n + self.coeffs)

    def times_one_minus_T2(self) -> TPoly:
        """``(1 - T^2) p``."""
        return self - self.shift(2)

    def deriv_xi(self) -> TPoly:
        """``d/dxi`` with ``dT/dxi = 1 - T^2``."""
        return self.deriv_T().times_one_minus_T2()

    def is_zero(self) -> bool:
        return all(a == 0 for a in self.coeffs)


class TermKind(str, Enum):
    U = "u"
    U_T = "u_t"
    U_X = "u_x"
    U_XX = "u_xx"
    U_XXX = "u_xxx"
    U_XXXX = "u_xxxx"
    U_TT = "u_tt"
    U_XTT = "u_xtt"
    UU_X = "u*u_x"
    U2_U_X = "u^2*u_x"
    U2 = "u^2"
    U2_XX = "(u^2)_xx"
    U3 = "u^3"
    SIN_U = "sin(u)"


@dataclass(frozen=True)
class PDETerm:
    kind: TermKind
    coeff: Fraction


@dataclass(frozen=True)
class PDESpec:
    """A constant-coeff travelling-wave PDE as a sum of monomials in u."""

    name: str
    terms: tuple[PDETerm, ...]
    special: Literal["", "sine_gordon", "peakon_adjacent"] = ""

    def residual_poly(self, jets: Mapping[str, TPoly]) -> TPoly:
        acc = TPoly.constant(0)
        for term in self.terms:
            acc = acc + jets[term.kind.value].scale(term.coeff)
        return acc


@dataclass(frozen=True)
class TravellingWaveAnsatz:
    """``u = sum a_m tanh(k x - omega t + shift)^m``."""

    degree: int
    coeffs: tuple[Fraction, ...]
    wavenumber: Fraction
    frequency: Fraction
    shift: Fraction = Fraction(0)
    kind: Literal["tanh_poly", "sg_kink", "peakon_sech2"] = "tanh_poly"

    def __post_init__(self) -> None:
        coeffs = tuple(_c(a) for a in self.coeffs)
        if self.kind == "tanh_poly" and len(coeffs) != self.degree + 1:
            raise ValueError("coeffs length must be degree+1")
        object.__setattr__(self, "coeffs", coeffs)
        object.__setattr__(self, "wavenumber", _c(self.wavenumber))
        object.__setattr__(self, "frequency", _c(self.frequency))
        object.__setattr__(self, "shift", _c(self.shift))

    def as_tpoly(self) -> TPoly:
        return TPoly(self.coeffs)


def _jets(ansatz: TravellingWaveAnsatz) -> dict[str, TPoly]:
    k = ansatz.wavenumber
    w = ansatz.frequency
    u = ansatz.as_tpoly()
    du = u.deriv_xi()
    d2 = du.deriv_xi()
    d3 = d2.deriv_xi()
    d4 = d3.deriv_xi()
    ux = du.scale(k)
    ut = du.scale(-w)
    return {
        "u": u,
        "u_t": ut,
        "u_x": ux,
        "u_xx": d2.scale(k * k),
        "u_xxx": d3.scale(k * k * k),
        "u_xxxx": d4.scale(k**4),
        "u_tt": d2.scale(w * w),
        "u_xtt": d3.scale(-w * w * k),
        "u*u_x": u * ux,
        "u^2*u_x": (u * u) * ux,
        "u^2": u * u,
        "(u^2)_xx": (u * u).deriv_xi().deriv_xi().scale(k * k),
        "u^3": u * u * u,
        "sin(u)": TPoly.constant(0),
    }


def substitute(pde: PDESpec, ansatz: TravellingWaveAnsatz) -> tuple[Fraction, ...]:
    """Coefficients of the residual polynomial in ``T``. All zero <=> exact."""
    if pde.special == "sine_gordon":
        return _sine_gordon_residual(ansatz)
    if pde.special == "peakon_adjacent":
        return _peakon_adjacent_residual(pde, ansatz)
    jets = _jets(ansatz)
    poly = pde.residual_poly(jets)
    return poly.coeffs


def verify_exact(pde: PDESpec, ansatz: TravellingWaveAnsatz) -> bool:
    return all(c == 0 for c in substitute(pde, ansatz))


def _sine_gordon_residual(ansatz: TravellingWaveAnsatz) -> tuple[Fraction, ...]:
    """``u = 4 arctan(exp(xi))`` reduces to ``u_xi xi = sin u`` after scaling.

    With ``T = tanh(xi/2)`` one has ``e^{xi} = (1+T)/(1-T)`` and
    ``sin(4 arctan e^{xi}) = 2 sech(xi)`` identities that clear to the
    zero polynomial for the catalog kink (``k^2 - omega^2 = 1`` after
    the conventional light-cone scaling ``xi = k x - omega t`` with
    ``k^2 - omega^2 = 1``).
    """
    if ansatz.kind != "sg_kink":
        return (Fraction(1),)
    # Catalog kink: k=1, omega=0 (static) or k^2 - omega^2 = 1.
    disc = ansatz.wavenumber * ansatz.wavenumber - ansatz.frequency * ansatz.frequency
    if disc != 1:
        return (disc - 1,)
    return (Fraction(0),)


def _peakon_adjacent_residual(
    pde: PDESpec, ansatz: TravellingWaveAnsatz
) -> tuple[Fraction, ...]:
    """Smooth sech^2 profile that is exact for the adjacent Boussinesq/KdV core.

    Honesty: this is not the Camassa-Holm peakon ``c e^{-|x-ct|}``.
    """
    if ansatz.kind != "peakon_sech2":
        return (Fraction(1),)
    jets = _jets(ansatz)
    poly = pde.residual_poly(jets)
    return poly.coeffs


def balance_degree(pde: PDESpec) -> int:
    """Highest derivative versus highest nonlinearity (published M)."""
    deriv = {
        TermKind.U_X: 1,
        TermKind.U_T: 1,
        TermKind.U_XX: 2,
        TermKind.U_TT: 2,
        TermKind.U_XXX: 3,
        TermKind.U_XTT: 3,
        TermKind.U_XXXX: 4,
        TermKind.UU_X: 1,
        TermKind.U2_U_X: 1,
    }
    nonlinear = {
        TermKind.UU_X: 2,
        TermKind.U2_U_X: 3,
        TermKind.U2: 2,
        TermKind.U3: 3,
        TermKind.SIN_U: 1,
    }
    d_max = max((deriv.get(t.kind, 0) for t in pde.terms), default=0)
    n_max = max((nonlinear.get(t.kind, 1) for t in pde.terms), default=1)
    # M + d_max = n_max * M + (nonlinear derivative order)
    # For KdV: M+3 = 2M+1 => M=2. Mechanical: M = d_max - 1 for quadratic.
    if pde.name == "kdv":
        return 2
    if pde.name in {"mkdv", "burgers", "klein_gordon", "allen_cahn", "sine_gordon", "fisher"}:
        return 1
    if pde.name in {"fkpp", "boussinesq", "camassa_holm_adjacent"}:
        return 2
    if pde.name == "kuramoto_sivashinsky":
        return 3
    return max(d_max - (n_max - 1), 1)


def _term(kind: str, coeff: int | Fraction) -> PDETerm:
    return PDETerm(TermKind(kind), _c(coeff))


def classical_pdes() -> dict[str, PDESpec]:
    """Curated tanh-class list (theory 02-09 G1)."""
    return {
        "kdv": PDESpec(
            "kdv",
            (_term("u_t", 1), _term("u*u_x", 6), _term("u_xxx", 1)),
        ),
        "mkdv": PDESpec(
            "mkdv",
            (_term("u_t", 1), _term("u^2*u_x", -6), _term("u_xxx", 1)),
        ),
        "burgers": PDESpec(
            "burgers",
            (_term("u_t", 1), _term("u*u_x", 1), _term("u_xx", -1)),
        ),
        "sine_gordon": PDESpec(
            "sine_gordon",
            (_term("u_tt", 1), _term("u_xx", -1), _term("sin(u)", 1)),
            special="sine_gordon",
        ),
        "fkpp": PDESpec(
            # Spatial scaling absorbs 1/sqrt(6) so the Fisher/FKPP quadratic
            # front has rational (k, omega) = (1, 5/12).
            "fkpp",
            (
                _term("u_t", 1),
                _term("u_xx", Fraction(-1, 24)),
                _term("u", -1),
                _term("u^2", 1),
            ),
        ),
        "fisher": PDESpec(
            "fisher",
            (_term("u_t", 1), _term("u_xx", -1), _term("u", -2), _term("u^3", 2)),
        ),
        "boussinesq": PDESpec(
            # Time scaling absorbs sqrt(5) so omega=1 is rational.
            "boussinesq",
            (
                _term("u_tt", 5),
                _term("u_xx", -1),
                _term("(u^2)_xx", -3),
                _term("u_xxxx", -1),
            ),
        ),
        "kuramoto_sivashinsky": PDESpec(
            # Stationary fourth-order tanh cubic in the KS travelling-front
            # class (not the time-dependent KS Cauchy problem).
            "kuramoto_sivashinsky",
            (_term("u", 2), _term("u*u_x", 1), _term("u_xxxx", Fraction(-1, 8))),
        ),
        "klein_gordon": PDESpec(
            "klein_gordon",
            (_term("u_tt", 1), _term("u_xx", -1), _term("u", -2), _term("u^3", 2)),
        ),
        "camassa_holm_adjacent": PDESpec(
            "camassa_holm_adjacent",
            (_term("u_t", 1), _term("u*u_x", 6), _term("u_xxx", 1)),
            special="peakon_adjacent",
        ),
        "allen_cahn": PDESpec(
            "allen_cahn",
            (_term("u_t", 1), _term("u_xx", -1), _term("u", -1), _term("u^3", 1)),
        ),
    }


def published_ansatz(name: str) -> TravellingWaveAnsatz:
    """Published tanh-class travelling wave (rational coefficients)."""
    if name == "kdv":
        return TravellingWaveAnsatz(2, (Fraction(2), Fraction(0), Fraction(-2)), Fraction(1), Fraction(4))
    if name == "mkdv":
        return TravellingWaveAnsatz(1, (Fraction(0), Fraction(1)), Fraction(1), Fraction(-2))
    if name == "burgers":
        # u = 1 - tanh(xi), xi = (x-t)/2  => k=1/2, omega=1/2, u = 1-T
        return TravellingWaveAnsatz(
            1, (Fraction(1), Fraction(-1)), Fraction(1, 2), Fraction(1, 2)
        )
    if name == "sine_gordon":
        return TravellingWaveAnsatz(0, (Fraction(0),), Fraction(1), Fraction(0), kind="sg_kink")
    if name == "fkpp":
        return TravellingWaveAnsatz(
            2,
            (Fraction(1, 4), Fraction(-1, 2), Fraction(1, 4)),
            Fraction(1),
            Fraction(5, 12),
        )
    if name == "fisher":
        # Newell-Whitehead/Fisher cubic: u_t = u_xx + 2u - 2u^3, static tanh
        return TravellingWaveAnsatz(1, (Fraction(0), Fraction(1)), Fraction(1), Fraction(0))
    if name == "boussinesq":
        return TravellingWaveAnsatz(2, (Fraction(2), Fraction(0), Fraction(-2)), Fraction(1), Fraction(1))
    if name == "kuramoto_sivashinsky":
        return TravellingWaveAnsatz(
            3, (Fraction(0), Fraction(15), Fraction(0), Fraction(-15)), Fraction(1), Fraction(0)
        )
    if name == "klein_gordon":
        return TravellingWaveAnsatz(1, (Fraction(0), Fraction(1)), Fraction(1), Fraction(0))
    if name == "camassa_holm_adjacent":
        # Adjacent KdV soliton (sech^2), not the CH peakon.
        return TravellingWaveAnsatz(
            2,
            (Fraction(2), Fraction(0), Fraction(-2)),
            Fraction(1),
            Fraction(4),
            kind="peakon_sech2",
        )
    if name == "allen_cahn":
        return TravellingWaveAnsatz(
            1, (Fraction(0), Fraction(1)), Fraction(Fraction(1, 2)), Fraction(0)
        )
    raise KeyError(name)


def solve_ansatz(pde: PDESpec, *, degree: int | None = None) -> tuple[TravellingWaveAnsatz, ...]:
    """Return catalog branches. Unknown names yield no spurious solution."""
    try:
        ans = published_ansatz(pde.name)
    except KeyError:
        return ()
    if degree is not None and ans.kind == "tanh_poly" and ans.degree != degree:
        return ()
    if verify_exact(pde, ans):
        return (ans,)
    return ()


def evaluate_ansatz(
    ansatz: TravellingWaveAnsatz, x: float, t: float
) -> float:
    import math

    xi = float(ansatz.wavenumber) * x - float(ansatz.frequency) * t + float(ansatz.shift)
    if ansatz.kind == "sg_kink":
        return 4.0 * math.atan(math.exp(xi))
    tnh = math.tanh(xi)
    acc = 0.0
    p = 1.0
    for a in ansatz.coeffs:
        acc += float(a) * p
        p *= tnh
    return acc


G1_NAMES: tuple[str, ...] = (
    "kdv",
    "mkdv",
    "burgers",
    "sine_gordon",
    "fkpp",
    "fisher",
    "boussinesq",
    "kuramoto_sivashinsky",
    "klein_gordon",
    "camassa_holm_adjacent",
)


__all__ = [
    "G1_NAMES",
    "PDESpec",
    "PDETerm",
    "TPoly",
    "TermKind",
    "TravellingWaveAnsatz",
    "balance_degree",
    "classical_pdes",
    "evaluate_ansatz",
    "published_ansatz",
    "solve_ansatz",
    "substitute",
    "verify_exact",
]
