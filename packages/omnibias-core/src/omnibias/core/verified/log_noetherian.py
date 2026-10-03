# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite, replayable Log-Noetherian certificate data.

This module implements only the finite algebra in the definition of an
LN-chain.  In particular, it does not prove that a supplied function is
holomorphic or bounded on a cell, that a symbolic radius is positive, or that
a Dulac/return map has an LN representation.  Those are separate analytic
obligations.

The exact replay checks

.. math::

   \partial_j^{\mathcal C} F_i =
   G_{i,j}(F_1,\ldots,F_N)

as identities of sparse polynomials over ``Q``.  Recorded claims are never
used as evidence: both the standard derivative and every composition are
recomputed and compared through :attr:`SparsePolynomial.terms`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, TypeAlias, cast, overload

from omnibias.core.proof.certificate import (
    NO_TRANSCENDENTAL_BACKEND,
    TRANSCEND_BACKEND_KEY,
    Cert,
    decode_interval,
    encode_interval,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.realization.polynomial import (
    AlgebraBudget,
    AlgebraBudgetExceeded,
    SparsePolynomial,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import certificate_mode, ln_iv

ExactRational: TypeAlias = int | Fraction
FiberKind: TypeAlias = Literal["Point", "Disc", "PuncturedDisc", "Annulus"]


def _fraction(value: ExactRational, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be an exact int or Fraction")
    return Fraction(value)


def _positive_fraction(value: ExactRational, name: str) -> Fraction:
    result = _fraction(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return result


@dataclass(frozen=True)
class ChainFunctionRef:
    """A positive-radius reference to one named chain function.

    ``scale`` is exact.  Positivity of the referenced function is deliberately
    not inferred by this syntactic substrate; it remains an analytic premise of
    the represented cell.
    """

    name: str
    scale: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("chain-function reference name must be nonempty")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(
            self,
            "scale",
            _positive_fraction(self.scale, "chain-function reference scale"),
        )

    def scaled(self, factor: ExactRational) -> ChainFunctionRef:
        return ChainFunctionRef(self.name, self.scale * _positive_fraction(factor, "factor"))


Radius: TypeAlias = Fraction | ChainFunctionRef


def _radius(value: Radius | int) -> Radius:
    if isinstance(value, ChainFunctionRef):
        return value
    return _positive_fraction(value, "radius")


def _scale_radius(value: Radius, factor: Fraction) -> Radius:
    if isinstance(value, ChainFunctionRef):
        return value.scaled(factor)
    return value * factor


def _ordered_radius_values(inner: Radius, outer: Radius) -> tuple[Fraction, Fraction] | None:
    if isinstance(inner, Fraction) and isinstance(outer, Fraction):
        return inner, outer
    if (
        isinstance(inner, ChainFunctionRef)
        and isinstance(outer, ChainFunctionRef)
        and inner.name == outer.name
    ):
        return inner.scale, outer.scale
    return None


@dataclass(frozen=True)
class LNFiber:
    """One LN fiber with its ordered radius data.

    The only admitted kinds are ``Point``, ``Disc``, ``PuncturedDisc`` and
    ``Annulus``.  Their radius arities are respectively 0, 1, 1 and 2.
    Annulus constants (or two scales of the same reference) are checked in
    strict inner/outer order.  Order between distinct symbolic functions is
    retained as an analytic premise and is not silently certified.
    """

    kind: FiberKind
    radii: tuple[Radius, ...] = ()

    def __post_init__(self) -> None:
        arities = {"Annulus": 2, "Disc": 1, "Point": 0, "PuncturedDisc": 1}
        if self.kind not in arities:
            raise ValueError(
                "fiber kind must be exactly Point, Disc, PuncturedDisc, or Annulus"
            )
        cleaned = tuple(_radius(value) for value in self.radii)
        object.__setattr__(self, "radii", cleaned)
        expected = arities[self.kind]
        if len(cleaned) != expected:
            raise ValueError(f"{self.kind} fiber requires exactly {expected} radii")
        if self.kind == "Annulus":
            ordered = _ordered_radius_values(cleaned[0], cleaned[1])
            if ordered is not None and ordered[0] >= ordered[1]:
                raise ValueError("annulus radii must satisfy inner < outer")

    def delta_extension(self, delta: Fraction) -> LNFiber:
        if self.kind == "Point":
            return self
        if self.kind in {"Disc", "PuncturedDisc"}:
            return LNFiber(self.kind, (_scale_radius(self.radii[0], 1 / delta),))
        inner = _scale_radius(self.radii[0], delta)
        outer = _scale_radius(self.radii[1], 1 / delta)
        ordered = _ordered_radius_values(inner, outer)
        if ordered is None:
            raise ValueError(
                "annulus delta-extension requires a certified positive annulus "
                "margin; distinct symbolic radius functions are incomparable"
            )
        if ordered[1] - ordered[0] <= 0:
            raise ValueError("annulus delta-extension has nonpositive annulus margin")
        return LNFiber("Annulus", (inner, outer))


@dataclass(frozen=True)
class CoordinateRef:
    """The coordinate multiplier ``z_axis`` in a logarithmic derivative."""

    axis: int

    def __post_init__(self) -> None:
        if type(self.axis) is not int or self.axis < 0:
            raise ValueError("coordinate axis must be a nonnegative integer")


@dataclass(frozen=True)
class LNStandardDerivation:
    """Finite encoding of ``r*d_z`` or ``z*d_z`` on one cell axis."""

    axis: int
    operator: Literal["radius_d_z", "z_d_z"]
    multiplier: Radius | CoordinateRef
    real_part: bool


@dataclass(frozen=True)
class LNCell:
    """An ordered iterated LN-cell description.

    ``real_part`` is only a marker for the real part of the same cell.  It does
    not assert that a point lies there or that any function is real-valued.
    """

    fibers: tuple[LNFiber, ...]
    real_part: bool = False

    def __post_init__(self) -> None:
        if type(self.real_part) is not bool:
            raise TypeError("real_part must be a boolean marker")
        if any(not isinstance(fiber, LNFiber) for fiber in self.fibers):
            raise TypeError("fibers must be an ordered tuple of LNFiber values")

    @property
    def dimension(self) -> int:
        return len(self.fibers)

    def delta_extension(self, delta: ExactRational) -> LNCell:
        """Return the fiberwise ``delta``-extension, for exact ``0 < delta < 1``."""
        d = _fraction(delta, "delta")
        if not 0 < d < 1:
            raise ValueError("delta must satisfy 0 < delta < 1")
        return LNCell(
            tuple(fiber.delta_extension(d) for fiber in self.fibers),
            real_part=self.real_part,
        )

    def standard_derivation(self, axis: int) -> LNStandardDerivation:
        """Encode ``r*d_z`` for a disc and ``z*d_z`` for every other fiber."""
        if type(axis) is not int or not 0 <= axis < self.dimension:
            raise ValueError("standard-derivation axis is out of range")
        fiber = self.fibers[axis]
        if fiber.kind == "Disc":
            return LNStandardDerivation(
                axis,
                "radius_d_z",
                fiber.radii[0],
                self.real_part,
            )
        return LNStandardDerivation(
            axis,
            "z_d_z",
            CoordinateRef(axis),
            self.real_part,
        )


@dataclass(frozen=True)
class LNResourceBudget:
    """Hard limits for every finite replay in this module."""

    max_dimension: int = 32
    max_functions: int = 128
    max_matrix_entries: int = 4096
    max_terms: int = 100_000
    max_degree: int = 64
    max_coefficient_bits: int = 8192
    max_composition_products: int = 2_000_000
    max_derivative_order: int = 64

    def __post_init__(self) -> None:
        values = (
            self.max_dimension,
            self.max_functions,
            self.max_matrix_entries,
            self.max_terms,
            self.max_degree,
            self.max_coefficient_bits,
            self.max_composition_products,
            self.max_derivative_order,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError("LN resource budgets must be positive integers")


DEFAULT_LN_BUDGET = LNResourceBudget()


@dataclass(frozen=True)
class NamedPolynomial:
    """One named exact-Q sparse polynomial representing a chain function."""

    name: str
    polynomial: SparsePolynomial

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("chain function name must be nonempty")
        if not isinstance(self.polynomial, SparsePolynomial):
            raise TypeError("chain function must be a SparsePolynomial")
        object.__setattr__(self, "name", self.name.strip())


def _polynomial_degree(polynomial: SparsePolynomial) -> int:
    return max((sum(index) for index, _ in polynomial.terms), default=0)


def _check_polynomial(polynomial: SparsePolynomial, budget: LNResourceBudget) -> None:
    if len(polynomial.terms) > budget.max_terms:
        raise AlgebraBudgetExceeded("LN polynomial term budget exceeded")
    if _polynomial_degree(polynomial) > budget.max_degree:
        raise AlgebraBudgetExceeded("LN polynomial degree budget exceeded")
    for _index, coefficient in polynomial.terms:
        bits = max(
            abs(coefficient.numerator).bit_length(),
            coefficient.denominator.bit_length(),
        )
        if bits > budget.max_coefficient_bits:
            raise AlgebraBudgetExceeded("LN rational coefficient bit budget exceeded")


@dataclass(frozen=True)
class LNChain:
    """A finite polynomial LN-chain candidate and its derivative witnesses.

    Matrix rows are indexed by chain function and columns by cell axis.
    ``closure_matrix[i][j]`` is ``G_ij`` in the chain variables, while
    ``claimed_derivatives[i][j]`` is a polynomial in the cell coordinates.
    Construction checks shape and budgets only; use :func:`replay_ln_chain` for
    the exact identities.
    """

    cell: LNCell
    functions: tuple[NamedPolynomial, ...]
    closure_matrix: tuple[tuple[SparsePolynomial, ...], ...]
    claimed_derivatives: tuple[tuple[SparsePolynomial, ...], ...]
    budget: LNResourceBudget = DEFAULT_LN_BUDGET

    def __post_init__(self) -> None:
        if not isinstance(self.cell, LNCell):
            raise TypeError("cell must be an LNCell")
        if not isinstance(self.budget, LNResourceBudget):
            raise TypeError("budget must be an LNResourceBudget")
        n_functions = len(self.functions)
        dimension = self.cell.dimension
        if dimension > self.budget.max_dimension:
            raise AlgebraBudgetExceeded("LN cell dimension budget exceeded")
        if n_functions > self.budget.max_functions:
            raise AlgebraBudgetExceeded("LN chain-function budget exceeded")
        if n_functions * dimension > self.budget.max_matrix_entries:
            raise AlgebraBudgetExceeded("LN closure-matrix entry budget exceeded")
        names = tuple(function.name for function in self.functions)
        if len(set(names)) != len(names):
            raise ValueError("chain function names must be unique")
        for function in self.functions:
            if function.polynomial.nvars != dimension:
                raise ValueError("chain-function polynomial has wrong variable count")
            _check_polynomial(function.polynomial, self.budget)
        if len(self.closure_matrix) != n_functions or any(
            len(row) != dimension for row in self.closure_matrix
        ):
            raise ValueError("closure_matrix must have shape functions x cell dimension")
        if len(self.claimed_derivatives) != n_functions or any(
            len(row) != dimension for row in self.claimed_derivatives
        ):
            raise ValueError(
                "claimed_derivatives must have shape functions x cell dimension"
            )
        total_terms = sum(
            len(polynomial.terms)
            for matrix in (self.closure_matrix, self.claimed_derivatives)
            for row in matrix
            for polynomial in row
        ) + sum(len(function.polynomial.terms) for function in self.functions)
        if total_terms > self.budget.max_terms:
            raise AlgebraBudgetExceeded("aggregate LN term budget exceeded")
        for row in self.closure_matrix:
            for polynomial in row:
                if polynomial.nvars != n_functions:
                    raise ValueError("closure polynomial must use the chain variables")
                _check_polynomial(polynomial, self.budget)
        for row in self.claimed_derivatives:
            for polynomial in row:
                if polynomial.nvars != dimension:
                    raise ValueError(
                        "claimed derivative polynomial has wrong variable count"
                    )
                _check_polynomial(polynomial, self.budget)
        references = {
            radius.name
            for fiber in self.cell.fibers
            for radius in fiber.radii
            if isinstance(radius, ChainFunctionRef)
        }
        missing = references.difference(names)
        if missing:
            raise ValueError(f"cell radius references unknown chain functions: {sorted(missing)}")


@dataclass(frozen=True)
class LNReplayFailure:
    """One exact closure-replay mismatch."""

    function: str
    axis: int
    stage: Literal["claimed_derivative", "closure_composition"]


@dataclass(frozen=True)
class LNReplayReport:
    """Result derived from exact polynomial terms, never from a recorded flag."""

    valid: bool
    failures: tuple[LNReplayFailure, ...]


def _composition_budget(chain: LNChain) -> AlgebraBudget:
    return AlgebraBudget(
        max_terms=chain.budget.max_terms,
        max_degree=chain.budget.max_degree,
        max_bits=chain.budget.max_coefficient_bits,
        max_products=chain.budget.max_composition_products,
    )


def _compose_closure(polynomial: SparsePolynomial, chain: LNChain) -> SparsePolynomial:
    dimension = chain.cell.dimension
    algebra_budget = _composition_budget(chain)
    result = SparsePolynomial.constant(dimension, 0, budget=algebra_budget)
    products = 0
    for index, coefficient in polynomial.terms:
        term = SparsePolynomial.constant(dimension, coefficient, budget=algebra_budget)
        for function, exponent in zip(chain.functions, index, strict=True):
            products += exponent
            if products > chain.budget.max_composition_products:
                raise AlgebraBudgetExceeded("LN composition product budget exceeded")
            if exponent:
                base = SparsePolynomial(
                    dimension,
                    dict(function.polynomial.terms),
                    budget=algebra_budget,
                )
                term = term * (base**exponent)
        result = result + term
        _check_polynomial(result, chain.budget)
    return result


def _standard_derivative(
    chain: LNChain,
    function: NamedPolynomial,
    axis: int,
) -> SparsePolynomial:
    derivative = function.polynomial.derivative(axis)
    derivation = chain.cell.standard_derivation(axis)
    multiplier = derivation.multiplier
    if isinstance(multiplier, CoordinateRef):
        return derivative * SparsePolynomial.variable(
            chain.cell.dimension,
            multiplier.axis,
            budget=function.polynomial.budget,
        )
    if isinstance(multiplier, Fraction):
        return derivative * multiplier
    named = {item.name: item.polynomial for item in chain.functions}
    return derivative * named[multiplier.name] * multiplier.scale


def replay_ln_chain(chain: LNChain) -> LNReplayReport:
    """Recompute every derivative and closure identity exactly over ``Q``."""
    failures: list[LNReplayFailure] = []
    for function_index, function in enumerate(chain.functions):
        for axis in range(chain.cell.dimension):
            actual = _standard_derivative(chain, function, axis)
            claimed = chain.claimed_derivatives[function_index][axis]
            if actual.terms != claimed.terms:
                failures.append(
                    LNReplayFailure(function.name, axis, "claimed_derivative")
                )
            composed = _compose_closure(
                chain.closure_matrix[function_index][axis],
                chain,
            )
            if composed.terms != claimed.terms:
                failures.append(
                    LNReplayFailure(function.name, axis, "closure_composition")
                )
    return LNReplayReport(valid=not failures, failures=tuple(failures))


def verify_ln_chain(chain: LNChain) -> bool:
    """Return the exact replay result for ``chain``."""
    return replay_ln_chain(chain).valid


@dataclass(frozen=True)
class LNFormat:
    """Finite format split into exact algebra and an enclosed sup contribution."""

    cell_format: Fraction
    combinatorial_part: Fraction
    sup_contribution: Interval

    @property
    def total(self) -> Interval:
        return Interval.from_rational(self.combinatorial_part) + self.sup_contribution


def _validate_sup_bound(sup_bound: Interval) -> None:
    if not isinstance(sup_bound, Interval):
        raise TypeError("sup_bound must be an Interval")
    if (
        not math.isfinite(sup_bound.lo)
        or not math.isfinite(sup_bound.hi)
        or sup_bound.lo < 0
    ):
        raise ValueError("sup_bound must be finite and nonnegative")


def ln_format(
    chain: LNChain,
    *,
    cell_format: ExactRational,
    sup_bound: Interval,
) -> LNFormat:
    r"""Compute the represented chain format without rounding exact-Q terms.

    The exact part is

    ``cell_format + dimension + N + sum_ij(deg(G_ij) + ||G_ij||_1)``.

    ``sup_bound`` is kept as a separate interval because this finite algebra
    cannot prove bounded holomorphy or compute ``sup |F_i|``.  The caller must
    supply that analytic enclosure independently.
    """
    if not verify_ln_chain(chain):
        raise ValueError("cannot assign LN format: exact closure replay failed")
    cell = _positive_fraction(cell_format, "cell_format")
    _validate_sup_bound(sup_bound)
    exact = cell + chain.cell.dimension + len(chain.functions)
    for row in chain.closure_matrix:
        for polynomial in row:
            exact += _polynomial_degree(polynomial)
            exact += sum(abs(coefficient) for _index, coefficient in polynomial.terms)
    return LNFormat(
        cell_format=cell,
        combinatorial_part=Fraction(exact),
        sup_contribution=sup_bound,
    )


def _finite_nonnegative_result(value: Interval) -> Interval:
    if not math.isfinite(value.lo) or not math.isfinite(value.hi):
        raise ValueError("bound overflowed; no finite certificate is available")
    if value.hi < 0:
        raise ArithmeticError("nonnegative bound computation produced a negative interval")
    return Interval(max(0.0, value.lo), value.hi)


def log_chart_derivative_bound(
    sup_bound: Interval,
    delta: ExactRational,
    order: int,
    *,
    margin_ratio: ExactRational = Fraction(1),
    budget: LNResourceBudget = DEFAULT_LN_BUDGET,
) -> Interval:
    r"""Cauchy bound in a logarithmic annulus chart.

    The extension ``A(delta*r1, r2/delta)`` supplies logarithmic distance
    ``log(1/delta)``.  Cauchy's estimate is first applied on circular disks
    with radii strictly below

    ``rho = margin_ratio * log(1/delta)``, with ``0 < margin_ratio <= 1``.

    Taking their radii increasingly to ``rho`` gives the endpoint estimate,
    so ``margin_ratio = 1`` is sound and is the default.  Ratios below one
    remain available when only a reduced margin has been certified.  The
    result encloses ``order! * M / rho**order``.
    """
    _validate_sup_bound(sup_bound)
    if type(order) is not int or order < 0:
        raise ValueError("order must be a nonnegative integer")
    if order > budget.max_derivative_order:
        raise AlgebraBudgetExceeded("log-chart derivative order budget exceeded")
    d = _fraction(delta, "delta")
    ratio = _fraction(margin_ratio, "margin_ratio")
    if not 0 < d < 1:
        raise ValueError("delta must satisfy 0 < delta < 1")
    if not 0 < ratio <= 1:
        raise ValueError("margin_ratio must satisfy 0 < margin_ratio <= 1")
    magnitude = Interval(0.0, sup_bound.hi)
    if order == 0:
        return magnitude
    with certificate_mode():
        distance = ln_iv(Interval.from_rational(1 / d))
    rho = distance * Interval.from_rational(ratio)
    if rho.lo <= 0 or not math.isfinite(rho.hi):
        raise ValueError("the supplied logarithmic strip margin is not positive and finite")
    bound = (
        Interval.from_rational(math.factorial(order))
        * magnitude
        / rho.pow_int(order)
    )
    return _finite_nonnegative_result(bound)


@overload
def monomial_eigenvalue_bound(
    alpha: ExactRational,
    order: int,
    sup_bound: None = None,
    *,
    budget: LNResourceBudget = DEFAULT_LN_BUDGET,
) -> Fraction: ...


@overload
def monomial_eigenvalue_bound(
    alpha: ExactRational,
    order: int,
    sup_bound: Interval,
    *,
    budget: LNResourceBudget = DEFAULT_LN_BUDGET,
) -> Interval: ...


def monomial_eigenvalue_bound(
    alpha: ExactRational,
    order: int,
    sup_bound: Interval | None = None,
    *,
    budget: LNResourceBudget = DEFAULT_LN_BUDGET,
) -> Fraction | Interval:
    r"""Return ``|alpha|**order`` for ``(z*d_z)^order z**alpha``.

    With ``sup_bound``, return its outward-rounded interval product with that
    exact factor.
    """
    exponent = _fraction(alpha, "alpha")
    if type(order) is not int or order < 0:
        raise ValueError("order must be a nonnegative integer")
    if order > budget.max_derivative_order:
        raise AlgebraBudgetExceeded("monomial derivative order budget exceeded")
    factor = abs(exponent) ** order
    if sup_bound is None:
        return factor
    _validate_sup_bound(sup_bound)
    if factor == 0:
        return Interval.point(0.0)
    return _finite_nonnegative_result(
        sup_bound * Interval.from_rational(factor)
    )


def _encode_fraction(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _encode_radius(radius: Radius) -> dict[str, object]:
    if isinstance(radius, Fraction):
        return {"kind": "rational", "value": _encode_fraction(radius)}
    return {
        "kind": "chain_function",
        "name": radius.name,
        "scale": _encode_fraction(radius.scale),
    }


def _encode_cell(cell: LNCell) -> dict[str, object]:
    return {
        "fibers": [
            {
                "kind": fiber.kind,
                "radii": [_encode_radius(radius) for radius in fiber.radii],
            }
            for fiber in cell.fibers
        ],
        "real_part": cell.real_part,
    }


def _encode_budget(budget: LNResourceBudget) -> dict[str, int]:
    return {
        "max_coefficient_bits": budget.max_coefficient_bits,
        "max_composition_products": budget.max_composition_products,
        "max_degree": budget.max_degree,
        "max_derivative_order": budget.max_derivative_order,
        "max_dimension": budget.max_dimension,
        "max_functions": budget.max_functions,
        "max_matrix_entries": budget.max_matrix_entries,
        "max_terms": budget.max_terms,
    }


def _encode_chain(chain: LNChain) -> dict[str, object]:
    return {
        "budget": _encode_budget(chain.budget),
        "cell": _encode_cell(chain.cell),
        "claimed_derivatives": [
            [polynomial.to_payload() for polynomial in row]
            for row in chain.claimed_derivatives
        ],
        "closure_matrix": [
            [polynomial.to_payload() for polynomial in row]
            for row in chain.closure_matrix
        ],
        "functions": [
            {"name": function.name, "polynomial": function.polynomial.to_payload()}
            for function in chain.functions
        ],
    }


def seal_ln_certificate(
    chain: LNChain,
    *,
    cell_format: ExactRational,
    sup_bound: Interval,
) -> Cert:
    """Seal a finite exact-Q closure replay and separately supplied sup bound."""
    format_report = ln_format(
        chain,
        cell_format=cell_format,
        sup_bound=sup_bound,
    )
    payload = {
        "type": "finite_log_noetherian_chain",
        "scope": "finite polynomial closure replay; analytic LN membership is not claimed",
        "chain": _encode_chain(chain),
        "format": {
            "cell_format": _encode_fraction(format_report.cell_format),
            "combinatorial_part": _encode_fraction(
                format_report.combinatorial_part
            ),
            "sup_contribution": encode_interval(format_report.sup_contribution),
        },
    }
    return make_certificate(
        claim="finite exact-Q Log-Noetherian chain closure replay",
        payload=payload,
        honesty={
            "dulac_map_log_noetherian_claim": False,
            "holomorphic_boundedness_claim": False,
            "unproven_claim": False,
        },
        meta={TRANSCEND_BACKEND_KEY: NO_TRANSCENDENTAL_BACKEND},
    )


def _coefficient_equalities(
    lhs: SparsePolynomial,
    rhs: SparsePolynomial,
) -> list[dict[str, list[int]]]:
    indices = sorted(set(dict(lhs.terms)) | set(dict(rhs.terms)))
    left = dict(lhs.terms)
    right = dict(rhs.terms)
    return [
        {
            "lhs": _encode_fraction(left.get(index, Fraction(0))),
            "rhs": _encode_fraction(right.get(index, Fraction(0))),
        }
        for index in indices
    ]


def seal_ln_chain_closure_obligation(chain: LNChain) -> Cert:
    """Seal coefficient equalities for the minimal Lean-kernel bridge.

    Python first replays the complete chain.  The emitted payload contains
    only finite rational coefficient equalities; it does not assert that an
    actual physical function belongs to the chain.
    """
    if not verify_ln_chain(chain):
        raise ValueError("cannot seal LN chain closure: exact replay failed")
    identities: list[dict[str, list[int]]] = []
    for function_index, function in enumerate(chain.functions):
        for axis in range(chain.cell.dimension):
            claimed = chain.claimed_derivatives[function_index][axis]
            actual = _standard_derivative(chain, function, axis)
            composed = _compose_closure(
                chain.closure_matrix[function_index][axis],
                chain,
            )
            identities.extend(_coefficient_equalities(actual, claimed))
            identities.extend(_coefficient_equalities(composed, claimed))
    if not identities:
        identities.append({"lhs": [0, 1], "rhs": [0, 1]})
    return make_certificate(
        claim="finite exact-Q LN chain coefficient closure",
        payload={
            "type": "ln_chain_closure",
            "identities": identities,
            "scope": "finite coefficient algebra; physical LN membership not claimed",
        },
        honesty={
            "dulac_map_log_noetherian_claim": False,
            "unproven_claim": False,
        },
        meta={TRANSCEND_BACKEND_KEY: NO_TRANSCENDENTAL_BACKEND},
    )


def seal_ln_format_bound_obligation(
    format_report: LNFormat,
    *,
    strict_upper: ExactRational,
) -> Cert:
    """Seal one strict rational bound on the combinatorial format part.

    The interval sup contribution deliberately stays outside this rational
    payload and must be checked by its enclosure replay.
    """
    upper = _fraction(strict_upper, "strict_upper")
    if format_report.combinatorial_part >= upper:
        raise ValueError("strict_upper must exceed the combinatorial format")
    return make_certificate(
        claim="finite strict rational LN format bound",
        payload={
            "type": "ln_format_bound",
            "inequalities": [
                {
                    "lhs": _encode_fraction(format_report.combinatorial_part),
                    "rhs": _encode_fraction(upper),
                }
            ],
            "scope": "rational combinatorial format only; sup enclosure is external",
        },
        honesty={
            "dulac_map_log_noetherian_claim": False,
            "holomorphic_boundedness_claim": False,
            "unproven_claim": False,
        },
        meta={TRANSCEND_BACKEND_KEY: NO_TRANSCENDENTAL_BACKEND},
    )


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _decode_fraction(value: object, name: str) -> Fraction:
    pair = _require_sequence(value, name)
    if len(pair) != 2:
        raise ValueError(f"{name} must contain numerator and denominator")
    numerator = _require_int(pair[0], f"{name}.numerator")
    denominator = _require_int(pair[1], f"{name}.denominator")
    if denominator == 0:
        raise ValueError(f"{name} denominator must be nonzero")
    return Fraction(numerator, denominator)


def _decode_budget(value: object) -> LNResourceBudget:
    data = _require_mapping(value, "budget")
    expected = {
        "max_coefficient_bits",
        "max_composition_products",
        "max_degree",
        "max_derivative_order",
        "max_dimension",
        "max_functions",
        "max_matrix_entries",
        "max_terms",
    }
    if set(data) != expected:
        raise ValueError("budget fields are incomplete or unexpected")
    return LNResourceBudget(
        max_dimension=_require_int(data["max_dimension"], "max_dimension"),
        max_functions=_require_int(data["max_functions"], "max_functions"),
        max_matrix_entries=_require_int(
            data["max_matrix_entries"], "max_matrix_entries"
        ),
        max_terms=_require_int(data["max_terms"], "max_terms"),
        max_degree=_require_int(data["max_degree"], "max_degree"),
        max_coefficient_bits=_require_int(
            data["max_coefficient_bits"], "max_coefficient_bits"
        ),
        max_composition_products=_require_int(
            data["max_composition_products"], "max_composition_products"
        ),
        max_derivative_order=_require_int(
            data["max_derivative_order"], "max_derivative_order"
        ),
    )


def _decode_radius(value: object) -> Radius:
    data = _require_mapping(value, "radius")
    kind = data.get("kind")
    if kind == "rational" and set(data) == {"kind", "value"}:
        return _positive_fraction(
            _decode_fraction(data["value"], "radius.value"),
            "radius",
        )
    if kind == "chain_function" and set(data) == {"kind", "name", "scale"}:
        name = data["name"]
        if not isinstance(name, str):
            raise TypeError("radius reference name must be a string")
        return ChainFunctionRef(
            name,
            _decode_fraction(data["scale"], "radius.scale"),
        )
    raise ValueError("invalid radius encoding")


def _decode_cell(value: object) -> LNCell:
    data = _require_mapping(value, "cell")
    if set(data) != {"fibers", "real_part"} or type(data["real_part"]) is not bool:
        raise ValueError("invalid cell encoding")
    fiber_values = _require_sequence(data["fibers"], "cell.fibers")
    fibers: list[LNFiber] = []
    for raw in fiber_values:
        fiber_data = _require_mapping(raw, "fiber")
        if set(fiber_data) != {"kind", "radii"}:
            raise ValueError("invalid fiber encoding")
        kind = fiber_data["kind"]
        if kind not in {"Point", "Disc", "PuncturedDisc", "Annulus"}:
            raise ValueError("invalid fiber kind")
        radii = tuple(
            _decode_radius(radius)
            for radius in _require_sequence(fiber_data["radii"], "fiber.radii")
        )
        fibers.append(LNFiber(cast(FiberKind, kind), radii))
    return LNCell(tuple(fibers), real_part=data["real_part"])


def _decode_polynomial(
    value: object,
    *,
    budget: LNResourceBudget,
) -> SparsePolynomial:
    data = _require_mapping(value, "polynomial")
    if set(data) != {"nvars", "terms"}:
        raise ValueError("invalid polynomial encoding")
    nvars = _require_int(data["nvars"], "polynomial.nvars")
    terms: dict[tuple[int, ...], Fraction] = {}
    for raw_term in _require_sequence(data["terms"], "polynomial.terms"):
        term = _require_sequence(raw_term, "polynomial term")
        if len(term) != 2:
            raise ValueError("polynomial term must contain index and coefficient")
        index = tuple(
            _require_int(item, "multi-index entry")
            for item in _require_sequence(term[0], "multi-index")
        )
        coefficient = _decode_fraction(term[1], "coefficient")
        if index in terms:
            raise ValueError("duplicate polynomial multi-index")
        terms[index] = coefficient
    algebra_budget = AlgebraBudget(
        max_terms=budget.max_terms,
        max_degree=budget.max_degree,
        max_bits=budget.max_coefficient_bits,
        max_products=budget.max_composition_products,
    )
    return SparsePolynomial(nvars, terms, budget=algebra_budget)


def _decode_matrix(
    value: object,
    *,
    name: str,
    budget: LNResourceBudget,
) -> tuple[tuple[SparsePolynomial, ...], ...]:
    return tuple(
        tuple(
            _decode_polynomial(polynomial, budget=budget)
            for polynomial in _require_sequence(row, f"{name} row")
        )
        for row in _require_sequence(value, name)
    )


def _decode_chain(value: object) -> LNChain:
    data = _require_mapping(value, "chain")
    expected = {
        "budget",
        "cell",
        "claimed_derivatives",
        "closure_matrix",
        "functions",
    }
    if set(data) != expected:
        raise ValueError("chain fields are incomplete or unexpected")
    budget = _decode_budget(data["budget"])
    functions: list[NamedPolynomial] = []
    for raw_function in _require_sequence(data["functions"], "functions"):
        function_data = _require_mapping(raw_function, "function")
        if set(function_data) != {"name", "polynomial"}:
            raise ValueError("invalid function encoding")
        name = function_data["name"]
        if not isinstance(name, str):
            raise TypeError("function name must be a string")
        functions.append(
            NamedPolynomial(
                name,
                _decode_polynomial(function_data["polynomial"], budget=budget),
            )
        )
    return LNChain(
        cell=_decode_cell(data["cell"]),
        functions=tuple(functions),
        closure_matrix=_decode_matrix(
            data["closure_matrix"],
            name="closure_matrix",
            budget=budget,
        ),
        claimed_derivatives=_decode_matrix(
            data["claimed_derivatives"],
            name="claimed_derivatives",
            budget=budget,
        ),
        budget=budget,
    )


def verify_ln_certificate(certificate: Mapping[str, object]) -> bool:
    """Verify digest, scope, exact closure replay, budgets, and format arithmetic."""
    try:
        if not verify_certificate_digest(certificate):
            return False
        if set(certificate) != {
            "claim",
            "digest",
            "honesty",
            "meta",
            "payload",
            "schema_version",
        }:
            return False
        if certificate["claim"] != "finite exact-Q Log-Noetherian chain closure replay":
            return False
        meta = _require_mapping(certificate["meta"], "meta")
        if meta.get(TRANSCEND_BACKEND_KEY) != NO_TRANSCENDENTAL_BACKEND:
            return False
        honesty = _require_mapping(certificate["honesty"], "honesty")
        if honesty != {
            "dulac_map_log_noetherian_claim": False,
            "holomorphic_boundedness_claim": False,
            "unproven_claim": False,
        }:
            return False
        payload = _require_mapping(certificate["payload"], "payload")
        if set(payload) != {"chain", "format", "scope", "type"}:
            return False
        if payload["type"] != "finite_log_noetherian_chain":
            return False
        if payload["scope"] != (
            "finite polynomial closure replay; analytic LN membership is not claimed"
        ):
            return False
        chain = _decode_chain(payload["chain"])
        if not verify_ln_chain(chain):
            return False
        format_data = _require_mapping(payload["format"], "format")
        if set(format_data) != {
            "cell_format",
            "combinatorial_part",
            "sup_contribution",
        }:
            return False
        cell_format = _decode_fraction(format_data["cell_format"], "cell_format")
        supplied_exact = _decode_fraction(
            format_data["combinatorial_part"],
            "combinatorial_part",
        )
        sup_mapping = _require_mapping(
            format_data["sup_contribution"],
            "sup_contribution",
        )
        if set(sup_mapping) != {"hi", "lo"}:
            return False
        if not all(isinstance(sup_mapping[key], str) for key in ("lo", "hi")):
            return False
        sup_bound = decode_interval(sup_mapping)
        report = ln_format(
            chain,
            cell_format=cell_format,
            sup_bound=sup_bound,
        )
        return (
            report.combinatorial_part == supplied_exact
            and report.sup_contribution == sup_bound
        )
    except (AlgebraBudgetExceeded, ArithmeticError, KeyError, TypeError, ValueError):
        return False


__all__ = [
    "ChainFunctionRef",
    "CoordinateRef",
    "DEFAULT_LN_BUDGET",
    "LNCell",
    "LNChain",
    "LNFiber",
    "LNFormat",
    "LNReplayFailure",
    "LNReplayReport",
    "LNResourceBudget",
    "LNStandardDerivation",
    "NamedPolynomial",
    "Radius",
    "ln_format",
    "log_chart_derivative_bound",
    "monomial_eigenvalue_bound",
    "replay_ln_chain",
    "seal_ln_certificate",
    "seal_ln_chain_closure_obligation",
    "seal_ln_format_bound_obligation",
    "verify_ln_certificate",
    "verify_ln_chain",
]
