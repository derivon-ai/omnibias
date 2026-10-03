# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Exact finite Dulac models and certified nonoscillation bounds.

The substitution ``x = exp(-kappa)`` sends
``x**alpha * log(x)**j`` to ``(-1)**j * kappa**j * exp(-alpha*kappa)``.
For rational exponents this is an exact
:class:`~omnibias.dynamics.cyclicity.ConfluentExponentialPolynomial`, whose
derivation-division chain gives a global isolated-zero bound. Interval
exponents use sign-stable derivative chains and the verified power
compensator.

These certificates concern the supplied finite expansion only. A declared
remainder is metadata: no physical return-map membership, uniform remainder
theorem, graphic capture, or finite-cyclicity theorem is inferred.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from math import comb, factorial, lcm
from typing import Literal

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.realization.polynomial import Rational, rational
from omnibias.core.verified.asymptotic_jet import power_compensator
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import certificate_mode, exp_iv, ln_iv
from omnibias.dynamics.cyclicity import (
    ConfluentExponentialPolynomial,
    ExponentialCyclicityCertificate,
    certify_exponential_cyclicity,
    verify_exponential_cyclicity,
)


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


@dataclass(frozen=True)
class RationalInterval:
    """Closed exact rational interval used for coefficients and exponents."""

    lo: Fraction
    hi: Fraction

    @classmethod
    def create(
        cls,
        value: Rational | Sequence[Rational],
    ) -> RationalInterval:
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) != 2:
                raise ValueError("an interval needs two endpoints")
            lo, hi = rational(value[0]), rational(value[1])
        else:
            lo = hi = rational(value)  # type: ignore[arg-type]
        return cls(lo, hi)

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("rational interval endpoints are reversed")

    @property
    def exact(self) -> bool:
        return self.lo == self.hi

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    @property
    def mid(self) -> Fraction:
        return (self.lo + self.hi) / 2

    def to_interval(self) -> Interval:
        lo = Interval.from_rational(self.lo)
        hi = Interval.from_rational(self.hi)
        return Interval(lo.lo, hi.hi)

    def to_payload(self) -> list[list[int]]:
        return [_q(self.lo), _q(self.hi)]


@dataclass(frozen=True)
class DulacTerm:
    """One ``coefficient * x**exponent * log(x)**log_power`` term."""

    exponent: RationalInterval
    log_power: int
    coefficient: RationalInterval

    @classmethod
    def create(
        cls,
        exponent: Rational | Sequence[Rational],
        log_power: int,
        coefficient: Rational | Sequence[Rational],
    ) -> DulacTerm:
        return cls(
            RationalInterval.create(exponent),
            log_power,
            RationalInterval.create(coefficient),
        )

    def __post_init__(self) -> None:
        if self.exponent.lo <= 0:
            raise ValueError("Dulac exponents must be strictly positive")
        if type(self.log_power) is not int or not 0 <= self.log_power <= 16:
            raise ValueError("log_power must be an integer from zero to sixteen")

    @property
    def exact(self) -> bool:
        return self.exponent.exact and self.coefficient.exact

    def to_payload(self) -> dict[str, object]:
        return {
            "exponent": self.exponent.to_payload(),
            "log_power": self.log_power,
            "coefficient": self.coefficient.to_payload(),
        }


@dataclass(frozen=True)
class DulacExpansion:
    """A finite declared Dulac expansion with an unproved remainder bound."""

    terms: tuple[DulacTerm, ...]
    truncation_order: int
    remainder_bound: Fraction

    @classmethod
    def create(
        cls,
        terms: Sequence[
            tuple[
                Rational | Sequence[Rational],
                int,
                Rational | Sequence[Rational],
            ]
        ],
        *,
        truncation_order: int,
        remainder_bound: Rational = 0,
    ) -> DulacExpansion:
        return cls(
            tuple(DulacTerm.create(*term) for term in terms),
            truncation_order,
            rational(remainder_bound),
        )

    def __post_init__(self) -> None:
        if type(self.truncation_order) is not int or self.truncation_order < 1:
            raise ValueError("truncation_order must be a positive integer")
        if self.remainder_bound < 0:
            raise ValueError("remainder_bound must be nonnegative")
        if not self.terms:
            raise ValueError("a Dulac expansion needs at least one term")

    @property
    def exact(self) -> bool:
        return all(term.exact for term in self.terms)

    @property
    def leading_terms(self) -> tuple[DulacTerm, ...]:
        nonzero = [
            term
            for term in self.terms
            if not (term.coefficient.lo == term.coefficient.hi == 0)
        ]
        return tuple(
            sorted(
                nonzero,
                key=lambda term: (
                    term.exponent.lo,
                    term.exponent.hi,
                    -term.log_power,
                ),
            )[:3]
        )

    @property
    def digest(self) -> str:
        return source_digest(self.to_payload())

    def to_payload(self) -> dict[str, object]:
        return {
            "terms": [term.to_payload() for term in self.terms],
            "truncation_order": self.truncation_order,
            "remainder_bound": _q(self.remainder_bound),
        }


def _merge_terms(terms: Sequence[DulacTerm]) -> tuple[DulacTerm, ...]:
    exact: dict[tuple[Fraction, int], Fraction] = {}
    boxed: list[DulacTerm] = []
    for term in terms:
        if term.exact:
            key = (term.exponent.lo, term.log_power)
            exact[key] = exact.get(key, Fraction(0)) + term.coefficient.lo
        else:
            boxed.append(term)
    merged = [
        DulacTerm.create(exponent, log_power, coefficient)
        for (exponent, log_power), coefficient in exact.items()
        if coefficient
    ]
    return tuple(
        sorted(
            (*merged, *boxed),
            key=lambda term: (
                term.exponent.lo,
                term.exponent.hi,
                -term.log_power,
            ),
        )
    )


def displacement_expansion(
    return_map: DulacExpansion,
    *,
    identity_coefficient: Rational = 1,
) -> DulacExpansion:
    """Return the finite displacement ``return_map - identity_coefficient*x``."""
    identity = DulacTerm.create(1, 0, -rational(identity_coefficient))
    terms = _merge_terms((*return_map.terms, identity))
    if not terms:
        terms = (DulacTerm.create(1, 0, 0),)
    return DulacExpansion(
        terms,
        return_map.truncation_order,
        return_map.remainder_bound,
    )


def dulac_to_confluent(expansion: DulacExpansion) -> ConfluentExponentialPolynomial:
    """Map an exact rational Dulac expansion through ``x=exp(-kappa)``."""
    if not expansion.exact:
        raise TypeError("exact rational exponents and coefficients are required")
    grouped: dict[Fraction, list[Fraction]] = {}
    for term in expansion.terms:
        exponent = -term.exponent.lo
        coefficients = grouped.setdefault(exponent, [])
        while len(coefficients) <= term.log_power:
            coefficients.append(Fraction(0))
        coefficients[term.log_power] += (
            (-1) ** term.log_power * term.coefficient.lo
        )
    return ConfluentExponentialPolynomial(tuple(grouped.items()))


def enclose_dulac_term(term: DulacTerm, x: RationalInterval) -> Interval:
    """Soundly enclose one term, including irrational exponent boxes.

    Positive log powers are evaluated through the confluent power
    compensator on its diagonal. The exponent remains an interval; it is never
    replaced by a representative floating-point value.
    """
    if x.lo <= 0 or x.hi > 1:
        raise ValueError("the boundary coordinate must satisfy 0 < x <= 1")
    exponent = term.exponent.to_interval()
    coefficient = term.coefficient.to_interval()
    xx = x.to_interval()
    if term.log_power == 0:
        with certificate_mode():
            value = exp_iv(exponent * ln_iv(xx))
    else:
        value = term.log_power * power_compensator(
            exponent,
            exponent,
            xx,
            a_order=term.log_power - 1,
        )
    return coefficient * value


def _derivation_rows(
    certificate: ExponentialCyclicityCertificate,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for current, derived in zip(
        certificate.chain,
        certificate.chain[1:],
        strict=False,
    ):
        pivot = current.terms[0][0]
        current_map = {exponent: coefficients for exponent, coefficients in current.terms}
        derived_map = {exponent: coefficients for exponent, coefficients in derived.terms}
        exponents = sorted(set(current_map) | set(derived_map))
        for exponent in exponents:
            coefficients = current_map.get(exponent, ())
            target = derived_map.get(exponent, ())
            size = max(len(coefficients), len(target))
            for degree in range(size):
                contributions: list[Fraction] = []
                if degree < len(coefficients):
                    contributions.append((exponent - pivot) * coefficients[degree])
                if degree + 1 < len(coefficients):
                    contributions.append(Fraction(degree + 1) * coefficients[degree + 1])
                expected = target[degree] if degree < len(target) else Fraction(0)
                denominator = lcm(
                    *(value.denominator for value in contributions),
                    expected.denominator,
                )
                rows.append(
                    {
                        "terms": [
                            [value.numerator, denominator // value.denominator]
                            for value in contributions
                        ]
                        or [[0, 1]],
                        "rhs": expected.numerator
                        * (denominator // expected.denominator),
                    }
                )
    return rows or [{"terms": [[0, 1]], "rhs": 0}]


@dataclass(frozen=True)
class DulacCyclicityCertificate:
    """Exact global zero bound for a finite rational Dulac displacement."""

    expansion: DulacExpansion
    transformed: ConfluentExponentialPolynomial
    exponential: ExponentialCyclicityCertificate
    upper_bound: int
    leading_terms: tuple[DulacTerm, ...]
    source_digest: str
    formal_seal: Cert


@dataclass(frozen=True)
class DulacFormalVerification:
    result: LeanCheckResult
    theorem_prover_verified: bool


def certify_dulac_cyclicity(
    displacement: DulacExpansion,
) -> DulacCyclicityCertificate:
    """Certify global nonoscillation of one exact finite Dulac model."""
    transformed = dulac_to_confluent(displacement)
    exponential = certify_exponential_cyclicity(transformed)
    seal = make_certificate(
        claim="Exact-Q derivation-division replay for a finite Dulac model.",
        payload={
            "type": "polynomial_identity_q",
            "source_digest": displacement.digest,
            "equations": _derivation_rows(exponential),
            "positive": [],
            "upper_bound": exponential.upper_bound,
            "formal_scope": (
                "finite rational derivation-division identities; analytic Rolle "
                "and physical return-map membership remain external"
            ),
        },
        honesty={
            "dulac_truncated_model_only": True,
            "physical_return_membership_proved": False,
            "uniform_remainder_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return DulacCyclicityCertificate(
        displacement,
        transformed,
        exponential,
        exponential.upper_bound,
        displacement.leading_terms,
        displacement.digest,
        seal,
    )


def verify_dulac_cyclicity(certificate: DulacCyclicityCertificate) -> bool:
    """Replay the transformed model, zero bound, and exact formal rows."""
    if (
        certificate.source_digest != certificate.expansion.digest
        or not verify_certificate_digest(certificate.formal_seal)
        or not verify_exponential_cyclicity(
            certificate.transformed,
            certificate.exponential,
        )
    ):
        return False
    try:
        expected = certify_dulac_cyclicity(certificate.expansion)
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected == certificate


def verify_dulac_cyclicity_formally(
    certificate: DulacCyclicityCertificate,
) -> DulacFormalVerification:
    """Run the Mathlib-free kernel on the exact derivation identities."""
    result = check_certificate(certificate.formal_seal)
    return DulacFormalVerification(result, result.verified)


def _derivative_coefficients(
    term: DulacTerm,
    order: int,
) -> tuple[RationalInterval, ...]:
    """Coefficients of ``D_kappa**order`` after removing the positive exponential."""
    degree = term.log_power
    output = [RationalInterval(Fraction(0), Fraction(0)) for _ in range(degree + 1)]
    exponent = term.exponent
    coefficient = term.coefficient
    signed_coefficient = RationalInterval(
        min(
            (-1) ** degree * coefficient.lo,
            (-1) ** degree * coefficient.hi,
        ),
        max(
            (-1) ** degree * coefficient.lo,
            (-1) ** degree * coefficient.hi,
        ),
    )
    a = exponent.to_interval()
    c = signed_coefficient.to_interval()
    for differentiated_power in range(min(order, degree) + 1):
        remaining_power = degree - differentiated_power
        scalar = Fraction(
            comb(order, differentiated_power)
            * factorial(degree),
            factorial(remaining_power),
        )
        interval = (
            Interval.from_rational(scalar)
            * c
            * (-a) ** (order - differentiated_power)
        )
        output[remaining_power] = RationalInterval(
            Fraction.from_float(interval.lo),
            Fraction.from_float(interval.hi),
        )
    return tuple(output)


def _global_derivative_order(
    expansion: DulacExpansion,
    max_derivative: int,
) -> tuple[int, int, tuple[Fraction, ...]] | None:
    """Find a sign-definite derivative on ``kappa > 0`` from coefficient signs."""
    for order in range(max_derivative + 1):
        coefficients = [
            coefficient
            for term in expansion.terms
            for coefficient in _derivative_coefficients(term, order)
            if not (coefficient.lo == coefficient.hi == 0)
        ]
        if coefficients and all(coefficient.lo > 0 for coefficient in coefficients):
            return order, 1, tuple(coefficient.lo for coefficient in coefficients)
        if coefficients and all(coefficient.hi < 0 for coefficient in coefficients):
            return order, -1, tuple(-coefficient.hi for coefficient in coefficients)
    return None


def _replace_interval(
    expansion: DulacExpansion,
    axis: int,
    interval: RationalInterval,
) -> DulacExpansion:
    term_index, field_index = divmod(axis, 2)
    if not 0 <= term_index < len(expansion.terms):
        raise ValueError("split axis out of range")
    terms = list(expansion.terms)
    term = terms[term_index]
    terms[term_index] = (
        replace(term, exponent=interval)
        if field_index == 0
        else replace(term, coefficient=interval)
    )
    return replace(expansion, terms=tuple(terms))


def _box(expansion: DulacExpansion) -> tuple[RationalInterval, ...]:
    return tuple(
        interval
        for term in expansion.terms
        for interval in (term.exponent, term.coefficient)
    )


def _box_payload(expansion: DulacExpansion) -> list[object]:
    return [interval.to_payload() for interval in _box(expansion)]


def _leaf_seal(
    expansion: DulacExpansion,
    order: int,
    sign: int,
    margins: tuple[Fraction, ...],
) -> Cert:
    return make_certificate(
        claim="Exact rational signs for a transformed derivative on kappa > 0.",
        payload={
            "type": "polynomial_identity_q",
            "source_digest": expansion.digest,
            "equations": [{"terms": [[sign, sign]], "rhs": 1}],
            "positive": [_q(margin) for margin in margins],
            "derivative_order": order,
            "formal_scope": (
                "finite coefficient signs; positivity of exponentials and the "
                "Rolle implication remain analytic"
            ),
        },
        honesty={
            "positive_kappa_transformed_derivative_sign": True,
            "dulac_truncated_model_only": True,
            "physical_return_membership_proved": False,
        },
        meta={"transcend_backend": "not_used"},
    )


@dataclass(frozen=True)
class DulacUniformCoverNode:
    """One proved/blocked leaf or one exact exponent/coefficient bisection."""

    expansion: DulacExpansion
    status: Literal["PROVED", "BLOCKED"]
    count: int | None
    derivative_order: int | None
    derivative_sign: int | None
    sign_seal: Cert | None = None
    split_axis: int | None = None
    split_cut: Fraction | None = None
    left: DulacUniformCoverNode | None = None
    right: DulacUniformCoverNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaves(self) -> tuple[DulacUniformCoverNode, ...]:
        if self.is_leaf:
            return (self,)
        if self.left is None or self.right is None:
            raise ArithmeticError("a cover split must have two children")
        return (*self.left.leaves(), *self.right.leaves())


def _tree_payload(node: DulacUniformCoverNode) -> dict[str, object]:
    if node.is_leaf:
        return {
            "kind": "leaf",
            "box": _box_payload(node.expansion),
            "count": node.count,
        }
    if (
        node.split_axis is None
        or node.split_cut is None
        or node.left is None
        or node.right is None
    ):
        raise ArithmeticError("invalid cover split")
    return {
        "kind": "split",
        "axis": node.split_axis,
        "cut": _q(node.split_cut),
        "left": _tree_payload(node.left),
        "right": _tree_payload(node.right),
    }


@dataclass(frozen=True)
class DulacUniformCoverCertificate:
    """Finite exact-Q parameter cover with one uniform truncated-model bound."""

    expansion: DulacExpansion
    max_derivative: int
    max_depth: int
    tree: DulacUniformCoverNode
    status: Literal["PROVED", "BLOCKED"]
    uniform_bound: int | None
    source_digest: str
    seal: Cert | None


@dataclass(frozen=True)
class DulacUniformFormalVerification:
    cover: LeanCheckResult
    leaves: tuple[LeanCheckResult, ...]
    theorem_prover_verified: bool


def _build_cover(
    expansion: DulacExpansion,
    *,
    depth: int,
    max_depth: int,
    max_derivative: int,
) -> DulacUniformCoverNode:
    proved = _global_derivative_order(expansion, max_derivative)
    if proved is not None:
        order, sign, margins = proved
        return DulacUniformCoverNode(
            expansion,
            "PROVED",
            order,
            order,
            sign,
            _leaf_seal(expansion, order, sign, margins),
        )
    widths = [interval.width for interval in _box(expansion)]
    if depth >= max_depth or not any(widths):
        return DulacUniformCoverNode(expansion, "BLOCKED", None, None, None)
    axis = max(range(len(widths)), key=widths.__getitem__)
    interval = _box(expansion)[axis]
    cut = interval.mid
    left_expansion = _replace_interval(
        expansion,
        axis,
        RationalInterval(interval.lo, cut),
    )
    right_expansion = _replace_interval(
        expansion,
        axis,
        RationalInterval(cut, interval.hi),
    )
    left = _build_cover(
        left_expansion,
        depth=depth + 1,
        max_depth=max_depth,
        max_derivative=max_derivative,
    )
    right = _build_cover(
        right_expansion,
        depth=depth + 1,
        max_depth=max_depth,
        max_derivative=max_derivative,
    )
    status: Literal["PROVED", "BLOCKED"] = (
        "PROVED" if left.status == right.status == "PROVED" else "BLOCKED"
    )
    return DulacUniformCoverNode(
        expansion,
        status,
        max(left.count or 0, right.count or 0) if status == "PROVED" else None,
        None,
        None,
        None,
        axis,
        cut,
        left,
        right,
    )


def certify_dulac_uniform_cover(
    expansion: DulacExpansion,
    *,
    max_derivative: int = 6,
    max_depth: int = 6,
) -> DulacUniformCoverCertificate:
    """Bisect blocked parameter boxes and certify a uniform derivative bound."""
    if expansion.exact:
        raise ValueError("a uniform cover requires a positive-width parameter box")
    if type(max_derivative) is not int or not 0 <= max_derivative <= 16:
        raise ValueError("max_derivative must be an integer from zero to sixteen")
    if type(max_depth) is not int or not 0 <= max_depth <= 16:
        raise ValueError("max_depth must be an integer from zero to sixteen")
    tree = _build_cover(
        expansion,
        depth=0,
        max_depth=max_depth,
        max_derivative=max_derivative,
    )
    if tree.status != "PROVED":
        return DulacUniformCoverCertificate(
            expansion,
            max_derivative,
            max_depth,
            tree,
            "BLOCKED",
            None,
            expansion.digest,
            None,
        )
    leaves = tree.leaves()
    bound = max(int(leaf.count or 0) for leaf in leaves)
    seal = make_certificate(
        claim="Exact rational box tiling for a uniform finite Dulac-model bound.",
        payload={
            "type": "box_cover_tiling",
            "source_digest": expansion.digest,
            "parent": _box_payload(expansion),
            "tree": _tree_payload(tree),
            "uniform_bound": bound,
            "formal_scope": (
                "exact parameter tiling and integer leaf bounds; transcendental "
                "kernel positivity, Rolle, remainder, and physical membership "
                "remain external"
            ),
        },
        honesty={
            "parameter_uniform_truncated_dulac_bound": True,
            "dulac_truncated_model_only": True,
            "physical_return_membership_proved": False,
            "uniform_remainder_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "full_hilbert16_solved": False,
        },
        meta={"transcend_backend": "not_used"},
    )
    return DulacUniformCoverCertificate(
        expansion,
        max_derivative,
        max_depth,
        tree,
        "PROVED",
        bound,
        expansion.digest,
        seal,
    )


def _verify_cover_tree(
    node: DulacUniformCoverNode,
    expected: DulacExpansion,
    bound: int,
) -> bool:
    if node.expansion != expected:
        return False
    if node.is_leaf:
        return bool(
            node.status == "PROVED"
            and node.count is not None
            and node.count <= bound
            and node.derivative_order == node.count
            and node.derivative_sign in (-1, 1)
            and node.sign_seal is not None
            and verify_certificate_digest(node.sign_seal)
        )
    if (
        node.split_axis is None
        or node.split_cut is None
        or node.left is None
        or node.right is None
    ):
        return False
    interval = _box(expected)[node.split_axis]
    if not interval.lo < node.split_cut < interval.hi:
        return False
    left = _replace_interval(
        expected,
        node.split_axis,
        RationalInterval(interval.lo, node.split_cut),
    )
    right = _replace_interval(
        expected,
        node.split_axis,
        RationalInterval(node.split_cut, interval.hi),
    )
    return _verify_cover_tree(node.left, left, bound) and _verify_cover_tree(
        node.right,
        right,
        bound,
    )


def verify_dulac_uniform_cover(
    certificate: DulacUniformCoverCertificate,
) -> bool:
    """Replay every bisection, leaf sign, and reported uniform bound."""
    if certificate.source_digest != certificate.expansion.digest:
        return False
    try:
        expected = certify_dulac_uniform_cover(
            certificate.expansion,
            max_derivative=certificate.max_derivative,
            max_depth=certificate.max_depth,
        )
    except (ArithmeticError, RuntimeError, TypeError, ValueError):
        return False
    if expected != certificate:
        return False
    if certificate.status != "PROVED":
        return certificate.seal is None and certificate.uniform_bound is None
    return bool(
        certificate.seal is not None
        and certificate.uniform_bound is not None
        and verify_certificate_digest(certificate.seal)
        and _verify_cover_tree(
            certificate.tree,
            certificate.expansion,
            certificate.uniform_bound,
        )
    )


def verify_dulac_uniform_cover_formally(
    certificate: DulacUniformCoverCertificate,
) -> DulacUniformFormalVerification:
    """Run Lean on the exact cover and every finite coefficient-sign leaf."""
    if certificate.status != "PROVED" or certificate.seal is None:
        raise ValueError("a proved sealed Dulac cover is required")
    cover = check_certificate(certificate.seal)
    leaves = tuple(
        check_certificate(leaf.sign_seal)
        for leaf in certificate.tree.leaves()
        if leaf.sign_seal is not None
    )
    verified = cover.verified and len(leaves) == len(certificate.tree.leaves()) and all(
        result.verified for result in leaves
    )
    return DulacUniformFormalVerification(cover, leaves, verified)


__all__ = [
    "DulacCyclicityCertificate",
    "DulacExpansion",
    "DulacFormalVerification",
    "DulacTerm",
    "DulacUniformCoverCertificate",
    "DulacUniformCoverNode",
    "DulacUniformFormalVerification",
    "RationalInterval",
    "certify_dulac_cyclicity",
    "certify_dulac_uniform_cover",
    "displacement_expansion",
    "dulac_to_confluent",
    "enclose_dulac_term",
    "verify_dulac_cyclicity",
    "verify_dulac_cyclicity_formally",
    "verify_dulac_uniform_cover",
    "verify_dulac_uniform_cover_formally",
]
