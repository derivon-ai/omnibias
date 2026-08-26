# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pairing / weak collapse: every ``<R, φ_g>`` enclosure is ``{0}``.

The surviving object is a **certified weak residual on a named finite
pack**, not a strong solution and not a continuum PDE. An odd residual
against even tests can collapse without ``R`` being the zero polynomial;
that is intended, and the honesty payload records
``not_a_strong_solution``.

Pairings are exact ``Q`` integrals of polynomial products on a compact
interval. A float inner product is not a certificate. Do not conflate
with founding bias collapse (``delta -> 0``) or temperature collapse
(``beta -> inf``).
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.collapse.schema import (
    CollapseOutcome,
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import (
    ObligationVerdict,
    is_singleton_zero,
)
from omnibias.core.proof.lift import as_fraction
from omnibias.core.verified.interval import Interval

PAIRING_SPEC = CollapseSpec(
    name="pairing",
    parameter="test_pack",
    limit="<R,φ_g>={0}",
    surviving_object="weak_residual_on_pack",
    failure="Inconclusive",
    home="omnibias.core.collapse.pairing",
    register="measure",
)

CoeffLike = int | float | Fraction


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="pairing")
    payload["pairing_collapse"] = True
    payload["not_a_strong_solution"] = True
    payload["continuum_parent_inferred"] = False
    return payload


def product_coeffs(
    left: Sequence[CoeffLike],
    right: Sequence[CoeffLike],
) -> tuple[Fraction, ...]:
    """Cauchy product of two coefficient sequences over ``Q``."""

    if not left or not right:
        return ()
    a = [as_fraction(c) for c in left]
    b = [as_fraction(c) for c in right]
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def integrate_polynomial(
    coeffs: Sequence[Fraction],
    lo: Fraction,
    hi: Fraction,
) -> Fraction:
    """Exact ``int_lo^hi p`` for a polynomial with ``Q`` coefficients."""

    if hi < lo:
        raise ValueError("domain must satisfy lo <= hi")
    acc = Fraction(0)
    for power, coeff in enumerate(coeffs):
        acc += coeff * (hi ** (power + 1) - lo ** (power + 1)) / (power + 1)
    return acc


def pairing_value(
    residual: Sequence[CoeffLike],
    test: Sequence[CoeffLike],
    lo: Fraction = Fraction(-1),
    hi: Fraction = Fraction(1),
) -> Fraction:
    """Exact pairing ``int_lo^hi R(x) φ(x) dx``."""

    return integrate_polynomial(product_coeffs(residual, test), lo, hi)


def pairing_collapse(
    residual: Sequence[CoeffLike],
    tests: Sequence[Sequence[CoeffLike]],
    *,
    lo: Fraction | int = -1,
    hi: Fraction | int = 1,
) -> ObligationVerdict:
    """Decide the finite weak statement ``<R, φ_g> = 0`` for every test."""

    if not tests:
        raise ValueError("pairing collapse needs at least one test function")
    a = as_fraction(lo)
    b = as_fraction(hi)
    boxes = [
        Interval.from_rational(pairing_value(residual, test, a, b))
        for test in tests
    ]
    if any(not box.contains_zero() for box in boxes):
        bad = next(box for box in boxes if not box.contains_zero())
        outcome = CollapseOutcome(
            status="excluded",
            spec_name="pairing",
            surviving="DISPROVED",
            residual=bad,
            detail="a test pairing excludes 0; not weakly zero on this pack",
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="DISPROVED",
            outcome=outcome,
            existential=False,
            evaluated=len(boxes),
            complete=True,
            detail=outcome.detail,
        )
    if all(is_singleton_zero(box) for box in boxes):
        outcome = CollapseOutcome(
            status="collapsed",
            spec_name="pairing",
            surviving="weak_residual_on_pack",
            residual=Interval.point(0.0),
            detail=(
                "every pairing on this pack is {0}; not a strong solution "
                "and not a continuum PDE"
            ),
            honesty=_honesty(),
        )
        return ObligationVerdict(
            status="PROVED",
            outcome=outcome,
            existential=False,
            evaluated=len(boxes),
            complete=True,
            detail=outcome.detail,
        )
    fat = next(box for box in boxes if not is_singleton_zero(box))
    outcome = CollapseOutcome(
        status="inconclusive",
        spec_name="pairing",
        surviving=None,
        residual=fat,
        detail="a pairing contains 0 with positive width; Inconclusive",
        honesty=_honesty(),
    )
    return ObligationVerdict(
        status="BLOCKED",
        outcome=outcome,
        existential=False,
        evaluated=len(boxes),
        complete=False,
        detail=outcome.detail,
    )


def _reseed() -> None:
    try:
        get_collapse("pairing")
    except KeyError:
        register_collapse(PAIRING_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "PAIRING_SPEC",
    "integrate_polynomial",
    "pairing_collapse",
    "pairing_value",
    "product_coeffs",
]
