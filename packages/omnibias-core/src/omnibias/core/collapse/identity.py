# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Contact / identity collapse: remainder enclosure of ``f - T_N``.

The surviving object is a **germ identity** on a compact, not a
derivative, not a 0/1 step, and not a point-plus-proof of a scalar
value. A float ``||R_N||`` is not a certificate. Exact ``Q`` coefficient
agreement is ``{0}`` without interval Horner (so a zero polynomial is
not widened by outward rounding). A nonzero difference is enclosed on
the domain; ``{0}`` proves, exclusion of ``0`` disproves, a fat zero
is ``BLOCKED``.

Do not conflate with founding bias collapse (``delta -> 0``),
temperature collapse (``beta -> inf``), or Enclosure Collapse: ``width -> 0``
of a sound enclosure, yielding a point plus a proof. Continuum PDE identities
are not inferred.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction

from omnibias.core.collapse.schema import (
    CollapseSpec,
    add_registry_hook,
    default_honesty,
    get_collapse,
    register_collapse,
)
from omnibias.core.collapse.verdict import ObligationVerdict, adjudicate_residual
from omnibias.core.proof.lift import as_fraction
from omnibias.core.verified.interval import Interval, IntervalLike

IDENTITY_SPEC = CollapseSpec(
    name="identity",
    parameter="remainder_order",
    limit="R_N={0}",
    surviving_object="germ_identity",
    failure="Inconclusive",
    home="omnibias.core.collapse.identity",
    register="verified",
)

CoeffLike = int | float | Fraction


def _honesty() -> dict[str, bool]:
    payload = default_honesty(spec_name="identity")
    payload["identity_collapse"] = True
    return payload


def difference_coeffs(
    left: Sequence[CoeffLike],
    right: Sequence[CoeffLike],
) -> tuple[Fraction, ...]:
    """Exact ``Q`` coefficient-wise ``left - right``, trailing zeros dropped."""

    n = max(len(left), len(right))
    out: list[Fraction] = []
    for index in range(n):
        a = as_fraction(left[index]) if index < len(left) else Fraction(0)
        b = as_fraction(right[index]) if index < len(right) else Fraction(0)
        out.append(a - b)
    while out and out[-1] == 0:
        out.pop()
    return tuple(out)


def evaluate_difference(
    coeffs: Sequence[Fraction],
    domain: Interval,
) -> Interval:
    """Sound Horner enclosure of a difference polynomial on ``domain``.

    The zero polynomial returns the exact point ``{0}`` so outward
    rounding cannot forge a fat zero.
    """

    if not coeffs:
        return Interval.point(0.0)
    acc = Interval.from_rational(coeffs[-1])
    for coeff in reversed(coeffs[:-1]):
        acc = acc * domain + Interval.from_rational(coeff)
    return acc


def _relabel(verdict: ObligationVerdict) -> ObligationVerdict:
    surviving: str | int | None
    if verdict.proved:
        surviving = "germ_identity"
    else:
        surviving = verdict.outcome.surviving
    outcome = replace(
        verdict.outcome,
        spec_name="identity",
        surviving=surviving,
        honesty=_honesty(),
    )
    return replace(verdict, outcome=outcome)


def identity_collapse(
    left: Sequence[CoeffLike],
    right: Sequence[CoeffLike],
    domain: IntervalLike,
) -> ObligationVerdict:
    """Decide whether ``left`` and ``right`` agree as polynomials on ``domain``."""

    boxed = domain if isinstance(domain, Interval) else Interval.from_value(domain)
    residual = evaluate_difference(difference_coeffs(left, right), boxed)
    return _relabel(adjudicate_residual(residual))


def remainder_collapse(
    coeffs: Sequence[CoeffLike],
    order: int,
    domain: IntervalLike,
) -> ObligationVerdict:
    """Decide whether ``coeffs`` equals its degree-``order`` truncation on ``domain``."""

    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    head = tuple(coeffs[: order + 1])
    return identity_collapse(coeffs, head, domain)


def _reseed() -> None:
    try:
        get_collapse("identity")
    except KeyError:
        register_collapse(IDENTITY_SPEC)


add_registry_hook(_reseed)


__all__ = [
    "IDENTITY_SPEC",
    "difference_coeffs",
    "evaluate_difference",
    "identity_collapse",
    "remainder_collapse",
]
