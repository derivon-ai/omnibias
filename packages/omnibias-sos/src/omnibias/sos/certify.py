# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""The sound part: rigorously certify a polynomial is a sum of squares.

:func:`certify_sos` runs the full propose-then-prove pipeline:

1. propose a floating-point PD Gram ``Q`` with the SDP (:mod:`omnibias.sos.solve`);
2. round + exactly project it onto ``z^T Q z = p`` over the rationals
   (:mod:`omnibias.sos.rounding`);
3. **prove** the rational Gram is positive definite with the repo's rigorous
   interval ``LDL^T`` (``omnibias.core.verified.eig_operator.interval_ldlt_pivots``):
   every pivot's lower endpoint ``> 0`` certifies ``Q`` PD, hence
   ``p(x) = z(x)^T Q z(x) >= 0`` for **all** ``x``.

Soundness over completeness: if the SDP is infeasible, the rational projection is
not exact, or no tried denominator yields a certified-PD Gram, the result is
``status == "inconclusive"`` -- never a false positivity claim.  (Because the
proof requires a *strictly* PD Gram, a polynomial that is SOS only with a
rank-deficient / boundary Gram -- e.g. one vanishing somewhere, like ``x^2 + y^2``
-- is reported inconclusive rather than proved; this preserves soundness.)
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from omnibias.core.verified.eig_operator import interval_ldlt_pivots
from omnibias.sos.monomials import SOSProblem, gram_products
from omnibias.sos.problem import Exponent, Polynomial, RationalPolynomial, SOSCertificate
from omnibias.sos.rounding import (
    CoefficientSource,
    RationalGram,
    exact_coefficient_residual,
    project_to_exact_gram,
)
from omnibias.sos.solve import GramProposer, solve_sos_gram

#: Denominators tried (ascending precision) when rationalising the float Gram.
DEFAULT_DENOMINATORS: tuple[int, ...] = (
    2**8,
    2**10,
    2**13,
    2**16,
    2**20,
    2**26,
    2**32,
    2**40,
)


def _inconclusive(n_vars: int, basis: Sequence[Exponent], detail: str) -> SOSCertificate:
    return SOSCertificate(
        status="inconclusive",
        n_vars=n_vars,
        basis=tuple(basis),
        gram=None,
        pivots=(),
        pd_margin=0.0,
        coeff_residual=float("nan"),
        detail=detail,
    )


def _gram_to_strings(gram: RationalGram) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(str(entry) for entry in row) for row in gram)


def _run_certification(
    float_poly: Polynomial,
    source: CoefficientSource,
    *,
    half_degree: int | None,
    denominators: Sequence[int],
    external: GramProposer | None,
) -> SOSCertificate:
    """Shared propose -> round -> prove loop; ``source`` supplies the exact targets."""
    n = float_poly.n_vars
    basis = SOSProblem.for_polynomial(float_poly, half_degree=half_degree).basis.exponents
    if not set(source.support) <= set(gram_products(basis)):
        return _inconclusive(
            n, basis,
            "polynomial has a monomial no basis product can build; it is not SOS in this basis",
        )

    proposal = solve_sos_gram(float_poly, basis, external=external)
    if proposal.status != "solved" or proposal.gram is None:
        return _inconclusive(n, basis, f"SDP proposer: {proposal.detail}")

    gram_float = [[float(v) for v in row] for row in proposal.gram]
    for denominator in denominators:
        rational = project_to_exact_gram(gram_float, source, basis, denominator=denominator)
        if exact_coefficient_residual(rational, source, basis) != 0:
            continue  # projection did not match exactly (should not happen); stay sound
        pivots = interval_ldlt_pivots(rational)
        if pivots is None or not all(p.lo > 0.0 for p in pivots):
            continue
        margin = min(p.lo for p in pivots)
        return SOSCertificate(
            status="proved",
            n_vars=n,
            basis=tuple(basis),
            gram=_gram_to_strings(rational),
            pivots=tuple((p.lo, p.hi) for p in pivots),
            pd_margin=margin,
            coeff_residual=0.0,
            detail=(
                f"SOS proof: exact rational Gram (denominator {denominator}) certified "
                f"positive definite by interval LDL^T (PD margin {margin:.3e})"
            ),
        )

    return _inconclusive(
        n, basis,
        "rational rounding could not certify a positive-definite Gram at the tried "
        "denominators (the polynomial may be SOS only with a rank-deficient Gram, or not SOS)",
    )


def certify_sos(
    polynomial: Polynomial,
    *,
    half_degree: int | None = None,
    denominators: Sequence[int] = DEFAULT_DENOMINATORS,
    external: GramProposer | None = None,
) -> SOSCertificate:
    r"""Rigorously certify ``polynomial(x) >= 0`` for all ``x`` via an SOS proof.

    Parameters
    ----------
    polynomial:
        The polynomial to certify nonnegative.
    half_degree:
        Half the SOS degree (the monomial basis is taken up to this degree);
        defaults to ``ceil(deg(p) / 2)``.
    denominators:
        Rational-rounding denominators tried in order until one yields a
        certified-PD Gram.
    external:
        Optional external SDP backend forwarded to
        :func:`omnibias.sos.solve.solve_sos_gram` (advisory; never trusted for the
        proof).

    Returns
    -------
    SOSCertificate
        ``status == "proved"`` is a sound universal-positivity proof carrying the
        exact rational Gram and its certified interval ``LDL^T`` pivots;
        ``status == "inconclusive"`` makes no claim.
    """
    return _run_certification(
        polynomial, polynomial,
        half_degree=half_degree, denominators=denominators, external=external,
    )


def certify_sos_rational(
    polynomial: RationalPolynomial,
    *,
    half_degree: int | None = None,
    denominators: Sequence[int] = DEFAULT_DENOMINATORS,
    external: GramProposer | None = None,
) -> SOSCertificate:
    r"""Certify a polynomial with **exact rational** coefficients is a sum of squares.

    Identical to :func:`certify_sos` but the coefficient-matching targets are the
    exact :class:`~fractions.Fraction` coefficients (the SDP still proposes on the
    float image).  This is what the auxiliary-functional method certifies, since
    its residual ``C - Phi - grad(V) . f`` must be represented exactly.
    """
    return _run_certification(
        polynomial.to_float(), polynomial,
        half_degree=half_degree, denominators=denominators, external=external,
    )


def is_sos(polynomial: Polynomial, *, half_degree: int | None = None) -> bool:
    """``True`` iff :func:`certify_sos` returns a sound proof of nonnegativity."""
    return certify_sos(polynomial, half_degree=half_degree).certified


def rational_gram(certificate: SOSCertificate) -> RationalGram | None:
    """Recover the exact rational Gram matrix from a proved certificate."""
    if certificate.gram is None:
        return None
    return [[Fraction(entry) for entry in row] for row in certificate.gram]


__all__ = [
    "DEFAULT_DENOMINATORS",
    "certify_sos",
    "certify_sos_rational",
    "is_sos",
    "rational_gram",
]
