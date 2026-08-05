# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Constrained positivity by a certified Putinar representation.

To prove ``p(x) >= 0`` for all ``x`` in the semialgebraic set
``S = {x : g_1(x) >= 0, ..., g_k(x) >= 0}`` it suffices to exhibit an identity

    p = s_0 + s_1 g_1 + ... + s_k g_k

with every ``s_i`` a sum of squares (Putinar's Positivstellensatz, fixed degree):
on ``S`` each ``s_i g_i >= 0`` and ``s_0 >= 0``, so ``p >= 0``.

The SDP proposes the SOS multipliers ``s_i`` (one PSD Gram block each, packed into
one block-diagonal Gram with the cross blocks pinned to zero).  The **proof** is
again rigorous: the float blocks are rounded and the *joint* coefficient identity
is projected exactly onto the rationals (a Gram entry now feeds several monomials,
so this uses the general :func:`~omnibias.sos.rounding.project_coefficients_exact`
rather than the disjoint-group shortcut), then every block is certified positive
definite by interval ``LDL^T``.  Anything short of that is ``inconclusive``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from omnibias.core.proof.certificate import Cert, positive_definite_certificate
from omnibias.core.verified.eig_operator import interval_ldlt_pivots
from omnibias.core.verified.interval import Interval
from omnibias.sos.certify import DEFAULT_DENOMINATORS
from omnibias.sos.honesty import GLOBAL_POLYNOMIAL, SOSScope, honesty_labels
from omnibias.sos.monomials import monomial_basis
from omnibias.sos.problem import Exponent, Polynomial
from omnibias.sos.rounding import project_coefficients_exact
from omnibias.sos.solve import _svec_dim, _svec_indices, solve_gram_program

_SQRT2 = float(np.sqrt(2.0))


@dataclass(frozen=True)
class SOSMultiplier:
    """One certified-SOS multiplier ``s_i`` in a Putinar representation."""

    constraint_index: int
    """``-1`` for the free term ``s_0``; otherwise the index of the constraint ``g_i``."""
    basis: tuple[Exponent, ...]
    gram: tuple[tuple[str, ...], ...]
    """Exact rational Gram of ``s_i`` (as strings)."""


@dataclass(frozen=True)
class PositivstellensatzCertificate:
    """Result of a constrained-positivity attempt via a Putinar representation."""

    status: str
    n_vars: int
    multipliers: tuple[SOSMultiplier, ...]
    pivots: tuple[tuple[float, float], ...]
    pd_margin: float
    detail: str

    @property
    def certified(self) -> bool:
        """Whether this is a sound proof of positivity on the constraint set."""
        return self.status == "proved"


def _frac(value: float, denominator: int) -> Fraction:
    return Fraction(int(round(float(value) * denominator)), denominator)


def _default_bases(
    polynomial: Polynomial, constraints: Sequence[Polynomial], half_degree: int | None
) -> tuple[int, list[tuple[Exponent, ...]]]:
    n = polynomial.n_vars
    if half_degree is None:
        degrees = [polynomial.degree(), *[g.degree() for g in constraints], 0]
        half_degree = (max(degrees) + 1) // 2
    bases = [monomial_basis(n, half_degree)]
    for g in constraints:
        reduced = half_degree - (max(g.degree(), 0) + 1) // 2
        bases.append(monomial_basis(n, max(reduced, 0)))
    return half_degree, bases


def _inconclusive(polynomial: Polynomial, detail: str) -> PositivstellensatzCertificate:
    return PositivstellensatzCertificate("inconclusive", polynomial.n_vars, (), (), 0.0, detail)


def certify_nonneg_on_set(
    polynomial: Polynomial,
    constraints: Sequence[Polynomial],
    *,
    half_degree: int | None = None,
    denominators: Sequence[int] = DEFAULT_DENOMINATORS,
) -> PositivstellensatzCertificate:
    r"""Certify ``polynomial >= 0`` on ``{g_i >= 0}`` via a Putinar representation.

    Parameters
    ----------
    polynomial:
        The polynomial to certify nonnegative on the constraint set.
    constraints:
        The polynomials ``g_i`` defining ``S = {x : g_i(x) >= 0}``.
    half_degree:
        Relaxation half-degree ``omega`` (the free multiplier ``s_0`` uses a basis
        up to ``omega``; each ``s_i`` up to ``omega - ceil(deg g_i / 2)``).
        Defaults to ``ceil(max deg / 2)``.

    Returns
    -------
    PositivstellensatzCertificate
        ``status == "proved"`` carries the exact rational SOS multipliers and their
        certified interval ``LDL^T`` pivots; ``status == "inconclusive"`` otherwise.
    """
    n = polynomial.n_vars
    constraints = list(constraints)
    extended = [Polynomial.constant(1.0, n), *constraints]  # s_0 multiplies g_0 := 1
    _omega, bases = _default_bases(polynomial, constraints, half_degree)

    sizes = [len(b) for b in bases]
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    total = offsets[-1]
    svec_idx = _svec_indices(total)
    svec_dim = _svec_dim(total)

    var_column: dict[tuple[int, int, int], int] = {}
    n_vars_lin = 0
    for block, basis in enumerate(bases):
        for r in range(len(basis)):
            for s in range(r, len(basis)):
                var_column[(block, r, s)] = n_vars_lin
                n_vars_lin += 1

    # Product monomials that can appear, plus every monomial of p (an unrepresentable
    # p-monomial then forces an all-zero == nonzero row, i.e. an honest infeasibility).
    alphas: set[Exponent] = set(polynomial.support)
    for block, basis in enumerate(bases):
        for r in range(len(basis)):
            for s in range(r, len(basis)):
                base_rs = tuple(basis[r][t] + basis[s][t] for t in range(n))
                for term in extended[block].support:
                    alphas.add(tuple(base_rs[t] + term[t] for t in range(n)))
    alpha_list = sorted(alphas)

    float_rows: list[list[float]] = []
    float_rhs: list[float] = []
    rational_rows: list[list[Fraction]] = []
    for alpha in alpha_list:
        srow = [0.0] * svec_dim
        rrow = [Fraction(0)] * n_vars_lin
        for block, basis in enumerate(bases):
            multiplier = extended[block]
            for r in range(len(basis)):
                for s in range(r, len(basis)):
                    needed = tuple(alpha[t] - basis[r][t] - basis[s][t] for t in range(n))
                    if any(v < 0 for v in needed):
                        continue
                    coeff = multiplier.coefficient(needed)
                    if coeff == 0.0:
                        continue
                    gr, gs = offsets[block] + r, offsets[block] + s
                    if r == s:
                        srow[svec_idx[(gr, gr)]] += coeff
                        rrow[var_column[(block, r, s)]] += Fraction(coeff)
                    else:
                        srow[svec_idx[(gr, gs)]] += _SQRT2 * coeff
                        rrow[var_column[(block, r, s)]] += 2 * Fraction(coeff)
        float_rows.append(srow)
        float_rhs.append(polynomial.coefficient(alpha))
        rational_rows.append(rrow)

    # Pin cross-block entries to zero so the big Gram is block diagonal.
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            for a in range(offsets[i], offsets[i + 1]):
                for b in range(offsets[j], offsets[j + 1]):
                    zero_row = [0.0] * svec_dim
                    zero_row[svec_idx[(a, b)]] = 1.0
                    float_rows.append(zero_row)
                    float_rhs.append(0.0)

    proposal = solve_gram_program(
        total, np.array(float_rows), np.array(float_rhs), np.zeros(svec_dim)
    )
    if proposal.status != "solved" or proposal.gram is None:
        return _inconclusive(polynomial, f"SDP proposer: {proposal.detail}")

    gram = proposal.gram
    target = [Fraction(polynomial.coefficient(alpha)) for alpha in alpha_list]
    for denominator in denominators:
        rounded: list[Fraction] = []
        for block, basis in enumerate(bases):
            for r in range(len(basis)):
                for s in range(r, len(basis)):
                    rounded.append(_frac(gram[offsets[block] + r][offsets[block] + s], denominator))
        exact = project_coefficients_exact(rounded, rational_rows, target)
        if exact is None:
            continue

        blocks: list[tuple[tuple[Exponent, ...], list[list[Fraction]], tuple[Interval, ...]]] = []
        pivots_all: list[Interval] = []
        ok = True
        for block, basis in enumerate(bases):
            size = len(basis)
            block_gram = [[Fraction(0)] * size for _ in range(size)]
            for r in range(size):
                for s in range(r, size):
                    block_gram[r][s] = block_gram[s][r] = exact[var_column[(block, r, s)]]
            pivots = interval_ldlt_pivots(block_gram)
            if pivots is None or not all(p.lo > 0.0 for p in pivots):
                ok = False
                break
            blocks.append((basis, block_gram, pivots))
            pivots_all.extend(pivots)
        if not ok:
            continue

        multipliers = tuple(
            SOSMultiplier(
                constraint_index=block - 1,
                basis=tuple(basis),
                gram=tuple(tuple(str(entry) for entry in row) for row in block_gram),
            )
            for block, (basis, block_gram, _piv) in enumerate(blocks)
        )
        margin = min(p.lo for p in pivots_all)
        return PositivstellensatzCertificate(
            status="proved",
            n_vars=n,
            multipliers=multipliers,
            pivots=tuple((p.lo, p.hi) for p in pivots_all),
            pd_margin=margin,
            detail=(
                f"Putinar proof: {len(multipliers)} certified-SOS multipliers "
                f"(denominator {denominator}), all Gram blocks PD (margin {margin:.3e})"
            ),
        )

    return _inconclusive(
        polynomial,
        "rational rounding could not certify all multiplier Gram blocks PD at the tried "
        "denominators (try a larger half_degree, or the statement may be false)",
    )


def is_nonneg_on_set(
    polynomial: Polynomial, constraints: Sequence[Polynomial], *, half_degree: int | None = None
) -> bool:
    """``True`` iff :func:`certify_nonneg_on_set` returns a sound proof."""
    return certify_nonneg_on_set(polynomial, constraints, half_degree=half_degree).certified


def seal_positivstellensatz_certificate(
    certificate: PositivstellensatzCertificate,
    *,
    claim: str,
    scope: SOSScope | None = None,
) -> Cert:
    """Seal a proved Putinar certificate into a v1 ``positive_definite`` certificate.

    The block-diagonal Gram's pivots are exactly the concatenation of the per-block
    pivots, so one ``allPivotsPos`` obligation certifies every multiplier PD at once.
    """
    if not certificate.certified:
        raise ValueError("cannot seal an inconclusive Positivstellensatz certificate")
    if scope is None:
        scope = SOSScope(GLOBAL_POLYNOMIAL)
    pivots = [Interval(lo, hi) for lo, hi in certificate.pivots]

    # The certified interval LDL^T pivots concatenate the per-multiplier block pivots,
    # so the block-diagonal Gram (multiplier order) factorises to exactly those pivots.
    # Storing it lets an independent replay re-derive the PD fact from the exact matrix.
    total = sum(len(mult.basis) for mult in certificate.multipliers)
    block_diag = [["0"] * total for _ in range(total)]
    offset = 0
    for mult in certificate.multipliers:
        size = len(mult.basis)
        for i in range(size):
            for j in range(size):
                block_diag[offset + i][offset + j] = mult.gram[i][j]
        offset += size

    meta = {
        "generator": "omnibias-sos",
        "sos": {
            "form": "positivstellensatz",
            "scope": scope.kind,
            "truncation_order": scope.truncation_order,
            "system": scope.system,
            "n_vars": certificate.n_vars,
            "num_multipliers": len(certificate.multipliers),
            "pd_margin": certificate.pd_margin,
            "gram": [list(row) for row in block_diag],
        },
    }
    return positive_definite_certificate(claim, pivots, honesty=honesty_labels(scope), meta=meta)


__all__ = [
    "PositivstellensatzCertificate",
    "SOSMultiplier",
    "certify_nonneg_on_set",
    "is_nonneg_on_set",
    "seal_positivstellensatz_certificate",
]
