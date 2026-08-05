# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Discrete equation / recurrence discovery -- the ``delta = 1`` twin of the jet search.

Where :mod:`omnibias.symbolic.discovery` reads a *continuous* jet ``[y, dy, d2y, ...]``
off a fitted field and compresses an implicit differential relation, this module
reads a *discrete* jet -- the leading forward differences ``Delta^k a`` of an integer
sequence -- and recovers the exact **linear recurrence with polynomial coefficients**
(a P-recursive / holonomic relation) the sequence satisfies:

.. math::

    \sum_{j=0}^{r} p_j(n)\, a_{n-j} = 0, \qquad p_j \in \mathbb{Q}[n].

Two engines, deliberately contrasted (the refinement program's baseline pair):

* :func:`discover_recurrence` -- an **exact rational** homogeneous solver. It clears
  denominators and calls the tested :func:`omnibias.symbolic.dimensional.integer_null_space`
  on the shifted-and-index-weighted design, so the leading coefficient ``p_0(n)`` may
  itself be a non-constant polynomial. This recovers genuinely P-recursive sequences
  (Catalan, factorial) that a monic least-squares fit cannot, and never suffers
  floating-point conditioning blow-up on long integer grids.
* :func:`discover_recurrence_least_squares` -- the **baseline**: build a float design
  with :func:`build_difference_relation_library` and hand it to the shared
  :func:`omnibias.symbolic.discovery.fit_sparse_equation` (STLSQ). It assumes a monic
  leading term (``a_n`` as the regression target), so it recovers C-finite recurrences
  (Fibonacci) and P-recursive ones whose *leading* coefficient is constant (factorial),
  but not Catalan, and it degrades numerically for large-magnitude columns.

Sequences that are **not** finitely P-recursive (Bell numbers, the partition function
``p(n)``) are *correctly* reported as such by :func:`discover_recurrence` (it returns
``None``); their genuine laws are full-history convolutions, verified separately by
:func:`verify_binomial_recurrence` (Bell) rather than forced into a fixed-order model.

Everything on the exact path runs in :class:`~fractions.Fraction` arithmetic, so a
recovered relation holds *identically*, not up to rounding. Honesty labels match
``omnibias-difference``: the exact recurrence / polynomial extraction is
**closed-form**; the least-squares baseline is **numerical**.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import factorial

import numpy as np
from omnibias.difference import (
    binomial_transform,
    falling_to_monomial,
    forward_difference,
    newton_forward_coeffs,
)
from omnibias.symbolic.dimensional import integer_null_space
from omnibias.symbolic.discovery import SparseEquation, fit_sparse_equation

__all__ = [
    "DifferenceJets",
    "RecurrenceRelation",
    "build_difference_relation_library",
    "discover_recurrence",
    "discover_recurrence_least_squares",
    "extract_difference_jets",
    "polynomial_from_samples",
    "verify_binomial_recurrence",
]

Number = int | Fraction


def _as_fractions(samples: Sequence[Number]) -> list[Fraction]:
    return [v if isinstance(v, Fraction) else Fraction(v) for v in samples]


# --------------------------------------------------------------------------- #
# Discrete jets                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DifferenceJets:
    r"""The discrete jet table ``columns[k][i] = Delta^k a(index[i])`` (exact).

    The finite-difference twin of :class:`omnibias.symbolic.JetBundle`: column ``k``
    is the ``k``-th forward difference of the sequence, anchored at each admissible
    index. For a sequence that is a degree-``d`` polynomial in ``n`` every column with
    ``k > d`` is identically zero -- the discrete analogue of a vanishing derivative.
    """

    index: tuple[int, ...]
    columns: tuple[tuple[Fraction, ...], ...]

    @property
    def max_order(self) -> int:
        return len(self.columns) - 1

    def column_name(self, order: int) -> str:
        if order == 0:
            return "a"
        return f"D{order}a"

    def as_float_design(self) -> np.ndarray:
        """Dense float array ``[len(index), max_order + 1]`` of the jet columns."""
        return np.asarray(
            [[float(col[i]) for col in self.columns] for i in range(len(self.index))],
            dtype=float,
        )


def extract_difference_jets(samples: Sequence[Number], max_order: int) -> DifferenceJets:
    r"""Leading forward differences ``Delta^k a`` up to ``max_order`` at every anchor.

    ``columns[k][i]`` is ``Delta^k`` of the sequence evaluated at ``index[i]``, computed
    in exact rational arithmetic via :func:`omnibias.difference.forward_difference`.
    Anchors run over ``0 .. len(samples) - 1 - max_order`` so every column has the same
    length (a rectangular jet table).
    """
    if max_order < 0:
        raise ValueError(f"max_order must be >= 0, got {max_order}")
    vals = _as_fractions(samples)
    if max_order >= len(vals):
        raise ValueError(
            f"max_order {max_order} needs at least {max_order + 1} samples, got {len(vals)}"
        )
    n_anchors = len(vals) - max_order
    index = tuple(range(n_anchors))
    columns: list[tuple[Fraction, ...]] = []
    for k in range(max_order + 1):
        columns.append(tuple(forward_difference(vals[i : i + k + 1], k)[0] for i in index))
    return DifferenceJets(index=index, columns=tuple(columns))


# --------------------------------------------------------------------------- #
# Recurrence relations                                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RecurrenceRelation:
    r"""A homogeneous P-recursive relation ``sum_j p_j(n) a_{n-j} = 0`` (exact).

    ``coefficients[j][d]`` is the ``n^d`` coefficient of the polynomial ``p_j`` that
    multiplies the lag-``j`` term ``a_{n-j}``. ``order`` is the recurrence order (max
    lag) and ``index_degree`` the max polynomial degree in ``n``.
    """

    order: int
    index_degree: int
    coefficients: tuple[tuple[Fraction, ...], ...]

    def coefficient_poly(self, lag: int, n: int) -> Fraction:
        """Evaluate the polynomial ``p_lag(n)`` at integer ``n``."""
        acc = Fraction(0)
        for degree, coef in enumerate(self.coefficients[lag]):
            acc += coef * Fraction(n) ** degree
        return acc

    def evaluate_residuals(self, samples: Sequence[Number]) -> list[Fraction]:
        r"""``sum_j p_j(n) a_{n-j}`` for every valid ``n`` (all zero iff satisfied)."""
        vals = _as_fractions(samples)
        residuals: list[Fraction] = []
        for n in range(self.order, len(vals)):
            acc = Fraction(0)
            for lag in range(self.order + 1):
                acc += self.coefficient_poly(lag, n) * vals[n - lag]
            residuals.append(acc)
        return residuals

    def max_abs_residual(self, samples: Sequence[Number]) -> Fraction:
        residuals = self.evaluate_residuals(samples)
        return max((abs(r) for r in residuals), default=Fraction(0))

    def is_satisfied_by(self, samples: Sequence[Number]) -> bool:
        return all(r == 0 for r in self.evaluate_residuals(samples))

    def pretty(self, symbol: str = "a") -> str:
        """Human-readable relation, e.g. ``(n + 1) a[n] + (-4 n + 2) a[n-1] = 0``."""
        pieces: list[str] = []
        for lag in range(self.order + 1):
            poly = _format_poly(self.coefficients[lag])
            if poly is None:
                continue
            term = f"{symbol}[n]" if lag == 0 else f"{symbol}[n-{lag}]"
            pieces.append(f"({poly}) {term}")
        body = " + ".join(pieces) if pieces else "0"
        return f"{body} = 0"


def _format_poly(coeffs: Sequence[Fraction]) -> str | None:
    """Render a polynomial in ``n``; return ``None`` if identically zero."""
    terms: list[str] = []
    for degree, coef in enumerate(coeffs):
        if coef == 0:
            continue
        c = _format_fraction(coef)
        if degree == 0:
            terms.append(c)
        elif degree == 1:
            terms.append("n" if coef == 1 else ("-n" if coef == -1 else f"{c} n"))
        else:
            terms.append(f"n^{degree}" if coef == 1 else f"{c} n^{degree}")
    if not terms:
        return None
    out = terms[0]
    for term in terms[1:]:
        out += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    return out


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


# --------------------------------------------------------------------------- #
# Exact rational discovery (the fix)                                          #
# --------------------------------------------------------------------------- #
def _clear_denominators(rows: list[list[Fraction]]) -> list[list[int]]:
    """Scale each row by the lcm of its denominators -> integer rows (null space kept)."""
    int_rows: list[list[int]] = []
    for row in rows:
        lcm = 1
        for value in row:
            den = value.denominator
            lcm = lcm * den // _gcd(lcm, den)
        int_rows.append([int(value * lcm) for value in row])
    return int_rows


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _vector_to_coefficients(
    vector: Sequence[int], order: int, index_degree: int
) -> tuple[tuple[Fraction, ...], ...]:
    width = index_degree + 1
    return tuple(
        tuple(Fraction(vector[lag * width + d]) for d in range(width))
        for lag in range(order + 1)
    )


def _poly_is_zero(coeffs: Sequence[Fraction]) -> bool:
    return all(c == 0 for c in coeffs)


def discover_recurrence(
    samples: Sequence[Number],
    *,
    max_order: int = 4,
    max_index_degree: int = 3,
    min_equation_margin: int = 2,
) -> RecurrenceRelation | None:
    r"""Recover the minimal exact P-recursive relation ``sum_j p_j(n) a_{n-j} = 0``.

    Searches ``(order, index_degree)`` in increasing order (order-major, so the
    lowest-order / lowest-degree relation wins) and, for each, solves for the exact
    homogeneous null space of the design whose columns are ``n^d a_{n-j}``
    (``j = 0 .. order``, ``d = 0 .. index_degree``). A **unique** (nullity-one) null
    vector whose leading (``a_n``) and trailing (``a_{n-order}``) coefficient
    polynomials are both non-trivial is returned as a :class:`RecurrenceRelation`;
    otherwise the search continues.

    Unlike :func:`discover_recurrence_least_squares` the leading coefficient ``p_0(n)``
    is free to be a non-constant polynomial, so genuinely P-recursive sequences whose
    natural relation is *not* monic -- e.g. Catalan ``(n+1)C_n - (4n-2)C_{n-1} = 0`` --
    are recovered exactly. Returns ``None`` when no finite-order polynomial recurrence
    up to the search bounds exists (e.g. Bell numbers / the partition function, which
    are not P-recursive); that ``None`` is a genuine finding, not a failure.
    """
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if max_index_degree < 0:
        raise ValueError(f"max_index_degree must be >= 0, got {max_index_degree}")
    vals = _as_fractions(samples)
    n_samples = len(vals)

    for order in range(1, max_order + 1):
        for index_degree in range(max_index_degree + 1):
            width = index_degree + 1
            unknowns = (order + 1) * width
            row_indices = list(range(order, n_samples))
            if len(row_indices) < unknowns + min_equation_margin:
                continue
            design: list[list[Fraction]] = []
            for n in row_indices:
                row: list[Fraction] = []
                for lag in range(order + 1):
                    a_shift = vals[n - lag]
                    for d in range(width):
                        row.append(Fraction(n) ** d * a_shift)
                design.append(row)
            null = integer_null_space(_clear_denominators(design))
            if len(null) != 1:
                continue  # 0 = no relation; >1 = ambiguous, prefer a simpler model
            coeffs = _vector_to_coefficients(null[0], order, index_degree)
            if _poly_is_zero(coeffs[0]) or _poly_is_zero(coeffs[order]):
                continue  # degenerate: not a genuine order-`order` recurrence
            relation = RecurrenceRelation(
                order=order, index_degree=index_degree, coefficients=coeffs
            )
            if relation.is_satisfied_by(vals):  # exact by construction; guard anyway
                return relation
    return None


# --------------------------------------------------------------------------- #
# Least-squares baseline (numerical)                                          #
# --------------------------------------------------------------------------- #
def build_difference_relation_library(
    samples: Sequence[Number],
    *,
    order: int,
    index_degree: int = 0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    r"""Float design for a monic recurrence fit: ``a_n`` vs ``{n^d a_{n-j}}``.

    Returns ``(design, term_names, target)`` where ``target[i] = a_n`` and each design
    column is ``n^d a_{n-j}`` for ``j = 1 .. order`` and ``d = 0 .. index_degree``.
    This is the discrete twin of :func:`omnibias.symbolic.build_jet_relation_library`;
    feed it to :func:`omnibias.symbolic.fit_sparse_equation` (see
    :func:`discover_recurrence_least_squares`). Because ``a_n`` is the regression
    target, the fit implicitly assumes a **monic** leading term -- the modelling
    limitation the exact :func:`discover_recurrence` removes.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    if index_degree < 0:
        raise ValueError(f"index_degree must be >= 0, got {index_degree}")
    vals = [float(v) for v in _as_fractions(samples)]
    n_samples = len(vals)
    if n_samples <= order:
        raise ValueError(f"need more than order={order} samples, got {n_samples}")
    rows: list[list[float]] = []
    target: list[float] = []
    names: list[str] = []
    for lag in range(1, order + 1):
        for d in range(index_degree + 1):
            names.append(f"a[n-{lag}]" if d == 0 else f"n^{d}*a[n-{lag}]")
    for n in range(order, n_samples):
        row: list[float] = []
        for lag in range(1, order + 1):
            for d in range(index_degree + 1):
                row.append((n**d) * vals[n - lag])
        rows.append(row)
        target.append(vals[n])
    return np.asarray(rows, dtype=float), names, np.asarray(target, dtype=float)


def discover_recurrence_least_squares(
    samples: Sequence[Number],
    *,
    order: int,
    index_degree: int = 0,
    alpha: float = 1e-10,
    threshold: float = 1e-6,
) -> SparseEquation:
    r"""Baseline recurrence fit via STLSQ (:func:`fit_sparse_equation`).

    Fits ``a_n = sum_{j,d} c_{j,d} n^d a_{n-j}`` by sequential thresholded ridge
    regression over the :func:`build_difference_relation_library` design. Recovers
    C-finite recurrences (Fibonacci) and P-recursive ones whose leading coefficient is
    constant (factorial, with ``index_degree=1``), but -- being monic in ``a_n`` and
    floating point -- it cannot represent a non-constant leading coefficient (Catalan)
    and is ill-conditioned for large-magnitude columns. Contrast with the exact
    :func:`discover_recurrence`.
    """
    design, names, target = build_difference_relation_library(
        samples, order=order, index_degree=index_degree
    )
    return fit_sparse_equation(design, target, names, alpha=alpha, threshold=threshold)


# --------------------------------------------------------------------------- #
# Closed-form polynomial extraction (Faulhaber and friends)                   #
# --------------------------------------------------------------------------- #
def polynomial_from_samples(samples: Sequence[Number]) -> tuple[Fraction, ...]:
    r"""Exact monomial coefficients of the polynomial interpolating integer samples.

    Assumes ``samples = [f(0), f(1), ...]`` for a polynomial ``f`` of degree
    ``< len(samples)``. Uses Newton's forward differences (exact rationals) to read the
    falling-factorial coefficients ``a_k = Delta^k f(0) / k!`` and converts them to the
    monomial basis via :func:`omnibias.difference.falling_to_monomial`. Trailing zeros
    (degrees above the true degree) are stripped.

    This is the **closed-form**, perfectly conditioned way to recover a Faulhaber power
    sum ``sum_{i=1}^{n} i^p`` (a degree-``p+1`` polynomial), where a float Vandermonde
    least-squares fit is numerically ill-conditioned for large ``n`` / ``p``.
    """
    vals = _as_fractions(samples)
    if not vals:
        raise ValueError("need at least one sample")
    newton = newton_forward_coeffs(vals)
    falling = tuple(newton[k] / factorial(k) for k in range(len(newton)))
    monomial = list(falling_to_monomial(falling))
    while len(monomial) > 1 and monomial[-1] == 0:
        monomial.pop()
    return tuple(monomial)


# --------------------------------------------------------------------------- #
# Full-history (non-P-recursive) laws                                         #
# --------------------------------------------------------------------------- #
def verify_binomial_recurrence(samples: Sequence[Number]) -> bool:
    r"""Exactly check the Bell recurrence ``B_{n+1} = sum_k C(n, k) B_k``.

    Bell numbers are **not** finitely P-recursive (so :func:`discover_recurrence`
    returns ``None`` for them); their genuine law is this full-history binomial
    convolution. Given ``samples = [B_0, B_1, ...]`` this returns ``True`` iff the
    binomial transform of ``[B_0 .. B_{m-1}]`` reproduces ``[B_1 .. B_m]`` exactly, via
    the exact :func:`omnibias.difference.binomial_transform`.
    """
    vals = _as_fractions(samples)
    if len(vals) < 2:
        raise ValueError("need at least two samples to check the binomial recurrence")
    transform = binomial_transform(vals[:-1])
    return all(transform[n] == vals[n + 1] for n in range(len(transform)))
