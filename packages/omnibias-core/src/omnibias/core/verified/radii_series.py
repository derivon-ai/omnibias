# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""One-sided radii-polynomial closure for a scalar quadratic fixed-point problem.

Existence gap this closes
--------------------------
:mod:`omnibias.core.verified.radii_spectral` glues the *two-sided*
:class:`~omnibias.core.verified.fourier.ValidatedFourierSeries` Banach algebra, a
symmetric bilinear form, and
:func:`~omnibias.core.verified.kantorovich.radii_polynomial_certificate` into one
Y0/Z0/Z1/Z2 pipeline for ``F(a) = ell*a + Q(a,a) - f = 0``. No analogous pipeline
existed for the *one-sided*
:class:`~omnibias.core.verified.sequence_space.ValidatedSeries` (audited by
grepping every ``radii_polynomial_certificate`` call site and reading ``ode.py``,
``series.py``, ``taylor_model.py``, and ``pde_certificate.py`` -- the closest
relative, ``pde_certificate.radii_polynomial_residual_certificate``, is a thin
wrapper that only *consumes* caller-precomputed ``(Y0,Z0,Z1,Z2)`` floats; it does
not assemble them from a ``ValidatedSeries`` problem, so it is not a duplicate of
what this module builds). This module is that minimal counterpart, scoped to the
same shape ``radii_spectral`` uses:

.. math::

    F(a) = \ell\,a + a * a - f = 0,

posed in the weighted one-sided Banach algebra
:class:`~omnibias.core.verified.sequence_space.ValidatedSeries` (``ell^1_\nu``
over ``n >= 0``). Here ``\ell`` is a **diagonal** symbol ``ell(n)`` (a
per-coefficient scalar multiplier -- the one-sided analogue of ``radii_spectral``'s
Fourier multiplier; a genuinely diagonal example in this monomial basis is the
Euler/scaling generator ``c0 + c1\,(x\,d/dx)``, whose eigenvalue on ``x^n`` is
``c0 + c1 n``), and the nonlinearity ``Q(a,a) := a * a`` is *exactly*
:meth:`~omnibias.core.verified.sequence_space.ValidatedSeries.__mul__` -- the
Banach-algebra convolution, not a caller-pluggable bilinear form -- so its bound
``C_Q = 1`` is exact (``||a*b||_nu <= ||a||_nu ||b||_nu``, proved in
:mod:`omnibias.core.verified.sequence_space`), not an estimate.

The Newton-Kantorovich operator and the split inverse
--------------------------------------------------------
From an approximate zero ``a_bar`` (a length ``N+1`` polynomial, coefficients
``0..N``) build an approximate inverse ``A`` of ``DF(a_bar)`` as a *split*
operator with truncation ``N``:

* on the finite block ``0 <= n <= N``: a numerical inverse ``A_N`` of the finite
  Jacobian block ``B_N`` (computed in floating point; rigor comes from *enclosing*
  every finite-input column exactly, never from trusting ``A_N`` itself);
* on the tail ``n > N``: the exact diagonal inverse ``1/ell(n)``, bounded by
  ``mu = sup_{n>N} |ell(n)|^{-1}`` (the caller's coercivity hypothesis, exactly
  :func:`omnibias.core.verified.banded.banded_tail_inverse_bound`'s diagonal
  ``s=0`` special case -- see :func:`constant_tail_inverse_bound`, which reuses it
  rather than re-deriving the same ``1/d_min`` formula).

Writing ``DF(a_bar)h = ell*h + 2*(a_bar * h)`` (``Q`` symmetric because
convolution commutes) and ``T(a) = a - A F(a)``, the radii-polynomial bounds are,
with ``||A|| = max(||A_N||_op, mu)`` and ``C_Q = 1``:

.. math::

    Y_0 &= \|A F(\bar a)\|_\nu, \\
    Z_0 &= \max_{0 \le n \le N}
            \frac{\| e_n - A\,DF(\bar a)\,e_n \|_\nu}{\nu^{n}}
            \quad(\text{finite-input columns, computed exactly}),\\
    Z_1 &= 2\,\|A\|\,\|\bar a\|_\nu
            \quad(\text{tail-input columns}),\\
    Z_2 &= \|A\|
            \quad(\tfrac12\,\text{Lipschitz constant of } DF).

Because ``a_bar`` has only ``N+1`` nonzero coefficients, embedding it (and every
unit column ``e_n``) as a *padded* ``ValidatedSeries`` of length ``2N+1`` (extra
coefficients set to the exact point ``0.0``, zero tail) keeps every product
computed by :meth:`ValidatedSeries.__mul__` **exact**: the true convolution of
two degree-``<=N`` polynomials has degree ``<=2N``, so it fits entirely inside the
kept block at truncation ``L=2N`` and every "overflow" term
:meth:`ValidatedSeries.__mul__` folds into its tail is a literal sum of exact
zeros. So ``F(a_bar)`` and every finite-input column ``DF(a_bar)e_n`` are exact
(no analytic approximation); only the tail bounds ``Z_1, Z_2`` (controlled by
``mu``) are analytic estimates. Feeding ``(Y_0, Z_0, Z_1, Z_2)`` to
:func:`~omnibias.core.verified.kantorovich.radii_polynomial_certificate` yields,
when a contracting radius ``r`` exists, a **true** zero ``a*`` of ``F`` with
``||a* - a_bar||_nu <= r`` (unique in that ball).

What this does and does not cover
----------------------------------
The linear part must be **diagonal** in the one-sided monomial basis (a
per-coefficient scalar ``ell(n)``); the nonlinearity is fixed to the algebra's own
square ``a*a``, not a general bilinear form. Both restrictions keep this a small,
honestly-scoped instance of the same machinery ``radii_spectral`` provides for the
two-sided Fourier case -- a **banded** (non-diagonal) linear part in either basis
is out of scope here exactly as it is there (see
:mod:`omnibias.core.verified.banded` for the tail-inverse-norm primitive a
banded-capable successor of *either* pipeline would need).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.core.verified.banded import banded_tail_inverse_bound
from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.kantorovich import RadiiCertificate, radii_polynomial_certificate
from omnibias.core.verified.sequence_space import ValidatedSeries

#: A diagonal (per-coefficient scalar) linear symbol ``n -> ell(n)``.
Symbol = Callable[[int], float]


@dataclass(frozen=True)
class SeriesProblem:
    r"""A quadratic one-sided problem ``F(a) = ell*a + a*a - f`` in ``ell^1_nu``."""

    trunc: int
    nu: float
    linear_symbol: Symbol
    tail_inverse_bound: float
    forcing: Sequence[IntervalLike] | None = None

    def __post_init__(self) -> None:
        if self.trunc < 1:
            raise ValueError("trunc N must be >= 1")
        if self.nu <= 0.0:
            raise ValueError("nu must be > 0 for the ell^1_nu algebra")
        if self.tail_inverse_bound < 0.0:
            raise ValueError("tail_inverse_bound mu must be non-negative")
        if self.forcing is not None and len(self.forcing) > self.work_trunc + 1:
            raise ValueError(
                f"forcing has {len(self.forcing)} coefficients, exceeding "
                f"work_trunc+1={self.work_trunc + 1}"
            )

    @property
    def work_trunc(self) -> int:
        """Internal truncation ``L = 2N`` at which residual / columns are exact."""
        return 2 * self.trunc


@dataclass(frozen=True)
class SeriesRadiiResult:
    """Bounds and (optional) certificate from the one-sided radii-polynomial closure."""

    y0: float
    z0: float
    z1: float
    z2: float
    a_op_norm: float
    residual_norm: float
    certificate: RadiiCertificate | None

    @property
    def proved(self) -> bool:
        return self.certificate is not None

    @property
    def radius(self) -> float | None:
        return None if self.certificate is None else self.certificate.radius


# --------------------------------------------------------------------------- #
# Diagonal linear helpers.
# --------------------------------------------------------------------------- #
def constant_symbol(value: float) -> Symbol:
    r"""The diagonal symbol ``ell(n) = value`` for every coefficient ``n`` (``value*I``)."""

    def m(n: int) -> float:
        return value

    return m


def constant_tail_inverse_bound(value: float) -> float:
    r"""Sound ``mu = 1/|value|`` for the constant symbol ``ell(n) = value != 0``.

    Reuses :func:`omnibias.core.verified.banded.banded_tail_inverse_bound` with a
    zero off-diagonal row sum: this problem's linear part is purely diagonal, so
    ``s = 0`` and the general banded-operator bound specialises *exactly* to the
    classical ``1/d_min`` formula
    (:func:`omnibias.core.verified.radii_spectral.laplacian_tail_inverse_bound`
    is the same specialisation on the Fourier side) rather than re-deriving it.
    """
    return banded_tail_inverse_bound(abs(value), 0.0).hi


# --------------------------------------------------------------------------- #
# Internal linear algebra (float real inverse + helpers).
# --------------------------------------------------------------------------- #
def _invert_real(mat: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse of a square real matrix (partial pivoting)."""
    n = len(mat)
    aug = [
        [mat[i][col] for col in range(n)] + [1.0 if c == i else 0.0 for c in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) == 0.0:
            raise ValueError("singular finite Jacobian block; cannot form approximate inverse")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [x / piv for x in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col], strict=True)]
    return [row[n:] for row in aug]


def _weight(nu: float, k: int) -> Interval:
    return Interval.point(nu).pow_int(k)


def _embed(coeffs: Sequence[IntervalLike], work_trunc: int, nu: float) -> ValidatedSeries:
    """Zero-pad ``coeffs`` to length ``work_trunc + 1`` with an exact zero tail."""
    values = list(coeffs)
    if len(values) > work_trunc + 1:
        raise ValueError(
            f"input has {len(values)} coefficients, exceeding work_trunc+1={work_trunc + 1}"
        )
    padded: list[IntervalLike] = [*values, *([0.0] * (work_trunc + 1 - len(values)))]
    return ValidatedSeries.from_coeffs(padded, nu, tail=0.0)


def _unit(n: int, work_trunc: int, nu: float) -> ValidatedSeries:
    """The basis series ``e_n`` embedded (exactly) at truncation ``work_trunc``."""
    return _embed([*([0.0] * n), 1.0], work_trunc, nu)


def _apply_diagonal_symbol(series: ValidatedSeries, symbol: Symbol) -> ValidatedSeries:
    """``ell * series`` for a finite (zero-tail) series and a diagonal symbol."""
    coeffs = [Interval.point(symbol(k)) * v for k, v in enumerate(series.coeffs)]
    return ValidatedSeries(coeffs, Interval.point(0.0), series.nu, series.chebyshev)


class _SplitInverse:
    """The approximate inverse ``A = A_N (finite) + 1/ell (tail)``."""

    def __init__(
        self,
        a_n: list[list[Interval]],
        symbol: Symbol,
        mu: float,
        work_trunc: int,
        nu: float,
        trunc: int,
    ) -> None:
        self.a_n = a_n
        self.symbol = symbol
        self.mu = mu
        self.work_trunc = work_trunc
        self.nu = nu
        self.trunc = trunc

    def apply(self, w: ValidatedSeries) -> ValidatedSeries:
        n = self.trunc + 1
        v = [w.coeffs[j] if j < len(w.coeffs) else Interval.point(0.0) for j in range(n)]
        out: list[Interval] = []
        for i in range(n):
            acc = Interval.point(0.0)
            for j in range(n):
                acc = acc + self.a_n[i][j] * v[j]
            out.append(acc)
        for k in range(n, self.work_trunc + 1):
            ck = w.coeffs[k] if k < len(w.coeffs) else Interval.point(0.0)
            out.append((Interval.point(1.0) / Interval.point(self.symbol(k))) * ck)
        tail = w.tail * Interval.point(self.mu)
        tail = Interval(max(tail.lo, 0.0), max(tail.hi, 0.0))
        return ValidatedSeries(out, tail, self.nu, False)

    def op_norm(self) -> float:
        """Weighted-ell1_nu operator norm ``max(||A_N||, mu)``."""
        n = self.trunc + 1
        weights = [_weight(self.nu, i) for i in range(n)]
        best = self.mu
        for j in range(n):
            col = Interval.point(0.0)
            for i in range(n):
                col = col + Interval.point(self.a_n[i][j].mag) * weights[i]
            best = max(best, (col / weights[j]).hi)
        return best


def _validate_a_bar(problem: SeriesProblem, a_bar: Sequence[IntervalLike]) -> None:
    """``a_bar`` must be supported on ``0..N`` for the ``L=2N`` exactness argument to hold.

    A longer ``a_bar`` would still be a sound ``ValidatedSeries`` element, but the
    finite Jacobian block ``B_N`` is only ever built from the ``N+1`` columns
    ``e_0..e_N``; silently accepting extra coefficients would drop them from the
    linearisation rather than degrade honestly, so this raises instead.
    """
    if len(a_bar) > problem.trunc + 1:
        raise ValueError(
            f"a_bar has {len(a_bar)} coefficients but trunc N={problem.trunc} allows "
            f"at most N+1={problem.trunc + 1} (a_bar must be supported on 0..N)"
        )


def evaluate_residual(problem: SeriesProblem, a_bar: Sequence[IntervalLike]) -> ValidatedSeries:
    r"""The residual ``F(a_bar) = ell*a_bar + a_bar*a_bar - f`` (exact at ``L = 2N``)."""
    _validate_a_bar(problem, a_bar)
    ab = _embed(a_bar, problem.work_trunc, problem.nu)
    residual = _apply_diagonal_symbol(ab, problem.linear_symbol) + (ab * ab)
    if problem.forcing is not None:
        residual = residual + _embed(problem.forcing, problem.work_trunc, problem.nu).scale(-1.0)
    return residual


@dataclass(frozen=True)
class _Assembled:
    """The linearised system ``DF(ab)`` and its split approximate inverse ``A``."""

    problem: SeriesProblem
    ab: ValidatedSeries
    df_cols: list[ValidatedSeries]
    inverse: _SplitInverse

    def df_apply(self, h: ValidatedSeries) -> ValidatedSeries:
        """``DF(ab) h`` for a finite ``h`` (coefficients ``0..N``, exact at ``L=2N``)."""
        out = ValidatedSeries.from_coeffs(
            [0.0] * (self.problem.work_trunc + 1), self.problem.nu
        )
        for k, col in enumerate(self.df_cols):
            hk = h.coeffs[k] if k < len(h.coeffs) else Interval.point(0.0)
            out = out + col.scale(hk)
        return out

    def defect_apply(self, h: ValidatedSeries) -> ValidatedSeries:
        """``(I - A DF(ab)) h`` -- the contraction defect (for soundness checks)."""
        return h + self.inverse.apply(self.df_apply(h)).scale(-1.0)


def _assemble(problem: SeriesProblem, a_bar: Sequence[IntervalLike]) -> _Assembled:
    _validate_a_bar(problem, a_bar)
    nu, work = problem.nu, problem.work_trunc
    symbol, mu = problem.linear_symbol, problem.tail_inverse_bound
    n = problem.trunc + 1

    ab = _embed(a_bar, work, nu)

    # Finite-input columns of DF(ab): DF(ab) e_n = ell(n) e_n + 2 (ab * e_n).
    df_cols: list[ValidatedSeries] = []
    for k in range(n):
        e_k = _unit(k, work, nu)
        df_cols.append(_apply_diagonal_symbol(e_k, symbol) + (ab * e_k).scale(2.0))

    # Finite Jacobian block B_N (rows/cols 0..N) -> float midpoint -> inverse.
    b_mid: list[list[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    for j, col in enumerate(df_cols):
        for i in range(n):
            b_mid[i][j] = col.coeffs[i].mid
    a_n_f = _invert_real(b_mid)
    a_n = [[Interval.point(a_n_f[i][j]) for j in range(n)] for i in range(n)]
    inverse = _SplitInverse(a_n, symbol, mu, work, nu, problem.trunc)
    return _Assembled(problem, ab, df_cols, inverse)


def series_radii_certificate(
    problem: SeriesProblem,
    a_bar: Sequence[IntervalLike],
    *,
    r_max: float = float("inf"),
) -> SeriesRadiiResult:
    r"""Attempt a radii-polynomial existence proof for ``F(a) = 0`` near ``a_bar``.

    Returns a :class:`SeriesRadiiResult` carrying the rigorous bounds
    ``(Y0, Z0, Z1, Z2)`` and, when a contracting radius exists, a sealed
    :class:`~omnibias.core.verified.kantorovich.RadiiCertificate` proving a *true*
    zero ``a*`` of ``F`` with ``||a* - a_bar||_nu <= r`` (unique there). ``a_bar``
    is the finite approximation (coefficients ``0..N``).
    """
    system = _assemble(problem, a_bar)
    inverse, df_cols, ab = system.inverse, system.df_cols, system.ab
    nu = problem.nu
    n = problem.trunc + 1
    c_q = 1.0  # exact Banach-algebra bound for Q(a,a) = a*a.

    # Z0 = max over finite input columns of || e_n - A DF(ab) e_n ||_nu / nu^n.
    z0 = Interval.point(0.0)
    for k in range(n):
        e_k = _unit(k, problem.work_trunc, nu)
        col = e_k + inverse.apply(df_cols[k]).scale(-1.0)
        colsum = col.norm() / _weight(nu, k)
        z0 = Interval(max(z0.lo, colsum.lo), max(z0.hi, colsum.hi))

    a_op = inverse.op_norm()
    ab_norm = ab.norm().hi
    z1 = (Interval.point(2.0) * Interval.point(a_op) * Interval.point(ab_norm)).hi
    z2 = Interval.point(a_op).hi * c_q

    residual = evaluate_residual(problem, a_bar)
    y0 = inverse.apply(residual).norm().hi
    residual_norm = residual.norm().hi

    cert = radii_polynomial_certificate(
        y0,
        z0.hi,
        z1,
        z2,
        r_max=r_max,
        claim="unique zero a* of F(a)=ell*a + a*a - f in B(a_bar, r) (one-sided ell^1_nu)",
    )
    return SeriesRadiiResult(y0, z0.hi, z1, z2, a_op, residual_norm, cert)


__all__ = [
    "SeriesProblem",
    "SeriesRadiiResult",
    "Symbol",
    "constant_symbol",
    "constant_tail_inverse_bound",
    "evaluate_residual",
    "series_radii_certificate",
]
