# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Infinite-dimensional radii-polynomial closure for a *quadratic* spectral problem.

This is the capstone of the spectral route: a **computer-assisted existence
proof** for a periodic solution of

.. math::

    F(a) = \ell\,a + Q(a, a) - f = 0,

posed in the weighted Banach algebra
:class:`~omnibias.core.verified.fourier.ValidatedFourierSeries` (``\ell^1_\nu`` over
``Z^d``).  Here ``\ell`` is a **diagonal** Fourier multiplier ``\ell(k)`` (the
dominant linear part), ``Q`` a bounded **symmetric bilinear** form (a convolution,
optionally dressed with Riesz / derivative multipliers -- the SQG advection
``Q(a,a) = R^\perp a \cdot \nabla a`` fits), and ``f`` a forcing.  Steady states of
2-D Euler / SQG-type models and semilinear elliptic problems all have this shape.

The Newton-Kantorovich operator and the split inverse
--------------------------------------------------------
From an approximate zero ``\bar a`` (a finite trigonometric polynomial supported on
``\|k\|_\infty \le N``) we build an approximate inverse ``A`` of ``DF(\bar a)`` as a
*split* operator with truncation ``N``:

* on the finite block ``\|k\|_\infty \le N``: a numerical inverse ``A_N`` of the
  finite Jacobian block ``B_N = P_N DF(\bar a) P_N`` (computed in floating point;
  the rigor comes from *enclosing* ``I - A_N B_N``, never from trusting ``A_N``);
* on the tail ``\|k\|_\infty > N``: the exact diagonal inverse ``1/\ell(k)``,
  bounded by ``\mu = \sup_{\|k\|_\infty > N} |\ell(k)|^{-1}`` (the caller's coercivity
  hypothesis).

Writing ``DF(\bar a)h = \ell h + 2 Q(\bar a, h)`` (symmetric ``Q``) and
``T(a) = a - A F(a)`` (so a fixed point of ``T`` is a zero of ``F``), the standard
radii-polynomial bounds are, with ``\|A\| = \max(\|A_N\|_{op}, \mu)`` and ``C_Q`` the
bilinear bound ``\|Q(u,v)\| \le C_Q \|u\|\,\|v\|``:

.. math::

    Y_0 &= \|A F(\bar a)\|_\nu, \\
    Z_0 &= \max_{\|k\|_\infty \le N}
            \frac{\| e_k - A\,DF(\bar a)\,e_k \|_\nu}{\nu^{\|k\|_1}}
            \quad(\text{finite-input columns, computed exactly}),\\
    Z_1 &= 2\,\|A\|\,C_Q\,\|\bar a\|_\nu
            \quad(\text{tail-input columns: } \|A\,2Q(\bar a, e_k)\| \le \|A\|\,2C_Q\|\bar a\|\,\nu^{\|k\|_1}),\\
    Z_2 &= \|A\|\,C_Q
            \quad(\tfrac12\,\text{Lipschitz of } DF:\ \|DF(a)-DF(b)\| \le 2C_Q\|a-b\|).

Because ``\bar a`` is finite, ``F(\bar a)`` and every finite-input column
``DF(\bar a)e_k = \ell(k)e_k + 2\,\bar a * e_k`` are supported on
``\|k\|_\infty \le 2N``; working at the internal truncation ``L = 2N`` keeps them
*exact* (zero tail), so ``Y_0`` and ``Z_0`` are computed with no analytic
approximation -- the only analytic estimates are the tail bounds ``Z_1, Z_2``
(controlled by ``\mu`` and ``C_Q``).  Feeding ``(Y_0, Z_0, Z_1, Z_2)`` to
:func:`~omnibias.core.verified.kantorovich.radii_polynomial_certificate` yields, when
a contracting radius ``r`` exists, a **true** zero ``a^\*`` of ``F`` with
``\|a^\* - \bar a\|_\nu \le r`` (unique in that ball).

What this does and does not cover
---------------------------------
The linear part must be **diagonal** (a Fourier multiplier).  This covers steady /
forced problems and constant-coefficient linear terms.  The *self-similar* scaling
operator ``\alpha + \beta\,x\cdot\nabla`` of a finite-time-singularity ansatz is
**not** diagonal (``x\cdot\nabla`` couples neighbouring modes); folding it in is the
remaining ingredient for the SQG self-similar profile and is left as documented
future work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.fourier import Symbol, ValidatedFourierSeries, Wavevector
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import RadiiCertificate, radii_polynomial_certificate

#: A symmetric bounded bilinear form on the Fourier algebra.
Bilinear = Callable[[ValidatedFourierSeries, ValidatedFourierSeries], ValidatedFourierSeries]


@dataclass(frozen=True)
class SpectralProblem:
    r"""A quadratic spectral problem ``F(a) = ell*a + Q(a,a) - f`` in ``l1_nu``."""

    dim: int
    trunc: int
    nu: float
    linear_symbol: Symbol
    tail_inverse_bound: float
    quadratic: Bilinear
    quadratic_norm: float
    forcing: Mapping[Wavevector, ComplexLike] | None = None

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.trunc < 1:
            raise ValueError("trunc N must be >= 1")
        if self.nu < 1.0:
            raise ValueError("nu must be >= 1 for the l1_nu algebra")
        if self.tail_inverse_bound < 0.0:
            raise ValueError("tail_inverse_bound mu must be non-negative")
        if self.quadratic_norm < 0.0:
            raise ValueError("quadratic_norm C_Q must be non-negative")

    @property
    def work_trunc(self) -> int:
        """Internal truncation ``L = 2N`` at which residual / columns are exact."""
        return 2 * self.trunc


@dataclass(frozen=True)
class SpectralRadiiResult:
    """Bounds and (optional) certificate from the spectral radii-polynomial closure."""

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
# Diagonal linear helpers (coercive Laplacian-type symbols).
# --------------------------------------------------------------------------- #
def laplacian_symbol(c0: float, c2: float) -> Symbol:
    r"""The real diagonal symbol ``ell(k) = c0 + c2 |k|^2`` (a screened Laplacian)."""

    def m(k: Wavevector) -> ComplexInterval:
        ksq = sum(ki * ki for ki in k)
        return ComplexInterval(
            Interval.from_value(c0) + Interval.from_value(c2) * Interval.from_rational(ksq),
            Interval.point(0.0),
        )

    return m


def laplacian_tail_inverse_bound(dim: int, trunc: int, c0: float, c2: float) -> float:
    r"""Sound ``mu = sup_{||k||inf > N} 1/(c0 + c2 |k|^2)`` for the screened Laplacian.

    On the tail ``max_i |k_i| >= N + 1`` so ``|k|^2 >= (N + 1)^2``; with ``c0, c2 >= 0``
    and ``c0 + c2 (N+1)^2 > 0`` the supremum is ``1/(c0 + c2 (N+1)^2)`` (outward rounded).
    """
    if c0 < 0.0 or c2 < 0.0:
        raise ValueError("c0 and c2 must be non-negative for a coercive symbol")
    denom = Interval.from_value(c0) + Interval.from_value(c2) * Interval.from_rational(
        (trunc + 1) ** 2
    )
    if denom.lo <= 0.0:
        raise ValueError("symbol is not coercive on the tail (c0 + c2 (N+1)^2 must be > 0)")
    return (Interval.point(1.0) / denom).hi


# --------------------------------------------------------------------------- #
# Internal linear algebra (float complex inverse + helpers).
# --------------------------------------------------------------------------- #
def _modes(dim: int, n: int) -> list[Wavevector]:
    """All ``k`` with ``||k||inf <= n`` (lexicographic)."""
    ranges = [range(-n, n + 1)] * dim
    out: list[Wavevector] = [()]
    for r in ranges:
        out = [prefix + (j,) for prefix in out for j in r]
    return out


def _invert_complex(mat: list[list[complex]]) -> list[list[complex]]:
    """Gauss-Jordan inverse of a square complex matrix (partial pivoting)."""
    n = len(mat)
    aug = [
        [mat[i][col] for col in range(n)] + [1.0 + 0j if c == i else 0j for c in range(n)]
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
                if factor != 0j:
                    aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col], strict=True)]
    return [row[n:] for row in aug]


def _weight(nu: float, k: Wavevector) -> Interval:
    return Interval.point(nu).pow_int(sum(abs(ki) for ki in k))


def _apply_finite_symbol(series: ValidatedFourierSeries, symbol: Symbol) -> ValidatedFourierSeries:
    """``ell * series`` for a finite (zero-tail) series and a diagonal symbol."""
    coeffs = {k: symbol(k) * v for k, v in series.coeffs.items()}
    return ValidatedFourierSeries(series.dim, series.trunc, series.nu, coeffs, Interval.point(0.0))


class _SplitInverse:
    """The approximate inverse ``A = A_N (finite) + 1/ell (tail)``."""

    def __init__(
        self,
        a_n: list[list[ComplexInterval]],
        modes: list[Wavevector],
        symbol: Symbol,
        mu: float,
        dim: int,
        work_trunc: int,
        nu: float,
        trunc: int,
    ) -> None:
        self.a_n = a_n
        self.modes = modes
        self.symbol = symbol
        self.mu = mu
        self.dim = dim
        self.work_trunc = work_trunc
        self.nu = nu
        self.trunc = trunc

    def apply(self, w: ValidatedFourierSeries) -> ValidatedFourierSeries:
        n = len(self.modes)
        v = [w.get(self.modes[j]) for j in range(n)]
        out: dict[Wavevector, ComplexInterval] = {}
        for i in range(n):
            acc = ComplexInterval.zero()
            for j in range(n):
                acc = acc + self.a_n[i][j] * v[j]
            out[self.modes[i]] = acc
        one = ComplexInterval.one()
        for m, val in w.coeffs.items():
            if any(abs(c) > self.trunc for c in m):  # tail: ||m||inf > N -> 1/ell(m)
                out[m] = (one / self.symbol(m)) * val
        tail = w.tail * Interval.point(self.mu)
        tail = Interval(max(tail.lo, 0.0), max(tail.hi, 0.0))
        return ValidatedFourierSeries(self.dim, self.work_trunc, self.nu, out, tail)

    def op_norm(self) -> float:
        """Weighted-l1 operator norm ``max(||A_N||, mu)``."""
        n = len(self.modes)
        best = self.mu
        for j in range(n):
            col = Interval.point(0.0)
            wj = _weight(self.nu, self.modes[j])
            for i in range(n):
                col = col + Interval.point(self.a_n[i][j].mag) * _weight(self.nu, self.modes[i])
            best = max(best, (col / wj).hi)
        return best


def _embed(
    coeffs: Mapping[Wavevector, ComplexLike], dim: int, work_trunc: int, nu: float
) -> ValidatedFourierSeries:
    return ValidatedFourierSeries.from_coeffs(coeffs, dim, work_trunc, nu)


def evaluate_residual(
    problem: SpectralProblem, a_bar: Mapping[Wavevector, ComplexLike]
) -> ValidatedFourierSeries:
    r"""The residual ``F(\bar a) = ell*\bar a + Q(\bar a, \bar a) - f`` (exact at ``L = 2N``)."""
    ab = _embed(a_bar, problem.dim, problem.work_trunc, problem.nu)
    residual = _apply_finite_symbol(ab, problem.linear_symbol) + problem.quadratic(ab, ab)
    if problem.forcing is not None:
        residual = residual - _embed(
            problem.forcing, problem.dim, problem.work_trunc, problem.nu
        )
    return residual


@dataclass(frozen=True)
class _Assembled:
    """The linearised system ``DF(ab)`` and its split approximate inverse ``A``."""

    problem: SpectralProblem
    ab: ValidatedFourierSeries
    modes: list[Wavevector]
    df_cols: list[ValidatedFourierSeries]
    inverse: _SplitInverse

    def df_apply(self, h: ValidatedFourierSeries) -> ValidatedFourierSeries:
        """``DF(ab) h`` for a finite ``h`` (supported on ``||k||inf <= N``)."""
        out = ValidatedFourierSeries.zero(self.problem.dim, self.problem.work_trunc, self.problem.nu)
        for col, k in zip(self.df_cols, self.modes, strict=True):
            out = out + col.scale(h.get(k))
        return out

    def defect_apply(self, h: ValidatedFourierSeries) -> ValidatedFourierSeries:
        """``(I - A DF(ab)) h`` -- the contraction defect (for soundness checks)."""
        return h - self.inverse.apply(self.df_apply(h))


def _assemble(
    problem: SpectralProblem, a_bar: Mapping[Wavevector, ComplexLike]
) -> _Assembled:
    dim, nu, work = problem.dim, problem.nu, problem.work_trunc
    symbol, mu = problem.linear_symbol, problem.tail_inverse_bound

    ab = _embed(a_bar, dim, work, nu)
    modes = _modes(dim, problem.trunc)
    n = len(modes)

    # finite-input columns of DF(ab): DF(ab) e_k = ell(k) e_k + 2 Q(ab, e_k).
    df_cols: list[ValidatedFourierSeries] = []
    for k in modes:
        e_k = _embed({k: 1.0}, dim, work, nu)
        df_cols.append(_apply_finite_symbol(e_k, symbol) + problem.quadratic(ab, e_k).scale(2.0))

    # finite Jacobian block B_N (rows/cols on ||.||inf <= N) -> float midpoint -> inverse.
    b_mid: list[list[complex]] = [[0j for _ in range(n)] for _ in range(n)]
    for j, col in enumerate(df_cols):
        for i, m in enumerate(modes):
            cij = col.get(m)
            b_mid[i][j] = complex(cij.re.mid, cij.im.mid)
    a_n_c = _invert_complex(b_mid)
    a_n = [[ComplexInterval.point(a_n_c[i][j]) for j in range(n)] for i in range(n)]
    inverse = _SplitInverse(a_n, modes, symbol, mu, dim, work, nu, problem.trunc)
    return _Assembled(problem, ab, modes, df_cols, inverse)


def quadratic_radii_certificate(
    problem: SpectralProblem,
    a_bar: Mapping[Wavevector, ComplexLike],
    *,
    r_max: float = float("inf"),
) -> SpectralRadiiResult:
    r"""Attempt a radii-polynomial existence proof for ``F(a)=0`` near ``\bar a``.

    Returns a :class:`SpectralRadiiResult` carrying the rigorous bounds
    ``(Y0, Z0, Z1, Z2)`` and, when a contracting radius exists, a sealed
    :class:`~omnibias.core.verified.kantorovich.RadiiCertificate` proving a *true*
    zero ``a^*`` of ``F`` with ``\|a^* - \bar a\|_\nu \le r`` (unique there).
    ``a_bar`` is the finite approximation (coefficients on ``\|k\|_\infty \le N``).
    """
    system = _assemble(problem, a_bar)
    inverse, modes, df_cols, ab = system.inverse, system.modes, system.df_cols, system.ab
    nu, c_q = problem.nu, problem.quadratic_norm

    # Z0 = max over finite input columns of || e_k - A DF(ab) e_k ||_nu / nu^|k|.
    z0 = Interval.point(0.0)
    for jk, k in enumerate(modes):
        e_k = _embed({k: 1.0}, problem.dim, problem.work_trunc, nu)
        col = e_k - inverse.apply(df_cols[jk])
        colsum = col.norm() / _weight(nu, k)
        z0 = Interval(max(z0.lo, colsum.lo), max(z0.hi, colsum.hi))

    a_op = inverse.op_norm()
    ab_norm = ab.norm().hi
    z1 = (Interval.point(2.0) * Interval.point(a_op) * Interval.point(c_q) * Interval.point(ab_norm)).hi
    z2 = (Interval.point(a_op) * Interval.point(c_q)).hi

    residual = evaluate_residual(problem, a_bar)
    y0 = inverse.apply(residual).norm().hi
    residual_norm = residual.norm().hi

    cert = radii_polynomial_certificate(
        y0,
        z0.hi,
        z1,
        z2,
        r_max=r_max,
        claim="unique zero a* of F(a)=ell a + Q(a,a) - f in B(a_bar, r) (l1_nu)",
    )
    return SpectralRadiiResult(y0, z0.hi, z1, z2, a_op, residual_norm, cert)


__all__ = [
    "Bilinear",
    "SpectralProblem",
    "SpectralRadiiResult",
    "evaluate_residual",
    "laplacian_symbol",
    "laplacian_tail_inverse_bound",
    "quadratic_radii_certificate",
]
