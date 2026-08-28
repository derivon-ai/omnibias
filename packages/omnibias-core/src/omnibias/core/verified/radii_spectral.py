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
The linear part is either **diagonal** (a Fourier multiplier ``ell``) or
**banded**: a :class:`BandedLinearPart` couples each mode ``k`` to a *finite*
set of neighbours ``k + d`` (e.g. the ``d = \pm 1`` couplings a
multiplication-by-``cos(x)`` term produces, the toy stand-in this module uses
for the *self-similar* scaling operator ``\alpha + \beta\,x\cdot\nabla`` of a
finite-time-singularity ansatz -- ``x\cdot\nabla`` is genuinely non-diagonal
because multiplication by ``x`` couples neighbouring Fourier modes).  The tail
piece of the split inverse (see below) needs a scalar bound ``mu``; for a
diagonal symbol that is the exact ``1/ell(k)`` supremum
(:func:`laplacian_tail_inverse_bound`), and for a banded linear part it is
:func:`tail_inverse_bound_from_banded`, built from
:func:`omnibias.core.verified.banded.banded_tail_inverse_bound` via the
nu-weighted row/column-sum bound :func:`banded_row_sum_bound_nu` (see that
function's docstring for exactly which operator norm this certifies). Either
way ``mu`` plugs into the *same* ``SpectralProblem.tail_inverse_bound`` field
and the *same* ``Z1``/``Z2`` formulas below -- the banded case only changes how
the finite-input columns ``DF(\bar a)e_k`` are built and how the split inverse
treats modes outside the numerically-inverted block ``\|k\|_\infty \le N``
(exactly per-mode for a diagonal symbol; folded into one aggregate ``mu``-scaled
magnitude for a banded one, since a non-diagonal tail has no elementary
per-mode inverse -- see :class:`BandedLinearPart` and ``_SplitInverse.apply``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from omnibias.core.verified.banded import banded_tail_inverse_bound
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.fourier import Symbol, ValidatedFourierSeries, Wavevector
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.kantorovich import RadiiCertificate, radii_polynomial_certificate

#: A symmetric bounded bilinear form on the Fourier algebra.
Bilinear = Callable[[ValidatedFourierSeries, ValidatedFourierSeries], ValidatedFourierSeries]


@dataclass(frozen=True)
class BandedLinearPart:
    r"""A finite-bandwidth linear part ``ell(k) e_k + sum_{d} c_d(k) e_{k+d}``.

    Generalises :attr:`SpectralProblem.linear_symbol` from a diagonal-only
    Fourier multiplier to a **banded** (non-diagonal) operator that couples
    each mode ``k`` to a finite set of neighbours ``k + d``.  This is a strict
    *alternative* value for ``linear_symbol`` -- every existing diagonal call
    site (a plain :data:`Symbol` callable) keeps working unchanged; only a
    caller that explicitly constructs a ``BandedLinearPart`` opts into the
    banded code path.

    ``diagonal`` is the diagonal part ``ell``, with the identical contract as
    the plain :data:`Symbol` this field replaces.  ``couplings`` maps each
    nonzero offset ``d`` (a :data:`Wavevector`) to a symbol ``c_d``, so mode
    ``k`` receives the extra contribution ``c_d(k) * a[k + d]`` for every
    ``d`` -- e.g. ``couplings = {(1,): c, (-1,): c}`` is the pair of
    nearest-neighbour couplings multiplication by ``cos(x)`` produces in a 1-D
    Fourier basis.  See :func:`constant_coefficient_band` for the common
    translation-invariant (``k``-independent ``c_d``) case.
    """

    diagonal: Symbol
    couplings: Mapping[Wavevector, Symbol]

    def __post_init__(self) -> None:
        for d in self.couplings:
            if all(di == 0 for di in d):
                raise ValueError("couplings must not include the zero offset; use `diagonal`")

    @property
    def bandwidth(self) -> int:
        """The sup-norm radius of the coupling offsets (``0`` if purely diagonal)."""
        if not self.couplings:
            return 0
        return max(max(abs(di) for di in d) for d in self.couplings)


def constant_coefficient_band(
    diagonal: Symbol, coupling_constants: Mapping[Wavevector, ComplexLike]
) -> BandedLinearPart:
    r"""Convenience :class:`BandedLinearPart` for *translation-invariant* couplings.

    ``coupling_constants`` maps each nonzero offset ``d`` to a **constant**
    (``k``-independent) complex coupling coefficient ``c_d`` -- the common case
    where the off-diagonal part comes from convolution with a fixed finite
    kernel (e.g. multiplication by a fixed trigonometric polynomial such as
    ``cos(x)``, which couples ``d = \pm 1`` with the constant coefficient
    ``1/2``).
    """
    # Each closure binds its own `_c` as a default argument so every offset
    # keeps its own constant rather than all closing over the loop's last value.
    couplings: dict[Wavevector, Symbol] = {
        tuple(d): (lambda k, _c=ComplexInterval.from_value(value): _c)
        for d, value in coupling_constants.items()
    }
    return BandedLinearPart(diagonal, couplings)


def banded_row_sum_bound_nu(coupling_bounds: Mapping[Wavevector, float], nu: float) -> float:
    r"""The ``ell^1_nu``-consistent off-diagonal row/column-sum bound for a banded tail.

    ``coupling_bounds`` maps each nonzero offset ``d`` to a rigorous upper
    bound on ``|c_d(k)|``, uniform over every tail row ``k`` (a
    translation-invariant, or merely uniformly-bounded, coupling).  Returns

    .. math::

        s = \sum_d \texttt{coupling\_bounds}[d] \cdot \nu^{\|d\|_1}

    (outward rounded), the ``off_diagonal_row_sum_upper`` that
    :func:`omnibias.core.verified.banded.banded_tail_inverse_bound` consumes
    when it is fed *this* quantity rather than a raw (unweighted) row sum.

    Why this is the right quantity for ``ell^1_nu`` (not just ``ell^infty``)
    --------------------------------------------------------------------------
    :mod:`omnibias.core.verified.banded`'s docstring is explicit that its bound
    certifies the operator norm induced by the *unweighted* sup norm, and
    documents two honest options for a caller who instead needs the weighted
    ``ell^1_nu`` operator norm :mod:`radii_spectral` actually uses (a
    *column*-sum criterion): reformulate everything in sup-norm, or recompute
    the row/column sum in the ``nu``-weighted quantity.  This function takes
    the second option.  For a fixed output column index ``j`` (i.e. the
    input coefficient at wavevector ``j``), the weighted contribution to every
    row ``k`` with ``k + d = j`` is ``|c_d(k)|\,\nu^{\|k\|_1}`` for
    ``k = j - d``; since ``\|j - d\|_1 \le \|j\|_1 + \|d\|_1`` (triangle
    inequality) and ``\nu \ge 1`` (so ``\nu^{(\cdot)}`` is non-decreasing),

    .. math::

        \nu^{\|j - d\|_1} \le \nu^{\|j\|_1}\,\nu^{\|d\|_1},

    so ``(1/\nu^{\|j\|_1}) \sum_d |c_d(j-d)|\,\nu^{\|j-d\|_1}
    \le \sum_d |c_d(j-d)|\,\nu^{\|d\|_1} \le \sum_d \texttt{coupling\_bounds}[d]
    \,\nu^{\|d\|_1} = s``, uniformly in ``j`` -- exactly the weighted
    column-sum bound the ``ell^1_nu`` operator norm needs.  This requires only
    a per-offset magnitude bound uniform over ``k``, not literal translation
    invariance.
    """
    if nu < 1.0:
        raise ValueError("nu must be >= 1 for the l1_nu algebra")
    terms = []
    for d, bound in coupling_bounds.items():
        if all(di == 0 for di in d):
            raise ValueError("coupling_bounds must not include the zero offset")
        b = float(bound)
        if b < 0.0:
            raise ValueError(f"coupling bound for offset {d!r} must be non-negative, got {b!r}")
        weight = Interval.point(float(nu)).pow_int(sum(abs(di) for di in d))
        terms.append(Interval.point(b) * weight)
    if not terms:
        return 0.0
    return sum_intervals(terms).hi


def tail_inverse_bound_from_banded(
    diag_lower: float, coupling_bounds: Mapping[Wavevector, float], nu: float
) -> float:
    r"""``mu`` for :attr:`SpectralProblem.tail_inverse_bound` from a banded tail.

    The banded-linear-part generalisation of :func:`laplacian_tail_inverse_bound`:
    combines :func:`banded_row_sum_bound_nu` (the ``ell^1_nu``-consistent
    off-diagonal sum) with
    :func:`omnibias.core.verified.banded.banded_tail_inverse_bound` (the
    Gershgorin / Neumann-series closed form) to produce the scalar tail
    operator-norm bound that plugs directly into the existing
    ``SpectralProblem(tail_inverse_bound=mu, ...)`` / ``_SplitInverse``
    machinery -- unchanged from how a diagonal symbol's ``mu`` is used.

    ``diag_lower`` is the caller's coercivity hypothesis
    ``inf_{\|k\|_\infty > N} |ell(k)| \ge \texttt{diag\_lower} > 0`` (the same
    proof obligation :func:`laplacian_tail_inverse_bound` discharges for the
    screened Laplacian, generalised to whatever diagonal part the caller's
    :class:`BandedLinearPart` uses).  ``coupling_bounds``/``nu`` are as in
    :func:`banded_row_sum_bound_nu`.  Raises :class:`ValueError` (propagated
    from :func:`~omnibias.core.verified.banded.banded_tail_inverse_bound`)
    when the resulting row-wise diagonal-dominance margin is not certifiably
    positive -- never fabricates a bound.
    """
    s = banded_row_sum_bound_nu(coupling_bounds, nu)
    return banded_tail_inverse_bound(diag_lower, s).hi


@dataclass(frozen=True)
class SpectralProblem:
    r"""A quadratic spectral problem ``F(a) = ell*a + Q(a,a) - f`` in ``l1_nu``."""

    dim: int
    trunc: int
    nu: float
    linear_symbol: Symbol | BandedLinearPart
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


def _apply_finite_banded(
    series: ValidatedFourierSeries, linear: BandedLinearPart, nu: float
) -> ValidatedFourierSeries:
    r"""``(L a)_k = ell(k) a_k + sum_d c_d(k) a_{k+d}`` for a finite (zero-tail) series.

    The output support is ``support(a) union {m - d : m in support(a), d in
    couplings}`` (every index that can receive a nonzero contribution from
    ``a``).  An output index landing outside ``series.trunc`` is folded into
    the returned tail by weighted magnitude, exactly as
    :meth:`ValidatedFourierSeries.__mul__` folds convolution overflow -- this
    only matters when the coupling bandwidth exceeds the truncation margin
    already reserved by the caller (e.g. ``work_trunc = 2N`` with a bandwidth
    ``<= N``, the regime :func:`quadratic_radii_certificate` uses, never
    overflows).  ``nu`` is the scalar weight (matching
    ``SpectralProblem.nu: float``; this module never uses the anisotropic
    per-axis form of :data:`~omnibias.core.verified.fourier.NuLike`).
    """
    trunc = series.trunc
    support = list(series.coeffs)
    out_indices: set[Wavevector] = set(support)
    for m in support:
        for d in linear.couplings:
            out_indices.add(tuple(mi - di for mi, di in zip(m, d, strict=True)))
    coeffs: dict[Wavevector, ComplexInterval] = {}
    overflow = Interval.point(0.0)
    for k in out_indices:
        acc = linear.diagonal(k) * series.get(k)
        for d, c_d in linear.couplings.items():
            shifted = tuple(ki + di for ki, di in zip(k, d, strict=True))
            neighbour = series.coeffs.get(shifted)
            if neighbour is not None:
                acc = acc + c_d(k) * neighbour
        if all(abs(kd) <= trunc for kd in k):
            coeffs[k] = acc
        else:
            overflow = overflow + Interval.point(acc.mag) * _weight(nu, k)
    return ValidatedFourierSeries(series.dim, trunc, series.nu, coeffs, _clip_nonneg(overflow))


def _apply_finite_linear(
    series: ValidatedFourierSeries, linear: Symbol | BandedLinearPart, nu: float
) -> ValidatedFourierSeries:
    """``linear * series`` for a finite (zero-tail) series; dispatches diagonal vs banded."""
    if isinstance(linear, BandedLinearPart):
        return _apply_finite_banded(series, linear, nu)
    return _apply_finite_symbol(series, linear)


def _clip_nonneg(iv: Interval) -> Interval:
    """Clamp a magnitude/tail enclosure to ``[0, inf)`` (outward rounding can dip below 0)."""
    return Interval(max(iv.lo, 0.0), max(iv.hi, 0.0))


class _SplitInverse:
    """The approximate inverse ``A = A_N (finite) + 1/ell (tail)``.

    ``linear`` may be a diagonal :data:`Symbol` (the original, byte-for-byte
    unchanged behaviour: the tail modes ``N < \\|m\\|_\\infty`` are inverted
    *exactly*, per mode, via ``1/ell(m)``) or a :class:`BandedLinearPart`. A
    banded tail has no elementary per-mode inverse (inverting a genuinely
    non-diagonal block would need solving a linear system, not a division),
    so instead every mode outside the numerically-inverted finite block --
    both the explicit ``N < \\|m\\|_\\infty \\le 2N`` shell and the continuum
    tail radius ``w.tail`` -- is folded into one aggregate magnitude scaled by
    the caller's ``mu`` bound. This is strictly *sound* (``mu`` bounds the
    banded tail operator's norm, so scaling any enclosed magnitude by it is a
    valid enclosure of that magnitude's image) but looser than the diagonal
    case's exact per-mode inverse on the shell -- the cost of admitting a
    non-diagonal linear part, not a soundness gap.
    """

    def __init__(
        self,
        a_n: list[list[ComplexInterval]],
        modes: list[Wavevector],
        linear: Symbol | BandedLinearPart,
        mu: float,
        dim: int,
        work_trunc: int,
        nu: float,
        trunc: int,
    ) -> None:
        self.a_n = a_n
        self.modes = modes
        self.linear = linear
        if isinstance(linear, BandedLinearPart):
            self.banded = True
            self.symbol: Symbol | None = None
        else:
            self.banded = False
            self.symbol = linear
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
        if self.banded:
            # No elementary per-mode inverse for a non-diagonal tail: fold
            # every mode with ||m||inf > N (the (N, 2N] shell plus w's own
            # continuum tail) into one mu-scaled aggregate magnitude.
            extra = Interval.point(0.0)
            for m, val in w.coeffs.items():
                if any(abs(c) > self.trunc for c in m):
                    extra = extra + Interval.point(val.mag) * _weight(self.nu, m)
            tail = (w.tail + extra) * Interval.point(self.mu)
        else:
            assert self.symbol is not None
            one = ComplexInterval.one()
            for m, val in w.coeffs.items():
                if any(abs(c) > self.trunc for c in m):  # tail: ||m||inf > N -> 1/ell(m)
                    out[m] = (one / self.symbol(m)) * val
            tail = w.tail * Interval.point(self.mu)
        tail = _clip_nonneg(tail)
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
    residual = _apply_finite_linear(ab, problem.linear_symbol, problem.nu) + problem.quadratic(
        ab, ab
    )
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
    linear, mu = problem.linear_symbol, problem.tail_inverse_bound

    ab = _embed(a_bar, dim, work, nu)
    modes = _modes(dim, problem.trunc)
    n = len(modes)

    # finite-input columns of DF(ab): DF(ab) e_k = ell(k) e_k + 2 Q(ab, e_k).
    df_cols: list[ValidatedFourierSeries] = []
    for k in modes:
        e_k = _embed({k: 1.0}, dim, work, nu)
        df_cols.append(
            _apply_finite_linear(e_k, linear, nu) + problem.quadratic(ab, e_k).scale(2.0)
        )

    # finite Jacobian block B_N (rows/cols on ||.||inf <= N) -> float midpoint -> inverse.
    b_mid: list[list[complex]] = [[0j for _ in range(n)] for _ in range(n)]
    for j, col in enumerate(df_cols):
        for i, m in enumerate(modes):
            cij = col.get(m)
            b_mid[i][j] = complex(cij.re.mid, cij.im.mid)
    a_n_c = _invert_complex(b_mid)
    a_n = [[ComplexInterval.point(a_n_c[i][j]) for j in range(n)] for i in range(n)]
    inverse = _SplitInverse(a_n, modes, linear, mu, dim, work, nu, problem.trunc)
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
    "BandedLinearPart",
    "Bilinear",
    "SpectralProblem",
    "SpectralRadiiResult",
    "banded_row_sum_bound_nu",
    "constant_coefficient_band",
    "evaluate_residual",
    "laplacian_symbol",
    "laplacian_tail_inverse_bound",
    "quadratic_radii_certificate",
    "tail_inverse_bound_from_banded",
]
