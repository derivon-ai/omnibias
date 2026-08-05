# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Rigorous activation enclosures -- interval *and* Taylor-model forms.

The smooth activations (``tanh`` / ``sigmoid`` / ``gaussian`` / exact ``GELU``)
are composed into a :class:`~omnibias.core.verified.taylor_model_mv.TaylorModelMV`
through the **closed-form derivative tower** -- one transcendental enclosure
yields every Taylor coefficient ``sigma^(k)(m)`` plus a Lagrange remainder
``sigma^(N+1)([range])``.  Keeping the polynomial shape (instead of collapsing to
a box at every layer) is what makes the verifier tighter than interval-bound
propagation.  The nonsmooth activations get sound relaxations: a ReLU
triangle/zonotope band and a group ``max`` enclosure.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.verified.interval import Interval, IntervalLike, _pred, _succ, hull
from omnibias.core.verified.sigma import sigma_tower_interval, sigma_value_interval
from omnibias.core.verified.taylor_model_mv import TaylorModelMV

# pi bracket (math.pi rounds down; the true pi is the next double up).
_PI = Interval(math.pi, _succ(math.pi))
#: rigorous enclosure of 1/sqrt(2*pi).
INV_SQRT_2PI = (Interval.point(2.0) * _PI).sqrt().reciprocal()

_MP: Any | None = None
_MP_RESOLVED = False


def _mpmath() -> Any | None:
    global _MP, _MP_RESOLVED
    if not _MP_RESOLVED:
        try:
            _MP = importlib.import_module("mpmath")
        except ImportError:  # pragma: no cover - environment dependent
            _MP = None
        _MP_RESOLVED = True
    return _MP


def _inv_factorial(k: int) -> Interval:
    return Interval.from_rational(Fraction(1, math.factorial(k)))


def _even_pow(iv: Interval, p: int) -> Interval:
    """Tight enclosure of ``{x**p : x in iv}`` (even powers are non-negative).

    ``Interval.pow_int`` multiplies term by term and so returns a symmetric
    ``[-a, a]`` for an even power of a symmetric interval; intersecting with
    ``[0, inf)`` recovers the true ``[0, a]`` and halves the Lagrange remainder.
    """
    r = iv.pow_int(p)
    if p % 2 == 0:
        return Interval(max(r.lo, 0.0), r.hi)
    return r


def _subdiv_tail(
    tail_deriv: Callable[[Interval, int], Interval], rng: Interval, n: int
) -> Interval:
    r"""Enclosure of ``f^(n)`` over ``rng``, sharpened by range subdivision.

    Interval evaluation of a high derivative over a wide range suffers severe
    dependency overestimation (e.g. ``tanh^(4)`` on ``[-0.8, 0.8]`` enclosed as
    ``+-25`` versus a true ``~4``).  Splitting ``rng`` into narrow pieces and
    taking the hull recovers a near-exact bound; the piece width shrinks the
    over-approximation roughly linearly.
    """
    width = rng.hi - rng.lo
    if width <= 0.0:
        return tail_deriv(rng, n)
    splits = min(128, max(1, math.ceil(width / 0.05)))
    step = width / splits
    pieces: list[IntervalLike] = []
    for i in range(splits):
        lo = rng.lo + i * step
        hi = rng.hi if i == splits - 1 else rng.lo + (i + 1) * step
        pieces.append(tail_deriv(Interval(lo, hi), n))
    return hull(pieces)


# --------------------------------------------------------------------------- #
# Gaussian CDF (Phi) -- the only primitive not already in core.verified.
# --------------------------------------------------------------------------- #
def _gauss_cdf_point(x: float) -> tuple[float, float]:
    """Rigorous double bracket of ``Phi(x)`` (mpmath preferred, libm fallback)."""
    mp = _mpmath()
    if mp is not None:
        with mp.workdps(60):
            y = mp.ncdf(mp.mpf(x))
        f = float(y)
        lo = f if mp.mpf(f) <= y else _pred(f)
        hi = f if mp.mpf(f) >= y else _succ(f)
        return _pred(lo), _succ(hi)
    v = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    lo = hi = v
    for _ in range(4):  # libm erf is < 1 ulp on mainstream platforms; pad 4.
        lo = _pred(lo)
        hi = _succ(hi)
    return lo, hi


def gauss_cdf_iv(x: Interval) -> Interval:
    """Guaranteed enclosure of the standard-normal CDF ``Phi`` (monotone increasing)."""
    lo, _ = _gauss_cdf_point(x.lo)
    _, hi = _gauss_cdf_point(x.hi)
    return Interval(max(lo, 0.0), min(hi, 1.0))


def gelu_enclosure(x: Interval) -> Interval:
    """Sound interval enclosure of exact GELU ``x * Phi(x)`` over ``x``."""
    return x * gauss_cdf_iv(x)


def max_enclosure(values: Sequence[Interval]) -> Interval:
    """Sound enclosure of ``max(values)`` (endpoint-wise)."""
    return Interval(max(v.lo for v in values), max(v.hi for v in values))


# --------------------------------------------------------------------------- #
# Taylor-model composition with a smooth scalar function.
# --------------------------------------------------------------------------- #
def compose_smooth(
    tm: TaylorModelMV,
    tower_at: Callable[[float, int], Sequence[Interval]],
    tail_deriv: Callable[[Interval, int], Interval],
    order: int,
) -> TaylorModelMV:
    r"""Compose a smooth scalar ``f`` with a Taylor model via its derivative tower.

    ``tower_at(m, N)`` returns ``(f(m), f'(m), ..., f^(N)(m))`` at the real
    expansion point ``m``; ``tail_deriv(rng, N+1)`` bounds ``f^(N+1)`` over a range.
    The result encloses ``f(tm(x))`` for every ``x`` in the box:

    .. math:: f(g) = \sum_{k=0}^N \frac{f^{(k)}(m)}{k!} (g-m)^k
                     + \frac{f^{(N+1)}(\xi)}{(N+1)!} (g-m)^{N+1}.
    """
    m = tm.coeffs[0].mid
    tower = tower_at(m, order)
    shifted = tm - m
    poly = TaylorModelMV.constant(tower[0], tm.center, tm.radius, tm.order)
    power = TaylorModelMV.constant(1.0, tm.center, tm.radius, tm.order)
    for k in range(1, order + 1):
        power = power * shifted
        poly = poly + power * (tower[k] * _inv_factorial(k))
    rng = hull([tm.bound(), Interval.point(m)])
    tail = _subdiv_tail(tail_deriv, rng, order + 1)
    h_range = tm.bound() - Interval.point(m)
    remainder = tail * _inv_factorial(order + 1) * _even_pow(h_range, order + 1)
    return TaylorModelMV(
        tm.center, tm.radius, tm.order, poly.coeffs, poly.remainder + remainder
    )


#: Collapse a neuron's polynomial to a constant once its bare polynomial bound is
#: this many times wider than the direct interval enclosure -- the signature of a
#: range past the activation's radius of convergence, where the Taylor remainder
#: blows up and the shape is worthless anyway.  Within the radius the shape is
#: kept (modest looseness pays off through cross-neuron cancellation downstream);
#: branch-and-bound shrinks any box back under the radius.  Tightness heuristic
#: only -- both branches are equally sound.
_DIVERGENCE_RATIO = 4.0


def _refine(result: TaylorModelMV, direct: Interval, tm: TaylorModelMV) -> TaylorModelMV:
    r"""Tighten a composed model with the direct interval enclosure ``direct``.

    Two sound moves, neither of which discards the polynomial shape in the regime
    that matters:

    * **Remainder intersection** -- the true remainder set
      ``{f(g(x)) - P(x)}`` lies in both ``result.remainder`` *and*
      ``direct - P_bound`` (since ``f(g(x)) in direct`` and ``P(x) in P_bound``);
      intersecting the two caps a growing tail without touching the coefficients.
    * **Divergence collapse** -- only when the bare polynomial bound itself is
      ``_DIVERGENCE_RATIO``x wider than ``direct`` do we fall back to the
      (tighter) constant intersection.
    """
    poly_bound = result._poly_bound()
    if poly_bound.width > _DIVERGENCE_RATIO * direct.width:
        total = poly_bound + result.remainder
        lo = max(total.lo, direct.lo)
        hi = min(total.hi, direct.hi)
        return TaylorModelMV.constant(Interval(lo, hi), tm.center, tm.radius, tm.order)
    cap = direct - poly_bound
    new_rem = Interval(
        max(result.remainder.lo, cap.lo), min(result.remainder.hi, cap.hi)
    )
    return TaylorModelMV(tm.center, tm.radius, tm.order, result.coeffs, new_rem)


def compose_sigma(tm: TaylorModelMV, name: str, order: int) -> TaylorModelMV:
    """Taylor-model composition for a core activation (``tanh`` / ``sigmoid`` / ``gaussian``)."""

    def tower_at(m: float, n: int) -> Sequence[Interval]:
        return sigma_tower_interval(name, Interval.point(m), n)

    def tail_deriv(rng: Interval, n: int) -> Interval:
        return sigma_tower_interval(name, rng, n)[n]

    result = compose_smooth(tm, tower_at, tail_deriv, order)
    return _refine(result, sigma_value_interval(name, tm.bound()), tm)


def compose_gelu(tm: TaylorModelMV, order: int) -> TaylorModelMV:
    r"""Taylor-model enclosure of exact GELU ``x * Phi(x)``.

    ``Phi`` is composed through its derivative tower
    (``Phi^{(k)} = (2\pi)^{-1/2} g^{(k-1)}`` for ``k>=1`` with
    ``g(z)=e^{-z^2/2}`` the core ``gaussian`` tower, and ``Phi(m)`` from
    :func:`gauss_cdf_iv`), then multiplied by the input model.
    """

    def phi_tower(m: float, n: int) -> Sequence[Interval]:
        out: list[Interval] = [gauss_cdf_iv(Interval.point(m))]
        if n >= 1:
            g = sigma_tower_interval("gaussian", Interval.point(m), n - 1)
            out.extend(INV_SQRT_2PI * g[k] for k in range(n))
        return out

    def phi_tail(rng: Interval, n: int) -> Interval:
        return INV_SQRT_2PI * sigma_tower_interval("gaussian", rng, n - 1)[n - 1]

    phi_tm = compose_smooth(tm, phi_tower, phi_tail, order)
    return _refine(tm * phi_tm, gelu_enclosure(tm.bound()), tm)


def relu_taylor(tm: TaylorModelMV) -> TaylorModelMV:
    r"""ReLU as a Taylor model: exact on stable neurons, a zonotope band otherwise.

    With pre-activation range ``[lo, hi]``: if ``lo >= 0`` ReLU is the identity; if
    ``hi <= 0`` it is zero; otherwise the linear relaxation
    ``relu(x) in lambda*(x-lo) + band`` (``lambda = hi/(hi-lo)``, ``band`` the
    rigorous range of ``relu(x) - lambda(x-lo)`` over the three breakpoints) keeps
    the affine dependence on the inputs instead of collapsing to a box.
    """
    b = tm.bound()
    lo, hi = b.lo, b.hi
    if lo >= 0.0:
        return tm
    if hi <= 0.0:
        return TaylorModelMV.constant(0.0, tm.center, tm.radius, tm.order)
    lam = hi / (hi - lo)
    lam_iv = Interval.point(lam)
    lo_iv, hi_iv = Interval.point(lo), Interval.point(hi)
    # gap(x) = relu(x) - lambda(x-lo); piecewise linear with breakpoints lo,0,hi.
    g_lo = Interval.point(0.0)  # relu(lo)=0, lambda(lo-lo)=0
    g_0 = -lam_iv * (Interval.point(0.0) - lo_iv)  # 0 - lambda(0-lo)
    g_hi = hi_iv - lam_iv * (hi_iv - lo_iv)  # hi - lambda(hi-lo)
    band = Interval(
        min(g_lo.lo, g_0.lo, g_hi.lo), max(g_lo.hi, g_0.hi, g_hi.hi)
    )
    linear = (tm - lo_iv) * lam_iv
    return TaylorModelMV(
        tm.center, tm.radius, tm.order, linear.coeffs, linear.remainder + band
    )


__all__ = [
    "INV_SQRT_2PI",
    "compose_gelu",
    "compose_sigma",
    "compose_smooth",
    "gauss_cdf_iv",
    "gelu_enclosure",
    "max_enclosure",
    "relu_taylor",
]
