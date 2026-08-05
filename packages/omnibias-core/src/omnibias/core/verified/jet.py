# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified (interval) directional jets -- the rigorous twin of the float jets.

This mirrors the exact-derivative jet kernels of :mod:`omnibias.jax.jet` /
:mod:`omnibias.torch.jet` (truncated-convolution Faà di Bruno composition) but
propagates :class:`~omnibias.core.verified.interval.Interval` Taylor
coefficients instead of floats.  Because the structure is identical, a float jet
and the midpoints of a verified jet agree to rounding; the verified version
additionally certifies an enclosure of every directional derivative.

Conventions (identical to the float kernels)
--------------------------------------------
* A *scalar jet* is ``list[Interval]`` of length ``N+1`` holding Taylor
  coefficients ``a_k = f^(k)(0)/k!`` of a path parameter ``t``.
* A *vector jet* is ``list[list[Interval]]`` of shape ``(N+1, D)`` (order-major),
  one inner list of components per Taylor order.
* :func:`jet_to_tower` rescales coefficient ``a_k`` by ``k!`` to recover the
  directional derivative ``d^k/dt^k f``.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from math import factorial

from omnibias.core.verified.interval import Interval, IntervalLike
from omnibias.core.verified.sigma import sigma_tower_interval

Jet = list[Interval]
VecJet = list[list[Interval]]


def _zero() -> Interval:
    return Interval.point(0.0)


def series_mul(a: Sequence[Interval], b: Sequence[Interval]) -> Jet:
    """Truncated Cauchy product of two equal-length interval series."""
    np1 = len(a)
    if len(b) != np1:
        raise ValueError(f"series length mismatch: {len(a)} vs {len(b)}")
    out: Jet = []
    for n in range(np1):
        acc = _zero()
        for i in range(n + 1):
            acc = acc + a[i] * b[n - i]
        out.append(acc)
    return out


def compose_jet(u_jet: Sequence[Interval], sigma_tower: Sequence[Interval]) -> Jet:
    """Compose an activation onto a scalar jet: ``b(t) = sigma(u(t))``.

    ``sigma_tower[k]`` must enclose ``sigma^(k)(u_jet[0])``.  Identical algorithm
    to :func:`omnibias.jax.jet.compose_jet`, evaluated in interval arithmetic.
    """
    np1 = len(u_jet)
    if len(sigma_tower) != np1:
        raise ValueError(
            f"sigma_tower order {len(sigma_tower) - 1} must match jet order {np1 - 1}"
        )
    zero = _zero()
    w: Jet = [zero] + [u_jet[j] for j in range(1, np1)]
    p: Jet = [Interval.point(1.0)] + [zero for _ in range(np1 - 1)]
    result: Jet = [sigma_tower[0]] + [zero for _ in range(np1 - 1)]
    fact = 1
    for k in range(1, np1):
        fact *= k
        p = series_mul(p, w)
        dk = sigma_tower[k] * Interval.from_rational(Fraction(1, fact))
        for n in range(np1):
            result[n] = result[n] + dk * p[n]
    return result


def affine_jet(
    z_jet: Sequence[Sequence[Interval]],
    weight: Sequence[Sequence[IntervalLike]],
    bias: Sequence[IntervalLike] | None = None,
) -> VecJet:
    """Push a vector jet through ``u = W z + b`` (acts per Taylor order)."""
    np1 = len(z_jet)
    d_out = len(weight)
    d_in = len(z_jet[0]) if np1 else 0
    out: VecJet = []
    for k in range(np1):
        row: list[Interval] = []
        for o in range(d_out):
            w_o = weight[o]
            if len(w_o) != d_in:
                raise ValueError("weight row width must match input dimension")
            acc = _zero()
            for i in range(d_in):
                acc = acc + Interval.from_value(w_o[i]) * z_jet[k][i]
            if k == 0 and bias is not None:
                acc = acc + Interval.from_value(bias[o])
            row.append(acc)
        out.append(row)
    return out


def layer_jet(
    z_jet: Sequence[Sequence[Interval]],
    weight: Sequence[Sequence[IntervalLike]],
    bias: Sequence[IntervalLike] | None,
    name: str,
) -> VecJet:
    """Push a vector jet through one ``sigma(W z + b)`` layer (interval)."""
    u_jet = affine_jet(z_jet, weight, bias)
    np1 = len(u_jet)
    order = np1 - 1
    width = len(u_jet[0]) if np1 else 0
    out: VecJet = [[_zero() for _ in range(width)] for _ in range(np1)]
    for o in range(width):
        u_series = [u_jet[k][o] for k in range(np1)]
        tower = sigma_tower_interval(name, u_series[0], order)
        composed = compose_jet(u_series, list(tower))
        for k in range(np1):
            out[k][o] = composed[k]
    return out


def path_jet(
    x0: Sequence[IntervalLike], v: Sequence[IntervalLike], order: int
) -> VecJet:
    """Vector input jet for the line ``x(t) = x0 + t v`` truncated at ``order``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    d = len(x0)
    if len(v) != d:
        raise ValueError("x0 and v must have the same dimension")
    rows: VecJet = [[Interval.from_value(x0[i]) for i in range(d)]]
    if order >= 1:
        rows.append([Interval.from_value(v[i]) for i in range(d)])
    for _ in range(order - 1):
        rows.append([_zero() for _ in range(d)])
    return rows[: order + 1]


def mlp_jet(
    x0: Sequence[IntervalLike],
    v: Sequence[IntervalLike],
    layers: Sequence[
        tuple[Sequence[Sequence[IntervalLike]], Sequence[IntervalLike] | None, str | None]
    ],
    order: int,
) -> VecJet:
    """Exact directional interval jet of a deep MLP along ``x(t) = x0 + t v``.

    Each layer is ``(W, b, name)``; ``name=None`` is a pure affine readout.
    """
    jet = path_jet(x0, v, order)
    for weight, bias, name in layers:
        jet = affine_jet(jet, weight, bias) if name is None else layer_jet(jet, weight, bias, name)
    return jet


def lhopital_ratio_iv(
    num_jet: Sequence[Interval], den_jet: Sequence[Interval], order: int = 1
) -> Interval:
    r"""Verified L'Hopital limit ``num_jet[order] / den_jet[order]`` (enclosure).

    Interval twin of :func:`omnibias.jax.jet.lhopital_ratio`: given rigorous
    enclosures of the Taylor coefficients of a ``0/0`` numerator and denominator
    that both vanish to order ``order - 1``, returns an :class:`Interval`
    enclosing the limit of the ratio. Raises :class:`ZeroDivisionError` (via
    :meth:`Interval.reciprocal`) when the leading denominator enclosure straddles
    zero -- i.e. the limit is not certified finite at this truncation order.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    max_order = min(len(num_jet), len(den_jet)) - 1
    if order > max_order:
        raise ValueError(f"order {order} exceeds available jet order {max_order}")
    return num_jet[order] / den_jet[order]


def jet_to_tower(jet: Sequence[Interval]) -> Jet:
    """Rescale a scalar jet ``a_k`` by ``k!`` to the derivative tower."""
    return [jet[k] * Interval.from_rational(factorial(k)) for k in range(len(jet))]


def tower_to_jet(tower: Sequence[Interval]) -> Jet:
    """Rescale a derivative tower by ``1/k!`` to the Taylor jet."""
    return [
        tower[k] * Interval.from_rational(Fraction(1, factorial(k)))
        for k in range(len(tower))
    ]


def derivative_jet(jet: Sequence[Interval]) -> Jet:
    r"""Taylor jet of ``f'`` from the jet of ``f`` (differentiation; FTC inverse).

    With ``a_k = f^(k)(0)/k!`` the derivative's Taylor coefficients are
    ``(f')^(k)(0)/k! = (k+1)\,a_{k+1}``, so the jet shortens by one order.  This is
    the exact inverse of :func:`antiderivative_jet`
    (``derivative_jet(antiderivative_jet(a)) == a``), the differentiation half of
    the Fundamental Theorem of Calculus in the jet register.
    """
    return [jet[k + 1] * Interval.from_rational(k + 1) for k in range(len(jet) - 1)]


def antiderivative_jet(jet: Sequence[Interval], constant: IntervalLike = 0.0) -> Jet:
    r"""Taylor jet of the antiderivative ``F(t) = constant + \int_0^t f`` (term-by-term FTC).

    With ``a_k = f^(k)(0)/k!`` the antiderivative's Taylor coefficients are
    ``A_0 = constant`` and ``A_m = a_{m-1}/m`` for ``m >= 1``, so the jet lengthens
    by one order.  This is the exact closed-form *integral tower* -- the two-sided
    partner of the derivative tower :func:`jet_to_tower` -- valid to every order for
    any activation; the constant of integration is a free parameter (default 0).
    """
    out: Jet = [Interval.from_value(constant)]
    out.extend(
        jet[m - 1] * Interval.from_rational(Fraction(1, m)) for m in range(1, len(jet) + 1)
    )
    return out


__all__ = [
    "Jet",
    "VecJet",
    "affine_jet",
    "antiderivative_jet",
    "compose_jet",
    "derivative_jet",
    "jet_to_tower",
    "layer_jet",
    "lhopital_ratio_iv",
    "mlp_jet",
    "path_jet",
    "series_mul",
    "tower_to_jet",
]
