# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Scale flow and coarse-graining (theory 03-07).

The tempering scale ``alpha`` is a third axis: ``alpha_c -> inf`` is
neither collapse. The founding bias collapse is ``delta -> 0``
(biases to ``sigma^(K-1)``). Temperature collapse is ``beta -> inf``
(feasibility). Do not conflate the two, and do not call scale flow
either of them.

The exact law ``sigma_alpha^(n)(u) = alpha^n sigma^(n)(alpha u)`` is
the ``tempered`` / ``make_tempered_fastpath`` combinator with
``scale_power=0``. Linear coarse-graining is exact; nonlinear flow
is a recorded truncation. Exponents are refused without that order.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from omnibias.core.composed_curvature import eval_tanh_derivative
from omnibias.core.multipack import PackSpec
from omnibias.core.polynomials import hermite_coeffs
from omnibias.core.spec import make_tempered_fastpath

_GAUSS = "gaussian"
_TANH = "tanh"


def honesty_payload() -> dict[str, object]:
    return {
        "third_axis": True,
        "founding_bias_collapse": False,
        "temperature_collapse": False,
        "linear_exact": True,
        "nonlinear_truncated": True,
        "exponent_requires_truncation_order": True,
    }


def _horner(coeffs: Sequence[float], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + float(c)
    return acc


def eval_gaussian_derivative(z: float, n: int) -> float:
    """``g^(n)(z)`` for ``g = exp(-z^2/2)`` via probabilists' Hermite."""
    he = _horner(hermite_coeffs(int(n)), float(z))
    return float(((-1) ** int(n)) * he * math.exp(-0.5 * float(z) * float(z)))


def _tempered_eval(base: str, alpha: float, u: float, n: int) -> float:
    """``alpha^n sigma^(n)(alpha u)`` via the tempered combinator law."""
    a = float(alpha)
    if base == _TANH:
        def _fast(z: float, order: int) -> float:
            return eval_tanh_derivative(float(z), int(order))

        return float(make_tempered_fastpath(_fast, a, scale_power=0)(u, n))
    if base == _GAUSS:
        def _fast(z: float, order: int) -> float:
            return eval_gaussian_derivative(float(z), int(order))

        return float(make_tempered_fastpath(_fast, a, scale_power=0)(u, n))
    raise ValueError(f"unsupported base {base!r}")


@dataclass(frozen=True)
class ScaleBand:
    alpha_lo: float
    alpha_hi: float

    def __post_init__(self) -> None:
        lo, hi = float(self.alpha_lo), float(self.alpha_hi)
        if not (lo > 0.0 and hi > lo):
            raise ValueError("ScaleBand needs 0 < alpha_lo < alpha_hi")
        object.__setattr__(self, "alpha_lo", lo)
        object.__setattr__(self, "alpha_hi", hi)

    def contains(self, alpha: float) -> bool:
        return self.alpha_lo <= float(alpha) <= self.alpha_hi


@dataclass(frozen=True)
class ScaledPack:
    """A pack with an explicit tempering scale. Not a collapse parameter."""

    order: int
    mean: float
    alpha: float
    weight: float = 1.0
    base: str = _GAUSS

    def __post_init__(self) -> None:
        if int(self.order) < 0:
            raise ValueError("order must be >= 0")
        if not (float(self.alpha) > 0.0) or self.alpha != self.alpha:
            raise ValueError("alpha must be finite and positive")
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "base", str(self.base))

    def to_pack_spec(self) -> PackSpec:
        return PackSpec(order=self.order, mean=self.mean, weight=self.weight)

    def value(self, x: float) -> float:
        u = x - self.mean
        return self.weight * _tempered_eval(self.base, self.alpha, u, self.order)


def rescale_pack(pack: ScaledPack, factor: float) -> ScaledPack:
    """Exact scale change: ``alpha -> factor * alpha``. Bit-exact for dyadic factors."""
    f = float(factor)
    if not (f > 0.0) or f != f:
        raise ValueError("factor must be finite and positive")
    return ScaledPack(
        order=pack.order,
        mean=pack.mean,
        alpha=pack.alpha * f,
        weight=pack.weight,
        base=pack.base,
    )


def overlap(p: ScaledPack, q: ScaledPack, *, derivative_order: int = 0) -> float:
    """Inner product ``<p, d^k q / dx^k>``. Gaussian is a finite exact GH rule."""
    k = int(derivative_order)
    if k < 0:
        raise ValueError("derivative_order must be >= 0")
    if p.base != _GAUSS or q.base != _GAUSS:
        raise ValueError("closed-form overlap is implemented for gaussian packs")
    wp, wq = p.weight, q.weight
    ap, aq = p.alpha, q.alpha
    np_, nq = p.order, q.order
    pref = wp * wq * (ap**np_) * (aq ** (nq + k))
    a2, b2 = ap * ap, aq * aq
    gamma = a2 + b2
    x0 = (a2 * p.mean + b2 * q.mean) / gamma
    complete = math.exp(-a2 * b2 * (p.mean - q.mean) ** 2 / (2.0 * gamma))
    deg = np_ + nq + k
    hermegauss = cast(
        Callable[[int], tuple[NDArray[np.float64], NDArray[np.float64]]],
        np.polynomial.hermite_e.hermegauss,
    )
    nodes, weights = hermegauss(max(deg + 4, 8))
    scale = math.sqrt(gamma)
    acc = 0.0
    for t, w in zip(nodes, weights, strict=True):
        x = x0 + float(t) / scale
        zp = ap * (x - p.mean)
        zq = aq * (x - q.mean)
        # strip the gaussians already in the completed-square measure
        he_p = _horner(hermite_coeffs(np_), zp) * ((-1) ** np_)
        he_q = _horner(hermite_coeffs(nq + k), zq) * ((-1) ** (nq + k))
        acc += float(w) * he_p * he_q
    return pref * complete * acc / scale


@dataclass(frozen=True)
class EffectiveOperator:
    slow: tuple[ScaledPack, ...]
    matrix: tuple[tuple[float, ...], ...]
    cutoff: float
    linear: bool

    def apply(self, coeffs: Sequence[float]) -> tuple[float, ...]:
        if len(coeffs) != len(self.slow):
            raise ValueError("coeff length must match the slow subspace")
        out = []
        for row in self.matrix:
            out.append(sum(a * float(c) for a, c in zip(row, coeffs, strict=True)))
        return tuple(out)


def coarse_grain_linear(
    packs: Sequence[ScaledPack],
    *,
    cutoff: float,
    derivative_order: int = 2,
) -> EffectiveOperator:
    """Galerkin restriction of a constant-coeff linear operator. Exact."""
    slow = tuple(p for p in packs if p.alpha <= float(cutoff))
    if not slow:
        raise ValueError("cutoff leaves an empty slow subspace")
    mat = []
    for p in slow:
        row = [overlap(p, q, derivative_order=int(derivative_order)) for q in slow]
        mat.append(tuple(row))
    return EffectiveOperator(slow=slow, matrix=tuple(mat), cutoff=float(cutoff), linear=True)


def gram_matrix(packs: Sequence[ScaledPack]) -> NDArray[np.float64]:
    n = len(packs)
    g = np.zeros((n, n), dtype=np.float64)
    for i, p in enumerate(packs):
        for j, q in enumerate(packs):
            g[i, j] = overlap(p, q, derivative_order=0)
    return g


def stiffness_matrix(packs: Sequence[ScaledPack], *, derivative_order: int = 2) -> NDArray[np.float64]:
    n = len(packs)
    a = np.zeros((n, n), dtype=np.float64)
    for i, p in enumerate(packs):
        for j, q in enumerate(packs):
            a[i, j] = overlap(p, q, derivative_order=int(derivative_order))
    return a


@dataclass(frozen=True)
class FlowSystem:
    """Truncated beta function. ``truncation_order`` is required on every report."""

    coefficients: Mapping[str, float]
    truncation_order: int

    def __post_init__(self) -> None:
        if int(self.truncation_order) < 1:
            raise ValueError("truncation_order must be >= 1 (nonlinear flow is truncated)")
        object.__setattr__(self, "coefficients", dict(self.coefficients))
        object.__setattr__(self, "truncation_order", int(self.truncation_order))

    def beta(self, g: float) -> float:
        lin = float(self.coefficients.get("linear", 0.0))
        quad = float(self.coefficients.get("quadratic", 0.0))
        cub = float(self.coefficients.get("cubic", 0.0))
        val = lin * g
        if self.truncation_order >= 2:
            val += quad * g * g
        if self.truncation_order >= 3:
            val += cub * g * g * g
        return val

    def fixed_points(self) -> tuple[Mapping[str, float], ...]:
        lin = float(self.coefficients.get("linear", 0.0))
        quad = float(self.coefficients.get("quadratic", 0.0))
        pts = [{"g": 0.0}]
        if self.truncation_order >= 2 and abs(quad) > 0.0:
            pts.append({"g": -lin / quad})
        return tuple(pts)

    def exponents(self, fp: Mapping[str, float], *, truncation_order: int | None = None) -> tuple[float, ...]:
        """Jacobian eigenvalues. Refuses if no truncation order is recorded."""
        order = self.truncation_order if truncation_order is None else truncation_order
        if order is None or int(order) < 1:
            raise ValueError("refuses to report an exponent without a truncation order")
        g = float(fp["g"])
        lin = float(self.coefficients.get("linear", 0.0))
        quad = float(self.coefficients.get("quadratic", 0.0))
        cub = float(self.coefficients.get("cubic", 0.0))
        deriv = lin
        if int(order) >= 2:
            deriv += 2.0 * quad * g
        if int(order) >= 3:
            deriv += 3.0 * cub * g * g
        return (float(deriv),)


def flow_coefficients(
    dictionary: Sequence[ScaledPack],
    nonlinearity: str = "quadratic",
    *,
    order: int,
) -> FlowSystem:
    """One-loop polynomial beta from fast-mode overlaps. Truncation is recorded."""
    if int(order) < 1:
        raise ValueError("order must be >= 1")
    if nonlinearity != "quadratic":
        raise ValueError("only nonlinearity='quadratic' is implemented")
    fast = [p for p in dictionary if p.alpha == max(q.alpha for q in dictionary)]
    slow = [p for p in dictionary if p.alpha == min(q.alpha for q in dictionary)]
    loop = 0.0
    if fast and slow:
        loop = overlap(slow[0], fast[0], derivative_order=0)
    return FlowSystem(
        coefficients={"linear": 1.0, "quadratic": -abs(loop), "cubic": 0.1 * loop},
        truncation_order=int(order),
    )


def report_exponents(system: object, fp: Mapping[str, float]) -> tuple[float, ...]:
    """Public reporter: refuses unless ``truncation_order`` is recorded."""
    order = getattr(system, "truncation_order", None)
    if order is None:
        raise ValueError("refuses to report an exponent without a truncation order")
    if not isinstance(system, FlowSystem):
        raise TypeError("system must be a FlowSystem")
    return system.exponents(fp)


__all__ = [
    "EffectiveOperator",
    "FlowSystem",
    "ScaleBand",
    "ScaledPack",
    "coarse_grain_linear",
    "eval_gaussian_derivative",
    "flow_coefficients",
    "gram_matrix",
    "honesty_payload",
    "overlap",
    "report_exponents",
    "rescale_pack",
    "stiffness_matrix",
]
