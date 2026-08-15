# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Weak-form test-function algebra (theory 02-04).

Analytic OMBU bumps have certified exponential tails, not compact support.
Exact integrals hold only for polynomial coefficient data on box windows;
otherwise quadrature runs on the coefficient factor and the path is recorded.
Boundary terms are bounded via :func:`~omnibias.core.mollifier.tail_bound`,
never dropped. SDF domains stay quadrature-near-boundary and are not claimed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

from omnibias.core.mollifier import MollifierSpec, tail_bound
from omnibias.core.multipack import PackSpec
from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs
from omnibias.core.scan import BankSpec
from omnibias.core.verified.interval import Interval


def _horner(coeffs: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def _sigma_n(base: str, z: float, n: int) -> float:
    if n < 0:
        raise ValueError(f"derivative order must be >= 0, got {n}")
    name = str(base).lower()
    if name in ("tanh", "sech"):
        return _horner(tanh_polynomial_coeffs(n), math.tanh(z))
    if name in ("sigmoid", "logistic"):
        if z >= 0.0:
            s = 1.0 / (1.0 + math.exp(-z))
        else:
            e = math.exp(z)
            s = e / (1.0 + e)
        return _horner(sigmoid_polynomial_coeffs(n), s)
    raise ValueError(f"unsupported test-function base {base!r}")


def _log_cosh(z: float) -> float:
    az = abs(z)
    return az + math.log1p(math.exp(-2.0 * az)) - math.log(2.0)


def _softplus(z: float) -> float:
    if z > 20.0:
        return z
    if z < -20.0:
        return math.exp(z)
    return math.log1p(math.exp(z))


def _antiderivative(base: str, z: float) -> float:
    name = str(base).lower()
    if name in ("tanh", "sech"):
        return _log_cosh(z)
    if name in ("sigmoid", "logistic"):
        return _softplus(z)
    raise ValueError(f"unsupported test-function base {base!r}")


def _scaled_primitive(base: str, z: float, n: int, alpha: float) -> float:
    """Antiderivative of ``sigma^{(n)}(alpha (x-mu))`` wrt ``x``, at this ``z=alpha(x-mu)``."""
    if n == 0:
        return _antiderivative(base, z) / alpha
    return _sigma_n(base, z, n - 1) / alpha


@dataclass(frozen=True)
class TestFunctionSpace:
    """Bank of OMBU bumps used as Petrov-Galerkin test functions."""

    __test__: ClassVar[bool] = False

    bank: BankSpec
    orders: tuple[int, ...]
    base: str = "tanh"
    window: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.orders:
            raise ValueError("orders must be non-empty")
        if any(int(n) < 0 for n in self.orders):
            raise ValueError("orders must be >= 0")
        if self.window is not None:
            lo, hi = self.window
            if not (hi > lo):
                raise ValueError(f"window needs lo < hi, got {self.window}")

    @property
    def size(self) -> int:
        return int(self.bank.n_offsets)

    def order_at(self, index: int) -> int:
        if index < 0 or index >= self.size:
            raise IndexError(f"test index {index} out of range [0, {self.size})")
        if len(self.orders) == 1:
            return int(self.orders[0])
        if len(self.orders) != self.size:
            raise ValueError("orders must be length 1 or match the bank")
        return int(self.orders[index])

    def scale_at(self, index: int) -> float:
        scales = self.bank.scales
        if len(scales) == 1:
            return float(scales[0])
        return float(scales[index])

    def offset_at(self, index: int) -> float:
        return float(self.bank.offsets[index])


@dataclass(frozen=True)
class WeakForm:
    """1-D Poisson-type form ``int a u' v' - int f v`` on a box.

    ``diffusion`` and ``source`` are polynomial coefficients (low degree first).
    Exact assembly requires polynomial data on a box window.
    """

    kind: Literal["poisson1d"] = "poisson1d"
    diffusion: tuple[float, ...] = (1.0,)
    source: tuple[float, ...] | None = None


def eval_test(space: TestFunctionSpace, index: int, x: float, *, deriv: int = 0) -> float:
    """``v^{(deriv)}(x)`` for test function ``index``."""
    n = space.order_at(index) + int(deriv)
    alpha = space.scale_at(index)
    mu = space.offset_at(index)
    z = alpha * (x - mu)
    return (alpha ** deriv) * _sigma_n(space.base, z, n)


def exact_moment(space: TestFunctionSpace, power: int, index: int) -> float:
    """``integral_window x^{power} v_index(x) dx`` by antiderivative differences.

    Requires ``order >= power`` so integration by parts stays inside the
    closed-form tower (no leftover ``int x^k sigma``).
    """
    if space.window is None:
        raise ValueError("exact_moment needs a box window")
    p, q = space.window
    n = space.order_at(index)
    j = int(power)
    if j < 0:
        raise ValueError(f"power must be >= 0, got {power}")
    if n < j:
        raise ValueError(
            f"exact_moment needs test order >= power (got order={n}, power={j})"
        )
    alpha = space.scale_at(index)
    mu = space.offset_at(index)
    return _moment_ibp(space.base, p, q, j, n, alpha, mu)


def _moment_ibp(
    base: str, lo: float, hi: float, power: int, order: int, alpha: float, mu: float
) -> float:
    z_lo = alpha * (lo - mu)
    z_hi = alpha * (hi - mu)
    if power == 0:
        return _scaled_primitive(base, z_hi, order, alpha) - _scaled_primitive(
            base, z_lo, order, alpha
        )
    v_lo = _scaled_primitive(base, z_lo, order, alpha)
    v_hi = _scaled_primitive(base, z_hi, order, alpha)
    boundary = (hi**power) * v_hi - (lo**power) * v_lo
    return boundary - float(power) * _moment_ibp(
        base, lo, hi, power - 1, order - 1, alpha, mu
    )


def boundary_bound(space: TestFunctionSpace, *, deriv_bound: float) -> Interval:
    """Certified tail bound on the dropped boundary term (01-05).

    Analytic bases are not compactly supported. The bound is
    ``2 * |deriv_bound| * tail_bound`` of a unit-scale mollifier at the
    window half-width, never assumed zero.
    """
    if space.window is None:
        raise ValueError("boundary_bound needs a box window")
    lo, hi = space.window
    half = 0.5 * (hi - lo)
    alpha = min(space.scale_at(i) for i in range(space.size))
    spec = MollifierSpec(
        base=space.base,
        scale=1.0 / float(alpha),
        packs=(PackSpec(order=0, mean=0.0, weight=1.0),),
    )
    tail = tail_bound(spec, half_width=half)
    two = Interval.from_rational(2)
    return two * tail.abs() * Interval.point(abs(float(deriv_bound)))


def poly_eval(coeffs: tuple[float, ...], x: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


__all__ = [
    "TestFunctionSpace",
    "WeakForm",
    "boundary_bound",
    "eval_test",
    "exact_moment",
    "poly_eval",
]
