# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""A fixed-box certified count for a Gaussian de Bruijn-style heat flow.

This module deliberately implements only the elementary entire model

.. math::

    H_t(z) = \int_0^\infty e^{-(1-t)u^2}\cos(zu)\,du,\qquad t < 1.

It obeys the de Bruijn heat equation ``d_t H_t = -d_z^2 H_t`` and uses the
same ``e^{t u^2}`` heat multiplier as a de Bruijn kernel representation.
It is **not** the Riemann-Xi de Bruijn--Newman function, and its fixed-box
counts make no statement about Lambda or the Riemann Hypothesis.

The finite integral is enclosed by the verified composite trapezoid rule.
For a contour image with ``|Im z| <= b`` and ``a = 1-t > 0``, the omitted
tail is bounded by

.. math::

    \int_U^\infty e^{-a u^2+b u}\,du
    \leq \frac{e^{-aU^2+bU}}{2aU-b},\qquad 2aU>b.

Thus the contract records one rectangle, truncation, and panel count; no
unproved far-field premise is hidden in the winding calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.core.collapse.winding import (
    contour_parameter_segments,
    integers_in,
    winding_enclosure_function,
)
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.quadrature import trapezoid_integral
from omnibias.core.verified.transcend import (
    cos_iv,
    exp_iv,
    require_rigorous_backend,
    sin_iv,
)


def _cos_complex(z: ComplexInterval) -> ComplexInterval:
    """Enclose ``cos(z)`` from real interval transcendental primitives."""
    cosh = (exp_iv(z.im) + exp_iv(-z.im)) * 0.5
    sinh = (exp_iv(z.im) - exp_iv(-z.im)) * 0.5
    return ComplexInterval(cos_iv(z.re) * cosh, -sin_iv(z.re) * sinh)


def _sin_complex(z: ComplexInterval) -> ComplexInterval:
    """Enclose ``sin(z)`` from real interval transcendental primitives."""
    cosh = (exp_iv(z.im) + exp_iv(-z.im)) * 0.5
    sinh = (exp_iv(z.im) - exp_iv(-z.im)) * 0.5
    return ComplexInterval(sin_iv(z.re) * cosh, cos_iv(z.re) * sinh)


def _integrand(z: ComplexInterval, u: Interval, a: float) -> ComplexInterval:
    q = exp_iv(-Interval.point(a) * u.pow_int(2))
    return ComplexInterval.from_value(q) * _cos_complex(z * ComplexInterval.from_value(u))


def _integrand_second_derivative(
    z: ComplexInterval, u: Interval, a: float
) -> ComplexInterval:
    """Enclose the second u-derivative of the truncated integrand."""
    a_iv = Interval.point(a)
    u_ci = ComplexInterval.from_value(u)
    zu = z * u_ci
    coefficient = (
        ComplexInterval.from_value(Interval.point(4.0) * a_iv * a_iv * u.pow_int(2) - 2.0 * a_iv)
        - z * z
    )
    derivative_term = (
        ComplexInterval.from_value(Interval.point(4.0) * a_iv * u)
        * z
        * _sin_complex(zu)
    )
    q = ComplexInterval.from_value(exp_iv(-a_iv * u.pow_int(2)))
    return q * (coefficient * _cos_complex(zu) + derivative_term)


@dataclass(frozen=True)
class GaussianHeatKernelContract:
    """Immutable finite contract for one rectangle and one quadrature tail."""

    t: float
    center: complex
    half_width: float
    half_height: float
    truncation: float
    panels: int = 256
    contour_segments: int = 64

    def __post_init__(self) -> None:
        if not math.isfinite(self.t) or self.t >= 1.0:
            raise ValueError("t must be finite and < 1 for the Gaussian tail bound")
        if self.half_width <= 0.0 or not math.isfinite(self.half_width):
            raise ValueError("half_width must be finite and > 0")
        if self.half_height <= 0.0 or not math.isfinite(self.half_height):
            raise ValueError("half_height must be finite and > 0")
        if self.truncation <= 0.0 or not math.isfinite(self.truncation):
            raise ValueError("truncation must be finite and > 0")
        if self.panels < 1:
            raise ValueError("panels must be >= 1")
        if self.contour_segments < 4 or self.contour_segments % 4:
            raise ValueError("contour_segments must be divisible by 4 and >= 4")
        b = abs(self.center.imag) + self.half_height
        if 2.0 * (1.0 - self.t) * self.truncation <= b:
            raise ValueError("truncation does not satisfy 2(1-t)U > max |Im z|")

    @property
    def a(self) -> float:
        """The positive Gaussian decay coefficient ``1-t``."""
        return 1.0 - self.t

    @property
    def max_imaginary_part(self) -> float:
        """Maximum ``|Im z|`` across the declared rectangle."""
        return abs(self.center.imag) + self.half_height


@dataclass(frozen=True)
class HeatKernelZeroCount:
    """Result of one fixed-box argument-principle computation."""

    count: int | None
    winding: Interval | None
    certified: bool
    detail: str
    contract: GaussianHeatKernelContract


def gaussian_heat_kernel_tail_bound(
    z: ComplexInterval, contract: GaussianHeatKernelContract
) -> Interval:
    """Rigorous upper bound on the omitted integral's modulus for ``z``."""
    b = z.im.mag
    denom = 2.0 * contract.a * contract.truncation - b
    if denom <= 0.0:
        raise ValueError("tail bound requires 2(1-t)U > max |Im z| on each image box")
    exponent = -contract.a * contract.truncation**2 + b * contract.truncation
    return exp_iv(Interval.point(exponent)) / Interval.point(denom)


def gaussian_heat_kernel_enclosure(
    z: ComplexInterval, contract: GaussianHeatKernelContract
) -> ComplexInterval:
    """Enclose the declared Gaussian heat kernel over a complex rectangle."""
    real_panels: list[Interval] = []
    imag_panels: list[Interval] = []
    for index in range(contract.panels):
        u0 = contract.truncation * index / contract.panels
        u1 = contract.truncation * (index + 1) / contract.panels
        left = _integrand(z, Interval.point(u0), contract.a)
        right = _integrand(z, Interval.point(u1), contract.a)
        derivative = _integrand_second_derivative(z, Interval(u0, u1), contract.a)
        real_panels.append(trapezoid_integral([left.re, right.re], u0, u1, derivative.re))
        imag_panels.append(trapezoid_integral([left.im, right.im], u0, u1, derivative.im))
    real = sum_intervals(real_panels)
    imag = sum_intervals(imag_panels)
    tail = gaussian_heat_kernel_tail_bound(z, contract).hi
    return ComplexInterval(
        real + Interval(-tail, tail),
        imag + Interval(-tail, tail),
    )


def count_gaussian_heat_kernel_zeros(
    contract: GaussianHeatKernelContract,
) -> HeatKernelZeroCount:
    """Certify the zeros of the declared Gaussian model inside one rectangle."""
    require_rigorous_backend()
    domains = contour_parameter_segments(
        contract.center,
        contract.half_width,
        segments=contract.contour_segments,
        contour="rectangle",
        half_width=contract.half_width,
        half_height=contract.half_height,
    )
    winding = winding_enclosure_function(
        lambda z: gaussian_heat_kernel_enclosure(z, contract),
        contract.center,
        contract.half_width,
        segments=contract.contour_segments,
        contour="rectangle",
        half_width=contract.half_width,
        half_height=contract.half_height,
    )
    if winding is None:
        return HeatKernelZeroCount(
            None,
            None,
            False,
            "contour image may contain zero; fixed contract is inconclusive",
            contract,
        )
    hits = integers_in(winding)
    if len(hits) != 1:
        return HeatKernelZeroCount(
            None,
            winding,
            False,
            "winding enclosure did not isolate a unique integer",
            contract,
        )
    # Retaining this explicit cover guards future refactors from treating a
    # pointwise evaluator as a contour certificate.
    assert len(domains) == contract.contour_segments
    return HeatKernelZeroCount(
        hits[0],
        winding,
        True,
        "fixed-rectangle Gaussian heat-kernel zero count certified",
        contract,
    )


__all__ = [
    "GaussianHeatKernelContract",
    "HeatKernelZeroCount",
    "count_gaussian_heat_kernel_zeros",
    "gaussian_heat_kernel_enclosure",
    "gaussian_heat_kernel_tail_bound",
]
