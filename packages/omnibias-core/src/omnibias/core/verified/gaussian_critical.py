# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified cubic charts for globally unique critical two-center decisions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from omnibias.core.gaussian_critical import GaussianCriticalSeries, _build_critical_series
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.verified.gaussian_affine import _dot, _validate_vector
from omnibias.core.verified.gaussian_heat import GaussianHeatMap, _rational
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_gaussian_affine import _intact, _linear
from omnibias.core.verified.inverse_gaussian_heat import _pair

if TYPE_CHECKING:
    from collections.abc import Sequence

    from omnibias.core.proof.certificate import Cert

Rational = int | Fraction
Vector = tuple[Fraction, ...]
Matrix = tuple[Vector, ...]


def _exact(value: Rational, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise TypeError(f"{name} must be exact int or Fraction, excluding bool")
    return Fraction(value)


def _preflight(model: GaussianHeatMap, order: int, arithmetic: _Arithmetic) -> None:
    dimension = model.dimension
    arithmetic._entries(64*(dimension+1)**2 + 64*(order+1) + 1024)
    arithmetic._work(256*(dimension+1)**2 + 128*(order+1)*(dimension+1) + 2048)


def _tail(chart: GaussianCriticalChart, radius: Fraction, arithmetic: _Arithmetic) -> Fraction:
    if chart.series.order == 0:
        return arithmetic.mul(radius, arithmetic.add(Fraction(1), chart.shape_radius))
    ratio = arithmetic._coefficient(radius/chart.parameter_radius)
    squared = arithmetic.mul(ratio, ratio)
    power = Fraction(1)
    for _ in chart.series.z_coefficients:
        power = arithmetic.mul(power, squared)
    return arithmetic._coefficient(arithmetic.mul(arithmetic.mul(radius, chart.shape_radius), power)
                                    /arithmetic.add(Fraction(1), -squared))


def _scalar_bounds(chart: GaussianCriticalChart, parameter: Fraction, arithmetic: _Arithmetic) -> tuple[Fraction, Fraction]:
    if chart.series.z_coefficients:
        squared = arithmetic.mul(parameter, parameter)
        value = chart.series.z_coefficients[-1]
        for coefficient in reversed(chart.series.z_coefficients[:-1]):
            value = arithmetic.add(coefficient, arithmetic.mul(squared, value))
        value = arithmetic.mul(parameter, value)
    else:
        value = Fraction(0)
    tail = _tail(chart, abs(parameter), arithmetic)
    return arithmetic.add(value, -tail), arithmetic.add(value, tail)


def _cube_root_interval(value: Fraction, precision: int, arithmetic: _Arithmetic) -> tuple[Fraction, Fraction]:
    """Enclose the real cube root by an exact, relatively scaled dyadic grid."""
    if value == 0:
        return Fraction(0), Fraction(0)
    arithmetic._work(32*(precision+1))
    if 3*precision+16 > arithmetic.limits.max_coefficient_bits:
        raise ValueError("cube-root precision exceeds the exact coefficient budget")
    magnitude = abs(value)
    exponent = (magnitude.numerator.bit_length()-magnitude.denominator.bit_length())//3-precision
    if abs(3*exponent)+1 > arithmetic.limits.max_coefficient_bits:
        raise ValueError("cube-root scaling exceeds the exact coefficient budget")
    unit = arithmetic._coefficient(Fraction(2)**exponent)
    scaled = arithmetic._coefficient(magnitude/arithmetic.mul(arithmetic.mul(unit, unit), unit))
    integer = scaled.numerator//scaled.denominator
    low, high = 0, 1 << ((integer.bit_length()+2)//3)
    while low+1 < high:
        middle = (low+high)//2
        number = arithmetic._coefficient(Fraction(middle))
        cube = arithmetic.mul(arithmetic.mul(number, number), number)
        if cube <= scaled:
            low = middle
        else:
            high = middle
    low_value = arithmetic.mul(Fraction(low), unit)
    low_cube = arithmetic.mul(arithmetic.mul(low_value, low_value), low_value)
    high_value = low_value if low_cube == magnitude else arithmetic.mul(Fraction(high), unit)
    high_cube = arithmetic.mul(arithmetic.mul(high_value, high_value), high_value)
    if not low_cube <= magnitude <= high_cube:
        raise ArithmeticError("exact cube-root enclosure replay failed")
    return (low_value, high_value) if value > 0 else (-high_value, -low_value)


@dataclass(frozen=True, init=False)
class GaussianCriticalChart:
    r"""Original-model inverse chart at exactly q=1 for two distinct centers.

    Write centers c-a,c+a, alpha=|a|²=2*nu*s, delta=b_plus-b_minus.
    With P=I-aa^T/alpha and w=s*a.target/alpha+delta/2, the UNIQUE GLOBAL
    inverse and target-tilted potential minimizer is
    c+s*P*target+a*(psi(w)-delta/2), where psi inverts y-tanh(y).
    This identity is global on real targets. The numerical certificate is
    local in w: it encloses chi(t)=psi(t³/3) for |t|<1/4, with no restriction
    on exact transverse target components beyond the arithmetic budgets.

    Curvature vanishes in the a direction at w=0. The original inverse has
    no finite derivative in that direction there. The analytic t-chart does
    not imply ordinary target differentiability or regular age derivatives.
    Neither arbitrary mixtures nor q>1 are covered by this certificate.
    """

    model: GaussianHeatMap
    series: GaussianCriticalSeries
    midpoint: Vector
    half_difference: Vector
    squared_half_distance: Fraction
    orthogonal_projector: Matrix
    bias_contrast: Fraction
    parameter_radius: Fraction
    shape_radius: Fraction
    complex_contraction_bound: Fraction
    certificate: Cert

    def __init__(self) -> None:
        raise TypeError("use certify_gaussian_critical_chart")

    def _arithmetic(self) -> _Arithmetic:
        arithmetic = _Arithmetic(self.model.budget)
        _preflight(self.model, self.series.order, arithmetic)
        if not _intact(self.certificate):
            raise ValueError("critical Gaussian chart certificate is damaged")
        return arithmetic

    def _query(
        self, parameter: Rational, transverse_target: Sequence[Rational] | None,
    ) -> tuple[Fraction, Vector, _Arithmetic]:
        value = _exact(parameter, "parameter")
        if abs(value) >= self.parameter_radius:
            raise ValueError("parameter must satisfy |parameter|<parameter_radius")
        vector = (Fraction(0),)*self.model.dimension if transverse_target is None else transverse_target
        _validate_vector(vector, self.model.dimension, "transverse_target")
        arithmetic = self._arithmetic()
        value = arithmetic.exact(value)
        transverse = tuple(arithmetic.exact(entry) for entry in vector)
        if _dot(self.half_difference, transverse, arithmetic):
            raise ValueError("transverse_target must be exactly orthogonal to the center difference")
        return value, transverse, arithmetic

    def target(self, parameter: Rational, transverse_target: Sequence[Rational] | None = None) -> Vector:
        """Return the exact target u+(a/s)*(t³/3-delta/2) of this chart."""
        value, transverse, arithmetic = self._query(parameter, transverse_target)
        cube = arithmetic.mul(arithmetic.mul(value, value), value)
        axial = arithmetic._coefficient(arithmetic.add(arithmetic._coefficient(cube/3),
                                                        -arithmetic._coefficient(self.bias_contrast/2))/self.model.time)
        return tuple(arithmetic.add(normal, arithmetic.mul(center, axial))
                     for normal, center in zip(transverse, self.half_difference, strict=True))

    def tail_bound(self, radius: Rational) -> Fraction:
        """Scalar chi Taylor error on |t|<=radius; multiply by |a_i| per coordinate.

        For an order-zero polynomial it is 9*radius/8. Otherwise the even
        z-series parity gives radius/8*(4*radius)^(2*len(z_coefficients))
        /(1-(4*radius)^2). The requested radius must lie in [0,1/4).
        """
        rho = _exact(radius, "radius")
        if not 0 <= rho < self.parameter_radius:
            raise ValueError("radius must satisfy 0<=radius<parameter_radius")
        arithmetic = self._arithmetic()
        return _tail(self, arithmetic.exact(rho), arithmetic)

    def _lift(
        self, transverse: Vector, bounds: tuple[Fraction, Fraction], arithmetic: _Arithmetic,
    ) -> tuple[Interval, ...]:
        lower, upper = bounds
        half_bias = arithmetic._coefficient(self.bias_contrast/2)
        result = []
        for center, direction, normal in zip(self.midpoint, self.half_difference, transverse, strict=True):
            anchor = arithmetic.add(arithmetic.add(center, arithmetic.mul(self.model.time, normal)),
                                    -arithmetic.mul(direction, half_bias))
            products = arithmetic.mul(direction, lower), arithmetic.mul(direction, upper)
            lo, hi = arithmetic.add(anchor, min(products)), arithmetic.add(anchor, max(products))
            result.append(Interval(_rational(lo).lo, _rational(hi).hi))
        return tuple(result)

    def enclose(self, parameter: Rational, transverse_target: Sequence[Rational] | None = None) -> tuple[Interval, ...]:
        """Enclose the unique global optimizer at target(parameter, transverse_target)."""
        value, transverse, arithmetic = self._query(parameter, transverse_target)
        return self._lift(transverse, _scalar_bounds(self, value, arithmetic), arithmetic)

    def enclose_target(self, target: Sequence[Rational], *, precision: int = 80) -> tuple[Interval, ...]:
        """Enclose the unique optimizer for a supplied exact target near the cusp.

        Require |3*w|<(1/4)^3, with w=s*a.target/|a|²+delta/2. An exact
        dyadic cube-root bracket for t³=3*w uses ``precision`` relative grid
        bits, including near zero. Monotonicity of chi bounds its value by
        certified series enclosures at the two endpoints. No fitted root or
        singular-Jacobian solve is used. If the parameter bracket reaches the
        analytic boundary, raise rather than extrapolate; a higher precision
        may then prove containment. Resource exhaustion and endpoint overflow
        also raise. This method does not certify targets outside the chart.
        """
        _validate_vector(target, self.model.dimension, "target")
        if type(precision) is not int:
            raise TypeError("precision must be an integer, excluding bool")
        if precision < 1:
            raise ValueError("precision must be positive")
        arithmetic = self._arithmetic()
        goal = tuple(arithmetic.exact(value) for value in target)
        projection = _dot(self.half_difference, goal, arithmetic)
        w = arithmetic.add(arithmetic._coefficient(arithmetic.mul(self.model.time, projection)/self.squared_half_distance),
                           arithmetic._coefficient(self.bias_contrast/2))
        radicand = arithmetic.mul(Fraction(3), w)
        radius_cube = arithmetic.mul(arithmetic.mul(self.parameter_radius, self.parameter_radius), self.parameter_radius)
        if abs(radicand) >= radius_cube:
            raise ValueError("target lies outside the certified critical chart")
        lower, upper = _cube_root_interval(radicand, precision, arithmetic)
        if max(abs(lower), abs(upper)) >= self.parameter_radius:
            raise ValueError("cube-root bracket reaches the analytic boundary; increase precision")
        transverse = _linear(self.orthogonal_projector, goal, arithmetic)
        lo = _scalar_bounds(self, lower, arithmetic)[0]
        hi = _scalar_bounds(self, upper, arithmetic)[1]
        return self._lift(transverse, (lo, hi), arithmetic)


def certify_gaussian_critical_chart(model: GaussianHeatMap, *, order: int = 9) -> GaussianCriticalChart | None:
    r"""Bind and certify the critical two-center inverse beyond the q<1 gate.

    Require exactly two distinct centers c-a,c+a and EXACT |a|²=2*nu*s.
    Bias contrast is arbitrary. Scalar y-tanh(y) is strictly increasing and
    surjective because its derivative is tanh²(y), positive except at zero.
    Orthogonal decomposition then proves a unique global inverse and minimum
    for every real target, despite singular curvature at the critical target.

    The numerical chart uses chi(t)=psi(t³/3)=t*z(t), H(t,z)=z³*A(t*z)
    -cosh(t*z), A(y)=3*(y*cosh(y)-sinh(y))/y³ with removable A(0)=1.
    Exact positive-series estimates prove contraction of z-H/3 on |t|<=1/4,
    |z-1|<=1/8. Even parity of z yields the explicit Cauchy tail. All compiled
    coefficients and proof inequalities use bounded rational arithmetic.

    Invalid model types or negative/noninteger order raise. Unsupported model
    families, exhausted budgets and failed exact replay return None. No
    numerical root solve, exponential, square root or matrix factorization
    occurs. The common certificate sealing policy still applies. A digest is
    not an external theorem-prover verification.
    """
    if not isinstance(model, GaussianHeatMap):
        raise TypeError("model must be a GaussianHeatMap")
    if type(order) is not int:
        raise TypeError("order must be an integer, excluding bool")
    if order < 0:
        raise ValueError("order must be nonnegative")
    if order > model.budget.max_order or model.mode_count != 2:
        return None
    arithmetic = _Arithmetic(model.budget)
    try:
        _preflight(model, order, arithmetic)
        direction = tuple(arithmetic._coefficient(arithmetic.add(plus, -minus)/2)
                          for minus, plus in zip(model.centers[0], model.centers[1], strict=True))
        alpha = _dot(direction, direction, arithmetic)
        if alpha <= 0 or alpha != arithmetic.mul(arithmetic.mul(Fraction(2), model.viscosity), model.time):
            return None
        midpoint = tuple(arithmetic.add(origin, delta) for origin, delta in zip(model.centers[0], direction, strict=True))
        bias = arithmetic.add(model.biases[1], -model.biases[0])
        projector = tuple(tuple(arithmetic.add(Fraction(int(i == j)),
                               -arithmetic._coefficient(arithmetic.mul(direction[i], direction[j])/alpha))
                               for j in range(model.dimension)) for i in range(model.dimension))
        if any(_linear(projector, direction, arithmetic)):
            return None
        series = _build_critical_series(order, model.budget, arithmetic)
        radius, shape = Fraction(1, 4), Fraction(1, 8)
        z_bound = arithmetic.add(Fraction(1), shape)
        y_bound = arithmetic.mul(radius, z_bound)
        y_squared = arithmetic.mul(y_bound, y_bound)
        complement = arithmetic.add(Fraction(1), -y_squared)
        z_squared = arithmetic.mul(z_bound, z_bound)
        z_cubed = arithmetic.mul(z_squared, z_bound)
        first = arithmetic.add(arithmetic.mul(Fraction(2), shape), arithmetic.mul(shape, shape))
        second = arithmetic._coefficient(arithmetic.mul(z_squared, y_squared)/arithmetic.mul(Fraction(10), complement))
        third_factor = arithmetic.add(arithmetic._coefficient(arithmetic.mul(radius, z_cubed)/15),
                                       arithmetic._coefficient(radius/3))
        third = arithmetic._coefficient(arithmetic.mul(third_factor, y_bound)/complement)
        contraction = arithmetic.add(arithmetic.add(first, second), third)
        radius_squared = arithmetic.mul(radius, radius)
        defect = arithmetic._coefficient(radius_squared/arithmetic.mul(Fraction(6), arithmetic.add(Fraction(1), -radius_squared)))
        available = arithmetic.mul(arithmetic.add(Fraction(1), -contraction), shape)
        if contraction >= Fraction(1, 3) or defect > available:
            return None
        from omnibias.core.proof.certificate import make_certificate

        certificate = make_certificate(
            claim="Globally unique critical two-center Gaussian inverse and a certified local cubic parameter chart.",
            payload={
                "type": "gaussian_critical_chart", "centers": [[_pair(value) for value in row] for row in model.centers],
                "biases": [_pair(value) for value in model.biases], "viscosity": _pair(model.viscosity), "time": _pair(model.time),
                "midpoint": [_pair(value) for value in midpoint], "half_difference": [_pair(value) for value in direction],
                "squared_half_distance": _pair(alpha), "bias_contrast": _pair(bias),
                "orthogonal_projector": [[_pair(value) for value in row] for row in projector],
                "coefficients": [_pair(value) for value in series.coefficients],
                "z_coefficients": [_pair(value) for value in series.z_coefficients], "order": order,
                "parameter_radius": _pair(radius), "shape_radius": _pair(shape),
                "complex_contraction_bound": _pair(contraction), "contraction_terms": [_pair(value) for value in (first, second, third)],
                "base_defect_bound": _pair(defect), "available_displacement": _pair(available),
                "identities": ["|a|²=2*viscosity*time", "P*a=0; P=I-a*a^T/|a|²",
                               "w=time*a.target/|a|²+bias_contrast/2",
                               "X=c+time*P*target+a*(psi(w)-bias_contrast/2)", "psi(w)-tanh(psi(w))=w"],
                "coefficient_identity": "u=t²; chi=t*z(u); (z+2*u*z')*(z-u/3)²=1; z(0)=1",
                "tail_formula": "order0:9*rho/8; otherwise rho/8*(4*rho)^(2*len(z_coefficients))/(1-(4*rho)^2), 0<=rho<1/4",
                "argument": "The original two-mode Gaussian mean is c+a*tanh(a.(x-c)/|a|²+delta/2). Orthogonal decomposition reduces the inverse to y-tanh(y)=w. Its derivative tanh²(y) is nonnegative and vanishes only at zero, so it is strictly increasing and surjective. The corresponding scalar potential is strictly convex and coercive; transverse directions are positive quadratic. Hence every real target has one global inverse and one global minimizer. The removable entire H(t,z)=z³*A(t*z)-cosh(t*z), A(y)=3*(y*cosh(y)-sinh(y))/y³, has H(0,z)=z³-1. Positive-series bounds A(Y)-1<=(cosh(Y)-1)/5, A'(Y)<=sinh(Y)/5, cosh(Y)-1<=Y²/[2*(1-Y²)] and sinh(Y)<=Y/(1-Y²) give the recorded contraction of T=z-H/3 on |t|<=1/4, |z-1|<=1/8 and its base defect. Uniform holomorphic contraction proves the analytic solution, uniquely even in t. Differentiating chi-tanh(chi)=t³/3 and using tanh(chi)=chi-t³/3 yields (z+2*u*z')*(z-u/3)²=1 for u=t². Each new z coefficient has the nonzero divisor 2*n+3, so this equation uniquely determines the same formal series; streaming two convolutions constructs it with quadratic rational arithmetic work. Cauchy bounds on z-1 with its even parity give the stated chi=t*z tail. On the real line it equals the globally unique inverse at w=t³/3. Exact dyadic cube-root brackets and monotonic chi permit direct target enclosures without a nonsingular target Jacobian.",
                "scope": "Exactly two distinct centers at q=1, fixed viscosity and physical time, fixed arbitrary source bias contrast. Global real uniqueness is proved, but numerical enclosures use only the local critical chart. Transverse targets can be arbitrary within exact budgets. The inverse is not differentiable in the axial target direction at w=0; analytic chart derivatives are derivatives with respect to t. No ordinary inverse hodograph PDE at the singular target, age-derivative regularization, arbitrary-mixture, q>1, or formal-checker verification claim.",
            }, honesty={"unproven_claim": False, "continuum_pde_claim": False}, meta={"transcend_backend": "not_used"},
        )
        result = object.__new__(GaussianCriticalChart)
        for name, stored in (("model", model), ("series", series), ("midpoint", midpoint), ("half_difference", direction),
                            ("squared_half_distance", alpha), ("orthogonal_projector", projector), ("bias_contrast", bias),
                            ("parameter_radius", radius), ("shape_radius", shape), ("complex_contraction_bound", contraction),
                            ("certificate", certificate)):
            object.__setattr__(result, name, stored)
        return result
    except (ArithmeticError, ValueError):
        return None


__all__ = ["GaussianCriticalChart", "certify_gaussian_critical_chart"]
