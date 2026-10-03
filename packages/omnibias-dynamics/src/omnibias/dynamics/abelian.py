# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Certified Abelian-integral zero counts for a cubic elliptic Hamiltonian.

The supported phase-one problem is

.. math::

   H=y^2+x^3+p x+q,\qquad
   \omega=(\alpha(H)+\beta(H)x)y\,dx,

with exact rational ``p < 0``, ``q`` and polynomials ``alpha`` and ``beta``. Exact
Picard--Fuchs algebra comes from :mod:`omnibias.holonomic`; initial periods
are enclosed by nonsingular interval quadrature after the cosine turning-point
substitution; complex values are enclosed by validated D-finite continuation.

The upper count is the argument-principle winding on one declared regular
rectangle.  The lower count is a tuple of disjoint Krawczyk simple-zero boxes.
Only equality of those independently certified counts earns an exact
instance count.  This is not a degree-uniform bound, not ``H(2) < infinity``,
and not a solution of Hilbert's sixteenth problem.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.collapse.winding import winding_collapse_function
from omnibias.core.proof.certificate import (
    Cert,
    make_certificate,
    verify_certificate_digest,
)
from omnibias.core.proof.lean_check import LeanCheckResult, check_certificate
from omnibias.core.proof.realization_replay import source_digest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.dfinite_continuation import (
    continue_dfinite,
    evaluate_complex_polynomial,
    realify_complex_matrix,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    KrawczykCertificate,
    krawczyk_certificate,
)
from omnibias.core.verified.linalg import matvec
from omnibias.core.verified.lohner import interval_matrix_exp
from omnibias.core.verified.transcend import PI_IV, cos_iv, sin_iv
from omnibias.holonomic import (
    PicardFuchsCertificate,
    certify_picard_fuchs,
    verify_picard_fuchs,
)
from omnibias.holonomic._core.rational_poly import (
    Poly,
    pderiv,
    peval,
    to_poly,
)

Rational = Fraction | int


def _fraction(value: Rational) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError("Abelian problem data must be exact rationals")
    return Fraction(value)


def _q(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _poly_payload(polynomial: Poly) -> list[list[int]]:
    return [_q(value) for value in polynomial]


@dataclass(frozen=True)
class AbelianProblem:
    """Exact cubic Hamiltonian, mixed-form factors, and rational base oval."""

    p: Fraction
    q: Fraction
    energy_factor: Poly
    base_h: Fraction
    base_roots: tuple[Fraction, Fraction, Fraction]
    quadrature_panels: int = 1024
    continuation_steps: int = 24
    continuation_order: int = 16
    moment_factor: Poly = ()
    declared_factor_roots: tuple[Fraction, ...] = ()

    def __post_init__(self) -> None:
        if self.p >= 0:
            raise ValueError("the real elliptic oval requires p < 0")
        if not self.energy_factor and not self.moment_factor:
            raise ValueError("at least one of energy_factor and moment_factor must be nonzero")
        if len(set(self.declared_factor_roots)) != len(self.declared_factor_roots):
            raise ValueError("declared_factor_roots must be distinct")
        for root in self.declared_factor_roots:
            if (
                peval(self.energy_factor, root) != 0
                and peval(self.moment_factor, root) != 0
            ):
                raise ValueError("every declared factor root must zero alpha or beta")
        if tuple(sorted(self.base_roots)) != self.base_roots:
            raise ValueError("base_roots must be strictly increasing")
        if len(set(self.base_roots)) != 3:
            raise ValueError("base_roots must be distinct")
        for root in self.base_roots:
            if root**3 + self.p * root + self.q != self.base_h:
                raise ValueError("each base root must exactly solve V(x)=base_h")
        if self.quadrature_panels < 16:
            raise ValueError("quadrature_panels must be >= 16")
        if self.continuation_steps < 1:
            raise ValueError("continuation_steps must be positive")
        if self.continuation_order < 4:
            raise ValueError("continuation_order must be >= 4")
        critical = 27 * (self.base_h - self.q) ** 2 + 4 * self.p**3
        if critical >= 0:
            raise ValueError("base_h must lie strictly inside the real period annulus")

    @classmethod
    def create(
        cls,
        *,
        p: Rational,
        q: Rational = 0,
        energy_factor: Sequence[Rational] = (1,),
        base_h: Rational,
        base_roots: Sequence[Rational],
        quadrature_panels: int = 1024,
        continuation_steps: int = 24,
        continuation_order: int = 16,
        moment_factor: Sequence[Rational] = (),
        declared_factor_roots: Sequence[Rational] = (),
    ) -> AbelianProblem:
        roots = tuple(_fraction(value) for value in base_roots)
        if len(roots) != 3:
            raise ValueError("exactly three base roots are required")
        return cls(
            p=_fraction(p),
            q=_fraction(q),
            energy_factor=to_poly(tuple(_fraction(value) for value in energy_factor)),
            base_h=_fraction(base_h),
            base_roots=(roots[0], roots[1], roots[2]),
            quadrature_panels=quadrature_panels,
            continuation_steps=continuation_steps,
            continuation_order=continuation_order,
            moment_factor=to_poly(tuple(_fraction(value) for value in moment_factor)),
            declared_factor_roots=tuple(
                _fraction(value) for value in declared_factor_roots
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "p": _q(self.p),
            "q": _q(self.q),
            "energy_factor": _poly_payload(self.energy_factor),
            "base_h": _q(self.base_h),
            "base_roots": [_q(value) for value in self.base_roots],
            "quadrature_panels": self.quadrature_panels,
            "continuation_steps": self.continuation_steps,
            "continuation_order": self.continuation_order,
            "moment_factor": _poly_payload(self.moment_factor),
            "declared_factor_roots": [
                _q(value) for value in self.declared_factor_roots
            ],
        }


def named_cubic_abelian_problem(
    *,
    quadrature_panels: int = 1024,
    continuation_steps: int = 24,
    continuation_order: int = 16,
) -> AbelianProblem:
    """The forced-factor cubic instance ``r(h)=h^2-1/64`` at energy zero."""

    return AbelianProblem.create(
        p=-1,
        q=0,
        energy_factor=(Fraction(-1, 64), 0, 1),
        base_h=0,
        base_roots=(-1, 0, 1),
        quadrature_panels=quadrature_panels,
        continuation_steps=continuation_steps,
        continuation_order=continuation_order,
        declared_factor_roots=(Fraction(-1, 8), Fraction(1, 8)),
    )


def named_genuine_cubic_abelian_problem(
    *,
    moment_weight: Rational = Fraction(1, 1000),
    quadrature_panels: int = 1024,
    continuation_steps: int = 24,
    continuation_order: int = 16,
) -> AbelianProblem:
    """A mixed one-form ``(h^2-1/64)y dx + epsilon*x*y dx``.

    The nonzero second coefficient destroys the planted factorization by
    ``h^2-1/64``.  The declared factor roots are retained only so a certified
    root box can prove that neither old planted root survived.
    """
    weight = _fraction(moment_weight)
    if weight == 0:
        raise ValueError("the genuine mixed problem requires beta != 0")
    return AbelianProblem.create(
        p=-1,
        q=0,
        energy_factor=(Fraction(-1, 64), 0, 1),
        moment_factor=(weight,),
        declared_factor_roots=(Fraction(-1, 8), Fraction(1, 8)),
        base_h=0,
        base_roots=(-1, 0, 1),
        quadrature_panels=quadrature_panels,
        continuation_steps=continuation_steps,
        continuation_order=continuation_order,
    )


def _interval_riemann_integral(
    function: Callable[[Interval], Interval], panels: int
) -> Interval:
    """Outward range-sum enclosure of ``integral_0^pi function(theta) dtheta``."""

    width = PI_IV / Interval.from_rational(panels)
    total = Interval.point(0.0)
    for index in range(panels):
        left = PI_IV * Interval.from_rational(Fraction(index, panels))
        right = PI_IV * Interval.from_rational(Fraction(index + 1, panels))
        theta = Interval(left.lo, right.hi)
        total = total + function(theta) * width
    return total


@dataclass(frozen=True)
class AbelianInitialData:
    """Validated base periods and initial derivative jets."""

    area: Interval
    moment: Interval
    j0: Interval
    j1: Interval
    area_jet: tuple[ComplexInterval, ...]
    integral_jet: tuple[ComplexInterval, ...]


@dataclass(frozen=True)
class ActionZeroFreeRectangleCertificate:
    """Exact reduction of a rectangle to the cubic action's zero-free strip."""

    left: Fraction
    right: Fraction
    critical_offset_squared: Fraction
    left_inside: bool
    right_inside: bool
    verified: bool

    def to_payload(self) -> dict[str, object]:
        """Return the finite exact domain checks stored in a count seal."""

        return {
            "left": _q(self.left),
            "right": _q(self.right),
            "critical_offset_squared": _q(self.critical_offset_squared),
            "left_inside": self.left_inside,
            "right_inside": self.right_inside,
            "verified": self.verified,
            "analytic_input": (
                "Euler representation of "
                "z*2F1(1/6,5/6;2;z) on C minus [1,infinity)"
            ),
        }


def certify_action_zero_free_rectangle(
    problem: AbelianProblem,
    *,
    contour_center: complex,
    half_width: float,
) -> ActionZeroFreeRectangleCertificate:
    r"""Prove that a rectangle lies in the distinguished action's zero-free strip.

    For ``s=sqrt(-p/3)`` and
    ``z=(h-(q-2s**3))/(4s**3)``, the Hamiltonian-oriented action is a
    positive constant times ``z*2F1(1/6,5/6;2;z)``.  Euler's integral
    representation makes the hypergeometric factor zero-free on the cut
    plane.  The exact checks below establish that the whole rectangle lies
    strictly between the two critical real energies, hence excludes both the
    center zero and the saddle cut.
    """

    if half_width <= 0.0 or not math.isfinite(half_width):
        raise ValueError("rectangle half-width must be positive and finite")
    if not math.isfinite(contour_center.real) or not math.isfinite(
        contour_center.imag
    ):
        raise ValueError("contour center must be finite")

    real_hull = Interval.point(contour_center.real) + Interval(
        -half_width,
        half_width,
    )
    left = Fraction.from_float(real_hull.lo)
    right = Fraction.from_float(real_hull.hi)
    critical_offset_squared = Fraction(-4 * problem.p**3, 27)

    left_inside = left >= problem.q or (
        (problem.q - left) ** 2 < critical_offset_squared
    )
    right_inside = right <= problem.q or (
        (right - problem.q) ** 2 < critical_offset_squared
    )
    return ActionZeroFreeRectangleCertificate(
        left=left,
        right=right,
        critical_offset_squared=critical_offset_squared,
        left_inside=left_inside,
        right_inside=right_inside,
        verified=bool(left_inside and right_inside),
    )


def certify_cubic_initial_data(
    problem: AbelianProblem,
    picard_fuchs: PicardFuchsCertificate | None = None,
) -> AbelianInitialData:
    """Enclose ``A``, ``B``, ``J0`` and ``J1`` after removing endpoint singularities.

    Here ``B = integral x*y dx`` on the same Hamiltonian-oriented oval.
    """

    pf = (
        certify_picard_fuchs(
            problem.p,
            problem.q,
            energy_factor=problem.energy_factor,
        )
        if picard_fuchs is None
        else picard_fuchs
    )
    if not pf.verified or not verify_picard_fuchs(pf):
        raise ValueError("an exact Picard--Fuchs certificate is required")

    root0, root1, root2 = problem.base_roots
    midpoint = Interval.from_rational((root1 + root2) / 2)
    radius = Interval.from_rational((root2 - root1) / 2)
    left_root = Interval.from_rational(root0)

    def coordinates(theta: Interval) -> tuple[Interval, Interval, Interval]:
        sine = sin_iv(theta)
        x = midpoint + radius * cos_iv(theta)
        gap = x - left_root
        if gap.lo <= 0.0:
            raise ArithmeticError("turning-point substitution lost the positive root gap")
        return sine, x, gap.sqrt()

    area_scale = Interval.from_rational(2) * radius.pow_int(2)

    def area_integrand(theta: Interval) -> Interval:
        sine, _x, root_gap = coordinates(theta)
        return area_scale * sine.pow_int(2) * root_gap

    def moment_integrand(theta: Interval) -> Interval:
        sine, x, root_gap = coordinates(theta)
        return area_scale * x * sine.pow_int(2) * root_gap

    def j0_integrand(theta: Interval) -> Interval:
        _sine, _x, root_gap = coordinates(theta)
        return Interval.from_rational(2) / root_gap

    def j1_integrand(theta: Interval) -> Interval:
        _sine, x, root_gap = coordinates(theta)
        return Interval.from_rational(2) * x / root_gap

    area = _interval_riemann_integral(area_integrand, problem.quadrature_panels)
    moment = _interval_riemann_integral(moment_integrand, problem.quadrature_panels)
    j0 = _interval_riemann_integral(j0_integrand, problem.quadrature_panels)
    j1 = _interval_riemann_integral(j1_integrand, problem.quadrature_panels)
    if area.lo <= 0.0 or j0.lo <= 0.0:
        raise ArithmeticError("validated oval area and period must be strictly positive")

    e = problem.base_h - problem.q
    area_from_basis = (
        Interval.from_rational(3 * e) * j0
        - Interval.from_rational(2 * problem.p) * j1
    ) / Interval.from_rational(5)
    moment_from_basis = (
        Interval.from_rational(Fraction(2 * problem.p**2, 21)) * j0
        + Interval.from_rational(Fraction(3 * e, 7)) * j1
    )
    try:
        area.intersect(area_from_basis)
        moment.intersect(moment_from_basis)
    except ValueError as exc:
        raise ArithmeticError(
            "base quadrature contradicts the exact cubic period reduction"
        ) from exc

    discriminant = 27 * e**2 + 4 * problem.p**3
    j0_prime = (
        Interval.from_rational(-9 * e) * j0
        + Interval.from_rational(6 * problem.p) * j1
    ) / Interval.from_rational(2 * discriminant)
    area_real_jet = (
        area,
        j0 / Interval.from_rational(2),
        j0_prime / Interval.from_rational(2),
    )
    if pf.area_operator.order != len(area_real_jet):
        raise ArithmeticError("unexpected area Picard--Fuchs order")

    factor_derivatives: list[Poly] = [problem.energy_factor]
    for _ in range(len(area_real_jet) - 1):
        factor_derivatives.append(pderiv(factor_derivatives[-1]))
    integral_real_jet: list[Interval] = []
    for order in range(pf.integral_operator.order):
        value = Interval.point(0.0)
        for factor_order in range(order + 1):
            derivative_value = peval(
                factor_derivatives[factor_order],
                problem.base_h,
            )
            value = value + (
                Interval.from_rational(math.comb(order, factor_order) * derivative_value)
                * area_real_jet[order - factor_order]
            )
        integral_real_jet.append(value)
    if len(integral_real_jet) != pf.integral_operator.order:
        raise ArithmeticError("failed to build the integral initial jet")

    return AbelianInitialData(
        area=area,
        moment=moment,
        j0=j0,
        j1=j1,
        area_jet=tuple(ComplexInterval.from_parts(value) for value in area_real_jet),
        integral_jet=tuple(ComplexInterval.from_parts(value) for value in integral_real_jet),
    )


@dataclass(frozen=True)
class AbelianEvaluator:
    """Certified complex evaluator backed by one Picard--Fuchs initial-value problem."""

    problem: AbelianProblem
    picard_fuchs: PicardFuchsCertificate
    initial_data: AbelianInitialData

    def evaluate(self, target: ComplexInterval | Interval | complex | float) -> tuple[ComplexInterval, ...]:
        target_box = ComplexInterval.from_value(target)
        result = continue_dfinite(
            self.picard_fuchs.area_operator.coeffs,
            float(self.problem.base_h),
            self.initial_data.area_jet,
            target_box,
            n_steps=self.problem.continuation_steps,
            order=self.problem.continuation_order,
        )
        factor_derivatives: list[Poly] = [self.problem.energy_factor]
        for _ in range(len(result.jet) - 1):
            factor_derivatives.append(pderiv(factor_derivatives[-1]))
        integral_jet: list[ComplexInterval] = []
        for derivative_order in range(len(result.jet)):
            value = ComplexInterval.zero()
            for factor_order in range(derivative_order + 1):
                value = value + (
                    math.comb(derivative_order, factor_order)
                    * evaluate_complex_polynomial(
                        factor_derivatives[factor_order],
                        target_box,
                    )
                    * result.jet[derivative_order - factor_order]
                )
            integral_jet.append(value)
        return tuple(integral_jet)

    def evaluate_real(self, target: Interval, derivative: int = 0) -> Interval:
        """Enclose a real derivative on a real energy interval."""

        jet = self.evaluate(ComplexInterval.from_parts(target))
        if not 0 <= derivative < len(jet):
            raise ValueError("derivative order exceeds the Picard--Fuchs jet")
        if not jet[derivative].im.contains_zero():
            raise ArithmeticError("real continuation lost conjugation symmetry")
        return jet[derivative].re


def _complex_rational(value: Rational) -> ComplexInterval:
    return ComplexInterval.from_parts(Interval.from_rational(_fraction(value)))


def _gauss_manin_period_matrix(
    problem: AbelianProblem,
    energy: ComplexInterval,
) -> tuple[tuple[ComplexInterval, ...], ...]:
    """The exact rank-two Gauss--Manin system on ``(J0, J1)``."""
    e = energy - _complex_rational(problem.q)
    discriminant = 27 * e * e + _complex_rational(4 * problem.p**3)
    if discriminant.modulus().lo <= 0.0:
        raise ZeroDivisionError("Gauss--Manin path may meet a critical energy")
    denominator = 2 * discriminant
    return (
        (
            (-9 * e) / denominator,
            _complex_rational(6 * problem.p) / denominator,
        ),
        (
            _complex_rational(2 * problem.p**2) / denominator,
            (9 * e) / denominator,
        ),
    )


def _gauss_manin_component_matrix(
    problem: AbelianProblem,
    energy: ComplexInterval,
) -> tuple[tuple[ComplexInterval, ...], ...]:
    """An overcomplete four-component representation on ``(A, B, J0, J1)``."""
    zero = ComplexInterval.zero()
    half = _complex_rational(Fraction(1, 2))
    period = _gauss_manin_period_matrix(problem, energy)
    return (
        (zero, zero, half, zero),
        (zero, zero, zero, half),
        (zero, zero, *period[0]),
        (zero, zero, *period[1]),
    )


def _period_state_from_basis(
    problem: AbelianProblem,
    energy: ComplexInterval,
    j0: ComplexInterval,
    j1: ComplexInterval,
) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval]:
    """Recover ``(A, B, J0, J1)`` from the exact cubic period identities."""
    e = energy - _complex_rational(problem.q)
    area = (
        _complex_rational(3) * e * j0
        - _complex_rational(2 * problem.p) * j1
    ) / _complex_rational(5)
    moment = (
        _complex_rational(Fraction(2 * problem.p**2, 21)) * j0
        + _complex_rational(Fraction(3, 7)) * e * j1
    )
    return area, moment, j0, j1


def _realify_state(values: Sequence[ComplexInterval]) -> list[Interval]:
    return [value.re for value in values] + [value.im for value in values]


def _complexify_state(values: Sequence[Interval]) -> tuple[ComplexInterval, ...]:
    if len(values) % 2:
        raise ValueError("a realified period state must have even dimension")
    size = len(values) // 2
    return tuple(
        ComplexInterval(values[index], values[index + size])
        for index in range(size)
    )


def _continue_period_state(
    problem: AbelianProblem,
    initial_data: AbelianInitialData,
    target: ComplexInterval,
) -> tuple[ComplexInterval, ...]:
    start = ComplexInterval.point(complex(float(problem.base_h), 0.0))
    delta = target - start
    whole_path = start + delta * ComplexInterval.from_value(Interval(0.0, 1.0))
    _gauss_manin_component_matrix(problem, whole_path)
    state = _realify_state((
        ComplexInterval.from_parts(initial_data.area),
        ComplexInterval.from_parts(initial_data.moment),
        ComplexInterval.from_parts(initial_data.j0),
        ComplexInterval.from_parts(initial_data.j1),
    ))
    step_size = Interval.from_rational(Fraction(1, problem.continuation_steps))
    for index in range(problem.continuation_steps):
        path_parameter = Interval.hull(
            Interval.from_rational(Fraction(index, problem.continuation_steps)),
            Interval.from_rational(Fraction(index + 1, problem.continuation_steps)),
        )
        energy_box = start + delta * ComplexInterval.from_value(path_parameter)
        generator = tuple(
            tuple(value * delta for value in row)
            for row in _gauss_manin_component_matrix(problem, energy_box)
        )
        real_generator = [
            [value * step_size for value in row]
            for row in realify_complex_matrix(generator)
        ]
        state = matvec(
            interval_matrix_exp(
                real_generator,
                order=problem.continuation_order,
            ),
            state,
        )
    return _complexify_state(state)


@dataclass(frozen=True)
class AbelianPeriodEvaluator:
    """Four-component view of the rank-two cubic Gauss--Manin system."""

    problem: AbelianProblem
    picard_fuchs: PicardFuchsCertificate
    initial_data: AbelianInitialData

    def evaluate_state(
        self,
        target: ComplexInterval | Interval | complex | float,
    ) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval]:
        """Enclose ``(A, B, J0, J1)`` at every point of ``target``."""
        values = _continue_period_state(
            self.problem,
            self.initial_data,
            ComplexInterval.from_value(target),
        )
        if len(values) != 4:
            raise ArithmeticError("four-component period view has the wrong size")
        reduced = _period_state_from_basis(
            self.problem,
            ComplexInterval.from_value(target),
            values[2],
            values[3],
        )
        try:
            values[0].re.intersect(reduced[0].re)
            values[0].im.intersect(reduced[0].im)
            values[1].re.intersect(reduced[1].re)
            values[1].im.intersect(reduced[1].im)
        except ValueError as exc:
            raise ArithmeticError(
                "four-component propagation violates the exact rank-two reduction"
            ) from exc
        return values[0], values[1], values[2], values[3]

    def evaluate(
        self,
        target: ComplexInterval | Interval | complex | float,
        derivative: int = 0,
    ) -> ComplexInterval:
        """Enclose the mixed Abelian integral or its first derivative."""
        if derivative not in (0, 1):
            raise ValueError("the mixed-period evaluator supports derivatives zero and one")
        target_box = ComplexInterval.from_value(target)
        area, moment, j0, j1 = self.evaluate_state(target_box)
        alpha = evaluate_complex_polynomial(self.problem.energy_factor, target_box)
        beta = evaluate_complex_polynomial(self.problem.moment_factor, target_box)
        if derivative == 0:
            return alpha * area + beta * moment
        alpha_prime = evaluate_complex_polynomial(
            pderiv(self.problem.energy_factor),
            target_box,
        )
        beta_prime = evaluate_complex_polynomial(
            pderiv(self.problem.moment_factor),
            target_box,
        )
        return (
            alpha_prime * area
            + alpha * j0 / 2
            + beta_prime * moment
            + beta * j1 / 2
        )

    def evaluate_real(self, target: Interval, derivative: int = 0) -> Interval:
        """Enclose a real value/derivative and check conjugation symmetry."""
        value = self.evaluate(ComplexInterval.from_parts(target), derivative)
        if not value.im.contains_zero():
            raise ArithmeticError("real continuation lost conjugation symmetry")
        return value.re


def build_abelian_evaluator(problem: AbelianProblem) -> AbelianEvaluator:
    """Build the exact operator and validated base data for ``problem``."""

    pf = certify_picard_fuchs(
        problem.p,
        problem.q,
        energy_factor=problem.energy_factor,
    )
    if not pf.verified:
        raise ArithmeticError("Picard--Fuchs certification failed")
    initial = certify_cubic_initial_data(problem, pf)
    return AbelianEvaluator(problem, pf, initial)


def build_abelian_period_evaluator(problem: AbelianProblem) -> AbelianPeriodEvaluator:
    """Build the overcomplete four-component view of the rank-two period system."""
    factor = problem.energy_factor if problem.energy_factor else (Fraction(1),)
    pf = certify_picard_fuchs(problem.p, problem.q, energy_factor=factor)
    if not pf.verified:
        raise ArithmeticError("Picard--Fuchs certification failed")
    initial = certify_cubic_initial_data(problem, pf)
    return AbelianPeriodEvaluator(problem, pf, initial)


def certify_simple_abelian_zero(
    evaluator: AbelianEvaluator,
    root: Rational,
    *,
    radius: float = 1e-3,
) -> KrawczykCertificate:
    """Prove one declared exact factor root is a unique simple zero."""

    exact_root = _fraction(root)
    if peval(evaluator.problem.energy_factor, exact_root) != 0:
        raise ValueError("the declared root must exactly zero the energy factor")

    root_float = float(exact_root)
    point_derivative = evaluator.evaluate_real(Interval.point(root_float), 1)
    if point_derivative.contains_zero():
        raise ArithmeticError("the Abelian derivative is not separated from zero")
    inverse = 1.0 / point_derivative.mid

    def function(values: list[Interval]) -> list[Interval]:
        return [evaluator.evaluate_real(values[0], 0)]

    def jacobian(values: list[Interval]) -> list[list[Interval]]:
        return [[evaluator.evaluate_real(values[0], 1)]]

    certificate = krawczyk_certificate(
        function,
        jacobian,
        [root_float],
        [[inverse]],
        radius,
    )
    if certificate is None:
        raise ArithmeticError("Krawczyk did not certify the declared simple zero")
    return certificate


def certify_simple_mixed_abelian_zero(
    evaluator: AbelianPeriodEvaluator,
    center: Rational,
    *,
    radius: float = 1e-3,
) -> KrawczykCertificate:
    """Prove a unique simple zero near an exact rational proposal center."""
    exact_center = _fraction(center)
    center_float = float(exact_center)
    point_derivative = evaluator.evaluate_real(
        Interval.point(center_float),
        1,
    )
    if point_derivative.contains_zero():
        raise ArithmeticError("the mixed Abelian derivative is not separated from zero")
    inverse = 1.0 / point_derivative.mid

    def function(values: list[Interval]) -> list[Interval]:
        return [evaluator.evaluate_real(values[0], 0)]

    def jacobian(values: list[Interval]) -> list[list[Interval]]:
        return [[evaluator.evaluate_real(values[0], 1)]]

    certificate = krawczyk_certificate(
        function,
        jacobian,
        [center_float],
        [[inverse]],
        radius,
    )
    if certificate is None:
        raise ArithmeticError("Krawczyk did not certify the mixed simple zero")
    return certificate


def _declared_factor_roots_excluded(
    problem: AbelianProblem,
    zeros: Sequence[KrawczykCertificate],
) -> bool:
    if not problem.declared_factor_roots:
        return False
    for certificate in zeros:
        if len(certificate.enclosure) != 1:
            return False
        lo, hi = certificate.enclosure[0]
        exact_lo, exact_hi = Fraction.from_float(lo), Fraction.from_float(hi)
        if any(exact_lo <= root <= exact_hi for root in problem.declared_factor_roots):
            return False
    return True


@dataclass(frozen=True)
class AbelianZeroCountCertificate:
    """Two-sided exact zero count on one declared complex rectangle."""

    problem: AbelianProblem
    contour_center: complex
    half_width: float
    half_height: float
    segments: int
    max_segments: int
    root_guesses: tuple[Fraction, ...]
    root_radius: float
    source_digest: str
    action_zero_free_domain: ActionZeroFreeRectangleCertificate
    picard_fuchs: PicardFuchsCertificate
    initial_data: AbelianInitialData
    upper_status: str
    upper_count: int | None
    winding_enclosure: Interval | None
    lower_zeros: tuple[KrawczykCertificate, ...]
    exact_count: int | None
    seal: Cert | None


@dataclass(frozen=True)
class AbelianFormalVerification:
    """Lean results for the two finite rational obligations only."""

    picard_fuchs: LeanCheckResult
    winding_integer: LeanCheckResult
    theorem_prover_verified: bool


def _count_source(
    problem: AbelianProblem,
    center: complex,
    half_width: float,
    half_height: float,
    segments: int,
    max_segments: int,
    roots: Sequence[Fraction],
    root_radius: float,
) -> dict[str, object]:
    return {
        "problem": problem.to_payload(),
        "contour": {
            "center": [center.real, center.imag],
            "half_width": half_width,
            "half_height": half_height,
            "segments": segments,
            "max_segments": max_segments,
        },
        "roots": [_q(root) for root in roots],
        "root_radius": root_radius,
    }


def certify_abelian_zero_count(
    problem: AbelianProblem,
    *,
    contour_center: complex = 0j,
    half_width: float = 0.2,
    half_height: float = 0.04,
    root_guesses: Sequence[Rational] = (Fraction(-1, 8), Fraction(1, 8)),
    root_radius: float = 1e-3,
    segments: int = 16,
    max_segments: int = 64,
) -> AbelianZeroCountCertificate:
    """Certify matching complex upper and real lower zero counts.

    When ``moment_factor`` is nonzero the four-component period view certifies the
    genuine mixed one-form ``alpha(h) A(h) + beta(h) B(h)``. Root proposals
    are then Krawczyk centers, not asserted exact roots.
    """

    if half_width <= 0.0 or half_height <= 0.0:
        raise ValueError("rectangle half-width and half-height must be positive")
    roots = tuple(_fraction(root) for root in root_guesses)
    action_domain = certify_action_zero_free_rectangle(
        problem,
        contour_center=contour_center,
        half_width=half_width,
    )
    if not action_domain.verified:
        raise ValueError(
            "rectangle is not proved inside the distinguished action's "
            "zero-free critical strip"
        )
    mixed = bool(problem.moment_factor)
    evaluator = (
        build_abelian_period_evaluator(problem)
        if mixed
        else build_abelian_evaluator(problem)
    )
    cache: dict[tuple[float, float, float, float], ComplexInterval] = {}

    def function(box: ComplexInterval) -> ComplexInterval:
        key = (box.re.lo, box.re.hi, box.im.lo, box.im.hi)
        if key not in cache:
            value = evaluator.evaluate(box)
            cache[key] = value if isinstance(value, ComplexInterval) else value[0]
        return cache[key]

    upper = winding_collapse_function(
        function,
        contour_center,
        max(half_width, half_height),
        expected=len(roots),
        segments=segments,
        max_segments=max_segments,
        contour="rectangle",
        half_width=half_width,
        half_height=half_height,
    )
    if isinstance(evaluator, AbelianPeriodEvaluator):
        lower = tuple(
            certify_simple_mixed_abelian_zero(
                evaluator,
                root,
                radius=root_radius,
            )
            for root in roots
        )
    else:
        lower = tuple(
            certify_simple_abelian_zero(evaluator, root, radius=root_radius)
            for root in roots
        )
    upper_count = (
        int(upper.outcome.surviving)
        if upper.proved and isinstance(upper.outcome.surviving, int)
        else None
    )
    exact_count = len(lower) if upper_count == len(lower) else None
    source = _count_source(
        problem,
        contour_center,
        half_width,
        half_height,
        segments,
        max_segments,
        roots,
        root_radius,
    )
    digest = source_digest(source)
    winding = upper.outcome.residual
    factor_roots_excluded = (
        _declared_factor_roots_excluded(problem, lower) if mixed else False
    )
    seal: Cert | None = None
    if exact_count is not None and winding is not None:
        seal = make_certificate(
            claim=(
                f"The declared Abelian integral has exactly {exact_count} zeros "
                "inside the declared regular contour; instance-level only."
            ),
            payload={
                "type": "winding_integer_isolation",
                "lo": winding.lo,
                "hi": winding.hi,
                "integer": exact_count,
                "source": source,
                "source_digest": digest,
                "action_zero_free_domain": action_domain.to_payload(),
                "picard_fuchs_digest": evaluator.picard_fuchs.seal["digest"],
                "lower_zero_digests": [
                    certificate.certificate["digest"] for certificate in lower
                ],
                "declared_factor_roots_excluded": factor_roots_excluded,
                "finite_formal_scope": "winding integer isolation only",
            },
            honesty={
                "infinitesimal_hilbert16_instance": True,
                "epsilon_effective": False,
                "uniform_degree_bound_claim": False,
                "h_n_finiteness_claim": False,
                "full_hilbert16_solved": False,
                "forced_factor_instance": not mixed,
                "genuine_mixed_one_form": mixed,
                "declared_factor_roots_excluded": factor_roots_excluded,
                "irrational_zeros_verified": False,
                "cubic_gauss_manin_differential_rank_two": True,
                "four_component_period_view_overcomplete": True,
                "petrov_uniform_theorem_reproduced": False,
                "analytic_enclosures_are_lean_verified": False,
            },
        )
    return AbelianZeroCountCertificate(
        problem=problem,
        contour_center=contour_center,
        half_width=half_width,
        half_height=half_height,
        segments=segments,
        max_segments=max_segments,
        root_guesses=roots,
        root_radius=root_radius,
        source_digest=digest,
        action_zero_free_domain=action_domain,
        picard_fuchs=evaluator.picard_fuchs,
        initial_data=evaluator.initial_data,
        upper_status=upper.status,
        upper_count=upper_count,
        winding_enclosure=winding,
        lower_zeros=lower,
        exact_count=exact_count,
        seal=seal,
    )


def verify_abelian_zero_count(
    certificate: AbelianZeroCountCertificate,
) -> bool:
    """Recompute the operator, enclosures, Krawczyk boxes, winding, and seal."""

    if certificate.exact_count is None or certificate.seal is None:
        return False
    try:
        if not verify_certificate_digest(certificate.seal):
            return False
        replay = certify_abelian_zero_count(
            certificate.problem,
            contour_center=certificate.contour_center,
            half_width=certificate.half_width,
            half_height=certificate.half_height,
            root_guesses=certificate.root_guesses,
            root_radius=certificate.root_radius,
            segments=certificate.segments,
            max_segments=certificate.max_segments,
        )
        original_fields = dict(certificate.__dict__)
        replay_fields = dict(replay.__dict__)
        original_pf = original_fields.pop("picard_fuchs")
        replay_pf = replay_fields.pop("picard_fuchs")
        return (
            original_fields == replay_fields
            and verify_picard_fuchs(original_pf)
            and verify_picard_fuchs(replay_pf)
            and original_pf.period_operator.coeffs == replay_pf.period_operator.coeffs
            and original_pf.area_operator.coeffs == replay_pf.area_operator.coeffs
            and original_pf.integral_operator.coeffs == replay_pf.integral_operator.coeffs
        )
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
        return False


def verify_abelian_zero_count_formally(
    certificate: AbelianZeroCountCertificate,
) -> AbelianFormalVerification:
    """Run the kernel on the syzygy and winding-isolation obligations.

    ``theorem_prover_verified`` is derived exclusively from two genuine
    successful kernel checks. It is never accepted from certificate payload
    or honesty metadata.
    """

    if certificate.picard_fuchs.seal is None or certificate.seal is None:
        raise ValueError("an exact sealed Abelian zero count is required")
    picard = check_certificate(certificate.picard_fuchs.seal)
    winding = check_certificate(certificate.seal)
    return AbelianFormalVerification(
        picard_fuchs=picard,
        winding_integer=winding,
        theorem_prover_verified=bool(picard.verified and winding.verified),
    )


@dataclass(frozen=True)
class RationalCoefficientInterval:
    """A closed exact rational coefficient interval (points are allowed)."""

    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "lo", _fraction(self.lo))
        object.__setattr__(self, "hi", _fraction(self.hi))
        if self.lo > self.hi:
            raise ValueError("coefficient interval endpoints must be ordered")

    @classmethod
    def create(
        cls,
        lo: Rational,
        hi: Rational | None = None,
    ) -> RationalCoefficientInterval:
        return cls(_fraction(lo), _fraction(lo if hi is None else hi))

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    @property
    def midpoint(self) -> Fraction:
        return (self.lo + self.hi) / 2

    def to_payload(self) -> list[list[int]]:
        return [_q(self.lo), _q(self.hi)]


@dataclass(frozen=True)
class AbelianCoefficientBox:
    """Independent rational boxes for ``alpha`` and ``beta`` coefficients."""

    alpha: tuple[RationalCoefficientInterval, ...]
    beta: tuple[RationalCoefficientInterval, ...]

    def __post_init__(self) -> None:
        if not self.alpha and not self.beta:
            raise ValueError("at least one coefficient is required")
        if not self.beta or all(
            coefficient.lo <= 0 <= coefficient.hi for coefficient in self.beta
        ):
            raise ValueError("the uniform mixed family requires beta != 0")

    @classmethod
    def create(
        cls,
        *,
        alpha: Sequence[Rational | Sequence[Rational]],
        beta: Sequence[Rational | Sequence[Rational]],
    ) -> AbelianCoefficientBox:
        def interval(value: Rational | Sequence[Rational]) -> RationalCoefficientInterval:
            if isinstance(value, Sequence) and not isinstance(value, str | bytes):
                if len(value) != 2:
                    raise ValueError("coefficient intervals need two endpoints")
                return RationalCoefficientInterval.create(value[0], value[1])
            return RationalCoefficientInterval.create(value)

        return cls(
            tuple(interval(value) for value in alpha),
            tuple(interval(value) for value in beta),
        )

    def _flat(self) -> tuple[RationalCoefficientInterval, ...]:
        return (*self.alpha, *self.beta)

    def variable_intervals(self) -> tuple[RationalCoefficientInterval, ...]:
        return tuple(value for value in self._flat() if value.width > 0)

    def split_variable(
        self,
        variable_axis: int,
    ) -> tuple[Fraction, AbelianCoefficientBox, AbelianCoefficientBox]:
        variable_indices = [
            index for index, value in enumerate(self._flat()) if value.width > 0
        ]
        if not 0 <= variable_axis < len(variable_indices):
            raise ValueError("split axis is not a varying coefficient")
        flat_index = variable_indices[variable_axis]
        flat = list(self._flat())
        selected = flat[flat_index]
        cut = selected.midpoint
        left_flat = list(flat)
        right_flat = list(flat)
        left_flat[flat_index] = RationalCoefficientInterval(selected.lo, cut)
        right_flat[flat_index] = RationalCoefficientInterval(cut, selected.hi)
        alpha_size = len(self.alpha)
        return (
            cut,
            AbelianCoefficientBox(
                tuple(left_flat[:alpha_size]),
                tuple(left_flat[alpha_size:]),
            ),
            AbelianCoefficientBox(
                tuple(right_flat[:alpha_size]),
                tuple(right_flat[alpha_size:]),
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "alpha": [value.to_payload() for value in self.alpha],
            "beta": [value.to_payload() for value in self.beta],
        }

    def variable_payload(self) -> list[list[list[int]]]:
        return [value.to_payload() for value in self.variable_intervals()]


def _evaluate_interval_coefficient_polynomial(
    coefficients: Sequence[RationalCoefficientInterval],
    energy: ComplexInterval,
) -> ComplexInterval:
    value = ComplexInterval.zero()
    for coefficient in reversed(coefficients):
        coefficient_box = ComplexInterval.from_parts(
            Interval.hull(
                Interval.from_rational(coefficient.lo),
                Interval.from_rational(coefficient.hi),
            )
        )
        value = value * energy + coefficient_box
    return value


@dataclass(frozen=True)
class IntervalCoefficientAbelianEvaluator:
    """Rank-two period evaluator with independently interval-valued coefficients."""

    periods: AbelianPeriodEvaluator
    coefficients: AbelianCoefficientBox

    def evaluate(
        self,
        target: ComplexInterval | Interval | complex | float,
    ) -> ComplexInterval:
        energy = ComplexInterval.from_value(target)
        area, moment, _j0, _j1 = self.periods.evaluate_state(energy)
        alpha = _evaluate_interval_coefficient_polynomial(
            self.coefficients.alpha,
            energy,
        )
        beta = _evaluate_interval_coefficient_polynomial(
            self.coefficients.beta,
            energy,
        )
        return alpha * area + beta * moment


@dataclass(frozen=True)
class AbelianUniformCoverNode:
    """One proved/blocked leaf or one exact coefficient-axis bisection."""

    box: AbelianCoefficientBox
    status: str
    count: int | None
    winding_enclosure: Interval | None
    winding_seal: Cert | None = None
    split_axis: int | None = None
    split_cut: Fraction | None = None
    left: AbelianUniformCoverNode | None = None
    right: AbelianUniformCoverNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def leaves(self) -> tuple[AbelianUniformCoverNode, ...]:
        if self.is_leaf:
            return (self,)
        if self.left is None or self.right is None:
            raise ArithmeticError("a cover split must have two children")
        return (*self.left.leaves(), *self.right.leaves())

    def formal_tree_payload(self) -> dict[str, object]:
        if self.is_leaf:
            if self.count is None:
                raise ValueError("a blocked leaf has no formal count")
            return {
                "kind": "leaf",
                "box": self.box.variable_payload(),
                "count": self.count,
            }
        if (
            self.split_axis is None
            or self.split_cut is None
            or self.left is None
            or self.right is None
        ):
            raise ArithmeticError("malformed coefficient-cover split")
        return {
            "kind": "split",
            "axis": self.split_axis,
            "cut": _q(self.split_cut),
            "left": self.left.formal_tree_payload(),
            "right": self.right.formal_tree_payload(),
        }


@dataclass(frozen=True)
class AbelianUniformCoverCertificate:
    """A finite rational coefficient cover carrying one uniform winding bound."""

    problem: AbelianProblem
    coefficient_box: AbelianCoefficientBox
    contour_center: complex
    half_width: float
    half_height: float
    segments: int
    max_segments: int
    expected_count: int
    max_depth: int
    tree: AbelianUniformCoverNode
    status: str
    uniform_bound: int | None
    source_digest: str
    seal: Cert | None


@dataclass(frozen=True)
class AbelianUniformFormalVerification:
    """Lean verdicts for exact tiling and every leaf winding integer."""

    box_cover: LeanCheckResult
    leaf_windings: tuple[LeanCheckResult, ...]
    theorem_prover_verified: bool


def _uniform_leaf(
    evaluator: AbelianPeriodEvaluator,
    coefficient_box: AbelianCoefficientBox,
    *,
    contour_center: complex,
    half_width: float,
    half_height: float,
    expected_count: int,
    segments: int,
    max_segments: int,
) -> AbelianUniformCoverNode:
    interval_evaluator = IntervalCoefficientAbelianEvaluator(
        evaluator,
        coefficient_box,
    )
    cache: dict[tuple[float, float, float, float], ComplexInterval] = {}

    def function(box: ComplexInterval) -> ComplexInterval:
        key = (box.re.lo, box.re.hi, box.im.lo, box.im.hi)
        if key not in cache:
            cache[key] = interval_evaluator.evaluate(box)
        return cache[key]

    upper = winding_collapse_function(
        function,
        contour_center,
        max(half_width, half_height),
        expected=expected_count,
        segments=segments,
        max_segments=max_segments,
        contour="rectangle",
        half_width=half_width,
        half_height=half_height,
    )
    count = (
        int(upper.outcome.surviving)
        if upper.proved and isinstance(upper.outcome.surviving, int)
        else None
    )
    winding = upper.outcome.residual
    winding_seal: Cert | None = None
    if count is not None and winding is not None:
        winding_seal = make_certificate(
            claim=(
                f"The interval-coefficient Abelian family has winding integer "
                f"{count} on this declared contour."
            ),
            payload={
                "type": "winding_integer_isolation",
                "lo": winding.lo,
                "hi": winding.hi,
                "integer": count,
                "coefficient_box": coefficient_box.to_payload(),
                "finite_formal_scope": "winding integer isolation only",
            },
            honesty={
                "parameter_uniform_abelian_leaf": True,
                "analytic_enclosures_are_lean_verified": False,
                "h_n_finiteness_claim": False,
                "full_hilbert16_solved": False,
            },
        )
    return AbelianUniformCoverNode(
        box=coefficient_box,
        status=upper.status,
        count=count,
        winding_enclosure=winding,
        winding_seal=winding_seal,
    )


def _verify_cover_tree(
    node: AbelianUniformCoverNode,
    expected: AbelianCoefficientBox,
    bound: int,
) -> bool:
    if node.box != expected:
        return False
    if node.is_leaf:
        if (
            node.status != "PROVED"
            or node.count is None
            or node.count > bound
            or node.winding_enclosure is None
            or node.winding_seal is None
            or not verify_certificate_digest(node.winding_seal)
        ):
            return False
        payload = node.winding_seal.get("payload", {})
        return (
            payload.get("type") == "winding_integer_isolation"
            and payload.get("integer") == node.count
            and payload.get("lo") == node.winding_enclosure.lo
            and payload.get("hi") == node.winding_enclosure.hi
        )
    if (
        node.split_axis is None
        or node.split_cut is None
        or node.left is None
        or node.right is None
    ):
        return False
    try:
        cut, left, right = expected.split_variable(node.split_axis)
    except ValueError:
        return False
    return (
        cut == node.split_cut
        and _verify_cover_tree(node.left, left, bound)
        and _verify_cover_tree(node.right, right, bound)
    )


def certify_abelian_uniform_cover(
    problem: AbelianProblem,
    coefficient_box: AbelianCoefficientBox,
    *,
    contour_center: complex = 0j,
    half_width: float = 0.2,
    half_height: float = 0.04,
    expected_count: int = 2,
    segments: int = 8,
    max_segments: int = 32,
    max_depth: int = 4,
) -> AbelianUniformCoverCertificate:
    """Bisect blocked coefficient leaves and certify one finite uniform bound."""
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError("expected_count must be a nonnegative integer")
    if type(max_depth) is not int or not 0 <= max_depth <= 16:
        raise ValueError("max_depth must be an integer from zero to sixteen")
    if not coefficient_box.variable_intervals():
        raise ValueError("a parameter-uniform cover needs a positive-width coefficient")
    action_domain = certify_action_zero_free_rectangle(
        problem,
        contour_center=contour_center,
        half_width=half_width,
    )
    if not action_domain.verified:
        raise ValueError("coefficient cover contour crosses a critical-energy boundary")
    evaluator = build_abelian_period_evaluator(problem)

    def recurse(
        box: AbelianCoefficientBox,
        depth: int,
    ) -> AbelianUniformCoverNode:
        leaf = _uniform_leaf(
            evaluator,
            box,
            contour_center=contour_center,
            half_width=half_width,
            half_height=half_height,
            expected_count=expected_count,
            segments=segments,
            max_segments=max_segments,
        )
        if leaf.status == "PROVED" or depth >= max_depth:
            return leaf
        variables = box.variable_intervals()
        axis = max(range(len(variables)), key=lambda index: variables[index].width)
        cut, left_box, right_box = box.split_variable(axis)
        return AbelianUniformCoverNode(
            box=box,
            status="SPLIT",
            count=None,
            winding_enclosure=leaf.winding_enclosure,
            winding_seal=None,
            split_axis=axis,
            split_cut=cut,
            left=recurse(left_box, depth + 1),
            right=recurse(right_box, depth + 1),
        )

    tree = recurse(coefficient_box, 0)
    leaves = tree.leaves()
    passed = all(
        leaf.status == "PROVED"
        and leaf.count is not None
        and leaf.count <= expected_count
        for leaf in leaves
    )
    source = {
        "problem": problem.to_payload(),
        "coefficient_box": coefficient_box.to_payload(),
        "contour": {
            "center": [contour_center.real, contour_center.imag],
            "half_width": half_width,
            "half_height": half_height,
            "segments": segments,
            "max_segments": max_segments,
        },
        "expected_count": expected_count,
        "max_depth": max_depth,
    }
    digest = source_digest(source)
    seal: Cert | None = None
    if passed and _verify_cover_tree(tree, coefficient_box, expected_count):
        seal = make_certificate(
            claim=(
                f"Every mixed Abelian integral in the declared rational coefficient "
                f"box has at most {expected_count} zeros inside the regular contour."
            ),
            payload={
                "type": "box_cover_tiling",
                "parent": coefficient_box.variable_payload(),
                "tree": tree.formal_tree_payload(),
                "uniform_bound": expected_count,
                "source": source,
                "source_digest": digest,
                "leaf_winding_digests": [
                    leaf.winding_seal["digest"]
                    for leaf in leaves
                    if leaf.winding_seal is not None
                ],
                "finite_formal_scope": (
                    "rational coefficient-box tiling and integer leaf bounds; "
                    "analytic winding enclosures are trusted inputs"
                ),
            },
            honesty={
                "parameter_uniform_abelian_bound": True,
                "finite_coefficient_box": True,
                "infinitesimal_hilbert16_family": True,
                "cubic_gauss_manin_differential_rank_two": True,
                "four_component_period_view_overcomplete": True,
                "h_n_finiteness_claim": False,
                "full_hilbert16_solved": False,
                "analytic_enclosures_are_lean_verified": False,
            },
        )
    return AbelianUniformCoverCertificate(
        problem=problem,
        coefficient_box=coefficient_box,
        contour_center=contour_center,
        half_width=half_width,
        half_height=half_height,
        segments=segments,
        max_segments=max_segments,
        expected_count=expected_count,
        max_depth=max_depth,
        tree=tree,
        status="PROVED" if seal is not None else "BLOCKED",
        uniform_bound=expected_count if seal is not None else None,
        source_digest=digest,
        seal=seal,
    )


def verify_abelian_uniform_cover(
    certificate: AbelianUniformCoverCertificate,
) -> bool:
    """Recompute every interval winding leaf and the exact rational tiling."""
    if certificate.seal is None or not verify_certificate_digest(certificate.seal):
        return False
    try:
        replay = certify_abelian_uniform_cover(
            certificate.problem,
            certificate.coefficient_box,
            contour_center=certificate.contour_center,
            half_width=certificate.half_width,
            half_height=certificate.half_height,
            expected_count=certificate.expected_count,
            segments=certificate.segments,
            max_segments=certificate.max_segments,
            max_depth=certificate.max_depth,
        )
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError):
        return False
    return replay == certificate


def verify_abelian_uniform_cover_formally(
    certificate: AbelianUniformCoverCertificate,
) -> AbelianUniformFormalVerification:
    """Run Lean on the rational cover and every leaf winding isolation."""
    if certificate.seal is None:
        raise ValueError("a proved uniform cover is required")
    box_cover = check_certificate(certificate.seal)
    leaves = certificate.tree.leaves()
    if any(leaf.winding_seal is None for leaf in leaves):
        raise ValueError("every proved cover leaf requires a winding seal")
    leaf_windings = tuple(
        check_certificate(leaf.winding_seal)
        for leaf in leaves
        if leaf.winding_seal is not None
    )
    return AbelianUniformFormalVerification(
        box_cover,
        leaf_windings,
        bool(box_cover.verified and all(result.verified for result in leaf_windings)),
    )


__all__ = [
    "AbelianCoefficientBox",
    "AbelianEvaluator",
    "AbelianFormalVerification",
    "AbelianInitialData",
    "AbelianPeriodEvaluator",
    "AbelianProblem",
    "AbelianUniformCoverCertificate",
    "AbelianUniformCoverNode",
    "AbelianUniformFormalVerification",
    "AbelianZeroCountCertificate",
    "ActionZeroFreeRectangleCertificate",
    "IntervalCoefficientAbelianEvaluator",
    "RationalCoefficientInterval",
    "build_abelian_evaluator",
    "build_abelian_period_evaluator",
    "certify_abelian_uniform_cover",
    "certify_abelian_zero_count",
    "certify_action_zero_free_rectangle",
    "certify_cubic_initial_data",
    "certify_simple_abelian_zero",
    "certify_simple_mixed_abelian_zero",
    "named_cubic_abelian_problem",
    "named_genuine_cubic_abelian_problem",
    "verify_abelian_uniform_cover",
    "verify_abelian_uniform_cover_formally",
    "verify_abelian_zero_count",
    "verify_abelian_zero_count_formally",
]
