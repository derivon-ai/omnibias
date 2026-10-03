# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Complex event-time branch on one cubic Hilbert-XVI normal-form cell.

Complex interval Newton isolates a unique event-time zero for every epsilon
in a declared complex rectangle.  An independent real stopped-event replay
proves that the corresponding zero is the first transverse physical crossing
for real epsilon in the cell.

The result concerns the cubic normal-form comparison field on one interior
cell.  It is not the full singular quadratic return family and does not prove
Log-Noetherian membership or G3.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.complex_ode import (
    ComplexTaylorSeries,
    integrate_complex_ivp,
)
from omnibias.core.verified.complex_rootfind import (
    ComplexNewtonResult,
    parametric_complex_interval_newton,
)
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.complex_normal_flow import (
    complex_epsilon_cell,
    complex_normal_field,
)
from omnibias.dynamics.honesty_inventory import build_honesty
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    StoppedEventResult,
    certify_stopped_event,
    verify_stopped_event,
)

__all__ = [
    "ComplexEventBranchReport",
    "certify_complex_event_branch",
    "certify_real_first_hit",
    "complex_event_derivative",
    "complex_event_value",
    "complex_time_domain",
    "event_epsilon_cell",
    "report",
]

_EPSILON = Fraction(1, 16)
_EPSILON_RADIUS = Fraction(1, 1_000_000)
_TARGET = Fraction(-629_534, 10_000_000)
_TIME_CENTER = Fraction(1, 2)
_TIME_REAL_RADIUS = Fraction(1, 20)
_TIME_IMAG_RADIUS = Fraction(1, 50)
_L = Fraction(9, 25)
_LAMBDA1 = Fraction(-2)


def _rational_interval(lo: Fraction, hi: Fraction) -> Interval:
    return Interval.hull(Interval.from_rational(lo), Interval.from_rational(hi))


def event_epsilon_cell() -> ComplexInterval:
    """Small complex epsilon cell on which the event branch is isolated."""
    return complex_epsilon_cell(
        center=_EPSILON,
        real_radius=_EPSILON_RADIUS,
        imag_radius=_EPSILON_RADIUS,
    )


def complex_time_domain() -> ComplexInterval:
    """Declared complex time rectangle around the real crossing."""
    return ComplexInterval(
        _rational_interval(
            _TIME_CENTER - _TIME_REAL_RADIUS,
            _TIME_CENTER + _TIME_REAL_RADIUS,
        ),
        _rational_interval(-_TIME_IMAG_RADIUS, _TIME_IMAG_RADIUS),
    )


def _scaled_normal_field(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    if len(series) != 4:
        raise ValueError("scaled normal field expects (V, h, epsilon, time)")
    v_coord, height, epsilon, time = series
    physical = complex_normal_field([v_coord, height, epsilon])
    return [
        time * physical[0],
        time * physical[1],
        0 * epsilon,
        0 * time,
    ]


def _complex_endpoint(
    time: ComplexInterval,
    epsilon: ComplexInterval,
) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval]:
    initial_v = -epsilon
    initial_h = 4 * epsilon * epsilon * epsilon
    result = integrate_complex_ivp(
        _scaled_normal_field,
        (initial_v, initial_h, epsilon, time),
        0.0,
        1.0,
        order=12,
        n_steps=8,
    )
    return result[0], result[1], result[2], result[3]


def complex_event_value(
    time: ComplexInterval,
    *,
    epsilon: ComplexInterval | None = None,
) -> ComplexInterval:
    """Enclose ``V(time, epsilon) - target`` on the complex parameter cell."""
    epsilon_box = event_epsilon_cell() if epsilon is None else epsilon
    return _complex_endpoint(time, epsilon_box)[0] - _TARGET


def complex_event_derivative(
    time: ComplexInterval,
    *,
    epsilon: ComplexInterval | None = None,
) -> ComplexInterval:
    """Enclose the exact autonomous identity ``dV/dtime = Vdot``."""
    epsilon_box = event_epsilon_cell() if epsilon is None else epsilon
    v_coord, height, epsilon_out, _ = _complex_endpoint(time, epsilon_box)
    series = [
        ComplexTaylorSeries([v_coord]),
        ComplexTaylorSeries([height]),
        ComplexTaylorSeries([epsilon_out]),
    ]
    return complex_normal_field(series)[0].coeffs[0]


def certify_complex_event_branch() -> ComplexNewtonResult:
    """Isolate the unique complex event-time branch over the epsilon cell."""
    epsilon = event_epsilon_cell()
    return parametric_complex_interval_newton(
        lambda time: complex_event_value(time, epsilon=epsilon),
        lambda time: complex_event_derivative(time, epsilon=epsilon),
        complex_time_domain(),
        max_iter=1,
    )


def _real_polynomial_flow() -> PolynomialFlow:
    nvars = 4
    v_coord = SparsePolynomial.variable(nvars, 0)
    height = SparsePolynomial.variable(nvars, 1)
    epsilon = SparsePolynomial.variable(nvars, 2)
    field_f = (
        SparsePolynomial.constant(nvars, -_L) * epsilon**3
        + SparsePolynomial.constant(nvars, _LAMBDA1) * epsilon**2 * v_coord
        - epsilon * v_coord**2
        + SparsePolynomial.constant(nvars, Fraction(1, 3))
        * epsilon
        * v_coord**3
    )
    field_g = -1 + epsilon * (v_coord - 1)
    return PolynomialFlow((field_f + height * field_g, -v_coord * height), 1)


def certify_real_first_hit() -> StoppedEventResult:
    """Certify the first real target crossing on the real epsilon slice."""
    epsilon_parameter = SparsePolynomial.variable(1, 0)
    v_coord = SparsePolynomial.variable(4, 0)
    epsilon_interval = _rational_interval(
        _EPSILON - _EPSILON_RADIUS,
        _EPSILON + _EPSILON_RADIUS,
    )
    request = StoppedEventRequest(
        flow=_real_polynomial_flow(),
        initial=(-epsilon_parameter, 4 * epsilon_parameter**3),
        parameters=(epsilon_interval,),
        target=PolynomialEvent(
            v_coord - _TARGET,
            direction=-1,
            name="complex_branch_real_slice",
        ),
        step=0.02,
        max_steps=30,
        order=10,
        derivative_order=2,
    )
    return certify_stopped_event(request)


def _complex_payload(value: ComplexInterval) -> dict[str, list[float]]:
    return {
        "re": [value.re.lo, value.re.hi],
        "im": [value.im.lo, value.im.hi],
    }


@dataclass(frozen=True)
class ComplexEventBranchReport:
    """Local complex branch plus independently replayed first real crossing."""

    branch: ComplexNewtonResult
    real_first_hit: StoppedEventResult
    complex_normal_event_branch_certified: bool
    real_first_hit_replayed: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-complex-event-branch-v1",
            "epsilon_cell": _complex_payload(event_epsilon_cell()),
            "time_domain": _complex_payload(complex_time_domain()),
            "event_target": str(_TARGET),
            "branch_status": self.branch.status,
            "branch_enclosure": _complex_payload(self.branch.enclosure),
            "complex_normal_event_branch_certified": (
                self.complex_normal_event_branch_certified
            ),
            "real_first_hit_status": self.real_first_hit.status,
            "real_first_hit_replayed": self.real_first_hit_replayed,
            "complex_physical_return_family_certified": False,
            "actual_return_ln_membership_proved": False,
            "g3_passed": False,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "Unique complex event-time branch and first real crossing for "
                "one interior epsilon cell of the cubic normal-form comparison "
                "field. Not the full singular quadratic return family, not "
                "uniform at sep=0 or r1=0, not LN membership, not G3, and not "
                "Hilbert XVI."
            ),
        }


def report() -> ComplexEventBranchReport:
    """Certify the local branch and preserve every global refusal."""
    branch = certify_complex_event_branch()
    real_hit = certify_real_first_hit()
    real_replay = bool(real_hit.certified) and verify_stopped_event(real_hit)
    branch_certified = (
        branch.status == "unique_root"
        and branch.unique_for_every_parameter
        and real_replay
    )
    honesty = build_honesty(
        complex_normal_event_branch_certified=branch_certified,
        complex_physical_return_family_certified=False,
        actual_return_ln_membership_proved=False,
        g3_passed=False,
        full_hilbert16_solved=False,
    )
    return ComplexEventBranchReport(
        branch=branch,
        real_first_hit=real_hit,
        complex_normal_event_branch_certified=branch_certified,
        real_first_hit_replayed=real_replay,
        honesty=honesty,
    )
