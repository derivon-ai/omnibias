# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Complex event branch spanning the cubic model's two separation limits.

With ``lambda1=-2`` and ``L=1-sep^2/4``, ``sep=0`` is the D--C double-root
limit and ``sep=2`` gives ``r1=(2-sep)/2=0`` at chart O.  This module
certifies one complex event-time branch on a single rectangle containing both
endpoints, and independently replays the first real crossing for every
``sep in [0,2]``.

The fixed target is deliberately a regular transverse ``V`` section of the
cubic comparison field.  It is not the singular physical entry/exit map whose
height sections, overlap matching, and bounded LN format remain open.
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
    "ComplexSeparationCoverReport",
    "certify_complex_separation_branch",
    "certify_real_separation_first_hit",
    "complex_separation_derivative",
    "complex_separation_event",
    "report",
    "separation_cell",
    "separation_time_domain",
]

_EPSILON = Fraction(1, 16)
_TARGET = Fraction(-629_534, 10_000_000)
_IMAG_RADIUS = Fraction(1, 1_000)
_LAMBDA1 = Fraction(-2)


def _rational_interval(lo: Fraction, hi: Fraction) -> Interval:
    return Interval.hull(Interval.from_rational(lo), Interval.from_rational(hi))


def separation_cell() -> ComplexInterval:
    """Complex rectangle containing both ``sep=0`` and ``sep=2``."""
    return ComplexInterval(
        _rational_interval(Fraction(0), Fraction(2)),
        _rational_interval(-_IMAG_RADIUS, _IMAG_RADIUS),
    )


def separation_time_domain() -> ComplexInterval:
    """Common complex event-time rectangle for the whole separation cell."""
    return ComplexInterval(
        _rational_interval(Fraction(1, 3), Fraction(2, 3)),
        _rational_interval(Fraction(-1, 20), Fraction(1, 20)),
    )


def _scaled_separation_field(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    if len(series) != 4:
        raise ValueError("scaled separation field expects (V, h, sep, time)")
    v_coord, height, separation, time = series
    ell = 1 - Fraction(1, 4) * separation * separation
    field_f = (
        -ell * _EPSILON**3
        + _LAMBDA1 * _EPSILON**2 * v_coord
        - _EPSILON * v_coord * v_coord
        + Fraction(1, 3) * _EPSILON * v_coord * v_coord * v_coord
    )
    field_g = -1 + _EPSILON * (v_coord - 1)
    return [
        time * (field_f + height * field_g),
        time * (-v_coord * height),
        0 * separation,
        0 * time,
    ]


def _complex_endpoint(
    time: ComplexInterval,
    separation: ComplexInterval,
) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval]:
    result = integrate_complex_ivp(
        _scaled_separation_field,
        (
            ComplexInterval.from_value(-_EPSILON),
            ComplexInterval.from_value(4 * _EPSILON**3),
            separation,
            time,
        ),
        0.0,
        1.0,
        order=12,
        n_steps=8,
    )
    return result[0], result[1], result[2], result[3]


def complex_separation_event(
    time: ComplexInterval,
    *,
    separation: ComplexInterval | None = None,
) -> ComplexInterval:
    """Enclose the target residual over the complex separation cell."""
    separation_box = separation_cell() if separation is None else separation
    return _complex_endpoint(time, separation_box)[0] - _TARGET


def complex_separation_derivative(
    time: ComplexInterval,
    *,
    separation: ComplexInterval | None = None,
) -> ComplexInterval:
    """Enclose ``dV/dtime`` for complex time and separation."""
    separation_box = separation_cell() if separation is None else separation
    v_coord, height, separation_out, _ = _complex_endpoint(
        time,
        separation_box,
    )
    ell = (
        ComplexInterval.one()
        - Fraction(1, 4) * separation_out * separation_out
    )
    field_f = (
        -ell * _EPSILON**3
        + _LAMBDA1 * _EPSILON**2 * v_coord
        - _EPSILON * v_coord * v_coord
        + Fraction(1, 3) * _EPSILON * v_coord * v_coord * v_coord
    )
    field_g = -1 + _EPSILON * (v_coord - 1)
    return field_f + height * field_g


def certify_complex_separation_branch() -> ComplexNewtonResult:
    """Isolate one event-time branch over the complete separation rectangle."""
    separation = separation_cell()
    return parametric_complex_interval_newton(
        lambda time: complex_separation_event(time, separation=separation),
        lambda time: complex_separation_derivative(time, separation=separation),
        separation_time_domain(),
        max_iter=1,
    )


def _real_polynomial_flow() -> PolynomialFlow:
    nvars = 4
    v_coord = SparsePolynomial.variable(nvars, 0)
    height = SparsePolynomial.variable(nvars, 1)
    separation = SparsePolynomial.variable(nvars, 2)
    one = SparsePolynomial.constant(nvars, 1)
    ell = one - Fraction(1, 4) * separation**2
    field_f = (
        -ell * _EPSILON**3
        + SparsePolynomial.constant(nvars, _LAMBDA1)
        * _EPSILON**2
        * v_coord
        - _EPSILON * v_coord**2
        + Fraction(1, 3) * _EPSILON * v_coord**3
    )
    field_g = SparsePolynomial.constant(nvars, -1) + _EPSILON * (v_coord - one)
    return PolynomialFlow((field_f + height * field_g, -v_coord * height), 1)


def certify_real_separation_first_hit() -> StoppedEventResult:
    """Prove the first real target crossing uniformly for ``sep in [0,2]``."""
    v_coord = SparsePolynomial.variable(4, 0)
    request = StoppedEventRequest(
        flow=_real_polynomial_flow(),
        initial=(
            SparsePolynomial.constant(1, -_EPSILON),
            SparsePolynomial.constant(1, 4 * _EPSILON**3),
        ),
        parameters=(Interval(0.0, 2.0),),
        target=PolynomialEvent(
            v_coord - _TARGET,
            direction=-1,
            name="separation_cover_real_slice",
        ),
        step=0.02,
        max_steps=40,
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
class ComplexSeparationCoverReport:
    """Uniform local event branch spanning D--C and chart-O model endpoints."""

    branch: ComplexNewtonResult
    derivative_modulus: Interval
    real_first_hit: StoppedEventResult
    complex_separation_event_cover_certified: bool
    real_first_hit_replayed: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-complex-separation-cover-v1",
            "separation_cell": _complex_payload(separation_cell()),
            "time_domain": _complex_payload(separation_time_domain()),
            "branch_status": self.branch.status,
            "branch_enclosure": _complex_payload(self.branch.enclosure),
            "derivative_modulus": [
                self.derivative_modulus.lo,
                self.derivative_modulus.hi,
            ],
            "d_c_endpoint_included": separation_cell().contains(0),
            "chart_o_endpoint_included": separation_cell().contains(2),
            "complex_separation_event_cover_certified": (
                self.complex_separation_event_cover_certified
            ),
            "real_first_hit_status": self.real_first_hit.status,
            "real_first_hit_replayed": self.real_first_hit_replayed,
            "physical_overlap_matching_proved": False,
            "actual_return_ln_membership_proved": False,
            "g3_passed": False,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "One regular V-event branch of the cubic comparison field "
                "spans sep=0 through sep=2. This does not construct or match "
                "the singular physical entry/exit sections, does not control "
                "the full return family, is not LN membership, not G3, and "
                "not Hilbert XVI."
            ),
        }


def report() -> ComplexSeparationCoverReport:
    """Certify the comparison branch and retain the physical overlap refusal."""
    branch = certify_complex_separation_branch()
    derivative_modulus = complex_separation_derivative(
        separation_time_domain()
    ).modulus()
    real_hit = certify_real_separation_first_hit()
    real_replay = bool(real_hit.certified) and verify_stopped_event(real_hit)
    certified = (
        branch.status == "unique_root"
        and branch.unique_for_every_parameter
        and derivative_modulus.lo > 0.0
        and real_replay
    )
    honesty = build_honesty(
        complex_separation_event_cover_certified=certified,
        physical_overlap_matching_proved=False,
        actual_return_ln_membership_proved=False,
        g3_passed=False,
        full_hilbert16_solved=False,
    )
    return ComplexSeparationCoverReport(
        branch=branch,
        derivative_modulus=derivative_modulus,
        real_first_hit=real_hit,
        complex_separation_event_cover_certified=certified,
        real_first_hit_replayed=real_replay,
        honesty=honesty,
    )
