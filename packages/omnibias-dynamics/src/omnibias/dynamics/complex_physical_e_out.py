# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Complex branches of the physical outgoing ``E_out`` matching section.

The cubic normal-form comparison orbit starts at ``(V,h)=(-eps,4 eps^3)``
with ``eps=nu=1/16`` and

``L = 1 - sep^2/4``.

The physical outgoing section is the matching-chart image of
``x=rho/nu``:

``E_out = V + rho + nu rho h + C nu^2 rho h^2 = 0``.

A finite rational cover of ``sep in [0,2]`` isolates the complex event-time
branch.  Adjacent cells use the same time domain and overlap at their common
parameter boundary, so uniqueness matches the local branches.  Independent
real stopped-event certificates establish first transverse crossing.

This is an outgoing branch of the cubic comparison field.  It is not the
incoming ``E_sigma`` branch, the full quadratic return map, a bounded-format
Log-Noetherian representation, G3, or Hilbert XVI.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, lru_cache
from itertools import pairwise

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
    "ComplexPhysicalEOutReport",
    "PhysicalEOutCell",
    "certify_complex_physical_e_out_cover",
    "certify_real_physical_e_out_cover",
    "complex_e_out_derivative",
    "complex_e_out_event",
    "physical_e_out_cells",
    "physical_e_out_time_domain",
    "report",
]

_EPSILON = Fraction(1, 16)
_RHO = Fraction(1, 4)
_C = Fraction(2)
_LAMBDA1 = Fraction(-2)
_ANCHOR_TIME = Fraction(30)
_SEPARATION_IMAG_RADIUS = Fraction(1, 10_000)
_DELTA_IMAG_RADIUS = Fraction(1, 50)
_SEPARATION_BREAKS = tuple(Fraction(index, 4) for index in range(9))


def _rational_interval(lo: Fraction, hi: Fraction) -> Interval:
    return Interval.hull(Interval.from_rational(lo), Interval.from_rational(hi))


@dataclass(frozen=True)
class PhysicalEOutCell:
    """One rational separation slab and its common complex time domain."""

    index: int
    separation: ComplexInterval
    time_delta: ComplexInterval


def physical_e_out_time_domain() -> ComplexInterval:
    """Complex ``T-30`` rectangle containing every outgoing event time."""
    return ComplexInterval(
        _rational_interval(Fraction(1, 4), Fraction(7, 4)),
        _rational_interval(-_DELTA_IMAG_RADIUS, _DELTA_IMAG_RADIUS),
    )


def physical_e_out_cells() -> tuple[PhysicalEOutCell, ...]:
    """Finite rational cover of the complete ``sep in [0,2]`` compact."""
    time_delta = physical_e_out_time_domain()
    imag = _rational_interval(
        -_SEPARATION_IMAG_RADIUS,
        _SEPARATION_IMAG_RADIUS,
    )
    return tuple(
        PhysicalEOutCell(
            index=index,
            separation=ComplexInterval(
                _rational_interval(left, right),
                imag,
            ),
            time_delta=time_delta,
        )
        for index, (left, right) in enumerate(pairwise(_SEPARATION_BREAKS))
    )


def _normal_field(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    if len(series) != 3:
        raise ValueError("normal field expects (V, h, sep)")
    v_coord, height, separation = series
    ell = 1 - Fraction(1, 4) * separation * separation
    field_f = (
        -ell * _EPSILON**3
        + _LAMBDA1 * _EPSILON**2 * v_coord
        - _EPSILON * v_coord * v_coord
        + Fraction(1, 3) * _EPSILON * v_coord * v_coord * v_coord
    )
    field_g = -1 + _EPSILON * (v_coord - 1)
    return [
        field_f + height * field_g,
        -v_coord * height,
        0 * separation,
    ]


def _scaled_local_field(
    series: list[ComplexTaylorSeries],
) -> list[ComplexTaylorSeries]:
    if len(series) != 4:
        raise ValueError("scaled local field expects (V, h, sep, delta)")
    v_coord, height, separation, delta = series
    physical = _normal_field([v_coord, height, separation])
    return [
        delta * physical[0],
        delta * physical[1],
        0 * separation,
        0 * delta,
    ]


@cache
def _complex_anchor(
    separation: ComplexInterval,
) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval]:
    result = integrate_complex_ivp(
        _normal_field,
        (
            ComplexInterval.from_value(-_EPSILON),
            ComplexInterval.from_value(4 * _EPSILON**3),
            separation,
        ),
        0.0,
        float(_ANCHOR_TIME),
        order=12,
        n_steps=192,
    )
    return result[0], result[1], result[2]


@cache
def _complex_endpoint(
    time_delta: ComplexInterval,
    separation: ComplexInterval,
) -> tuple[ComplexInterval, ComplexInterval, ComplexInterval, ComplexInterval]:
    v_anchor, h_anchor, separation_anchor = _complex_anchor(separation)
    result = integrate_complex_ivp(
        _scaled_local_field,
        (v_anchor, h_anchor, separation_anchor, time_delta),
        0.0,
        1.0,
        order=12,
        n_steps=16,
    )
    return result[0], result[1], result[2], result[3]


def complex_e_out_event(
    time_delta: ComplexInterval,
    *,
    separation: ComplexInterval,
) -> ComplexInterval:
    """Enclose the physical ``E_out`` residual on one complex cell."""
    v_coord, height, _, _ = _complex_endpoint(time_delta, separation)
    return (
        v_coord
        + _RHO
        + _EPSILON * _RHO * height
        + _C * _EPSILON**2 * _RHO * height * height
    )


def complex_e_out_derivative(
    time_delta: ComplexInterval,
    *,
    separation: ComplexInterval,
) -> ComplexInterval:
    """Enclose the exact time derivative of physical ``E_out``."""
    v_coord, height, separation_out, _ = _complex_endpoint(
        time_delta,
        separation,
    )
    vdot, hdot, _ = (
        component.coeffs[0]
        for component in _normal_field(
            [
                ComplexTaylorSeries([v_coord]),
                ComplexTaylorSeries([height]),
                ComplexTaylorSeries([separation_out]),
            ]
        )
    )
    return (
        vdot
        + _EPSILON * _RHO * hdot
        + 2 * _C * _EPSILON**2 * _RHO * height * hdot
    )


@lru_cache(maxsize=1)
def certify_complex_physical_e_out_cover() -> tuple[ComplexNewtonResult, ...]:
    """Isolate the outgoing complex event branch on every separation cell."""
    results: list[ComplexNewtonResult] = []
    for cell in physical_e_out_cells():
        def event(
            time: ComplexInterval,
            *,
            separation: ComplexInterval = cell.separation,
        ) -> ComplexInterval:
            return complex_e_out_event(time, separation=separation)

        def derivative(
            time: ComplexInterval,
            *,
            separation: ComplexInterval = cell.separation,
        ) -> ComplexInterval:
            return complex_e_out_derivative(time, separation=separation)

        results.append(
            parametric_complex_interval_newton(
                event,
                derivative,
                cell.time_delta,
                max_iter=3,
            )
        )
    return tuple(results)


def _real_polynomial_flow() -> PolynomialFlow:
    nvars = 4
    v_coord = SparsePolynomial.variable(nvars, 0)
    height = SparsePolynomial.variable(nvars, 1)
    separation = SparsePolynomial.variable(nvars, 2)
    one = SparsePolynomial.constant(nvars, 1)
    ell = one - Fraction(1, 4) * separation**2
    field_f = (
        -ell * _EPSILON**3
        + _LAMBDA1 * _EPSILON**2 * v_coord
        - _EPSILON * v_coord**2
        + Fraction(1, 3) * _EPSILON * v_coord**3
    )
    field_g = -one + _EPSILON * (v_coord - one)
    return PolynomialFlow(
        (field_f + height * field_g, -v_coord * height),
        1,
    )


def _real_e_out_event() -> PolynomialEvent:
    nvars = 4
    v_coord = SparsePolynomial.variable(nvars, 0)
    height = SparsePolynomial.variable(nvars, 1)
    event = (
        v_coord
        + SparsePolynomial.constant(nvars, _RHO)
        + SparsePolynomial.constant(nvars, _EPSILON * _RHO) * height
        + SparsePolynomial.constant(nvars, _C * _EPSILON**2 * _RHO)
        * height**2
    )
    return PolynomialEvent(event, direction=-1, name="physical_E_out")


@lru_cache(maxsize=1)
def certify_real_physical_e_out_cover() -> tuple[StoppedEventResult, ...]:
    """Certify first real ``E_out`` crossing on the same separation cover."""
    flow = _real_polynomial_flow()
    target = _real_e_out_event()
    initial = (
        SparsePolynomial.constant(1, -_EPSILON),
        SparsePolynomial.constant(1, 4 * _EPSILON**3),
    )
    results: list[StoppedEventResult] = []
    for left, right in pairwise(_SEPARATION_BREAKS):
        request = StoppedEventRequest(
            flow=flow,
            initial=initial,
            parameters=(_rational_interval(left, right),),
            target=target,
            step=0.5,
            max_steps=64,
            order=6,
            derivative_order=0,
        )
        results.append(certify_stopped_event(request))
    return tuple(results)


def _rectangles_overlap(left: ComplexInterval, right: ComplexInterval) -> bool:
    return bool(
        max(left.re.lo, right.re.lo) <= min(left.re.hi, right.re.hi)
        and max(left.im.lo, right.im.lo) <= min(left.im.hi, right.im.hi)
    )


def _complex_payload(value: ComplexInterval) -> dict[str, list[float]]:
    return {
        "re": [value.re.lo, value.re.hi],
        "im": [value.im.lo, value.im.hi],
    }


@dataclass(frozen=True)
class ComplexPhysicalEOutReport:
    """Matched complex outgoing branches and real first-hit replays."""

    cells: tuple[PhysicalEOutCell, ...]
    branches: tuple[ComplexNewtonResult, ...]
    derivative_moduli: tuple[Interval, ...]
    real_first_hits: tuple[StoppedEventResult, ...]
    adjacent_branches_matched: bool
    complex_physical_e_out_cover_certified: bool
    real_first_hit_cover_replayed: bool
    honesty: Mapping[str, object]

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-complex-physical-e-out-v1",
            "anchor_time": str(_ANCHOR_TIME),
            "cells": [
                {
                    "index": cell.index,
                    "separation": _complex_payload(cell.separation),
                    "time_delta_domain": _complex_payload(cell.time_delta),
                    "branch_status": branch.status,
                    "branch_delta_enclosure": _complex_payload(branch.enclosure),
                    "derivative_modulus": [modulus.lo, modulus.hi],
                    "real_first_hit_status": real_hit.status,
                }
                for cell, branch, modulus, real_hit in zip(
                    self.cells,
                    self.branches,
                    self.derivative_moduli,
                    self.real_first_hits,
                    strict=True,
                )
            ],
            "d_c_endpoint_included": self.cells[0].separation.contains(0),
            "chart_o_endpoint_included": self.cells[-1].separation.contains(2),
            "adjacent_branches_matched": self.adjacent_branches_matched,
            "complex_physical_e_out_cover_certified": (
                self.complex_physical_e_out_cover_certified
            ),
            "real_first_hit_cover_replayed": self.real_first_hit_cover_replayed,
            "incoming_physical_branch_certified": False,
            "physical_overlap_matching_proved": False,
            "actual_return_ln_membership_proved": False,
            "g3_passed": False,
            "full_hilbert16_solved": False,
            "honesty": dict(self.honesty),
            "scope": (
                "A finite complex cover isolates and matches the physical "
                "outgoing E_out branch of the cubic comparison field across "
                "sep=0 through sep=2, with independent real first-hit replay. "
                "The incoming E_sigma branch, the full quadratic return map, "
                "bounded LN format, G3, and Hilbert XVI remain open."
            ),
        }


@lru_cache(maxsize=1)
def report() -> ComplexPhysicalEOutReport:
    """Build the physical outgoing cover without promoting a global gate."""
    cells = physical_e_out_cells()
    branches = certify_complex_physical_e_out_cover()
    derivative_moduli = tuple(
        complex_e_out_derivative(
            cell.time_delta,
            separation=cell.separation,
        ).modulus()
        for cell in cells
    )
    real_hits = certify_real_physical_e_out_cover()
    real_replay = all(
        bool(result.certified) and verify_stopped_event(result)
        for result in real_hits
    )
    adjacent_matched = all(
        _rectangles_overlap(left.enclosure, right.enclosure)
        for left, right in pairwise(branches)
    )
    certified = (
        len(cells) > 0
        and cells[0].separation.contains(0)
        and cells[-1].separation.contains(2)
        and all(
            branch.status == "unique_root"
            and branch.unique_for_every_parameter
            for branch in branches
        )
        and all(modulus.lo > 0.0 for modulus in derivative_moduli)
        and adjacent_matched
        and real_replay
    )
    honesty = build_honesty(
        complex_physical_e_out_cover_certified=certified,
        physical_overlap_matching_proved=False,
        actual_return_ln_membership_proved=False,
        g3_passed=False,
        full_hilbert16_solved=False,
    )
    return ComplexPhysicalEOutReport(
        cells=cells,
        branches=branches,
        derivative_moduli=derivative_moduli,
        real_first_hits=real_hits,
        adjacent_branches_matched=adjacent_matched,
        complex_physical_e_out_cover_certified=certified,
        real_first_hit_cover_replayed=real_replay,
        honesty=honesty,
    )
