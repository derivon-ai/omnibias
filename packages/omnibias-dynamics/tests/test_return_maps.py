# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""First-hit existence, exclusion, and physical event derivatives."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from fractions import Fraction

import pytest
from omnibias.core.realization.polynomial import SparsePolynomial as Poly
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    certify_stopped_event,
    polynomial_interval,
    verify_stopped_event,
)


def _constant_flow_request(**kwargs: object) -> StoppedEventRequest:
    x = Poly.variable(2, 0)
    base = StoppedEventRequest(
        flow=PolynomialFlow((Poly.constant(2, 1),)),
        initial=(Poly.constant(0, 0),),
        parameters=(),
        target=PolynomialEvent(x - 1, direction=1),
        step=0.125,
        max_steps=16,
        derivative_order=0,
    )
    return replace(base, **kwargs)


def test_regular_positive_flow_and_replay() -> None:
    result = certify_stopped_event(_constant_flow_request())
    assert result.certified, result.reason
    assert result.time_bracket is not None and result.time_bracket.contains(1.0)
    assert result.return_box is not None and result.return_box[0].contains(1.0)
    assert result.return_box[0].width < 1e-12
    assert result.normal_velocity is not None and result.normal_velocity.lo > 0
    assert verify_stopped_event(result)
    assert not verify_stopped_event(replace(result, return_box=(Interval.point(2.0),)))
    assert not verify_stopped_event(replace(result, request=_constant_flow_request(step=0.25)))
    assert not verify_stopped_event(replace(result, source_fingerprint="invented"))


def test_parameterized_initial_section_return_derivatives() -> None:
    # x' = 1, y' = 2, x(0)=-1, y(0)=h; hit x=0 at t=1.
    x = Poly.variable(4, 0)
    h = Poly.variable(1, 0)
    request = StoppedEventRequest(
        flow=PolynomialFlow((Poly.constant(4, 1), Poly.constant(4, 2)), 1),
        initial=(Poly.constant(1, -1), h),
        parameters=(Interval(-0.1, 0.1),),
        target=PolynomialEvent(x, direction=1),
        step=0.125,
        max_steps=16,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.return_jacobian is not None and result.return_hessian is not None
    assert result.event_time_gradient is not None and result.event_time_hessian is not None
    assert result.event_time_gradient[0].contains(0)
    assert result.event_time_hessian[0][0].contains(0)
    assert result.return_jacobian[0][0].contains(0)
    assert result.return_jacobian[1][0].contains(1)
    assert result.return_jacobian[1][0].width < 1e-10
    assert result.return_hessian[1][0][0].contains(0)


def test_moving_section_field_parameters_and_mixed_derivatives() -> None:
    # x'=a, y'=x; g=x-1-b*t. Then tau=1/(a-b),
    # x(tau)=a/(a-b), y(tau)=a/(2*(a-b)^2).
    x, a, b, t = (Poly.variable(5, i) for i in (0, 2, 3, 4))
    request = StoppedEventRequest(
        flow=PolynomialFlow((a, x), 2),
        initial=(Poly.constant(2, 0), Poly.constant(2, 0)),
        parameters=(Interval(1.9, 2.1), Interval(0.2, 0.3)),
        target=PolynomialEvent(x - 1 - b * t, direction=1),
        step=0.05,
        max_steps=20,
        order=6,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.time_bracket is not None and result.return_box is not None
    assert result.event_time_gradient is not None and result.event_time_hessian is not None
    assert result.return_jacobian is not None and result.return_hessian is not None
    points = [(1.9 + 0.2 * i / 16, 0.2 + 0.1 * j / 16) for i in range(17) for j in range(17)]
    rng = random.Random(731)
    points.extend((rng.uniform(1.9, 2.1), rng.uniform(0.2, 0.3)) for _ in range(200))
    for av, bv in points:
        q = av - bv
        assert result.time_bracket.contains(1 / q)
        assert result.return_box[0].contains(av / q)
        assert result.return_box[1].contains(av / (2 * q**2))
        for index, truth in enumerate((-1 / q**2, 1 / q**2)):
            assert result.event_time_gradient[index].contains(truth)
        tau_hess = ((2 / q**3, -2 / q**3), (-2 / q**3, 2 / q**3))
        jac = ((-bv / q**2, av / q**2), (-(av + bv) / (2 * q**3), av / q**3))
        hess = (
            ((2 * bv / q**3, -(av + bv) / q**3), (-(av + bv) / q**3, 2 * av / q**3)),
            (((av + 2 * bv) / q**4, -(2 * av + bv) / q**4),
             (-(2 * av + bv) / q**4, 3 * av / q**4)),
        )
        for i in range(2):
            for j in range(2):
                assert result.event_time_hessian[i][j].contains(tau_hess[i][j])
                assert result.return_jacobian[i][j].contains(jac[i][j])
                for k in range(2):
                    assert result.return_hessian[i][j][k].contains(hess[i][j][k])


def test_nonlinear_initial_embedding_contributes_mixed_jet() -> None:
    a, b = Poly.variable(2, 0), Poly.variable(2, 1)
    x = Poly.variable(4, 0)
    request = StoppedEventRequest(
        flow=PolynomialFlow((Poly.constant(4, 1),), 2),
        initial=(a * b,),
        parameters=(Interval(0.1, 0.2), Interval(0.3, 0.4)),
        target=PolynomialEvent(x - 1),
        step=0.125,
        max_steps=12,
        order=4,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.event_time_hessian is not None
    assert result.event_time_hessian[0][1].contains(-1)
    assert result.event_time_hessian[0][1].width < 1e-10
    assert result.return_hessian is not None
    assert result.return_hessian[0][0][1].contains(0)


def test_cubic_nontransverse_crossing_is_unresolved() -> None:
    x, y = Poly.variable(3, 0), Poly.variable(3, 1)
    request = StoppedEventRequest(
        flow=PolynomialFlow((Poly.constant(3, 1), 3 * x**2)),
        initial=(Poly.constant(0, Fraction(-1, 2)), Poly.constant(0, Fraction(-1, 8))),
        parameters=(), target=PolynomialEvent(y), step=1.0, max_steps=1,
        order=4, derivative_order=0,
    )
    result = certify_stopped_event(request)
    assert result.status == "unresolved"
    assert "transverse" in result.reason
    assert result.slabs[0].start_values[0].hi < 0 < result.slabs[0].end_values[0].lo
    assert result.slabs[0].normal_velocities[0].contains_zero()
    assert not verify_stopped_event(result)


def test_multiple_crossings_in_one_step_cannot_certify_a_later_first_hit() -> None:
    x = Poly.variable(2, 0)
    event = PolynomialEvent((x - 1) * (x - 2) * (x - 3))
    coarse = certify_stopped_event(_constant_flow_request(target=event, step=4.0, max_steps=1))
    assert coarse.status == "unresolved"
    fine = certify_stopped_event(_constant_flow_request(target=event, step=0.0625, max_steps=64))
    assert fine.certified, fine.reason
    assert fine.time_bracket is not None and fine.time_bracket.contains(1)
    assert fine.time_bracket.hi < 2


def test_hidden_earlier_pair_with_equal_endpoint_signs_is_not_skipped() -> None:
    x = Poly.variable(2, 0)
    event = PolynomialEvent((x - Fraction(1, 4)) * (x - Fraction(3, 4)) * (x - Fraction(5, 4)))
    result = certify_stopped_event(_constant_flow_request(target=event, step=1.0, max_steps=2))
    assert result.status == "unresolved"
    assert len(result.slabs) == 1  # refuses the first unresolved slab
    assert result.slabs[0].start_values[0].hi < 0
    assert result.slabs[0].end_values[0].hi < 0


def test_flow_blowup_is_an_unresolved_validated_computation() -> None:
    x = Poly.variable(2, 0)
    request = _constant_flow_request(
        flow=PolynomialFlow((x**2,)), initial=(Poly.constant(0, 1),),
        target=PolynomialEvent(x - 5), step=2.0, max_steps=1,
    )
    result = certify_stopped_event(request)
    assert result.status == "unresolved" and "validated computation stopped" in result.reason


def test_competing_stop_before_target_cannot_be_ignored() -> None:
    x = Poly.variable(2, 0)
    request = _constant_flow_request(competitors=(PolynomialEvent(x - Fraction(1, 2), name="exit"),))
    result = certify_stopped_event(request)
    assert result.status == "unresolved" and "competing" in result.reason
    assert result.return_box is None


def test_finite_horizon_exclusion_and_bracket_incompletion_differ() -> None:
    excluded = certify_stopped_event(_constant_flow_request(max_steps=2))
    assert excluded.status == "excluded"
    incomplete = certify_stopped_event(_constant_flow_request(step=1.0, max_steps=1))
    assert incomplete.status == "unresolved"


def test_initial_identity_departure_and_positive_return() -> None:
    # Exact initial incidence; first positive whole-line return is at pi.
    x, y = Poly.variable(4, 0), Poly.variable(4, 1)
    h = Poly.variable(1, 0)
    request = StoppedEventRequest(
        flow=PolynomialFlow((y, -x), 1),
        initial=(Poly.constant(1, 0), h), parameters=(Interval(1, 1.001),),
        target=PolynomialEvent(x, direction=-1), step=0.0625, max_steps=60, order=8,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.slabs[0].target_action == "initial_departure"
    assert result.time_bracket is not None and result.time_bracket.contains(math.pi)
    assert result.return_box is not None and result.return_box[1].contains(-1)
    assert result.return_jacobian is not None and result.return_jacobian[1][0].contains(-1)
    opposite = certify_stopped_event(replace(request, target=PolynomialEvent(x, direction=1)))
    assert opposite.status == "unresolved" and "opposite" in opposite.reason


def test_guarded_oscillator_returns_after_full_period_with_physical_jets() -> None:
    x, y = Poly.variable(4, 0), Poly.variable(4, 1)
    h = Poly.variable(1, 0)
    request = StoppedEventRequest(
        flow=PolynomialFlow((y, -x), 1),
        initial=(Poly.constant(1, 0), h), parameters=(Interval(1, 1.0001),),
        target=PolynomialEvent(x, direction=1, guard=y),
        step=0.0625, max_steps=110, order=8,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.time_bracket is not None and result.time_bracket.contains(2 * math.pi)
    assert result.time_bracket.lo > 6
    assert any(s.target_action == "guard_exclusion" for s in result.slabs)
    assert result.return_box is not None and result.return_box[1].contains(1)
    assert result.return_jacobian is not None and result.return_jacobian[1][0].contains(1)
    assert result.return_hessian is not None and result.return_hessian[1][0][0].contains(0)
    assert result.event_time_gradient is not None and result.event_time_gradient[0].contains(0)


def test_uncertain_guard_does_not_certify_or_skip_a_possible_event() -> None:
    x = Poly.variable(2, 0)
    result = certify_stopped_event(_constant_flow_request(target=PolynomialEvent(x - 1, guard=x - 1)))
    assert result.status == "unresolved" and "guard" in result.reason
    ignored = certify_stopped_event(_constant_flow_request(
        target=PolynomialEvent(x - 1, guard=Poly.constant(2, -1))))
    assert ignored.status == "excluded"


def test_initial_grazing_or_competing_incidence_remains_unresolved() -> None:
    x = Poly.variable(2, 0)
    grazing = certify_stopped_event(_constant_flow_request(target=PolynomialEvent(x**2)))
    assert grazing.status == "unresolved" and "departure" in grazing.reason
    competitor = certify_stopped_event(_constant_flow_request(competitors=(PolynomialEvent(x),)))
    assert competitor.status == "unresolved" and "initial source" in competitor.reason


def test_exact_interval_provider_and_input_contracts() -> None:
    p = Poly(1, {(3,): Fraction(2, 3), (0,): Fraction(1, 10)})
    bound = polynomial_interval(p, (Interval(-0.3, 0.4),))
    assert bound.contains(float(p.evaluate((Fraction(1, 7),))))
    with pytest.raises(ValueError, match="dimension"):
        polynomial_interval(p, ())
    with pytest.raises(ValueError, match="clock"):
        PolynomialFlow((Poly.constant(1, 1),))
    with pytest.raises(ValueError, match="direction"):
        PolynomialEvent(Poly.constant(2, 1), direction=2)
    with pytest.raises(ValueError, match="finite"):
        _constant_flow_request(step=math.inf)
    with pytest.raises(TypeError, match="binary float"):
        _constant_flow_request(step=Fraction(1, 10))


def test_first_derivative_only_has_no_manufactured_second_jet() -> None:
    p = Poly.variable(1, 0)
    x = Poly.variable(3, 0)
    request = StoppedEventRequest(
        flow=PolynomialFlow((Poly.constant(3, 1),), 1), initial=(p,),
        parameters=(Interval(0, 0.01),), target=PolynomialEvent(x - 1),
        derivative_order=1, order=4,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.event_time_gradient is not None and result.event_time_gradient[0].contains(-1)
    assert result.event_time_hessian is None and result.return_hessian is None
