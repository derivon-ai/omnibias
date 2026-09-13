# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Independent nonlinear and nonautonomous checks of first-event sensitivities."""

import random
from dataclasses import replace
from fractions import Fraction

import pytest
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.core.verified.interval import Interval as I
from omnibias.dynamics.return_maps import (
    PolynomialEvent,
    PolynomialFlow,
    StoppedEventRequest,
    _advance,
    _JetSystem,
    certify_stopped_event,
    verify_stopped_event,
)

mp = pytest.importorskip("mpmath")


def _contains(box, truth):
    assert mp.mpf(box.lo) <= truth <= mp.mpf(box.hi), (box, truth)


def test_exponential_flow_moving_section_all_mixed_event_derivatives():
    # x'=a*x, x(0)=1, x(tau)=b: tau=log(b)/a and the stopped state is b.
    x, a, b = (P.variable(4, i) for i in range(3))
    request = StoppedEventRequest(
        PolynomialFlow((a*x,), 2), (P.constant(2, 1),), (I(1, 1.001), I(1.5, 1.501)),
        PolynomialEvent(x-b), step=0.02, max_steps=30, order=8,
    )
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.time_bracket and result.return_box
    assert result.event_time_gradient and result.event_time_hessian
    assert result.return_jacobian and result.return_hessian
    rng = random.Random(16041)
    points = [(1+i/4000, 1.5+j/4000) for i in range(5) for j in range(5)]
    points += [(rng.uniform(1, 1.001), rng.uniform(1.5, 1.501)) for _ in range(20)]
    with mp.workdps(90):
        for av, bv in points:
            aa, bb = mp.mpf(av), mp.mpf(bv)
            logb = mp.log(bb)
            _contains(result.time_bracket, logb/aa)
            _contains(result.return_box[0], bb)
            derivatives = (-logb/aa**2, 1/(aa*bb))
            second = ((2*logb/aa**3, -1/(aa**2*bb)),
                      (-1/(aa**2*bb), -1/(aa*bb**2)))
            for i in range(2):
                _contains(result.event_time_gradient[i], derivatives[i])
                _contains(result.return_jacobian[0][i], i)
                for j in range(2):
                    _contains(result.event_time_hessian[i][j], second[i][j])
                    _contains(result.return_hessian[0][i][j], 0)
    assert verify_stopped_event(result)
    assert not verify_stopped_event(replace(result, event_time_hessian=((I.point(0),)*2,)*2))


def test_explicit_clock_polynomial_and_frozen_parameter_clamping():
    # x'=clock²+p; x=t³/3+p*t. At x=1, tau_p=-tau/(tau²+p),
    # tau_pp=2*p*tau/(tau²+p)^3. Mixed clock/state terms are essential.
    x, p, clock = (P.variable(3, i) for i in range(3))
    request = StoppedEventRequest(
        PolynomialFlow((clock*clock+p,), 1), (P.constant(1, 0),), (I(0.5, 0.501),),
        PolynomialEvent(x-1), step=0.03, max_steps=60, order=6,
    )
    system = _JetSystem(request)
    state = system.initial()
    for index in range(12):
        tube, state = _advance(system, state, index)
        exact_clock = Fraction(request.step)*(index+1)
        assert state[1] == request.parameters[0]
        assert state[2] == I.from_rational(exact_clock)
        assert tube[1] == request.parameters[0]
        assert state[system.si(1, 0)] == I.point(1)
        assert state[system.si(2, 0)] == I.point(0)
        assert state[system.qi(1, 0, 0)] == I.point(0)
        assert state[system.qi(2, 0, 0)] == I.point(0)
    result = certify_stopped_event(request)
    assert result.certified, result.reason
    assert result.time_bracket and result.event_time_gradient and result.event_time_hessian
    assert result.return_box and result.return_jacobian and result.return_hessian
    rng = random.Random(16042)
    with mp.workdps(90):
        for parameter in [0.5+i/16000 for i in range(17)] + [rng.uniform(0.5, 0.501) for _ in range(20)]:
            pp = mp.mpf(parameter)
            tau = mp.findroot(lambda t, coefficient=pp: t**3/3+coefficient*t-1, 1.1)
            denominator = tau*tau+pp
            _contains(result.time_bracket, tau)
            _contains(result.event_time_gradient[0], -tau/denominator)
            _contains(result.event_time_hessian[0][0], 2*pp*tau/denominator**3)
            _contains(result.return_box[0], 1)
            _contains(result.return_jacobian[0][0], 0)
            _contains(result.return_hessian[0][0][0], 0)
