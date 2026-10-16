# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Prescribed periodic data, derivative towers, and independent heat solutions."""

from fractions import Fraction

import pytest
from omnibias.core.exp_polynomial import ExpPolynomial
from omnibias.core.periodic_heat import PeriodicHeatPlan
from omnibias.core.rational_field import RationalField
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.periodic_heat import enclose_burgers_jet, enclose_heat_derivative


def _contains(mp, interval, value):
    assert mp.mpf(interval.lo) <= value <= mp.mpf(interval.hi)


def _scalar_reference(mp, x, time, viscosity, derivative=0, cutoff=50):
    # exp(cos x) has Fourier coefficients I_n(1). This computes the true
    # heat solution without using the finite Taylor-in-potential compiler.
    result = mp.besseli(0, 1) if derivative == 0 else mp.mpf(0)
    for n in range(1, cutoff + 1):
        result += 2 * mp.besseli(n, 1) * mp.exp(-viscosity*n*n*time) * n**derivative * mp.cos(
            n*x + derivative*mp.pi/2)
    return result


def test_prescribed_sine_initial_data_and_raw_high_order_derivatives():
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_velocity({}, {1: 1}, mean=Fraction(1, 3),
                                         viscosity=Fraction(1, 2), degree=24)
    with mp.workdps(80):
        for x in (0.125, 0.5):
            jet = enclose_burgers_jet(plan, (Interval.point(x),), Interval.point(0.0), order=6)
            for order in range(7):
                expected = mp.sin(mp.mpf(x) + order*mp.pi/2)
                if order == 0:
                    expected += mp.mpf(1)/3
                _contains(mp, jet.derivative((order,))[0], expected)


def test_periodic_solution_has_both_spatial_slope_signs():
    pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_velocity({}, {1: 1}, viscosity=Fraction(1, 2), degree=18)
    ascending = enclose_burgers_jet(plan, (Interval.point(0.0),), Interval.point(0.0), order=1)
    descending = enclose_burgers_jet(plan, (Interval.point(3.0),), Interval.point(0.0), order=1)
    assert ascending.derivative((1,))[0].lo > 0.0
    assert descending.derivative((1,))[0].hi < 0.0


def test_true_heat_derivatives_and_galilean_velocity_against_bessel_solution():
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_velocity({}, {1: 1}, mean=Fraction(1, 3),
                                         viscosity=Fraction(1, 2), degree=18)
    x, time = 0.375, 0.25
    with mp.workdps(80):
        for order in range(4):
            bound = enclose_heat_derivative(plan, (Interval.point(x),), Interval.point(time), (order,))
            expected = _scalar_reference(mp, mp.mpf(x), mp.mpf(time), mp.mpf(1)/2, order)
            doubled = _scalar_reference(mp, mp.mpf(x), mp.mpf(time), mp.mpf(1)/2, order, cutoff=100)
            assert abs(expected - doubled) < mp.mpf("1e-70")
            _contains(mp, bound.value, expected)
        shifted = mp.mpf(x) - mp.mpf(time)/3

        def velocity(position):
            return mp.mpf(1)/3 - _scalar_reference(mp, position, mp.mpf(time), mp.mpf(1)/2, 1) / (
                _scalar_reference(mp, position, mp.mpf(time), mp.mpf(1)/2))

        jet = enclose_burgers_jet(plan, (Interval.point(x),), Interval.point(time), order=3)
        for order in range(4):
            _contains(mp, jet.derivative((order,))[0], mp.diff(velocity, shifted, order))


def test_increasing_expansion_degree_tightens_solution_for_fixed_prescribed_data():
    mp = pytest.importorskip("mpmath")
    widths = []
    intervals = []
    for degree in (6, 10, 14):
        plan = PeriodicHeatPlan.from_velocity({}, {1: 1}, viscosity=Fraction(1, 2), degree=degree)
        interval = enclose_burgers_jet(plan, (Interval.point(0.5),), Interval.point(0.5)).value[0]
        intervals.append(interval)
        widths.append((Interval.point(interval.hi) - Interval.point(interval.lo)).hi)
    with mp.workdps(80):
        expected = -_scalar_reference(mp, mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("0.5"), 1) / (
            _scalar_reference(mp, mp.mpf("0.5"), mp.mpf("0.5"), mp.mpf("0.5")))
        for interval in intervals:
            _contains(mp, interval, expected)
    assert widths[1] < widths[0]/100
    assert widths[2] < widths[1]/100
    assert widths[2] < 1e-7


def test_missing_mean_mode_is_not_damped_as_a_high_frequency_tail():
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_velocity({}, {1: 1}, viscosity=Fraction(1, 2), degree=1)
    bound = enclose_heat_derivative(plan, (Interval.point(0.0),), Interval.point(100.0))
    with mp.workdps(80):
        expected = _scalar_reference(mp, mp.mpf(0), mp.mpf(100), mp.mpf("0.5"))
        _contains(mp, bound.value, expected)
        # Q_1(cos x) has mean1; exp(cos x) has meanI_0(1)>1.26.
        # Heat cannot damp this omitted zero-frequency contribution.
        assert expected - mp.mpf(bound.finite_part.hi) > mp.mpf("0.26")
    assert bound.remainder_bound > 0.26


def test_coupled_two_dimensional_prescribed_potential_and_evolved_velocity():
    mp = pytest.importorskip("mpmath")
    plan = PeriodicHeatPlan.from_log_potential(
        2, {(1, 0): 1, (1, 1): Fraction(1, 2)}, viscosity=Fraction(1, 2), degree=14,
        drift=(Fraction(1, 4), Fraction(-1, 3)))
    x, y, time = 0.25, -0.125, 0.375
    at_initial = enclose_burgers_jet(plan, (Interval.point(x), Interval.point(y)), Interval.point(0.0), order=1)
    evolved = enclose_burgers_jet(plan, (Interval.point(x), Interval.point(y)), Interval.point(time), order=1)
    with mp.workdps(80):
        mx, my = mp.mpf(x), mp.mpf(y)
        initial = (mp.mpf(1)/4 + mp.sin(mx) + mp.sin(mx + my)/2,
                   -mp.mpf(1)/3 + mp.sin(mx + my)/2)
        for interval, expected in zip(at_initial.value, initial, strict=True):
            _contains(mp, interval, expected)
        # Both velocity components depend on both physical coordinates.
        assert at_initial.derivative((0, 1))[0].lo > 0.0
        assert at_initial.derivative((1, 0))[1].lo > 0.0
        sx, sy = mx - mp.mpf(time)/4, my + mp.mpf(time)/3
        heat = [mp.mpf(0), mp.mpf(0), mp.mpf(0)]
        bessel_one = {n: mp.besseli(abs(n), 1) for n in range(-50, 51)}
        bessel_half = {n: mp.besseli(abs(n), mp.mpf("0.5")) for n in range(-25, 26)}
        # exp(cos x + cos(x+y)/2) has coefficient I_(kx-ky)(1) I_ky(1/2).
        for kx in range(-25, 26):
            for ky in range(-25, 26):
                coefficient = bessel_one[kx-ky] * bessel_half[ky]
                factor = coefficient * mp.exp(-mp.mpf(time)*(kx*kx + ky*ky)/2)
                phase = kx*sx + ky*sy
                heat[0] += factor * mp.cos(phase)
                heat[1] -= kx * factor * mp.sin(phase)
                heat[2] -= ky * factor * mp.sin(phase)
        expected = (mp.mpf(1)/4 - heat[1]/heat[0], -mp.mpf(1)/3 - heat[2]/heat[0])
        for interval, value in zip(evolved.value, expected, strict=True):
            _contains(mp, interval, value)


def test_compiled_finite_heat_field_has_exact_burgers_identity_on_its_regular_domain():
    # Real polynomial lift (t,c,s), with c=cos(x), s=sin(x). The physical
    # derivative is -s D_c+c D_s, not either coordinate derivative alone.
    nu = Fraction(1, 2)
    plan = PeriodicHeatPlan.from_log_potential(1, {(1,): 1}, {(1,): Fraction(1, 4)},
                                              viscosity=nu, degree=3)
    one = ExpPolynomial.constant(1, 3)
    c = ExpPolynomial.monomial((0, 1, 0))
    s = ExpPolynomial.monomial((0, 0, 1))
    harmonic = {0: (one, one.scale(0))}
    for n in range(1, 4):
        real, imaginary = harmonic[n-1]
        harmonic[n] = (real*c - imaginary*s, real*s + imaginary*c)
    tau = one.scale(0)
    for (frequency,), real, imaginary in plan.heat_coefficients:
        if frequency < 0:
            continue
        if frequency == 0:
            assert imaginary == 0
            tau += one.scale(real)
        else:
            a, b = harmonic[frequency]
            tau += (a.scale(2*real) - b.scale(2*imaginary)) * ExpPolynomial.exponential(
                (-nu*frequency**2, 0, 0))

    def dx(field):
        return -s*field.differentiate(1) + c*field.differentiate(2)

    assert (tau.differentiate(0) - dx(dx(tau)).scale(nu)).is_zero()
    velocity = RationalField(dx(tau), tau).scale(-2*nu)
    c_field, s_field = RationalField.from_polynomial(c), RationalField.from_polynomial(s)

    def velocity_dx(field):
        return -s_field*field.differentiate(1) + c_field*field.differentiate(2)

    residual = (velocity.differentiate(0) + velocity*velocity_dx(velocity)
                - velocity_dx(velocity_dx(velocity)).scale(nu))
    assert residual.is_zero()
    assert residual.nonzero_conditions  # A finite Taylor potential need not be globally positive.
