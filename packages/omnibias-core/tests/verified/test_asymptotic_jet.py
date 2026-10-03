# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent high-precision and exact oracles for confluent scale calculus."""

import math
import random
from fractions import Fraction as Q

import pytest
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.core.verified.asymptotic_jet import (
    fixed_product_jet,
    power_compensator,
    signed_root_primitive,
    verify_fixed_product_derivative,
    weighted_scale_derivative,
)
from omnibias.core.verified.interval import Interval as I

mp = pytest.importorskip("mpmath")


def contains(box, truth):
    assert mp.mpf(box.lo) <= truth <= mp.mpf(box.hi), (box, truth)


def power_reference(a, b, x, p=0, q=0, k=0):
    def value(aa, bb, logx):
        d = aa - bb
        return (mp.exp(bb * logx) * mp.expm1(d * logx) / d
                if d else mp.exp(aa * logx) * logx)
    return mp.diff(value, (mp.mpf(a), mp.mpf(b), mp.log(mp.mpf(x))), (p, q, k))


@pytest.mark.parametrize("a,b,x", [
    (0.0, 0.0, 0.5), (0.75, 0.75, 0.125), (1e-30, -1e-30, 0.01),
    (1.0, 1.0 + 1e-12, 0.001), (-0.1, 0.2, 0.7), (-2.0, 3.0, 2.0),
])
@pytest.mark.parametrize("orders", [(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 2), (0, 0, 4)])
def test_power_mixed_derivatives_include_diagonal_and_tiny_opposite_splits(a, b, x, orders):
    p, q, k = orders
    with mp.workdps(110):
        box = power_compensator(a, b, x, a_order=p, b_order=q, log_order=k)
        contains(box, power_reference(a, b, x, p, q, k))


def test_power_diagonal_is_analytic_extension_and_retains_independent_parameter_axes():
    with mp.workdps(110):
        a, x = mp.mpf(0.5), mp.mpf(0.125)
        contains(power_compensator(float(a), float(a), float(x)), x**a * mp.log(x))
        # partial_a C on a=b is HALF the derivative of the diagonal function.
        box = power_compensator(float(a), float(a), float(x), a_order=1)
        independent = x**a * mp.log(x)**2 / 2
        contains(box, independent)
        assert not (mp.mpf(box.lo) <= 2*independent <= mp.mpf(box.hi))
        for order in (0, 1, 2, 3):
            contains(power_compensator(0, 0, 1, log_order=order), int(order == 1))


def test_power_parameter_boxes_dense_grid_and_random_including_diagonal():
    rng = random.Random(16031)
    boxes = [(I(-0.01, 0.01), I(-0.02, 0.02), I(0.2, 0.3)),
             (I(-2, -1.8), I(1.1, 1.2), I(1.8, 2.0))]
    with mp.workdps(100):
        for aa, bb, xx in boxes:
            for p, q, k in ((0, 0, 0), (1, 1, 2)):
                result = power_compensator(aa, bb, xx, a_order=p, b_order=q, log_order=k)
                points = [(a, b, x) for a in (aa.lo, aa.mid, aa.hi)
                          for b in (bb.lo, bb.mid, bb.hi) for x in (xx.lo, xx.mid, xx.hi)]
                points += [(rng.uniform(aa.lo, aa.hi), rng.uniform(bb.lo, bb.hi),
                            rng.uniform(xx.lo, xx.hi)) for _ in range(12)]
                for a, b, x in points:
                    contains(result, power_reference(a, b, x, p, q, k))


def test_power_underflow_and_explicit_series_tail():
    with mp.workdps(130):
        for a, b in ((2.0, 2.0), (2.0, 2.000001)):
            result = power_compensator(a, b, 1e-200)
            truth = power_reference(a, b, 1e-200)
            contains(result, truth)
            assert truth < 0 and all(math.isfinite(v) for v in (result.lo, result.hi))
        coarse = power_compensator(0.1, 0, 0.2, terms=1)
        fine = power_compensator(0.1, 0, 0.2, terms=20)
        truth = power_reference(0.1, 0, 0.2)
        contains(coarse, truth)
        contains(fine, truth)
        assert fine.width < coarse.width / 1000


def root_reference(delta, value, p=0, endpoint_order=0):
    def primitive(d, v):
        if d > 0:
            return mp.atan(mp.sqrt(d)*v)/mp.sqrt(d)
        if d < 0:
            return mp.atanh(mp.sqrt(-d)*v)/mp.sqrt(-d)
        return v
    return mp.diff(primitive, (mp.mpf(delta), mp.mpf(value)), (p, endpoint_order))


@pytest.mark.parametrize("delta,value", [(0., 0.7), (1e-30, -0.7), (-1e-30, 0.7),
                                         (0.1, 1.0), (-0.1, -1.0), (2., 1.0), (-0.8, 1.0)])
@pytest.mark.parametrize("p,endpoint", [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (2, 1)])
def test_signed_root_regimes_against_atan_atanh(delta, value, p, endpoint):
    with mp.workdps(100):
        result = signed_root_primitive(delta, value, delta_order=p, endpoint_order=endpoint)
        contains(result, root_reference(delta, value, p, endpoint))


def test_signed_root_interval_zero_crossing_and_negative_endpoint():
    rng = random.Random(16032)
    with mp.workdps(100):
        for domain in (I(-0.1, 0.1), I(-0.8, -0.7)):
            vv = I(-1, 0.9)
            for p in (0, 1, 2):
                result = signed_root_primitive(domain, vv, delta_order=p)
                points = [(d, v) for d in (domain.lo, domain.mid, domain.hi)
                          for v in (vv.lo, vv.mid, vv.hi)]
                points += [(rng.uniform(domain.lo, domain.hi), rng.uniform(vv.lo, vv.hi))
                           for _ in range(20)]
                for d, v in points:
                    contains(result, root_reference(d, v, p))


def test_signed_root_exact_diagonal_series_remainder_and_pole_refusal():
    with mp.workdps(100):
        for p in range(5):
            result = signed_root_primitive(0, Q(-3, 4), delta_order=p)
            exact = (-1)**p * Q(math.factorial(p), 2*p+1) * Q(-3, 4)**(2*p+1)
            contains(result, mp.mpf(exact.numerator)/exact.denominator)
        coarse = signed_root_primitive(-0.2, 1, terms=1)
        contains(coarse, root_reference(-0.2, 1))
        assert signed_root_primitive(-0.2, 1).width < coarse.width/1000
    for delta, v in ((-1, 1), (-1, -1), (I(-1, 1), 1), (-0.5, I(-2, 0))):
        with pytest.raises(ValueError, match="pole"):
            signed_root_primitive(delta, v)


def test_fixed_product_derivatives_retain_acceleration_and_reject_wrong_claim():
    w, u = SparsePolynomial.variable(2, 0), SparsePolynomial.variable(2, 1)
    product = w*u
    assert not weighted_scale_derivative(product, order=2).terms
    assert not verify_fixed_product_derivative(product, -2*product, order=2)
    assert verify_fixed_product_derivative(product, SparsePolynomial.constant(2, 0), order=2)
    assert fixed_product_jet(product, I(0.1, 0.2), I(0.4, 0.5), order=4)[1:] == (I.point(0),)*4
    polynomial = w**2 + 3*w*u + Q(2, 3)*u**3 - w*u**2
    with mp.workdps(100):
        for omega, value in ((0.125, 0.25), (0.01, 0.3), (0.4, 0.2)):
            jet = fixed_product_jet(polynomial, omega, value, order=6)
            def reference(s, w0=omega, u0=value):
                a, b = mp.mpf(w0)*mp.exp(-s), mp.mpf(u0)*mp.exp(s)
                return a*a + 3*a*b + mp.mpf(2)/3*b**3 - a*b*b
            for k, enclosure in enumerate(jet):
                contains(enclosure, mp.diff(reference, 0, k))


def test_fixed_product_extra_axes_are_fixed_and_invalid_requests_fail():
    p = SparsePolynomial(3, {(2, 1, 3): 2})
    assert weighted_scale_derivative(p, order=3, omega_axis=2, u_axis=0).terms == (
        ((2, 1, 3), Q(-2)),
    )
    with pytest.raises(ValueError, match="two-variable"):
        fixed_product_jet(p, 1, 1)
    with pytest.raises(ValueError, match="positive"):
        fixed_product_jet(SparsePolynomial.constant(2, 1), 0, 1)
    with pytest.raises(ValueError, match="distinct"):
        weighted_scale_derivative(p, omega_axis=1, u_axis=1)
    with pytest.raises(ValueError):
        power_compensator(0, 0, I(0, 1))
    with pytest.raises(ValueError):
        power_compensator(0, 0, 1, a_order=True)
    with pytest.raises(ValueError, match="budget"):
        power_compensator(0, 0, 1, a_order=17)
    with pytest.raises(NotImplementedError):
        signed_root_primitive(0, 1, endpoint_order=2)
    with pytest.raises(ValueError):
        signed_root_primitive(0, 1, terms=0)


def test_confluent_power_refuses_conditional_transcendental_backend(monkeypatch):
    from omnibias.core.verified import transcend
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError, match="rigorous"):
        power_compensator(0, 0, 0.5)
