# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Positive derivative tails and certified physical-age derivative contracts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import mpmath as mp
import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_aging import certify_gaussian_inverse_aging
from omnibias.core.verified.gaussian_aging_derivative import (
    GaussianAgingDerivative,
    _rising_tail,
    certify_gaussian_aging_derivative,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval


def _mp(value):
    rational = F(value)
    return mp.mpf(rational.numerator)/rational.denominator


def _source(order=8, **kwargs):
    model = GaussianHeatMap(((-1,), (1,)), (0, F(1, 3)), time=16, **kwargs)
    result = certify_gaussian_inverse_aging(model, (Interval.point(0.25),), order=order)
    assert result is not None
    return result


@pytest.mark.parametrize("order", (0, 1, 2, 4, 8))
@pytest.mark.parametrize("start", (1, 3, 9))
@pytest.mark.parametrize("ratio", (F(0), F(1, 64), F(1, 3), F(15, 16)))
def test_positive_tail_matches_independent_rational_function_derivative(order, start, ratio):
    actual = _rising_tail(ratio, start, order, _Arithmetic(RidgeBudget()))
    with mp.workdps(100):
        r = _mp(ratio)
        # d^k[r^(K+k-1)/(1-r)]*r generates rising-factorial coefficients.
        reference = r*mp.diff(lambda x: x**(start+order-1)/(1-x), r, order)
        assert mp.almosteq(_mp(actual), reference, rel_eps=mp.mpf("1e-90"), abs_eps=mp.mpf("1e-90"))


def test_first_derivative_tail_has_closed_form():
    source = _source(8)
    result = certify_gaussian_aging_derivative(source)
    assert result is not None
    r, n = 3*source.model.contraction_bound, source.order
    expected = r**(n+1)*((n+1)-n*r)/(16*(1-r)**2)
    assert result.approximation_error_bound == expected
    assert result.enclosure[0].lo < result.polynomial_enclosure[0].lo
    assert result.polynomial_enclosure[0].hi < result.enclosure[0].hi


def test_order_zero_retains_original_value_bounds():
    source = _source(1)
    result = certify_gaussian_aging_derivative(source, 0)
    assert result is not None
    assert result.polynomial_enclosure == source.predictor_enclosure
    assert result.enclosure == source.enclosure
    assert result.approximation_error_bound == source.approximation_error_bound


@pytest.mark.parametrize("order", (1, 2, 5, 8))
def test_derivatives_above_truncation_degree_remain_nonzero(order):
    source = _source(1)
    result = certify_gaussian_aging_derivative(source, order)
    assert result is not None
    with mp.workdps(90):
        mean = mp.tanh(mp.mpf(1)/8+mp.mpf(1)/6)
        first_coefficient = mean*(1-mean*mean)/32
        reference = (-1)**order*mp.factorial(order)*first_coefficient/16**order
        if order == 1:
            reference += mp.mpf(1)/4
        assert reference != 0
        assert _mp(result.polynomial_enclosure[0].lo) <= reference <= _mp(result.polynomial_enclosure[0].hi)


@pytest.mark.parametrize("order", (0, 1, 2, 12))
def test_affine_derivatives_are_exact_without_backend(monkeypatch, order):
    model = GaussianHeatMap(((-10**400,),), (10**400,), time=10**400)
    source = certify_gaussian_inverse_aging(model, (Interval.point(1),), order=8)
    assert source is not None
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    result = certify_gaussian_aging_derivative(source, order)
    assert result is not None
    expected = (Interval.point(int(order == 1)),)
    assert result.polynomial_enclosure == result.enclosure == expected
    assert result.approximation_error_bound == 0


def test_derivative_uses_only_source_provenance_and_snapshots_dependency(monkeypatch):
    source = _source()
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    result = certify_gaussian_aging_derivative(source, 2)
    assert result is not None and transcend.libm_fallback_used()
    assert result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    source.certificate["payload"]["order"] = 0
    assert verify_certificate_digest(result.certificate)
    assert not verify_certificate_digest(source.certificate)


@pytest.mark.parametrize("damage", ("payload", "unserializable", "metadata", "backend"))
def test_damaged_dependency_is_refused(damage):
    source = _source()
    if damage == "payload":
        source.certificate["payload"]["order"] = 0
    elif damage == "unserializable":
        source.certificate["payload"]["order"] = object()
    else:
        if damage == "metadata":
            source.certificate["meta"] = []
        else:
            source.certificate["meta"]["transcend_backend"] = "unverified"
        source.certificate.update(seal_certificate(source.certificate))
    assert certify_gaussian_aging_derivative(source) is None


@pytest.mark.parametrize("order,error", [(True, TypeError), (1.0, TypeError), (-1, ValueError)])
def test_invalid_orders_raise(order, error):
    with pytest.raises(error):
        certify_gaussian_aging_derivative(_source(), order)


def test_factory_frozen_record_and_resource_refusal():
    with pytest.raises(TypeError):
        GaussianAgingDerivative()
    with pytest.raises(TypeError):
        certify_gaussian_aging_derivative(None)
    source = _source()
    result = certify_gaussian_aging_derivative(source)
    assert result is not None
    with pytest.raises(FrozenInstanceError):
        result.order = 2
    assert certify_gaussian_aging_derivative(source, 129) is None
    constrained = _source(1, budget=RidgeBudget(max_coefficient_bits=256))
    assert certify_gaussian_aging_derivative(constrained, 80) is None


def test_strict_backend_policy_remains_in_force(monkeypatch):
    source = _source()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        certify_gaussian_aging_derivative(source)
