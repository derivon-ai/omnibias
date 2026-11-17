# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Derivative bounds, exact edge cases, and certificate provenance contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from math import factorial

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
from omnibias.core.verified.gaussian_chebyshev_derivative import (
    GaussianInverseDerivative,
    certify_gaussian_inverse_derivative,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube


def _approximation(*, degree=12, direction=1, affine=False, budget=None, width=F(1, 4), **kwargs):
    centers = ((1,), (1,)) if affine else ((-1,), (1,))
    model = GaussianHeatMap(centers, (0, F(1, 3)), time=2, budget=budget)
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    result = approximate_gaussian_inverse(tube, (F(1, 4),), (direction,), half_width=width,
                                          degree=degree, ellipse_parameter=2, **kwargs)
    assert result is not None
    return result


@pytest.fixture(scope="module")
def approximation():
    return _approximation(degree=24)


def test_order_zero_preserves_the_exact_polynomial_and_value_error(approximation):
    derivative = certify_gaussian_inverse_derivative(approximation, 0)
    assert derivative is not None
    assert derivative.order == 0 and derivative.degree == approximation.degree
    assert derivative.uniform_error_bounds == approximation.uniform_error_bounds
    assert derivative.coefficient_error_bounds == approximation.coefficient_error_bounds
    assert derivative.interpolation_error_bound == approximation.interpolation_error_bound
    for t in (F(-1, 4), F(1, 13), F(1, 4)):
        assert derivative.evaluate(t) == approximation.evaluate(t)
        assert derivative.enclose(t) == approximation.enclose(t)


@pytest.mark.parametrize("order", (1, 2, 3, 5))
def test_coefficient_uncertainty_is_amplified_by_exact_endpoint_derivative_norm(approximation, order):
    derivative = certify_gaussian_inverse_derivative(approximation, order)
    assert derivative is not None
    expected = F(0)
    for n in range(order, approximation.degree + 1):
        norm = F(1)
        for j in range(order):
            norm *= F(n*n-j*j, 2*j+1)
        nominal = F(approximation.coefficients[n][0])
        box = approximation.coefficient_enclosures[n][0]
        delta = max(abs(nominal-F(box.lo)), abs(F(box.hi)-nominal))
        expected += delta * norm / approximation.half_width**order
    assert derivative.coefficient_error_bounds == (expected,)
    assert derivative.uniform_error_bounds == (expected + derivative.interpolation_error_bound,)


@pytest.mark.parametrize("order", (2, 3, 6))
def test_derivative_beyond_degree_has_zero_polynomial_but_nonzero_true_tail(order):
    source = _approximation(degree=1)
    derivative = certify_gaussian_inverse_derivative(source, order)
    assert derivative is not None
    assert derivative.coefficients == ((0,),)
    assert derivative.evaluate(F(1, 17)) == (0,)
    assert derivative.coefficient_error_bounds == (0,)
    assert derivative.interpolation_error_bound > 0
    # For K=k the exact generating function is
    # sum C(n,k) r^n = 2^(k-1)*k!*r^k*(1+r)/(1-r)^(2k+1).
    # Ratios cancel the common outward M and isolate raw derivative factors.
    first = certify_gaussian_inverse_derivative(source, 2)
    assert first is not None
    r = F(1, 2)
    tail = 2**(order-1)*factorial(order)*r**order*(1+r)/(1-r)**(2*order+1)
    tail2 = 2*factorial(2)*r**2*(1+r)/(1-r)**5
    assert derivative.interpolation_error_bound / first.interpolation_error_bound == tail/tail2/source.half_width**(order-2)


@pytest.mark.parametrize("order", (0, 1, 2, 7))
def test_exact_affine_case_keeps_trend_and_uses_no_backend(monkeypatch, order):
    source = _approximation(affine=True, direction=3)
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    derivative = certify_gaussian_inverse_derivative(source, order)
    assert derivative is not None
    assert derivative.uniform_error_bounds == (0,)
    for t in (F(-1, 4), F(1, 17), F(1, 4)):
        expected = F(3, 2)+6*t if order == 0 else F(6) if order == 1 else F(0)
        assert derivative.evaluate(t) == (expected,)
        box = derivative.enclose(t)[0]
        assert F(box.lo) <= expected <= F(box.hi)
    assert derivative.certificate["meta"]["transcend_backend"] == "not_used"


@pytest.mark.parametrize("order", (1, 2, 9))
def test_constant_target_path_has_exact_zero_derivatives_despite_sample_uncertainty(order):
    source = _approximation(direction=0)
    assert source.coefficient_error_bounds[0] > 0
    derivative = certify_gaussian_inverse_derivative(source, order)
    assert derivative is not None
    assert derivative.uniform_error_bounds == (0,)
    assert derivative.evaluate(F(-1, 4)) == derivative.evaluate(F(1, 4)) == (0,)


def test_construction_inherits_backend_without_requiring_new_transcendentals(approximation, monkeypatch):
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    derivative = certify_gaussian_inverse_derivative(approximation, 2)
    assert derivative is not None
    assert derivative.certificate["meta"]["transcend_backend"] == "mpmath"
    assert derivative.certificate["meta"]["construction_arithmetic"] == "exact_rational"
    assert transcend.libm_fallback_used()
    assert verify_certificate_digest(derivative.certificate)
    assert "theorem_prover_verified" not in derivative.certificate["honesty"]


def test_dependency_is_checked_and_snapshotted():
    source = _approximation(degree=3)
    derivative = certify_gaussian_inverse_derivative(source, 1)
    assert derivative is not None
    saved = deepcopy(derivative.certificate)
    source.certificate["payload"]["degree"] = 999
    assert certify_gaussian_inverse_derivative(source, 1) is None
    assert derivative.certificate == saved
    assert verify_certificate_digest(derivative.certificate)


@pytest.mark.parametrize("damage", ("nonserializable", "metadata"))
def test_malformed_dependency_is_refused_without_hiding_public_input_errors(damage):
    source = _approximation(degree=1)
    if damage == "nonserializable":
        source.certificate["payload"]["damaged"] = object()
    else:
        source.certificate["meta"] = []
        source.certificate.update(seal_certificate(source.certificate))
    assert certify_gaussian_inverse_derivative(source, 1) is None


def test_strict_backend_policy_is_retained(approximation, monkeypatch):
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_MPMATH", None)
    monkeypatch.setattr(transcend, "_MPMATH_RESOLVED", True)
    with pytest.raises(RuntimeError):
        certify_gaussian_inverse_derivative(approximation, 1)


@pytest.mark.parametrize("order,error", [(-1, ValueError), (True, TypeError), (1.0, TypeError)])
def test_invalid_orders_raise(approximation, order, error):
    with pytest.raises(error):
        certify_gaussian_inverse_derivative(approximation, order)


def test_factory_frozen_record_and_parameter_domain(approximation):
    with pytest.raises(TypeError):
        certify_gaussian_inverse_derivative(None, 1)
    with pytest.raises(TypeError):
        GaussianInverseDerivative()
    derivative = certify_gaussian_inverse_derivative(approximation, 2)
    assert derivative is not None
    with pytest.raises(FrozenInstanceError):
        derivative.order = 0
    for method in (derivative.evaluate, derivative.enclose):
        for t in (F(-1, 3), F(1, 3)):
            with pytest.raises(ValueError):
                method(t)
        for t in (0.0, True):
            with pytest.raises(TypeError):
                method(t)


@pytest.mark.parametrize("budget,order,width", [
    (RidgeBudget(max_order=2), 3, F(1, 4)),
    (RidgeBudget(max_matrix_entries=1000), 128, F(1, 4)),
    (RidgeBudget(max_elimination_work=3000), 128, F(1, 4)),
    (RidgeBudget(max_coefficient_bits=64), 64, F(1, 1000)),
])
def test_resource_failure_is_inconclusive(budget, order, width):
    source = _approximation(affine=True, degree=1, budget=budget, width=width)
    assert certify_gaussian_inverse_derivative(source, order) is None
