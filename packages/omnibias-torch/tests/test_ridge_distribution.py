# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent quadrature, weak integration by parts and trainable geometry."""

from fractions import Fraction as F
from math import factorial, pi, sqrt

import mpmath as mp
import numpy as np
import pytest
import torch
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_distribution import RidgeDistribution, collapse_dirac_cluster
from omnibias.core.ridge_system import RidgeSystem
from omnibias.torch.ridge_distribution import gaussian_weak
from scipy.integrate import quad


@pytest.mark.parametrize("index", [-7, -4, -2, -1, 0, 1, 2, 5])
@pytest.mark.parametrize("mean", [-1.3, 0.0, 1.7])
def test_pairing_against_high_precision_integrals_and_derivatives(index, mean) -> None:
    field = RidgeDistribution((1,), ((index, (1,)),))
    value = gaussian_weak(field, torch.tensor(mean, dtype=torch.float64), 1.3).item()
    with mp.workdps(55):
        m, v = mp.mpf(mean), mp.mpf("1.3")
        def density(z):
            return mp.exp(-(z-m)**2/(2*v)) / mp.sqrt(2*mp.pi*v)
        if index < 0:
            order = -index - 1
            expected = (-1)**order * mp.diff(density, 0, order)
        else:
            expected = mp.quad(lambda z: z**index*density(z)/factorial(index), [0, mp.inf])
    assert value == pytest.approx(float(expected), rel=3e-11, abs=3e-13)


def test_weak_poisson_source_matches_integration_by_parts() -> None:
    relu = RidgeDistribution((1,), ((1, (1,)),))
    operator = RidgeSystem.from_operators(((OperatorSpec.laplacian(),),), (1,))
    source = relu.apply(operator)
    means = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64)
    actual = gaussian_weak(source, means, 0.4)[:, 0].numpy()
    for mean, value in zip(means.tolist(), actual, strict=True):
        def integrand(z, mean=mean):
            density = np.exp(-(z-mean)**2/0.8)/sqrt(0.8*pi)
            return z * (((z-mean)**2/0.4**2)-1/0.4)*density
        expected, _ = quad(integrand, 0, np.inf, epsabs=1e-12)
        assert value == pytest.approx(expected, rel=2e-11, abs=2e-12)


def test_scalar_bias_projected_variance_and_second_parameter_gradients() -> None:
    field = RidgeDistribution((1,), ((-3, (1,)), (1, (2,))))
    mean = torch.tensor([-0.3, 0.5], dtype=torch.float64, requires_grad=True)
    variance = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    bias = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    def fn(m, v, b):
        return gaussian_weak(field, m, v, b)
    assert torch.autograd.gradcheck(fn, (mean, variance, bias))
    assert torch.autograd.gradgradcheck(fn, (mean, variance, bias))


def test_normal_scaling_is_pullback_not_unit_surface_measure() -> None:
    # A standard 2-D Gaussian paired with delta(3x+4y) is p_(3X+4Y)(0),
    # i.e. 1/(5 sqrt(2pi)), not the unit-surface answer 1/sqrt(2pi).
    field = RidgeDistribution((3, 4), ((-1, (1,)),))
    actual = gaussian_weak(field, torch.tensor(0.0, dtype=torch.float64), 25).item()
    assert actual == pytest.approx(1/(5*sqrt(2*pi)))


def test_zero_map_preserves_zero_parameter_gradients_and_dtype() -> None:
    zero = RidgeDistribution((1,), (), outputs=2)
    mean = torch.tensor([0.1, 0.2], dtype=torch.float32, requires_grad=True)
    variance = torch.tensor(1.0, dtype=mean.dtype, requires_grad=True)
    value = gaussian_weak(zero, mean, variance)
    assert value.dtype == torch.float32 and value.shape == (2, 2)
    gradients = torch.autograd.grad(value.sum(), (mean, variance))
    assert all(torch.all(g == 0) for g in gradients)


def test_invalid_domain_and_shape_refusals() -> None:
    field = RidgeDistribution((1,), ((-1, (1,)),))
    values = gaussian_weak(field, torch.zeros(4), torch.tensor([0., -1., float("inf"), 1.]))
    assert torch.isnan(values[:3]).all() and torch.isfinite(values[3]).all()
    with pytest.raises(TypeError):
        gaussian_weak(field, torch.tensor([1]), 1.)
    with pytest.raises(ValueError, match="scalar"):
        gaussian_weak(field, torch.zeros(2), 1., torch.zeros(2))


@pytest.mark.parametrize("order", range(7))
def test_cluster_bound_against_independent_high_precision_pairing(order) -> None:
    shifts, strengths = (-F(1, 80), F(1, 100)), (F(-2), F(3))
    collapse = collapse_dirac_cluster((1,), shifts, strengths, max_order=order,
                                      min_probe_std=F(1, 5))
    with mp.workdps(70):
        def exact(value):
            return mp.mpf(value.numerator)/value.denominator
        for mean, std in ((mp.mpf("-0.4"), mp.mpf("0.2")),
                          (mp.mpf("0.3"), mp.mpf("0.7"))):
            def density(shift, mean=mean, std=std):
                return mp.exp(-(mean+shift)**2/(2*std**2))/(mp.sqrt(2*mp.pi)*std)
            reference = sum(exact(a)*density(exact(s)) for a, s in zip(strengths, shifts, strict=True))
            confluent = sum(exact(row[0])*mp.diff(density, 0, -index-1)
                            for index, row in collapse.field.terms)
            assert abs(reference-confluent) <= exact(collapse.weak_error_bound)
