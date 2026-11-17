# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original constrained parameter derivatives and independent symmetry laws."""

from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core._rational_linear import RationalLDLT
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_affine_adjoint import compile_gaussian_affine_adjoint
from omnibias.core.verified.gaussian_affine_parameters import certify_gaussian_affine_parameter_vjp
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.mean_map import MeanMap


def _geometry(centers=((-1, -2), (1, 2)), biases=(F(9, 4), F(1, 4)),
              constraints=((0, 1),), rhs=(1,), **kwargs):
    source = GaussianHeatMap(centers, biases, **kwargs)
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None
    geometry = compile_gaussian_affine_adjoint(restriction)
    assert geometry is not None
    return geometry


def _contains(interval, value):
    assert F(interval.lo) <= F(value) <= F(interval.hi)


def _box_contains(box, values):
    for interval, value in zip(box, values, strict=True):
        _contains(interval, value)


def _fixture_expected(result):
    assert result is not None
    _box_contains(result.root_enclosure, (0, 1))
    _box_contains(result.target_gradient, (2, 0))
    _box_contains(result.posterior, (F(1, 2), F(1, 2)))
    _box_contains(result.primal_multiplier, (-1,))
    _box_contains(result.adjoint_multiplier, (5,))
    _box_contains(result.rhs_gradient, (5,))
    _box_contains(result.constraint_gradient[0], (2, -5))
    _box_contains(result.bias_gradient, (-1, 1))
    _box_contains(result.center_gradient[0], (F(1, 2), F(-3, 2)))
    _box_contains(result.center_gradient[1], (F(1, 2), F(-1, 2)))
    _contains(result.viscosity_gradient, -2)
    _contains(result.time_gradient, -2)


@pytest.mark.parametrize("point,proposal", [((0, 1), (2, 0)), ((0.1, 1.5), (1.9, 100)), ((3, -5), (-4, 0))])
def test_exact_original_parameter_gradient_with_untrusted_proposals(point, proposal):
    geometry = _geometry()
    assert geometry.restriction.source_model.contraction_bound == F(5, 2)
    assert geometry.restriction.model.contraction_bound == F(1, 2)
    _fixture_expected(certify_gaussian_affine_parameter_vjp(geometry, (0, 0), point, (1, 3), proposal))


def test_normal_upstream_has_nonzero_constraint_derivatives():
    result = certify_gaussian_affine_parameter_vjp(_geometry(), (0, 0), (0, 1), (0, 3), (100, -100))
    assert result is not None and result.root_certificate is not None
    assert result.target_gradient == (Interval.point(0),)*2
    assert result.center_gradient == ((Interval.point(0),)*2,)*2
    assert result.bias_gradient == (Interval.point(0),)*2
    assert result.viscosity_gradient == result.time_gradient == Interval.point(0)
    assert result.posterior == () and result.primal_multiplier is None
    _box_contains(result.rhs_gradient, (3,))
    _box_contains(result.constraint_gradient[0], (0, -3))


def test_zero_tangent_diameter_retains_posterior_and_individual_center_gradients():
    geometry = _geometry(centers=((3, -2), (3, 2)))
    assert geometry.restriction.model.contraction_bound == 0
    result = certify_gaussian_affine_parameter_vjp(geometry, (2, 7), (999, -999), (2, 3), (-100, 100))
    assert result is not None and result.root_certificate is None
    _box_contains(result.root_enclosure, (5, 1))
    _box_contains(result.target_gradient, (2, 0))
    _box_contains(result.posterior, (F(1, 2), F(1, 2)))
    _box_contains(result.primal_multiplier, (6,))
    _box_contains(result.rhs_gradient, (3,))
    _box_contains(result.constraint_gradient[0], (-27, -3))
    for row in result.center_gradient:
        _box_contains(row, (1, 0))
    assert result.bias_gradient == (Interval.point(0),)*2
    assert result.viscosity_gradient == Interval.point(0)
    _contains(result.time_gradient, 4)


def test_single_center_unconstrained_reduces_to_translation_and_scaling():
    geometry = _geometry(centers=((2, -3),), biases=(17,), constraints=(), rhs=(), time=3)
    result = certify_gaussian_affine_parameter_vjp(geometry, (1, -2), (0, 0), (4, -1), (0, 0))
    assert result is not None
    _box_contains(result.root_enclosure, (5, -9))
    _box_contains(result.target_gradient, (12, -3))
    _box_contains(result.center_gradient[0], (4, -1))
    assert result.constraint_gradient == result.rhs_gradient == result.adjoint_multiplier == ()
    assert result.primal_multiplier == ()
    assert result.viscosity_gradient == Interval.point(0)
    _contains(result.time_gradient, 6)


def test_complete_constraints_are_independent_of_gaussian_model_and_target():
    geometry = _geometry(constraints=((1, 2), (2, -1)), rhs=(3, 1))
    result = certify_gaussian_affine_parameter_vjp(geometry, (100, -50), (90, 100), (2, 3), (10, -20))
    assert result is not None and result.root_certificate is None
    assert result.posterior == () and result.primal_multiplier is None
    _box_contains(result.root_enclosure, (1, 1))
    _box_contains(result.rhs_gradient, (F(8, 5), F(1, 5)))
    _box_contains(result.constraint_gradient[0], (F(-8, 5), F(-8, 5)))
    _box_contains(result.constraint_gradient[1], (F(-1, 5), F(-1, 5)))
    assert result.target_gradient == (Interval.point(0),)*2


def test_original_parameter_symmetries_and_constraint_row_gauge():
    geometry = _geometry(centers=((-1, -2), (1, 2), (2, -1)), biases=(0, F(1, 3), F(-1, 4)),
                         constraints=((1, 2),), time=4)
    target, upstream = (F(1, 7), F(-1, 11)), (2, -3)
    result = certify_gaussian_affine_parameter_vjp(geometry, target, (0.25, 0), upstream, (7, -3.5))
    assert result is not None
    zero = Interval.point(0)
    # Common bias shift leaves the optimizer fixed.
    _contains(sum(result.bias_gradient, zero), 0)
    # Translating every center and the feasible set translates the optimizer.
    for i in range(2):
        value = sum((row[i] for row in result.center_gradient), zero)
        value += geometry.restriction.constraints[0][i].numerator*result.rhs_gradient[0]
        _contains(value, upstream[i])
    # Heat-age / viscosity scaling identity derived by rescaling nu*s.
    value = 4*result.time_gradient-result.viscosity_gradient
    dot = sum((result.target_gradient[i]*Interval(float(target[i]), float(target[i])) for i in range(2)), zero)
    assert not (value.hi < dot.lo or dot.hi < value.lo)
    # Replacing C,d by alpha*C,alpha*d changes no feasible point.
    row = geometry.restriction.constraints[0]
    value = sum((int(row[i])*result.constraint_gradient[0][i] for i in range(2)), zero)
    value += int(geometry.restriction.rhs[0])*result.rhs_gradient[0]
    _contains(value, 0)


def test_query_has_no_iteration_factorization_or_dense_covariance(monkeypatch):
    geometry = _geometry()

    def fail(*args, **kwargs):
        pytest.fail("a residual certificate query must not solve or form a covariance matrix")

    monkeypatch.setattr(RationalLDLT, "solve", fail)
    monkeypatch.setattr(GaussianHeatMap, "scaled_jacobian", fail)
    monkeypatch.setattr(MeanMap, "jacobian", fail)
    monkeypatch.setattr(import_module("omnibias.core.verified.inverse_gaussian_heat"), "inverse_gaussian_heat", fail)
    monkeypatch.setattr(import_module("omnibias.core.verified._gaussian_moments"), "gaussian_moments", fail)
    _fixture_expected(certify_gaussian_affine_parameter_vjp(geometry, (0, 0), (0, 1), (1, 3), (2, 0)))


def test_common_bias_and_spatial_gauges_preserve_the_relevant_gradients():
    baseline = certify_gaussian_affine_parameter_vjp(_geometry(), (0, 0), (0, 1), (1, 3), (2, 0))
    assert baseline is not None
    shift, gauge = 2**20, 10**100
    geometry = _geometry(centers=((-1+shift, -2+shift), (1+shift, 2+shift)),
                         biases=(gauge+F(9, 4), gauge+F(1, 4)), rhs=(1+shift,))
    result = certify_gaussian_affine_parameter_vjp(geometry, (0, 0), (shift, 1+shift), (1, 3), (2, 0))
    assert result is not None
    _box_contains(result.target_gradient, (2, 0))
    _box_contains(result.bias_gradient, (-1, 1))
    _box_contains(result.center_gradient[0], (F(1, 2), F(-3, 2)))
    _box_contains(result.center_gradient[1], (F(1, 2), F(-1, 2)))
    _box_contains(result.rhs_gradient, (5,))
    _box_contains(result.constraint_gradient[0], (2-5*shift, -5*(1+shift)))
    _contains(result.time_gradient, -2)
    _contains(result.viscosity_gradient, -2)
