# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact scalar laws and covariance geometry for joint Gaussian directions."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_grouped_directional import (
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_grouped_equilibrium import (
    certify_gaussian_grouped_equilibrium,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_joint_directional import (
    certify_gaussian_joint_directional,
)
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_projected_equilibrium import (
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.interval import Interval

ZERO = Interval.point(0)


def _scalar(active=()):
    source = GaussianHeatMap(((-1,), (1,)), (F(1, 4), F(1, 4)))
    base = restrict_gaussian_heat(source, (), ())
    assert base is not None
    primal = compile_gaussian_polyhedral_face(base, ((1,),), (0,))
    assert primal is not None
    equilibrium = certify_gaussian_grouped_equilibrium(primal, (0,), (0,))
    assert equilibrium is not None and equilibrium.weak_indices == (0,)
    face = compile_gaussian_grouped_critical_face(equilibrium, active)
    assert face is not None
    return face


def _projected(cross=False):
    centers = ((-1, 0, -int(cross)), (1, 0, int(cross)), (0, -1, 4-int(cross)), (0, 1, 4+int(cross)))
    biases = tuple(F(group)+sum(F(a)**2 for a in row)/4 for row, group in zip(centers, (0, 0, 1, 1), strict=True))
    source = GaussianHeatMap(centers, biases)
    base = restrict_gaussian_heat(source, ((0, 0, 1),), (0,))
    assert base is not None
    primal = compile_gaussian_polyhedral_face(base, ((1, 2, 0), (0, 1, 0)), (0, 0), (0,))
    assert primal is not None
    equilibrium = certify_gaussian_projected_equilibrium(primal, (1, 2, 0), (0, 0, 0))
    assert equilibrium is not None
    face = compile_gaussian_grouped_critical_face(equilibrium)
    assert face is not None
    return face


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _contains_exact(interval, value):
    assert F(interval.lo) <= value <= F(interval.hi)


@pytest.mark.parametrize("xi,beta", [
    (-1, (0, 0)), (1, (0, 0)), (0, (0, 2)), (0, (2, 0)),
    (-1, (0, 2)), (1, (2, 0)),
    (F(-1, 3), (F(2, 5), F(7, 5))), (F(-1, 3), (F(2, 5), F(1, 5))),
])
def test_collapsed_scalar_joint_derivative_is_the_exact_clamped_affine_forcing(xi, beta):
    forcing = F(xi)+(F(beta[1])-beta[0])/2
    expected = min(2*forcing, F(0))
    active = (0,) if forcing > 0 else ()
    face = _scalar(active)
    result = certify_gaussian_joint_directional(face, (xi,), beta, (float(expected),))
    assert result is not None
    assert result.target_direction == (xi,) and result.bias_direction == beta
    assert result.relative_bias_direction == (0, beta[1]-beta[0])
    assert result.group_bias_means == ((F(beta[1])-beta[0])/2,)
    assert result.group_bias_covariances == (((F(beta[1])-beta[0])/2,),)
    _contains_exact(result.forcing_enclosure[0], forcing)
    _contains_exact(result.enclosure[0], expected)
    assert abs(result.value[0]-expected) <= result.error_bound
    assert result.error_bound < F(1, 10**10)
    if active:
        assert result.value == (0,) and result.error_bound == 0
        _contains_exact(result.equality_multipliers[0], forcing)
        assert result.weak_slacks == (ZERO,)
    elif forcing < 0:
        assert result.weak_slacks[0].lo > 0
        assert result.equality_multipliers == ()
    if forcing:
        wrong = _scalar(() if active else (0,))
        assert certify_gaussian_joint_directional(wrong, (xi,), beta, (float(2*forcing),)) is None


@pytest.mark.parametrize("active", [(), (0,)])
def test_exact_joint_boundary_cancellation_returns_zero_without_a_finite_activity_claim(active):
    result = certify_gaussian_joint_directional(_scalar(active), (-1,), (0, 2), (37.0,))
    assert result is not None and result.value == (0,)
    assert result.forcing_enclosure == result.projected_forcing_enclosure == (ZERO,)
    assert result.zero_projected_forcing
    assert result.error_bound == result.residual_norm_bound == 0
    assert result.enclosure == result.residual_enclosure == (ZERO,)
    assert result.weak_multipliers == result.weak_slacks == (ZERO,)
    # The nonlinear root departs at cubic order; this certificate asserts
    # the directional derivative only, so both critical faces can certify.


@pytest.mark.parametrize("cross", [False, True])
def test_within_group_and_between_group_joint_covariances_have_distinct_projected_roles(cross):
    mp = pytest.importorskip("mpmath")
    face = _projected(cross)
    result = certify_gaussian_joint_directional(face, (1, 0, 0), (0, 2, -1, 1), (1.0, -0.5, 0.0))
    assert result is not None
    assert result.group_bias_means == (1, 0)
    assert result.group_bias_covariances == ((1, 0, int(cross)), (0, 1, int(cross)))
    with mp.workdps(100):
        e = mp.exp(1)
        p, q = 1/(1+e), e/(1+e)
        forcing = (1+p, q, int(cross)-4*p*q)
        projected = ((4*(1+p)-2*q)/5, (-2*(1+p)+q)/5, mp.mpf(0))
        normal = (forcing[2], (1+p+2*q)/5)
        for intervals, expected in ((result.forcing_enclosure, forcing), (result.projected_forcing_enclosure, projected),
                                    (result.normal_forcing_enclosure, normal)):
            for interval, value in zip(intervals, expected, strict=True):
                assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
        assert abs(forcing[2]) > mp.mpf("0.1")
        assert not result.zero_projected_forcing
        assert result.value[2] == 0 and result.coordinate_error_bounds[2] == 0


@pytest.mark.parametrize("cross", [False, True])
def test_group_constant_bias_change_keeps_exact_zero_primal_response_and_nonzero_normal_qp_multiplier(cross):
    mp = pytest.importorskip("mpmath")
    face = _projected(cross)
    result = certify_gaussian_joint_directional(face, (0, 0, 0), (0, 0, 1, 1), (5.0, -3.0, 99.0))
    assert result is not None
    assert result.group_bias_means == (0, 1)
    assert result.group_bias_covariances == ((0, 0, 0), (0, 0, 0))
    assert result.projected_forcing_enclosure == (ZERO,)*3
    assert result.zero_projected_forcing and result.value == (0, 0, 0)
    assert result.error_bound == 0 and result.residual_enclosure == (ZERO,)*3
    assert result.forcing_enclosure[:2] == (ZERO, ZERO)
    with mp.workdps(100):
        normal_force = 4*mp.exp(1)/(1+mp.exp(1))**2
        interval = result.equality_multipliers[0]
        assert _mp(mp, interval.lo) <= normal_force <= _mp(mp, interval.hi)
    assert result.equality_multipliers[0].lo > 0
    assert result.equality_multipliers[1] == ZERO
    assert result.weak_slacks == result.weak_multipliers == (ZERO, ZERO)


@pytest.mark.parametrize("kind", ["scalar", "projected"])
def test_constant_bias_direction_reduces_to_existing_target_certificate_without_rounding_a_huge_gauge(kind):
    face = _scalar() if kind == "scalar" else _projected(True)
    direction, proposal = ((-1,), (-2.0,)) if kind == "scalar" else ((1, 0, 0), (1.0, -0.5, 0.0))
    count = face.equilibrium.face.base_restriction.source_model.mode_count
    ordinary = certify_gaussian_grouped_directional(face, direction, proposal)
    joint = certify_gaussian_joint_directional(face, direction, (10**400,)*count, proposal)
    assert ordinary is not None and joint is not None
    assert joint.relative_bias_direction == (0,)*count
    assert all(all(value == 0 for value in row) for row in joint.group_bias_covariances)
    for field in ("value", "projected_proposal", "enclosure", "error_bound", "coordinate_error_bounds",
                  "residual_enclosure", "residual_norm_bound", "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(ordinary, field) == getattr(joint, field)


@pytest.mark.parametrize("scale", [F(1, 4), F(2), F(8)])
def test_scalar_joint_response_is_positively_homogeneous_but_not_additive(scale):
    free, active = _scalar(), _scalar((0,))
    negative = certify_gaussian_joint_directional(free, (0,), (0, -2*scale), (float(-2*scale),))
    positive = certify_gaussian_joint_directional(active, (0,), (0, 2*scale), (0.0,))
    zero = certify_gaussian_joint_directional(free, (0,), (0, 0), (0.0,))
    assert negative is not None and positive is not None and zero is not None
    assert negative.value == (-2*scale,) and positive.value == zero.value == (0,)
    assert negative.value[0]+positive.value[0] != zero.value[0]
    _contains_exact(negative.enclosure[0], -2*scale)
    _contains_exact(positive.equality_multipliers[0], scale)
