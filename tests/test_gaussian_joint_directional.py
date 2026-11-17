# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint target/bias responses against original Gaussian moments and KKT."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_grouped_directional import (
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_joint_directional import (
    certify_gaussian_joint_directional,
)
from omnibias.core.verified.interval import Interval
from test_gaussian_grouped_directional import (
    _assert_enclosed,
    _critical_qp,
    _hessian,
)
from test_gaussian_grouped_directional import _equilibrium as _shared_equilibrium
from test_gaussian_polyhedral_directional import _mp, _nonlinear_minimum
from test_gaussian_projected_directional import _equilibrium as _projected_equilibrium


def _joint_forcing(mp, source, point, target_direction, bias_direction):
    """Original per-mode covariance, independently of grouped forcing code."""
    hessian, posterior = _hessian(mp, source, point)
    centers = tuple(tuple(_mp(mp, value) for value in row) for row in source.centers)
    # Covariance is unchanged by a common bias direction. Remove it in
    # exact arithmetic before crossing the mpmath boundary, even at 10**400.
    beta = tuple(_mp(mp, value-bias_direction[0]) for value in bias_direction)
    mean_beta = sum(p*b for p, b in zip(posterior, beta, strict=True))
    mean = tuple(sum(p*row[i] for p, row in zip(posterior, centers, strict=True)) for i in range(source.dimension))
    covariance = tuple(sum(p*(row[i]-mean[i])*(b-mean_beta)
                           for p, row, b in zip(posterior, centers, beta, strict=True)) for i in range(source.dimension))
    forcing = tuple(_mp(mp, xi)+value/_mp(mp, source.time)
                    for xi, value in zip(target_direction, covariance, strict=True))
    return forcing, covariance, hessian


def _perturbed(data, step, bias_direction):
    source = data[0]
    changed = GaussianHeatMap(source.centers, tuple(b+step*db for b, db in zip(source.biases, bias_direction, strict=True)),
                              viscosity=source.viscosity, time=source.time, budget=source.budget)
    return (changed,)+data[1:]


@pytest.mark.parametrize("kind", ["axis", "rotated", "projected", "projected_cross"])
@pytest.mark.parametrize("sign", [1, -1])
def test_joint_critical_qp_and_original_parameter_perturbed_secants(kind, sign):
    mp = pytest.importorskip("mpmath")
    projected = kind.startswith("projected")
    equilibrium, data = _projected_equilibrium(kind == "projected_cross") if projected else _shared_equilibrium(kind)
    source = data[0]
    direction = (sign,)+(0,)*(source.dimension-1)
    beta = (0, 2, -1, 1) if projected else (0, 1, 0, 0)
    with mp.workdps(110):
        forcing, _, hessian = _joint_forcing(mp, source, data[6], direction, beta)
        (_, selected, expected, multipliers), _, _ = _critical_qp(mp, data, forcing)
        if projected and selected == (1,):
            # Independent full-rank constraints force w=0 and E.T*eta=zeta.
            assert max(abs(value) for value in expected) < mp.mpf("1e-100")
            expected = (mp.mpf(0),)*3
            multipliers = (forcing[2], forcing[0], forcing[1]-2*forcing[0])
        face = compile_gaussian_grouped_critical_face(equilibrium, selected)
        assert face is not None
        result = certify_gaussian_joint_directional(face, direction, beta, tuple(float(value) for value in expected))
        assert result is not None
        _assert_enclosed(mp, result.enclosure, expected)
        _assert_enclosed(mp, result.equality_multipliers, multipliers)
        actual_error = mp.sqrt(sum((_mp(mp, value)-true)**2 for value, true in zip(result.value, expected, strict=True)))
        assert actual_error <= _mp(mp, result.error_bound)
        assert result.error_bound < F(1, 10**9)
        projector = mp.matrix([[_mp(mp, value) for value in row] for row in face.projector])
        _assert_enclosed(mp, result.forcing_enclosure, forcing)
        _assert_enclosed(mp, result.projected_forcing_enclosure, projector*mp.matrix(forcing))
        normal_forcing = tuple(sum(_mp(mp, a)*b for a, b in zip(row, forcing, strict=True)) for row in face.left_inverse)
        _assert_enclosed(mp, result.normal_forcing_enclosure, normal_forcing)
        residual = projector*(hessian*mp.matrix([_mp(mp, value) for value in result.value])-mp.matrix(forcing))
        _assert_enclosed(mp, result.residual_enclosure, residual)
        if projected:
            assert result.value[2] == 0 and result.enclosure[2] == Interval.point(0)
            assert source.contraction_bound > 1
        errors = []
        for step in (F(1, 100), F(1, 1000), F(1, 10000)):
            epsilon = _mp(mp, step)
            wanted = tuple(_mp(mp, value)+epsilon*_mp(mp, delta) for value, delta in zip(data[5], direction, strict=True))
            root = _nonlinear_minimum(mp, _perturbed(data, step, beta), wanted)
            secant = tuple((value-_mp(mp, base))/epsilon for value, base in zip(root, data[6], strict=True))
            errors.append(mp.sqrt(sum((value-true)**2 for value, true in zip(secant, expected, strict=True))))
        assert errors[-1] < mp.mpf("2e-4")
        if errors[0] > mp.mpf("1e-60"):
            assert errors[1] < errors[0]/8 and errors[2] < errors[1]/8


@pytest.mark.parametrize("cross", [False, True])
def test_between_group_bias_forcing_changes_normal_dual_even_when_primal_response_is_zero(cross):
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _projected_equilibrium(cross)
    face = compile_gaussian_grouped_critical_face(equilibrium)
    assert face is not None
    result = certify_gaussian_joint_directional(face, (0, 0, 0), (0, 0, 1, 1), (37.0, -19.0, 11.0))
    assert result is not None and result.value == (0, 0, 0)
    assert result.error_bound == 0 and result.enclosure == (Interval.point(0),)*3
    with mp.workdps(110):
        forcing, covariance, _ = _joint_forcing(mp, data[0], data[6], (0, 0, 0), (0, 0, 1, 1))
        expected = 4*mp.exp(1)/(1+mp.exp(1))**2
        assert abs(covariance[2]-expected) < mp.mpf("1e-100")
        assert covariance[:2] == (mp.mpf(0), mp.mpf(0))
        _assert_enclosed(mp, result.equality_multipliers, (expected, mp.mpf(0)))
        assert result.equality_multipliers[0].lo > 0
        assert forcing[2] > 0


def test_common_bias_direction_gauge_is_removed_exactly_before_interval_conversion():
    mp = pytest.importorskip("mpmath")
    equilibrium, data = _projected_equilibrium(True)
    beta = (0, 2, -1, 1)
    with mp.workdps(110):
        forcing, _, _ = _joint_forcing(mp, data[0], data[6], (1, 0, 0), beta)
        (_, selected, expected, _), _, _ = _critical_qp(mp, data, forcing)
        face = compile_gaussian_grouped_critical_face(equilibrium, selected)
        proposal = tuple(float(value) for value in expected)
        base = certify_gaussian_joint_directional(face, (1, 0, 0), beta, proposal)
        shifted = certify_gaussian_joint_directional(face, (1, 0, 0), tuple(10**400+value for value in beta), proposal)
        assert base is not None and shifted is not None
        for field in ("value", "enclosure", "error_bound", "coordinate_error_bounds", "residual_enclosure", "residual_norm_bound",
                      "equality_multipliers", "weak_multipliers", "weak_slacks"):
            assert getattr(base, field) == getattr(shifted, field)


def test_zero_joint_derivative_does_not_imply_the_wall_remains_active_at_finite_steps():
    mp = pytest.importorskip("mpmath")
    with mp.workdps(100):
        # For v(t)=-t and bias difference 2t, the original stationarity
        # equation is x-tanh(x/2+t)=-t. Its free root is negative for t>0,
        # but x(t)/t -> 0 and x(t)/t^3 -> -2/3.
        scaled_cubic_errors = []
        for denominator in (8, 32, 128, 512):
            t = mp.mpf(1)/denominator
            root = mp.findroot(lambda x, t=t: x-mp.tanh(x/2+t)+t, (-t**3, mp.mpf(0)))
            assert root < 0 and abs(root-mp.tanh(root/2+t)+t) < mp.mpf("1e-90")
            assert -t+mp.tanh(t) < 0  # Wall multiplier would be negative.
            scaled_cubic_errors.append(abs(root/t**3+mp.mpf(2)/3))
            assert abs(root/t) < t*t
        assert all(later < earlier/15 for earlier, later in zip(scaled_cubic_errors[:-1], scaled_cubic_errors[1:], strict=True))
        assert scaled_cubic_errors[-1] < mp.mpf("1e-5")
