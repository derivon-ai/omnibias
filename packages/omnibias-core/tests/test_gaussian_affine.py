# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact affine Gaussian geometry, independent density identities and refusals."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import gaussian_affine, transcend
from omnibias.core.verified.gaussian_affine import (
    GaussianAffineRestriction,
    InconsistentAffineConstraints,
    restrict_gaussian_heat,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _model(**kwargs):
    return GaussianHeatMap(((4, 1), (-4, -1)), (F(9, 4), F(25, 4)), **kwargs)


def _matvec(matrix, vector):
    return tuple(sum(a*b for a, b in zip(row, vector, strict=True)) for row in matrix)


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def test_redundant_constraints_remove_actual_ambient_nonconvexity():
    source = _model()
    restriction = restrict_gaussian_heat(source, ((1, 0), (2, 0), (-3, 0)), (1, 2, -3))
    assert restriction is not None and restriction.source_model is source
    assert restriction.constraint_rank == restriction.tangent_dimension == 1
    assert restriction.dimension == 2
    assert restriction.anchor == (1, 0)
    assert restriction.projector == ((0, 0), (0, 1))
    assert restriction.normal_cost_differences == (0, 16)
    assert restriction.model.centers == ((1, 1), (1, -1))
    assert restriction.model.biases == (0, 0)
    assert source.contraction_bound == F(17, 2)
    assert restriction.model.contraction_bound == F(1, 2)
    # At (1,0) the original weights are equal, giving covariance a*a^T
    # with a=(4,1). The ambient potential Hessian has eigenvalue 1-17/2.
    assert 1-source.contraction_bound == F(-15, 2)
    assert restriction.project_target((99, F(1, 3))) == (0, F(1, 3))
    assert restriction.project_point((-5, F(2, 7))) == (1, F(2, 7))


def test_oblique_minimum_norm_anchor_and_projector_are_rational():
    anchor = (F(3, 5), F(6, 5))
    centers = tuple(tuple(anchor[i]+sign*(4*(1, 2)[i]+F(1, 4)*(2, -1)[i])
                          for i in range(2)) for sign in (1, -1))
    source = GaussianHeatMap(centers, (0, 0))
    restriction = restrict_gaussian_heat(source, ((1, 2),), (3,))
    assert restriction is not None
    assert restriction.anchor == anchor
    assert restriction.projector == ((F(4, 5), F(-2, 5)), (F(-2, 5), F(1, 5)))
    assert restriction.normal_cost_differences == (0, 0)
    assert restriction.model.centers == ((F(11, 10), F(19, 20)), (F(1, 10), F(29, 20)))
    assert restriction.model.biases == (0, 0)  # The shared normal energy is gauged away.
    assert source.contraction_bound == F(1285, 32)
    assert restriction.model.contraction_bound == F(5, 32)
    for point in ((0, 0), (7, -3), (F(-1, 7), F(3, 11))):
        projected = restriction.project_point(point)
        assert projected[0]+2*projected[1] == 3
        assert restriction.project_point(projected) == projected
        difference = tuple(F(x)-y for x, y in zip(point, projected, strict=True))
        assert _matvec(restriction.projector, difference) == (0, 0)


def test_density_gauge_and_projected_velocity_against_direct_high_precision_gaussians():
    mp = pytest.importorskip("mpmath")
    source = GaussianHeatMap(((0, 0), (1, 0), (0, 2), (2, 1)),
                             (F(2, 7), F(-3, 11), F(1, 4), F(2, 3)), viscosity=F(3, 2), time=F(5, 4))
    restriction = restrict_gaussian_heat(source, ((1, 2),), (3,))
    assert restriction is not None
    # For this plane the squared normal distance is (a_x+2*a_y-3)^2/5.
    costs = tuple(F((a+2*b-3)**2, 5) for a, b in source.centers)
    assert restriction.normal_cost_differences == tuple(value-costs[0] for value in costs)
    with mp.workdps(90):
        nu, time = _mp(mp, source.viscosity), _mp(mp, source.time)

        def direct(model, point):
            centers = [tuple(_mp(mp, value) for value in row) for row in model.centers]
            weights = [mp.exp(_mp(mp, b)-sum((x-a)**2 for x, a in zip(point, center, strict=True))/(4*nu*time))
                       for b, center in zip(model.biases, centers, strict=True)]
            partition = sum(weights)
            mean = tuple(sum(w*a[axis] for w, a in zip(weights, centers, strict=True))/partition for axis in range(2))
            return partition, tuple((x-a)/time for x, a in zip(point, mean, strict=True))

        for parameter in (-10, -1, 0, 3, 7):
            point = (_mp(mp, F(3, 5)+2*parameter), _mp(mp, F(6, 5)-parameter))
            original_density, original_velocity = direct(source, point)
            density, velocity = direct(restriction.model, point)
            expected_gauge = -_mp(mp, source.biases[0])+_mp(mp, costs[0])/(4*nu*time)
            assert abs(mp.log(density/original_density)-expected_gauge) < mp.mpf("1e-80")
            expected_velocity = tuple(sum(_mp(mp, p)*v for p, v in zip(row, original_velocity, strict=True))
                                      for row in restriction.projector)
            assert max(abs(a-b) for a, b in zip(velocity, expected_velocity, strict=True)) < mp.mpf("1e-80")


@pytest.mark.parametrize("constraints,rhs", [((), ()), (((0, 0),), (0,)), (((0, 0), (0, 0)), (0, 0))])
def test_zero_rank_reuses_source_model_without_needing_contraction(constraints, rhs):
    source = _model()
    restriction = restrict_gaussian_heat(source, constraints, rhs)
    assert restriction is not None and restriction.model is source
    assert restriction.constraint_rank == 0 and restriction.tangent_dimension == 2
    assert restriction.anchor == (0, 0) and restriction.projector == ((1, 0), (0, 1))
    assert restriction.normal_cost_differences == (0, 0)
    assert restriction.certificate["payload"]["density_gauge"] == "1"
    assert restriction.project_target((F(1, 3), 2)) == (F(1, 3), 2)


@pytest.mark.parametrize("constraints,rhs", [(((1, 0), (2, 0)), (1, 3)),
                                           (((0, 0),), (1,)), (((1, 0), (0, 0)), (2, -1))])
def test_exact_inconsistency_raises_distinct_public_exception(constraints, rhs):
    with pytest.raises(InconsistentAffineConstraints, match="inconsistent"):
        restrict_gaussian_heat(_model(), constraints, rhs)


def test_full_rank_uses_exact_cancellation_beyond_binary64_resolution():
    large = 2**60
    assert float(large) == float(large+1)
    restriction = restrict_gaussian_heat(_model(), ((large, large), (large, large+1)), (0, -1))
    assert restriction is not None and restriction.constraint_rank == 2
    assert restriction.anchor == (1, -1) and restriction.tangent_dimension == 0
    assert restriction.projector == ((0, 0), (0, 0))
    assert restriction.model.centers == ((1, -1),)*2
    assert restriction.model.contraction_bound == 0
    assert restriction.project_point((100, -200)) == (1, -1)
    assert restriction.project_target((100, -200)) == (0, 0)


def test_equivalent_row_permutations_rescalings_and_duplicates_preserve_geometry_and_model():
    source = GaussianHeatMap(((0, 0, 0), (1, -2, 3)), (0, F(1, 3)))
    base = restrict_gaussian_heat(source, ((1, 2, 3), (2, 1, -1)), (4, -1))
    equivalent = restrict_gaussian_heat(source, ((-6, -3, 3), (F(1, 7), F(2, 7), F(3, 7)), (2, 4, 6)),
                                        (3, F(4, 7), 8))
    assert base is not None and equivalent is not None
    for name in ("anchor", "projector", "normal_cost_differences", "constraint_rank"):
        assert getattr(base, name) == getattr(equivalent, name)
    assert base.model.centers == equivalent.model.centers
    assert base.model.biases == equivalent.model.biases


def test_factorizes_nullspace_gram_once_and_skips_factorization_at_extreme_ranks(monkeypatch):
    calls = 0
    original = gaussian_affine.factor_positive_definite

    def tracked(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(gaussian_affine, "factor_positive_definite", tracked)
    source = GaussianHeatMap(((0, 0, 0), (1, 2, 3)), (0, 0))
    assert restrict_gaussian_heat(source, ((1, 2, 3),), (1,)) is not None
    assert calls == 1
    assert restrict_gaussian_heat(source, (), ()) is not None
    assert restrict_gaussian_heat(source, ((0, 0, 0),), (0,)) is not None
    assert restrict_gaussian_heat(source, ((1, 0, 0), (0, 1, 0), (0, 0, 1)), (1, 2, 3)) is not None
    assert calls == 1


def test_common_normal_energy_and_bias_are_not_squared_or_rounded_before_cancellation():
    huge = 2**3000
    assert (huge*huge).bit_length() > RidgeBudget().max_coefficient_bits
    source = GaussianHeatMap(((huge, 1), (huge, -1)), (huge, huge))
    restriction = restrict_gaussian_heat(source, ((1, 0),), (0,))
    assert restriction is not None
    assert restriction.model.centers == ((0, 1), (0, -1))
    assert restriction.model.biases == (0, 0)
    assert restriction.normal_cost_differences == (0, 0)
    assert restriction.certificate["payload"]["normal_reference"] == [[huge, 1], [0, 1]]


def test_immutable_snapshots_and_factory_only_construction():
    constraints, rhs = [[1, 2]], [3]
    restriction = restrict_gaussian_heat(_model(), constraints, rhs)
    constraints[0][0], rhs[0] = 99, 99
    assert restriction.constraints == ((1, 2),) and restriction.rhs == (3,)
    with pytest.raises(FrozenInstanceError):
        restriction.anchor = (0, 0)
    with pytest.raises(TypeError, match="use restrict_gaussian_heat"):
        GaussianAffineRestriction()


@pytest.mark.parametrize("constraints,rhs,error", [(None, (), TypeError), ("", (), TypeError),
    (((1, 0),), None, TypeError), (((1, 0),), (), ValueError), (((1,),), (0,), ValueError),
    ((None,), (0,), TypeError), (((True, 0),), (0,), TypeError), (((1.0, 0),), (0,), TypeError),
    (((1, 0),), (True,), TypeError), (((1, 0),), (0.0,), TypeError)])
def test_malformed_or_inexact_constraint_metadata_raises(constraints, rhs, error):
    with pytest.raises(error):
        restrict_gaussian_heat(_model(), constraints, rhs)


def test_source_model_type_validation():
    with pytest.raises(TypeError, match="GaussianHeatMap"):
        restrict_gaussian_heat(object(), (), ())


@pytest.mark.parametrize("method", ["project_target", "project_point"])
@pytest.mark.parametrize("values,error", [((), ValueError), ((0,), ValueError), (None, TypeError),
                                         ((True, 0), TypeError), ((0.0, 1), TypeError),
                                         ((2**5000, 0), ValueError)])
def test_projection_methods_validate_exact_vectors_and_bound_coefficients(method, values, error):
    restriction = restrict_gaussian_heat(_model(), ((1, 2),), (3,))
    with pytest.raises(error):
        getattr(restriction, method)(values)


def test_budget_refusals_cover_storage_work_input_and_gram_intermediates():
    for budget in (RidgeBudget(max_matrix_entries=200), RidgeBudget(max_elimination_work=3000)):
        source = GaussianHeatMap(((0, 0),), (0,), budget=budget)
        assert restrict_gaussian_heat(source, ((1, 2),), (3,)) is None
    source = GaussianHeatMap(((0, 0),), (0,), budget=RidgeBudget(max_coefficient_bits=12))
    assert restrict_gaussian_heat(source, ((1024, 1023),), (0,)) is None  # Gram denominator grows.
    assert restrict_gaussian_heat(_model(), ((2**5000, 0),), (0,)) is None
    restriction = restrict_gaussian_heat(_model(), ((1, 2),), (3,))
    with pytest.raises(ValueError, match="bit budget"):
        restriction.project_target((2**4095, 0))  # Input fits; multiplication by 4/5 does not.


def test_generated_model_construction_is_charged_to_the_same_work_budget():
    # Source construction alone fits this budget. Projector work plus creating
    # the replacement Gaussian/mean maps must not receive independent resets.
    source = GaussianHeatMap(((0, 0),), (0,), budget=RidgeBudget(max_elimination_work=5000))
    assert restrict_gaussian_heat(source, ((1, 0),), (0,)) is None


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_certificate_needs_no_transcendentals_and_preserves_sticky_history(monkeypatch, sticky):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    restriction = restrict_gaussian_heat(_model(), ((1, 2),), (3,))
    assert restriction is not None
    assert restriction.certificate["meta"]["transcend_backend"] == "not_used"
    assert certificate_is_unconditional(restriction.certificate)
    assert schema_errors_v1(restriction.certificate) == []
    assert transcend.libm_fallback_used() is sticky
    assert "theorem_prover_verified" not in restriction.certificate["honesty"]


def test_shared_strict_sealing_policy_is_preserved(monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        restrict_gaussian_heat(_model(), ((1, 2),), (3,))
    assert transcend.strict_backend() is True


@pytest.mark.parametrize("field", ["source_centers", "source_biases", "time", "viscosity", "constraints", "rhs",
                                   "constraint_rank", "anchor", "projector", "normal_reference",
                                   "normal_cost_differences", "centers", "biases", "identities", "density_gauge"])
def test_certificate_binds_original_problem_transformation_and_exact_witness(field):
    restriction = restrict_gaussian_heat(_model(), ((1, 2),), (3,))
    assert verify_certificate_digest(restriction.certificate)
    changed = deepcopy(restriction.certificate)
    changed["payload"][field] = [123, 456]
    assert not verify_certificate_digest(changed)
