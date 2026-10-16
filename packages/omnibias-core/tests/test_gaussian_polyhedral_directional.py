# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact response identities and directional geometry of critical cones."""

from fractions import Fraction as F

import pytest
from omnibias.core.verified.gaussian_affine import restrict_gaussian_heat
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral import compile_gaussian_polyhedral_face
from omnibias.core.verified.gaussian_polyhedral_directional import (
    certify_gaussian_polyhedral_directional,
    compile_gaussian_critical_face,
)
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    certify_gaussian_polyhedral_equilibrium,
)


def _dot(left, right):
    return sum((a*b for a, b in zip(left, right, strict=True)), F(0))


def _equilibrium(source, inequalities, upper, active=(), *, target=None, point=None, equalities=(), rhs=()):
    dimension = source.dimension
    target = (0,)*dimension if target is None else target
    point = (0,)*dimension if point is None else point
    base = restrict_gaussian_heat(source, equalities, rhs)
    assert base is not None
    face = compile_gaussian_polyhedral_face(base, inequalities, upper, active)
    assert face is not None
    equilibrium = certify_gaussian_polyhedral_equilibrium(face, target, point)
    assert equilibrium is not None
    return equilibrium


def _coupled(*, redundant=False):
    source = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0))
    inequalities = ((1, 1), (2, 2), (-1, 0)) if redundant else ((1, 1), (-1, 0))
    upper = (0, 0, 2) if redundant else (0, 2)
    return _equilibrium(source, inequalities, upper)


def _compile(equilibrium, active=()):
    face = compile_gaussian_critical_face(equilibrium, active)
    assert face is not None
    _assert_identities(face)
    return face


def _assert_identities(face):
    """Replay complete linear stationarity, independent of a sample direction."""
    dimension = len(face.response)
    columns = tuple(tuple(row[j] for row in face.response) for j in range(dimension))
    assert columns == face.response
    for j, column in enumerate(columns):
        assert all(_dot(row, column) == 0 for row in face.constraints)
        for i in range(dimension):
            left = _dot(face.equilibrium.hessian[i], column)
            left += sum(row[i]*multipliers[j] for row, multipliers in
                        zip(face.constraints, face.multiplier_response, strict=True))
            assert left == int(i == j)
    weak_offset = len(face.equilibrium.face.base_restriction.constraints)+len(face.equilibrium.strong_indices)
    expected = tuple(face.multiplier_response[weak_offset+i] for i in range(len(face.active_weak_indices)))
    expected += tuple(tuple(-_dot(face.equilibrium.face.inequalities[i], column) for column in columns)
                      for i in face.equilibrium.weak_indices if i not in face.active_weak_indices)
    assert face.region_inequalities == expected


def test_coupled_response_multiplier_and_region_matrices_have_exact_closed_forms():
    equilibrium = _coupled()
    assert equilibrium.hessian == ((F(1, 2), 0), (0, 1))
    assert equilibrium.strong_indices == () and equilibrium.weak_indices == (0,)
    free, active = _compile(equilibrium), _compile(equilibrium, (0,))
    assert free.response == ((2, 0), (0, 1))
    assert free.multiplier_response == ()
    assert free.region_inequalities == ((-2, -1),)
    assert active.response == ((F(2, 3), F(-2, 3)), (F(-2, 3), F(2, 3)))
    assert active.multiplier_response == ((F(2, 3), F(1, 3)),)
    assert active.region_inequalities == active.multiplier_response
    result = certify_gaussian_polyhedral_directional(active, (1, 0))
    assert result is not None and result.value == (F(2, 3), F(-2, 3))
    assert result.equality_multipliers == (F(2, 3),)
    assert result.weak_multipliers == (F(2, 3), 0)
    assert result.weak_slacks == (0, 0)
    assert result.quadratic_value == F(-1, 3)


@pytest.mark.parametrize("direction", [F(-3), F(-1, 3), F(0), F(1, 5), F(7)])
def test_scalar_weak_boundary_is_exact_directional_clamping(direction):
    equilibrium = _equilibrium(GaussianHeatMap(((0,),), (0,), time=2), ((1,),), (0,))
    face = _compile(equilibrium, (0,) if direction >= 0 else ())
    result = certify_gaussian_polyhedral_directional(face, (direction,))
    assert result is not None and result.value == (min(2*direction, 0),)
    assert result.weak_multipliers == (max(direction, 0),)
    assert result.weak_slacks == (max(-2*direction, 0),)
    assert result.quadratic_value == (-direction**2 if direction < 0 else 0)


@pytest.mark.parametrize("scale", [F(0), F(1, 3), F(2), F(7, 2)])
def test_positive_homogeneity_includes_zero_and_scales_the_quadratic_value(scale):
    face = _compile(_coupled(), (0,))
    baseline = certify_gaussian_polyhedral_directional(face, (2, -1))
    result = certify_gaussian_polyhedral_directional(face, (2*scale, -scale))
    assert baseline is not None and result is not None
    for field in ("value", "equality_multipliers", "weak_multipliers", "weak_slacks"):
        assert getattr(result, field) == tuple(scale*value for value in getattr(baseline, field))
    assert result.quadratic_value == scale**2*baseline.quadratic_value


def test_directional_map_is_not_additive_or_an_ordinary_jacobian():
    equilibrium = _coupled()
    positive = certify_gaussian_polyhedral_directional(_compile(equilibrium, (0,)), (1, 0))
    negative = certify_gaussian_polyhedral_directional(_compile(equilibrium), (-1, 0))
    zero = certify_gaussian_polyhedral_directional(_compile(equilibrium), (0, 0))
    assert positive is not None and negative is not None and zero is not None
    assert positive.value == (F(2, 3), F(-2, 3))
    assert negative.value == (-2, 0) and zero.value == (0, 0)
    assert tuple(a+b for a, b in zip(positive.value, negative.value, strict=True)) != zero.value


def test_shared_region_boundary_agrees_and_wrong_regions_refuse():
    equilibrium = _coupled()
    free, active = _compile(equilibrium), _compile(equilibrium, (0,))
    for direction in ((1, -2), (0, 0), (F(-1, 3), F(2, 3))):
        left = certify_gaussian_polyhedral_directional(free, direction)
        right = certify_gaussian_polyhedral_directional(active, direction)
        assert left is not None and right is not None
        assert left.value == right.value == (2*direction[0], direction[1])
        assert left.quadratic_value == right.quadratic_value
        assert left.weak_multipliers == right.weak_multipliers == (0, 0)
    assert certify_gaussian_polyhedral_directional(free, (1, 0)) is None
    assert certify_gaussian_polyhedral_directional(active, (-1, 0)) is None


def test_strong_row_directional_multipliers_can_be_negative():
    source = GaussianHeatMap(((0, 0),), (0,), time=3)
    equilibrium = _equilibrium(source, ((1, 0), (0, 1)), (0, 0), (0,), target=(1, 0))
    assert equilibrium.strong_indices == (0,) and equilibrium.weak_indices == (1,)
    free, corner = _compile(equilibrium), _compile(equilibrium, (1,))
    assert free.response == ((0, 0), (0, 3)) and free.multiplier_response == ((1, 0),)
    inward = certify_gaussian_polyhedral_directional(free, (-5, -2))
    outward = certify_gaussian_polyhedral_directional(corner, (-5, 2))
    assert inward is not None and inward.value == (0, -6)
    assert inward.equality_multipliers == (-5,) and inward.weak_slacks == (0, 6)
    assert inward.quadratic_value == -6
    assert outward is not None and outward.value == (0, 0)
    assert outward.equality_multipliers == (-5, 2)
    assert outward.weak_multipliers == (0, 2)
    assert certify_gaussian_polyhedral_directional(corner, (-5, -2)) is None


def test_redundant_unselected_weak_rows_remain_feasible_and_allow_alternative_multipliers():
    equilibrium = _coupled(redundant=True)
    assert equilibrium.weak_indices == (0, 1)
    first, second = _compile(equilibrium, (0,)), _compile(equilibrium, (1,))
    assert first.response == second.response
    assert first.multiplier_response == ((F(2, 3), F(1, 3)),)
    assert second.multiplier_response == ((F(1, 3), F(1, 6)),)
    assert first.region_inequalities == ((F(2, 3), F(1, 3)), (0, 0))
    assert second.region_inequalities == ((F(1, 3), F(1, 6)), (0, 0))
    a = certify_gaussian_polyhedral_directional(first, (1, 0))
    b = certify_gaussian_polyhedral_directional(second, (1, 0))
    assert a is not None and b is not None and a.value == b.value == (F(2, 3), F(-2, 3))
    assert a.weak_multipliers == (F(2, 3), 0, 0)
    assert b.weak_multipliers == (0, F(1, 3), 0)
    assert a.weak_slacks == b.weak_slacks == (0, 0, 0)
    inward = certify_gaussian_polyhedral_directional(_compile(equilibrium), (-1, 0))
    assert inward is not None and inward.weak_slacks == (2, 4, 0)
    with pytest.raises(ValueError, match="full row rank"):
        compile_gaussian_critical_face(equilibrium, (0, 1))


@pytest.mark.parametrize("direction,active", [((2, 3), (0,)), ((-2, 3), (1,))])
def test_opposing_weak_rows_force_a_subspace_without_selecting_redundant_rows(direction, active):
    source = GaussianHeatMap(((-1, 0), (1, 0)), (0, 0))
    equilibrium = _equilibrium(source, ((1, 0), (-1, 0)), (0, 0))
    face = _compile(equilibrium, active)
    result = certify_gaussian_polyhedral_directional(face, direction)
    assert result is not None and result.value == (0, 3)
    assert result.weak_slacks == (0, 0)
    assert result.weak_multipliers[active[0]] == 2
    assert face.response == ((0, 0), (0, 1))


def test_refused_multiplier_region_is_inconclusive_even_when_its_response_is_correct():
    equilibrium = _equilibrium(GaussianHeatMap(((0,),), (0,)), ((1,), (-1,)), (0, 0))
    positive, negative = _compile(equilibrium, (0,)), _compile(equilibrium, (1,))
    assert positive.response == negative.response == ((0,),)
    # The feasible set is the singleton 0, so BOTH linear responses give
    # the true derivative. Only the second face supplies a nonnegative
    # selected weak multiplier for direction -1; refusal is inconclusive.
    assert _dot(positive.response[0], (-1,)) == 0
    assert certify_gaussian_polyhedral_directional(positive, (-1,)) is None
    result = certify_gaussian_polyhedral_directional(negative, (-1,))
    assert result is not None and result.value == (0,)
    assert result.weak_multipliers == (0, 1) and result.weak_slacks == (0, 0)


def test_no_constraints_or_inequalities_give_the_original_inverse_hessian():
    equilibrium = _equilibrium(GaussianHeatMap(((-1, 0), (1, 0)), (0, 0)), (), ())
    face = _compile(equilibrium)
    assert face.constraints == face.multiplier_response == face.region_inequalities == ()
    assert face.response == ((2, 0), (0, 1))
    result = certify_gaussian_polyhedral_directional(face, (F(2, 3), F(-4, 5)))
    assert result is not None and result.value == (F(4, 3), F(-4, 5))
    assert result.equality_multipliers == result.weak_multipliers == result.weak_slacks == ()
    assert result.quadratic_value == F(-172, 225)


def test_full_rank_strong_cone_has_zero_response_for_every_direction():
    equilibrium = _equilibrium(GaussianHeatMap(((0, 0),), (0,)), ((1, 0), (0, 1)), (0, 0), (0, 1), target=(1, 1))
    face = _compile(equilibrium)
    assert equilibrium.weak_indices == () and equilibrium.strong_indices == (0, 1)
    assert face.response == ((0, 0), (0, 0))
    assert face.multiplier_response == ((1, 0), (0, 1)) and face.region_inequalities == ()
    result = certify_gaussian_polyhedral_directional(face, (-5, -7))
    assert result is not None and result.value == (0, 0)
    assert result.equality_multipliers == (-5, -7)
    assert result.weak_multipliers == result.weak_slacks == (0, 0)
    assert result.quadratic_value == 0


def test_full_rank_base_equalities_remove_large_normal_directions_exactly():
    source = GaussianHeatMap(((-1, -2), (1, 2)), (0, 0))
    equilibrium = _equilibrium(source, (), (), equalities=((1, 2), (2, -1)), rhs=(0, 0))
    face = _compile(equilibrium)
    assert source.contraction_bound == F(5, 2)
    assert face.response == ((0, 0), (0, 0))
    assert face.multiplier_response == ((F(1, 5), F(2, 5)), (F(2, 5), F(-1, 5)))
    direction = (2**200, -(2**180))
    result = certify_gaussian_polyhedral_directional(face, direction)
    assert result is not None and result.value == (0, 0)
    assert result.equality_multipliers == tuple(_dot(row, direction) for row in face.multiplier_response)
