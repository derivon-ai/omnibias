# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent segment/box projections and exact proof/resource contracts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.polyhedral_projection import (
    PolyhedralProjection,
    PolyhedralProjector,
    _Arithmetic,
    _project_exact,
    _query_work_bound,
    _source_slots,
    compile_polyhedral_projector,
)


def _projector(dimension, rows=(), bounds=(), **kwargs):
    result = compile_polyhedral_projector(dimension, rows, bounds, **kwargs)
    assert result is not None
    return result


def _project(projector, point):
    result = projector.project(point)
    assert result is not None
    assert all(sum(a*x for a, x in zip(row, result.point, strict=True)) <= bound
               for row, bound in zip(projector.inequalities, projector.upper_bounds, strict=True))
    assert all(value >= 0 for value in result.multipliers)
    for j in range(projector.dimension):
        assert result.point[j]-result.source_point[j]+sum(row[j]*eta for row, eta in zip(projector.inequalities, result.multipliers, strict=True)) == 0
    for row, bound, eta in zip(projector.inequalities, projector.upper_bounds, result.multipliers, strict=True):
        assert eta*(bound-sum(a*x for a, x in zip(row, result.point, strict=True))) == 0
    assert result.squared_distance == sum((x-z)**2 for x, z in zip(result.point, result.source_point, strict=True))
    return result


def test_unconstrained_high_dimension_is_identity_with_empty_basis():
    projector = _projector(17)
    point = tuple(F(i, 19) for i in range(17))
    result = _project(projector, point)
    assert projector.enumerated_bases == len(projector.bases) == 1
    assert result.point == point and result.squared_distance == 0
    assert result.active_indices == result.multipliers == ()


@pytest.mark.parametrize("point", [(0, 0, 0), (5, -3, 2), (F(1, 7), F(1, 3), F(-2, 5))])
def test_oblique_halfspace_matches_closed_form_projection(point):
    row, bound = (F(1), F(-2), F(3)), F(2, 3)
    projector = _projector(3, (row,), (bound,))
    multiplier = max(F(0), (sum(a*z for a, z in zip(row, point, strict=True))-bound)/sum(a*a for a in row))
    expected = tuple(z-multiplier*a for z, a in zip(point, row, strict=True))
    result = _project(projector, point)
    assert result.point == expected and result.multipliers == (multiplier,)


@pytest.mark.parametrize("point", [(-3, 2), (2, -1), (F(1, 3), F(2, 7)), (7, 8), (1, 0)])
def test_box_projection_is_coordinate_clamp(point):
    projector = _projector(2, ((1, 0), (0, 1), (-1, 0), (0, -1)), (1, 1, 0, 0))
    result = _project(projector, point)
    assert result.point == tuple(min(F(1), max(F(0), value)) for value in point)
    assert projector.enumerated_bases == 11 and len(projector.bases) == 9
    assert any(len(basis.active_indices) == 2 for basis in projector.bases)


def _segment_projection(point, left, right):
    difference = tuple(b-a for a, b in zip(left, right, strict=True))
    parameter = sum((z-a)*v for z, a, v in zip(point, left, difference, strict=True))/sum(v*v for v in difference)
    parameter = min(F(1), max(F(0), parameter))
    return tuple(a+parameter*v for a, v in zip(left, difference, strict=True))


@pytest.mark.parametrize("point", [(-2, -1), (-1, 2), (0, 5), (3, 0), (3, 4), (1, 1), (F(1, 3), F(2, 5))])
def test_triangle_matches_independent_nearest_segment_oracle(point):
    point = tuple(F(value) for value in point)
    projector = _projector(2, ((-1, 0), (0, -1), (3, 2)), (0, 0, 6))
    vertices = ((F(0), F(0)), (F(2), F(0)), (F(0), F(3)))
    if point[0] >= 0 and point[1] >= 0 and 3*point[0]+2*point[1] <= 6:
        expected = point
    else:
        candidates = [_segment_projection(point, vertices[i], vertices[(i+1) % 3]) for i in range(3)]
        expected = min(candidates, key=lambda p: sum((a-b)**2 for a, b in zip(p, point, strict=True)))
    assert _project(projector, point).point == expected


def test_opposite_rows_project_to_affine_line_and_redundancy_is_supported():
    projector = _projector(2, ((1, 2), (-1, -2), (2, 4), (0, 0)), (F(1, 3), F(-1, 3), F(2, 3), 0))
    for point in ((0, 0), (4, -1), (F(1, 3), 0)):
        multiplier = (point[0]+2*point[1]-F(1, 3))/5
        expected = (point[0]-multiplier, point[1]-2*multiplier)
        result = _project(projector, point)
        assert result.point == expected
        assert sum(a*x for a, x in zip((1, 2), result.point, strict=True)) == F(1, 3)


def test_full_rank_active_corner_and_zero_rows():
    projector = _projector(2, ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)), (F(1, 3), F(-1, 3), -2, 2, 1))
    assert _project(projector, (10, 20)).point == (F(1, 3), F(-2))
    assert _project(projector, (-10, -20)).point == (F(1, 3), F(-2))


@pytest.mark.parametrize("rows,bounds", [(((1,), (-1,)), (0, -1)), (((0,),), (-1,))])
def test_infeasible_sets_produce_no_projection_without_false_witness(rows, bounds):
    projector = _projector(1, rows, bounds)
    assert projector.project((0,)) is None and projector.project((999,)) is None


def test_large_exact_cancellation_never_rounds_the_source_point():
    big = 10**400
    projector = _projector(2, ((1, 1),), (F(1, 3),))
    result = _project(projector, (big, -big+1))
    assert result.point == (big-F(1, 3), -big+F(2, 3))
    assert result.multipliers == (F(1, 3),) and result.squared_distance == F(2, 9)


def test_projection_is_firmly_nonexpansive_on_exact_queries():
    projector = _projector(2, ((-1, 0), (0, -1), (3, 2)), (0, 0, 6))
    sources = ((F(-3), F(2)), (F(5), F(-1)), (F(1, 7), F(2, 3)), (F(5), F(5)))
    outputs = tuple(_project(projector, point).point for point in sources)
    for i, left in enumerate(outputs):
        for j, right in enumerate(outputs):
            difference = tuple(a-b for a, b in zip(left, right, strict=True))
            assert sum(x*x for x in difference) <= sum(x*(a-b) for x, a, b in zip(difference, sources[i], sources[j], strict=True))


def test_projected_convex_hull_is_not_convex_hull_of_projected_vertices():
    projector = _projector(2, ((-1, 0), (0, -1)), (0, 0))
    assert _project(projector, (-1, 1)).point == (0, 1)
    assert _project(projector, (1, -1)).point == (1, 0)
    assert _project(projector, (0, 0)).point == (0, 0)  # Not on x+y=1.


def test_input_snapshot_frozen_result_and_nested_certificate_copy():
    rows, bounds = [[1, 0], [0, 1]], [1, 1]
    projector = _projector(2, rows, bounds)
    rows[0][0], bounds[0] = 99, -99
    result = _project(projector, (2, 3))
    assert result.point == (1, 1)
    assert verify_certificate_digest(projector.certificate) and verify_certificate_digest(result.certificate)
    projector.certificate["payload"]["dimension"] = 9
    assert verify_certificate_digest(result.certificate)
    assert projector.project((0, 0)) is None
    with pytest.raises(FrozenInstanceError):
        result.point = (0, 0)
    with pytest.raises(TypeError):
        PolyhedralProjector()
    with pytest.raises(TypeError):
        PolyhedralProjection()


def test_kernel_reuses_factors_without_hashing_or_sealing(monkeypatch):
    projector = _projector(2, ((-1, 0), (0, -1), (1, 1)), (0, 0, 1))
    module = import_module("omnibias.core.verified.polyhedral_projection")
    proof = import_module("omnibias.core.proof.certificate")
    monkeypatch.setattr(module, "factor_positive_definite", lambda *_: pytest.fail("query refactorized"))
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("kernel hashed source"))
    monkeypatch.setattr(proof, "make_certificate", lambda **_: pytest.fail("kernel sealed a callback"))
    for point in ((F(2), F(3)), (F(-1), F(-2)), (F(2), F(-1))):
        meter = _Arithmetic(projector.budget)
        result = _project_exact(projector, point, meter)
        assert result is not None and meter.used <= _query_work_bound(projector)
    assert _source_slots(projector) > 0


def test_complete_binomial_preflight_precedes_any_combination_allocation(monkeypatch):
    module = import_module("omnibias.core.verified.polyhedral_projection")
    monkeypatch.setattr(module, "combinations", lambda *_: pytest.fail("enumeration began before complete reservation"))
    assert compile_polyhedral_projector(4, ((1, 1, 1, 1),)*20, (1,)*20) is None


def test_query_preflight_refuses_before_source_hash(monkeypatch):
    projector = _projector(2, ((-1, 0), (0, -1)), (0, 0), budget=RidgeBudget(max_matrix_entries=4300))
    module = import_module("omnibias.core.verified.polyhedral_projection")
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("unreserved query proof scan"))
    assert projector.project((1, 1)) is None


def test_shared_meter_cumulative_work_and_bit_failures_are_explicit():
    projector = _projector(1, ((1,),), (0,))
    meter = _Arithmetic(RidgeBudget(max_elimination_work=10))
    with pytest.raises(ValueError, match="work"):
        _project_exact(projector, (F(1),), meter)
    low_bits = _projector(1, ((1,),), (0,), budget=RidgeBudget(max_coefficient_bits=8))
    assert low_bits.project((2**20,)) is None


@pytest.mark.parametrize("sticky", [False, True])
def test_rational_projection_needs_no_backend_and_preserves_history(sticky, monkeypatch):
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    result = _project(_projector(1, ((1,),), (0,)), (1,))
    assert result.point == (0,) and result.certificate["meta"]["transcend_backend"] == "not_used"
    assert transcend.libm_fallback_used() is sticky


def test_shared_strict_sealing_policy_is_not_bypassed(monkeypatch):
    projector = _projector(1)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        projector.project((0,))
    assert transcend.strict_backend()


@pytest.mark.parametrize("dimension,error", [(True, TypeError), (1.0, TypeError), (0, ValueError), (-1, ValueError)])
def test_invalid_dimension_raises(dimension, error):
    with pytest.raises(error):
        compile_polyhedral_projector(dimension, (), ())


@pytest.mark.parametrize("rows,bounds,error", [(((1.0,),), (0,), TypeError), (((True,),), (0,), TypeError),
                                             (((1,),), (0.0,), TypeError), (((1, 2),), (0,), ValueError),
                                             (((1,),), (), ValueError), ("x", (), TypeError)])
def test_invalid_exact_metadata_raises(rows, bounds, error):
    with pytest.raises(error):
        compile_polyhedral_projector(1, rows, bounds)
    with pytest.raises(error):
        compile_polyhedral_projector(1, rows, bounds, budget=RidgeBudget(max_matrix_entries=1, max_elimination_work=1))


@pytest.mark.parametrize("point", [(1.0,), (True,), (), (1, 2)])
def test_invalid_query_point_raises(point):
    with pytest.raises((TypeError, ValueError)):
        _projector(1).project(point)
