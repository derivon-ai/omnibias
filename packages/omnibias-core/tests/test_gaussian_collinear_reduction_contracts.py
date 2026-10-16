# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original-mixture geometry and complete collinear reduction contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_collinear_reduction import (
    GaussianCollinearPiece,
    GaussianCollinearReduction,
    compile_gaussian_collinear_reduction,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap


def _axis_model(*, order=(0, 1, 2), gauge=0, shift=(0, 0), budget=None):
    centers = tuple((F(value)+shift[0], F(shift[1])) for value in (-2, 0, 3))
    biases = (F(1, 3)+gauge, F(-1, 5)+gauge, F(2, 7)+gauge)
    return GaussianHeatMap(tuple(centers[i] for i in order), tuple(biases[i] for i in order), budget=budget)


def _compile(rows=((1, -1),), bounds=(0,), target=(0, 2), **kwargs):
    result = compile_gaussian_collinear_reduction(_axis_model(**kwargs), rows, bounds, target)
    assert result is not None
    return result


def _cover(result, y):
    return [(index, piece) for index, piece in enumerate(result.pieces)
            if (piece.lower is None or y >= piece.lower) and (piece.upper is None or y <= piece.upper)]


def _parameter_for_point(reduction, point):
    return sum(a*(x-c) for a, x, c in zip(reduction.direction, point, reduction.origin, strict=True))/(2*reduction.model.viscosity*reduction.model.time)


def test_nonuniform_three_mode_exact_coordinates_logits_and_scale():
    model = GaussianHeatMap(((1, -2), (3, -1), (4, F(-1, 2))), (F(1, 3), F(-1, 5), F(2, 7)), viscosity=F(3, 4), time=2)
    result = compile_gaussian_collinear_reduction(model, (), (), (F(1, 4), F(-1, 3)))
    assert result is not None
    assert result.origin == (1, -2) and result.direction == (2, 1)
    assert result.center_coordinates == (0, 1, F(3, 2))
    assert result.logit_offsets == (0, F(-41, 30), F(-323, 168))
    assert result.mean_bounds == (0, F(3, 2))
    assert result.projector == ((F(1, 5), F(-2, 5)), (F(-2, 5), F(4, 5)))
    assert result.axial_scale == F(3, 5) and result.field == F(1, 15)
    assert result.axial_scale != 1/model.contraction_bound
    assert model.contraction_bound == (result.mean_bounds[1]-result.mean_bounds[0])**2/(4*result.axial_scale)
    piece = result.pieces[0]
    assert piece.point_intercept == (F(41, 30), F(-41, 15))
    assert piece.point_slope == (F(6, 5), F(3, 5))
    assert piece.search_interval == (F(1, 9), F(47, 18))


def test_logits_match_independent_original_squared_gaussian_contrasts():
    result = _compile()
    model = result.model
    for y in (F(-7), F(0), F(2), F(4), F(9, 2), F(10)):
        for index, _ in _cover(result, y):
            point = result.lift(index, y)
            reference_cost = sum((x-a)**2 for x, a in zip(point, model.centers[0], strict=True))
            for j, center in enumerate(model.centers):
                cost = sum((x-a)**2 for x, a in zip(point, center, strict=True))
                original = model.biases[j]-model.biases[0]-(cost-reference_cost)/(4*model.viscosity*model.time)
                assert original == result.center_coordinates[j]*y+result.logit_offsets[j]


def test_independent_original_transverse_projection_and_switch_cost():
    result = _compile()
    assert len(result.pieces) == 2 and result.enumerated_bases == 2
    free = next(piece for piece in result.pieces if piece.active_indices == ())
    active = next(piece for piece in result.pieces if piece.active_indices == (0,))
    assert (free.lower, free.upper) == (None, 4) and free.point_intercept == (-2, 2) and free.point_slope == (1, 0)
    assert free.transverse_cost == (0, 0, 0) and free.search_interval == (0, 4)
    assert (active.lower, active.upper) == (4, None) and active.point_intercept == (-2, -2) and active.point_slope == (1, 1)
    assert active.transverse_cost == (8, -4, F(1, 2))
    assert active.scalar_curvature == 1 and active.scalar_field == 2
    assert active.search_interval == (4, F(9, 2))
    for y in (F(-100), F(0), F(4), F(9, 2), F(100)):
        covering = _cover(result, y)
        assert len(covering) == (2 if y == 4 else 1)
        for index, _ in covering:
            point = result.lift(index, y)
            assert point == (y-2, max(y-2, F(2)))
            cost = (point[1]-2)**2/2
            assert result.transverse_value(index, y) == cost
            # Every feasible point in this fiber has second coordinate >=y-2;
            # the returned coordinate is the projection of the preferred 2.
            for alternative in (max(y-2, F(2))+F(1, 3), max(y-2, F(2))+3):
                assert cost <= (alternative-2)**2/2


def test_mode_permutation_changes_parameter_gauge_but_not_original_qp_solution():
    reductions = [_compile(order=permutation) for permutation in ((0, 1, 2), (2, 1, 0), (1, 2, 0))]
    assert len({item.direction for item in reductions}) == 3
    for coordinate in (F(-10), F(0), F(2), F(5, 2), F(10)):
        for reduction in reductions:
            y = _parameter_for_point(reduction, (coordinate, 0))
            covering = _cover(reduction, y)
            assert covering
            for index, _ in covering:
                assert reduction.lift(index, y) == (coordinate, max(coordinate, F(2)))
                assert reduction.transverse_value(index, y) == max(coordinate-2, F(0))**2/2


def test_duplicate_positions_and_first_nonzero_direction_are_preserved():
    model = GaussianHeatMap(((3, -1), (3, -1), (1, 0), (0, F(1, 2))), (0, 1, 2, 3))
    result = compile_gaussian_collinear_reduction(model, (), (), (0, 0))
    assert result is not None and result.direction == (-2, 1)
    assert result.center_coordinates == (0, 0, 1, F(3, 2))
    assert result.logit_offsets[:2] == (0, 1)
    assert result.mean_bounds == (0, F(3, 2)) and len(result.logit_offsets) == model.mode_count


def test_huge_common_bias_and_translation_preserve_exact_reduced_geometry():
    shift = (2**1500, 2**1600)
    plain = _compile()
    moved = _compile(bounds=(shift[0]-shift[1],), gauge=2**2500, shift=shift)
    for name in ("direction", "projector", "axial_scale", "field", "center_coordinates", "logit_offsets", "mean_bounds"):
        assert getattr(plain, name) == getattr(moved, name)
    assert len(plain.pieces) == len(moved.pieces)
    for index, (left, right) in enumerate(zip(plain.pieces, moved.pieces, strict=True)):
        for name in ("active_indices", "lower", "upper", "point_slope", "transverse_cost", "scalar_curvature", "scalar_field", "search_interval"):
            assert getattr(left, name) == getattr(right, name)
        parameter = 0 if left.active_indices == () else 5
        assert moved.lift(index, parameter) == tuple(x+c for x, c in zip(plain.lift(index, parameter), shift, strict=True))


def test_huge_exact_transverse_target_cancels_without_squaring_irrelevant_common_cost():
    huge = 2**2500
    model = GaussianHeatMap(((0, 0), (2, 1), (6, 3)), (0, 0, 0))
    result = compile_gaussian_collinear_reduction(model, (), (), (huge, -2*huge))
    assert result is not None and result.field == 0
    assert result.pieces[0].transverse_cost == (0, 0, 0)
    assert result.lift(0, 0) == (huge, -2*huge)
    assert result.transverse_value(0, 1) == 0


@pytest.mark.parametrize("rows,bounds,feasible", [(((0, 0),), (0,), True), (((0, 0),), (1,), True), (((0, 0),), (-1,), False)])
def test_zero_rows_retain_exact_feasibility_meaning(rows, bounds, feasible):
    result = _compile(rows, bounds, (0, 0))
    assert bool(result.pieces) is feasible and result.feasible is feasible
    assert verify_certificate_digest(result.certificate)


def test_redundant_rows_lower_dimensional_fibers_and_complete_basis_count():
    result = _compile(((0, 1), (0, -1), (0, 2)), (0, 0, 0), (0, 2))
    assert result.enumerated_bases == 4 and result.feasible
    for y in (F(-5), F(0), F(5)):
        assert _cover(result, y)
        for index, _ in _cover(result, y):
            assert result.lift(index, y) == (y-2, 0)
            assert result.transverse_value(index, y) == 2


def test_projected_zero_rows_can_leave_one_scalar_point():
    result = _compile(((1, 0), (-1, 0)), (F(1, 3), F(-1, 3)))
    assert len(result.pieces) == 1
    piece = result.pieces[0]
    assert piece.lower == piece.upper == F(7, 3)
    assert piece.search_interval == (F(7, 3), F(7, 3))
    assert result.lift(0, F(7, 3)) == (F(1, 3), 2)


@pytest.mark.parametrize("row,bound,endpoint", [((1,), -10, -8), ((-1,), -10, 12)])
def test_one_dimensional_disjoint_search_strip_keeps_the_boundary(row, bound, endpoint):
    model = GaussianHeatMap(((-2,), (0,), (3,)), (0, 0, 0))
    result = compile_gaussian_collinear_reduction(model, (row,), (bound,), (0,))
    assert result is not None and len(result.pieces) == 1
    assert result.pieces[0].search_interval == (endpoint, endpoint)
    assert result.lift(0, endpoint) == (endpoint-2,)


@pytest.mark.parametrize("centers", [((0, 0),), ((0, 0), (0, 0)), ((0, 0), (1, 0), (0, 1)), ((0, 0), (1, 0), (2, F(1, 2**1000)))])
def test_unsupported_or_exactly_noncollinear_models_refuse_before_piece_enumeration(centers, monkeypatch):
    model = GaussianHeatMap(centers, (0,)*len(centers))
    module = import_module("omnibias.core.verified.gaussian_collinear_reduction")
    monkeypatch.setattr(module, "_compile_pieces", lambda *_a, **_k: pytest.fail("unsupported family reached piece enumeration"))
    assert compile_gaussian_collinear_reduction(model, (), (), (0, 0)) is None


def test_factory_records_metadata_snapshots_and_digest_checked_queries():
    model = _axis_model()
    rows, bounds, target = [[1, -1]], [0], [0, 2]
    result = compile_gaussian_collinear_reduction(model, rows, bounds, target)
    assert result is not None and result.model is model
    saved = deepcopy(result.certificate)
    rows[0][0], bounds[0], target[0] = 99, 99, 99
    assert result.inequalities == ((1, -1),) and result.upper_bounds == (0,) and result.target == (0, 2)
    assert result.certificate == saved and schema_errors_v1(result.certificate) == []
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    for cls in (GaussianCollinearReduction, GaussianCollinearPiece):
        with pytest.raises(TypeError):
            cls()
    with pytest.raises(FrozenInstanceError):
        result.field = F(0)
    with pytest.raises(FrozenInstanceError):
        result.pieces[0].lower = F(0)
    result.certificate["payload"]["field"] = [99, 1]
    assert not verify_certificate_digest(result.certificate) and verify_certificate_digest(saved)
    for method in (result.lift, result.transverse_value):
        with pytest.raises(ValueError, match="damaged"):
            method(0, 0)


@pytest.mark.parametrize("rows,bounds,target,error", [(None, (), (0, 0), TypeError), (((1,),), (0,), (0, 0), ValueError),
    (((1, -1),), (), (0, 0), ValueError), (((1.0, -1),), (0,), (0, 0), TypeError),
    (((True, -1),), (0,), (0, 0), TypeError), (((1, -1),), (False,), (0, 0), TypeError),
    (((1, -1),), (0,), (0,), ValueError), (((1, -1),), (0,), (0.0, 0), TypeError)])
def test_invalid_exact_metadata_precedes_preflight(rows, bounds, target, error, monkeypatch):
    model = _axis_model()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid metadata reached preflight"))
    with pytest.raises(error):
        compile_gaussian_collinear_reduction(model, rows, bounds, target)


@pytest.mark.parametrize("method", ["lift", "transverse_value"])
@pytest.mark.parametrize("index,value,error", [(-1, 0, IndexError), (99, 0, IndexError), (True, 0, TypeError),
    (0, 0.0, TypeError), (0, True, TypeError), (0, 5, ValueError)])
def test_query_exact_metadata_and_whole_piece_membership(method, index, value, error):
    result = _compile()
    with pytest.raises(error):
        getattr(result, method)(index, value)


def test_bad_model_type_raises():
    with pytest.raises(TypeError):
        compile_gaussian_collinear_reduction(None, (), (), (0,))


def test_negative_support_coordinates_change_the_proved_search_strip():
    model = GaussianHeatMap(((0,), (3,), (-2,)), (0, 0, 0))
    result = compile_gaussian_collinear_reduction(model, (), (), (0,))
    assert result is not None
    assert result.center_coordinates == (0, 1, F(-2, 3)) and result.mean_bounds == (F(-2, 3), 1)
    assert result.axial_scale == F(2, 9)
    assert result.pieces[0].search_interval == (F(-3), F(9, 2))


def test_full_combinatorial_preflight_precedes_shared_enumerator(monkeypatch):
    centers = ((0,)*6, (1, 0, 0, 0, 0, 0), (3, 0, 0, 0, 0, 0))
    model = GaussianHeatMap(centers, (0, 0, 0))
    module = import_module("omnibias.core.verified.gaussian_collinear_reduction")
    monkeypatch.setattr(module, "_compile_pieces", lambda *_: pytest.fail("unbounded enumeration started"))
    monkeypatch.setattr(_Arithmetic, "exact", lambda *_: pytest.fail("failed preflight copied metadata"))
    assert compile_gaussian_collinear_reduction(model, ((0, 1, 0, 0, 0, 0),)*30, (0,)*30, (0,)*6) is None


def test_all_original_modes_are_reserved_before_geometry_and_enumeration(monkeypatch):
    limits = RidgeBudget(max_matrix_entries=10_000)
    model = GaussianHeatMap(tuple((j % 3,) for j in range(128)), (0,)*128, budget=limits)
    module = import_module("omnibias.core.verified.gaussian_collinear_reduction")
    monkeypatch.setattr(module, "_compile_pieces", lambda *_: pytest.fail("unreserved mode metadata reached enumeration"))
    monkeypatch.setattr(_Arithmetic, "exact", lambda *_: pytest.fail("unreserved mode metadata copied"))
    assert compile_gaussian_collinear_reduction(model, (), (), (0,)) is None


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_resource_preflight_happens_before_exact_geometry(guard, monkeypatch):
    model = _axis_model()

    def exhausted(*args):
        raise ValueError("test aggregate resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "add", lambda *_: pytest.fail("failed preflight reached geometry"))
    assert compile_gaussian_collinear_reduction(model, ((1, -1),), (0,), (0, 2)) is None


@pytest.mark.parametrize("operation", ["solve_affine", "factor_positive_definite"])
def test_resource_failure_after_first_piece_never_returns_a_partial_cover(operation, monkeypatch):
    model = _axis_model()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")
    called = []

    def exhausted(*args):
        called.append(1)
        raise ValueError("test late exact resource exhaustion")

    monkeypatch.setattr(module, operation, exhausted)
    assert compile_gaussian_collinear_reduction(model, ((1, -1),), (0,), (0, 2)) is None
    assert called


def test_generic_transverse_helper_and_rank_checks_share_parent_arithmetic(monkeypatch):
    model = _axis_model()
    module = import_module("omnibias.core.verified.gaussian_collinear_reduction")
    shared = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")
    original_compile, original_rank = module._compile_pieces, shared.solve_affine
    identities, counts = [], []

    def compile_pieces(*args):
        arithmetic = next(value for value in args if isinstance(value, _Arithmetic))
        identities.append(id(arithmetic))
        counts.append(arithmetic.used)
        result = original_compile(*args)
        counts.append(arithmetic.used)
        return result

    def rank(*args):
        identities.append(id(args[-1]))
        return original_rank(*args)

    monkeypatch.setattr(module, "_compile_pieces", compile_pieces)
    monkeypatch.setattr(shared, "solve_affine", rank)
    result = compile_gaussian_collinear_reduction(model, ((1, -1),), (0,), (0, 2))
    assert result is not None and len(identities) == 2 and len(set(identities)) == 1
    assert 0 < counts[0] < counts[1]


def test_exact_metadata_and_query_bit_budget_failures_are_explicit():
    model = _axis_model()
    assert compile_gaussian_collinear_reduction(model, ((2**5000, -1),), (0,), (0, 2)) is None
    assert compile_gaussian_collinear_reduction(model, ((1, -1),), (2**5000,), (0, 2)) is None
    assert compile_gaussian_collinear_reduction(model, ((1, -1),), (0,), (0, 2**5000)) is None
    result = _compile((), (), (0, 0))
    for method in (result.lift, result.transverse_value):
        with pytest.raises(ValueError, match="bit"):
            method(0, 2**5000)


@pytest.mark.parametrize("method", ["lift", "transverse_value"])
def test_query_preflight_precedes_dependency_digest(method, monkeypatch):
    result = _compile()
    module = import_module("omnibias.core.verified.gaussian_collinear_reduction")

    def exhausted(*args):
        raise ValueError("test query budget exhaustion")

    monkeypatch.setattr(RidgeBudget, "_entries", exhausted)
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("unreserved query scanned the certificate"))
    with pytest.raises(ValueError, match="budget"):
        getattr(result, method)(0, 0)


def test_query_reservation_counts_modes_even_when_geometry_has_one_piece(monkeypatch):
    small_model = GaussianHeatMap(((0,), (1,), (2,)), (0, 0, 0))
    large_model = GaussianHeatMap(tuple((j % 3,) for j in range(128)), (0,)*128)
    small = compile_gaussian_collinear_reduction(small_model, (), (), (0,))
    large = compile_gaussian_collinear_reduction(large_model, (), (), (0,))
    assert small is not None and large is not None and len(small.pieces) == len(large.pieces) == 1
    entries, work = [], []
    original_entries, original_work = RidgeBudget._entries, RidgeBudget._work

    def reserve(self, count):
        entries.append(count)
        return original_entries(self, count)

    def charge(self, count):
        work.append(count)
        return original_work(self, count)

    monkeypatch.setattr(RidgeBudget, "_entries", reserve)
    monkeypatch.setattr(RidgeBudget, "_work", charge)
    small.lift(0, 0)
    small_entries, small_work = entries[0], work[0]
    entries.clear()
    work.clear()
    large.lift(0, 0)
    assert entries[0] > small_entries+128 and work[0] > small_work+128


def test_queries_need_no_fresh_factorization_or_enumeration(monkeypatch):
    result = _compile()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")

    def fail(*args, **kwargs):
        pytest.fail("compiled collinear query attempted a solve or enumeration")

    for name in ("combinations", "solve_affine", "factor_positive_definite"):
        monkeypatch.setattr(module, name, fail)
    assert result.lift(0, 0) == (-2, 2) and result.transverse_value(0, 0) == 0
    active = next(i for i, piece in enumerate(result.pieces) if piece.active_indices == (0,))
    assert result.lift(active, 5) == (3, 3) and result.transverse_value(active, 5) == F(1, 2)


@pytest.mark.parametrize("sticky", [False, True])
def test_pure_exact_compiler_preserves_state_without_optional_backend(sticky, monkeypatch):
    model = _axis_model()
    monkeypatch.setattr(transcend, "exp_iv", lambda *_: pytest.fail("exact compiler attempted exponential evaluation"))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    result = compile_gaussian_collinear_reduction(model, ((1, -1),), (0,), (0, 2))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    result.lift(0, 0)
    assert not transcend.strict_backend() and transcend.libm_fallback_used() is sticky


def test_shared_strict_backend_requirement_is_not_bypassed(monkeypatch):
    model = _axis_model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_collinear_reduction(model, (), (), (0, 0))
    assert transcend.strict_backend()


def test_shared_strict_sticky_policy_is_preserved(monkeypatch):
    model = _axis_model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        compile_gaussian_collinear_reduction(model, (), (), (0, 0))
    assert transcend.strict_backend() and transcend.libm_fallback_used()
