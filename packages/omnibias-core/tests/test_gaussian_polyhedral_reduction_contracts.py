# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Complete exact transverse covers, budget guards and immutable model binding."""

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
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    GaussianPolyhedralReduction,
    compile_gaussian_polyhedral_reduction,
)


def _model(dimension=2, *, budget=None, gauge=0, shift=None, viscosity=1):
    offset = (0,)*dimension if shift is None else shift
    centers = tuple(tuple(value+(sign*2 if i == 0 else 0) for i, value in enumerate(offset)) for sign in (-1, 1))
    return GaussianHeatMap(centers, (gauge, gauge), viscosity=viscosity, budget=budget)


def _compile(inequalities=((1, -1),), bounds=(0,), target=(0, 2), **kwargs):
    result = compile_gaussian_polyhedral_reduction(_model(**kwargs), inequalities, bounds, target)
    assert result is not None
    return result


def _cover(result, y):
    return [(index, piece) for index, piece in enumerate(result.pieces)
            if (piece.lower is None or piece.lower <= y) and (piece.upper is None or y <= piece.upper)]


def test_exact_active_switch_and_overlapping_endpoint_cover():
    result = _compile()
    assert result.midpoint == (0, 0) and result.half_difference == (2, 0)
    assert result.projector == ((0, 0), (0, 1)) and result.inverse_q == F(1, 2)
    assert result.field == result.bias_contrast == 0
    assert len(result.pieces) == 2
    free = next(piece for piece in result.pieces if piece.active_indices == ())
    active = next(piece for piece in result.pieces if piece.active_indices == (0,))
    assert (free.lower, free.upper) == (None, 2)
    assert free.point_intercept == (0, 2) and free.point_slope == (1, 0)
    assert free.transverse_cost == (0, 0, 0) and free.scalar_curvature == F(1, 2) and free.scalar_field == 0
    assert (active.lower, active.upper) == (2, None)
    assert active.point_intercept == (0, 0) and active.point_slope == (1, 1)
    assert active.transverse_cost == (2, -2, F(1, 2))
    assert active.scalar_curvature == 1 and active.scalar_field == 1
    for y in (F(-100), F(0), F(1), F(2), F(5, 2), F(100)):
        covering = _cover(result, y)
        assert covering
        assert len(covering) == (2 if y == 2 else 1)
        for index, _ in covering:
            assert result.lift(index, y) == (y, max(y, F(2)))
            assert result.transverse_value(index, y) == max(y-2, F(0))**2/2


def test_upper_wall_truncates_the_complete_cover_and_duplicate_rows_are_safe():
    plain = _compile(((1, -1), (0, 1)), (0, 3))
    duplicated = _compile(((1, -1), (0, 1), (2, -2)), (0, 3, 0))
    for y in (F(-100), F(0), F(2), F(5, 2), F(3), F(4)):
        for result in (plain, duplicated):
            covering = _cover(result, y)
            assert bool(covering) is (y <= 3)
            for index, _ in covering:
                assert result.lift(index, y) == (y, max(y, F(2)))
    assert len(duplicated.pieces) >= len(plain.pieces)


def test_empty_constraints_give_one_unbounded_piece_and_queries_use_full_domain():
    result = _compile((), (), (F(1, 3), F(2, 3)))
    assert len(result.pieces) == 1
    piece = result.pieces[0]
    assert piece.active_indices == () and piece.lower is None and piece.upper is None
    assert piece.point_intercept == (0, F(2, 3)) and piece.point_slope == (1, 0)
    assert piece.transverse_cost == (0, 0, 0)
    assert piece.search_interval[0] <= piece.search_interval[1]
    assert result.lift(0, 10**3) == (1000, F(2, 3))
    assert result.transverse_value(0, 10**3) == 0


@pytest.mark.parametrize("bound,expected", [(0, True), (1, True), (-1, False)])
def test_zero_rows_preserve_feasibility_or_prove_exact_infeasibility(bound, expected):
    result = _compile(((0, 0),), (bound,), (0, 0))
    assert bool(result.pieces) is expected
    if expected:
        assert len(result.pieces) == 1 and result.pieces[0].active_indices == ()
    assert verify_certificate_digest(result.certificate)


def test_projected_zero_rows_restrict_scalar_domain_instead_of_disappearing():
    result = _compile(((1, 0), (-1, 0)), (F(3, 2), F(1, 2)), (0, 0))
    assert len(result.pieces) == 1
    piece = result.pieces[0]
    assert piece.active_indices == () and (piece.lower, piece.upper) == (F(-1, 2), F(3, 2))
    assert result.lift(0, F(-1, 2)) == (F(-1, 2), 0)
    assert result.lift(0, F(3, 2)) == (F(3, 2), 0)


def test_lower_dimensional_transverse_feasibility_needs_no_slater_condition():
    # x2=0 is feasible although neither inequality can be strictly satisfied;
    # one independent active normal suffices for the nonzero KKT multiplier.
    result = _compile(((0, 1), (0, -1)), (0, 0), (0, 2))
    assert result.pieces
    for y in (F(-9), F(0), F(9)):
        covering = _cover(result, y)
        assert covering
        for index, piece in covering:
            assert result.lift(index, y) == (y, 0)
            assert result.transverse_value(index, y) == 2
            assert len(piece.active_indices) <= 1


def test_singleton_scalar_feasible_domain_is_retained_exactly():
    result = _compile(((1, 0), (-1, 0)), (3, -3), (0, 2))
    assert len(result.pieces) == 1
    piece = result.pieces[0]
    assert piece.lower == piece.upper == 3 and piece.search_interval == (3, 3)
    assert result.lift(0, 3) == (3, 2)
    assert not _cover(result, F(299, 100)) and not _cover(result, F(301, 100))


def test_empty_piece_tuple_is_a_proof_of_infeasibility_not_a_resource_refusal():
    result = _compile(((0, 1), (0, -1)), (0, -1), (0, 0))
    assert result.pieces == () and verify_certificate_digest(result.certificate)
    with pytest.raises((ValueError, IndexError)):
        result.lift(0, 0)


@pytest.mark.parametrize("row,bound,endpoint", [((1,), -10, -10), ((-1,), -10, 10)])
def test_one_dimensional_disjoint_stationary_strip_retains_boundary(row, bound, endpoint):
    result = compile_gaussian_polyhedral_reduction(_model(1), (row,), (bound,), (0,))
    assert result is not None and len(result.pieces) == 1
    piece = result.pieces[0]
    assert piece.active_indices == () and piece.transverse_cost == (0, 0, 0)
    assert piece.search_interval == (endpoint, endpoint)
    assert result.lift(0, endpoint) == (endpoint,)


@pytest.mark.parametrize("viscosity", [F(1, 4), 1, 2, 4])
def test_exact_reduction_does_not_require_a_contractive_or_nonconvex_regime(viscosity):
    result = _compile((), (), (0, 0), viscosity=viscosity)
    assert result.inverse_q == viscosity/2
    assert len(result.pieces) == 1


def test_common_bias_and_large_coordinate_gauges_preserve_transverse_costs():
    plain = _compile()
    shift = (2**1500, 2**1600)
    moved = _compile(((1, -1),), (shift[0]-shift[1],), (0, 2), gauge=2**2500, shift=shift)
    assert moved.midpoint == shift and moved.bias_contrast == plain.bias_contrast == 0
    assert len(moved.pieces) == len(plain.pieces)
    for index, (left, right) in enumerate(zip(plain.pieces, moved.pieces, strict=True)):
        assert left.active_indices == right.active_indices
        for name in ("lower", "upper", "point_slope", "transverse_cost", "scalar_curvature", "scalar_field", "search_interval"):
            assert getattr(left, name) == getattr(right, name)
        sample = F(0) if left.active_indices == () else F(3)
        assert moved.lift(index, sample) == tuple(a+b for a, b in zip(plain.lift(index, sample), shift, strict=True))


def test_factory_record_snapshot_certificate_binding_and_immutable_pieces():
    inequalities, bounds, target = [[1, -1]], [0], [0, 2]
    model = _model()
    result = compile_gaussian_polyhedral_reduction(model, inequalities, bounds, target)
    assert result is not None and result.model is model
    saved = deepcopy(result.certificate)
    inequalities[0][0], bounds[0], target[0] = 99, 99, 99
    assert result.inequalities == ((1, -1),) and result.upper_bounds == (0,) and result.target == (0, 2)
    assert result.certificate == saved
    with pytest.raises(TypeError):
        GaussianPolyhedralReduction()
    with pytest.raises(FrozenInstanceError):
        result.field = F(0)
    with pytest.raises(FrozenInstanceError):
        result.pieces[0].lower = F(0)
    assert verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == [] and certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    result.certificate["payload"]["target"] = [[99, 1], [99, 1]]
    assert not verify_certificate_digest(result.certificate) and verify_certificate_digest(saved)
    for method in (result.lift, result.transverse_value):
        with pytest.raises(ValueError, match="damaged"):
            method(0, 0)


@pytest.mark.parametrize("inequalities,bounds,target,error", [
    (None, (), (0, 0), TypeError), (((1,),), (0,), (0, 0), ValueError),
    (((1, -1),), (), (0, 0), ValueError), (((1.0, -1),), (0,), (0, 0), TypeError),
    (((True, -1),), (0,), (0, 0), TypeError), (((1, -1),), (False,), (0, 0), TypeError),
    (((1, -1),), (0.0,), (0, 0), TypeError), (((1, -1),), (0,), (0,), ValueError),
    (((1, -1),), (0,), (0, F(1, 2), 1), ValueError), (((1, -1),), (0,), (0.0, 2), TypeError),
])
def test_bad_exact_metadata_is_rejected_before_preflight(inequalities, bounds, target, error, monkeypatch):
    model = _model()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid metadata reached preflight"))
    with pytest.raises(error):
        compile_gaussian_polyhedral_reduction(model, inequalities, bounds, target)


def test_bad_model_type_and_unsupported_families():
    with pytest.raises(TypeError):
        compile_gaussian_polyhedral_reduction(None, (), (), (0,))
    for model in (GaussianHeatMap(((0,),), (0,)), GaussianHeatMap(((0,), (0,)), (0, 1)),
                  GaussianHeatMap(((-1,), (0,), (1,)), (0, 0, 0))):
        assert compile_gaussian_polyhedral_reduction(model, (), (), (0,)) is None


@pytest.mark.parametrize("method", ["lift", "transverse_value"])
@pytest.mark.parametrize("index,parameter,error", [(True, 0, TypeError), (0.0, 0, TypeError),
    (-1, 0, IndexError), (99, 0, IndexError), (0, False, TypeError), (0, 0.0, TypeError),
    (0, None, TypeError), (0, 3, ValueError)])
def test_queries_validate_indices_exact_parameters_and_piece_membership(method, index, parameter, error):
    result = _compile()
    with pytest.raises(error):
        getattr(result, method)(index, parameter)


def test_complete_enumeration_count_includes_dependent_and_infeasible_bases():
    result = _compile(((1, -1), (2, -2), (0, 0)), (0, 0, 1))
    assert result.enumerated_bases == 4  # Empty plus all three singleton subsets.
    assert result.feasible
    assert len(result.pieces) == 3  # The projected-zero active row is discarded.
    with pytest.raises(TypeError):
        type(result.pieces[0])()
    empty = _compile(((0, 0),), (-1,), (0, 0))
    assert not empty.feasible and empty.enumerated_bases == 2


def test_combinatorial_refusal_happens_before_any_subset_or_matrix_arithmetic(monkeypatch):
    model = _model(6)
    inequalities = ((0, 1, 0, 0, 0, 0),)*30
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")

    def fail(*args, **kwargs):
        pytest.fail("combinatorial preflight allowed enumeration or exact data allocation")

    monkeypatch.setattr(module, "combinations", fail)
    monkeypatch.setattr(_Arithmetic, "exact", fail)
    assert compile_gaussian_polyhedral_reduction(model, inequalities, (0,)*30, (0,)*6) is None


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_aggregate_preflight_refuses_before_geometry(guard, monkeypatch):
    model = _model()

    def exhausted(*args):
        raise ValueError("test resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "add", lambda *_: pytest.fail("failed preflight reached arithmetic"))
    assert compile_gaussian_polyhedral_reduction(model, ((1, -1),), (0,), (0, 2)) is None


@pytest.mark.parametrize("operation", ["solve_affine", "factor_positive_definite"])
def test_resource_failure_after_a_valid_piece_is_never_returned_as_partial_cover(operation, monkeypatch):
    model = _model()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")
    calls = []

    def exhausted(*args):
        calls.append(1)
        raise ValueError("test exact arithmetic resource exhaustion")

    monkeypatch.setattr(module, operation, exhausted)
    assert compile_gaussian_polyhedral_reduction(model, ((1, -1),), (0,), (0, 2)) is None
    assert calls  # The empty-active feasible piece was visited before this call.


def test_rank_and_factor_solve_work_share_the_constructor_meter(monkeypatch):
    model = _model()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")
    original_rank, original_factor = module.solve_affine, module.factor_positive_definite
    meters, counts = [], []

    def rank(*args):
        arithmetic = args[-1]
        meters.append(id(arithmetic))
        counts.append(arithmetic.used)
        value = original_rank(*args)
        counts.append(arithmetic.used)
        return value

    def factor(*args):
        arithmetic = args[-1]
        meters.append(id(arithmetic))
        counts.append(arithmetic.used)
        value = original_factor(*args)
        counts.append(arithmetic.used)
        return value

    monkeypatch.setattr(module, "solve_affine", rank)
    monkeypatch.setattr(module, "factor_positive_definite", factor)
    result = compile_gaussian_polyhedral_reduction(model, ((1, -1), (2, -2)), (0, 0), (0, 2))
    assert result is not None and result.enumerated_bases == 3
    assert len(set(meters)) == 1 and len(meters) == 4
    assert all(a <= b for a, b in zip(counts, counts[1:], strict=False)) and counts[-1] > counts[0]


def test_exact_input_and_query_bit_budget_failures_are_explicit():
    assert compile_gaussian_polyhedral_reduction(_model(), ((2**5000, -1),), (0,), (0, 2)) is None
    assert compile_gaussian_polyhedral_reduction(_model(), ((1, -1),), (2**5000,), (0, 2)) is None
    assert compile_gaussian_polyhedral_reduction(_model(), ((1, -1),), (0,), (0, 2**5000)) is None
    result = _compile((), (), (0, 0))
    for method in (result.lift, result.transverse_value):
        with pytest.raises(ValueError, match="bit"):
            method(0, 2**5000)


@pytest.mark.parametrize("method", ["lift", "transverse_value"])
def test_query_preflight_precedes_certificate_digest(method, monkeypatch):
    result = _compile()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")

    def exhausted(*args):
        raise ValueError("test query resource exhaustion")

    monkeypatch.setattr(RidgeBudget, "_entries", exhausted)
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("failed preflight reached certificate hashing"))
    with pytest.raises(ValueError, match="resource"):
        getattr(result, method)(0, 0)


def test_query_reservation_scales_with_original_constraint_metadata_not_only_pieces(monkeypatch):
    small = compile_gaussian_polyhedral_reduction(_model(1), ((1,),), (10,), (0,))
    large = compile_gaussian_polyhedral_reduction(_model(1), ((1,),)*128, (10,)*128, (0,))
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


def test_queries_need_no_new_enumeration_rank_test_or_factorization(monkeypatch):
    result = _compile()
    module = import_module("omnibias.core.verified.gaussian_polyhedral_reduction")

    def fail(*args, **kwargs):
        pytest.fail("compiled query attempted a fresh rank test, solve or enumeration")

    for operation in ("combinations", "solve_affine", "factor_positive_definite"):
        monkeypatch.setattr(module, operation, fail)
    assert result.lift(0, 0) == (0, 2) and result.transverse_value(0, 0) == 0
    active_index = next(i for i, piece in enumerate(result.pieces) if piece.active_indices == (0,))
    assert result.lift(active_index, 3) == (3, 3) and result.transverse_value(active_index, 3) == F(1, 2)


@pytest.mark.parametrize("sticky", [False, True])
def test_exact_compiler_needs_no_transcendental_backend_and_preserves_history(sticky, monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "exp_iv", lambda *_: pytest.fail("exact reduction attempted exponential evaluation"))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    result = compile_gaussian_polyhedral_reduction(model, ((1, -1),), (0,), (0, 2))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    assert result.lift(0, 0) == (0, 2)
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


def test_exact_compiler_does_not_bypass_shared_strict_sealing(monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        compile_gaussian_polyhedral_reduction(model, ((1, -1),), (0,), (0, 2))
    assert transcend.strict_backend()


def test_exact_compiler_does_not_clear_sticky_history_in_strict_mode(monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        compile_gaussian_polyhedral_reduction(model, ((1, -1),), (0,), (0, 2))
    assert transcend.strict_backend() and transcend.libm_fallback_used()
