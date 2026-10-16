# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent root/error checks for bounded certified posterior proposals."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from math import isfinite

import pytest
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    seal_certificate,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_entropy_solver import (
    GaussianEntropySolve,
    solve_gaussian_entropy,
)
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _unique(*, centers=((0,), (1,)), biases=(0, 0), viscosity=1, time=1,
            target=None, rows=(), bounds=(), reference=None, budget=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    goal = (F(1, 5),)*model.dimension if target is None else target
    domain = certify_gaussian_posterior_domain(model, projector, goal)
    assert domain is not None
    result = certify_gaussian_entropy_unique(domain, reference=reference)
    assert result is not None
    return result


def _solve(unique=None, **kwargs):
    result = solve_gaussian_entropy(_unique() if unique is None else unique, **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _posterior(mp, unique, point):
    model = unique.domain.model
    logits = [_mp(mp, bias)-sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
              for center, bias in zip(model.centers, model.biases, strict=True)]
    weights = [mp.exp(value-max(logits)) for value in logits]
    total = sum(weights)
    return tuple(value/total for value in weights)


def _energy(mp, unique, point):
    model = unique.domain.model
    logits = [_mp(mp, bias-model.biases[0])-sum((x-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
              for center, bias in zip(model.centers, model.biases, strict=True)]
    peak = max(logits)
    return -2*_mp(mp, model.viscosity)*(peak+mp.log(sum(mp.exp(value-peak) for value in logits)))-sum(
        _mp(mp, v)*(x-_mp(mp, c)) for v, x, c in zip(unique.domain.target, point, model.centers[0], strict=True))


def _unconstrained_root(mp, unique):
    model = unique.domain.model

    def equation(*point):
        weights = _posterior(mp, unique, point)
        return tuple(x-_mp(mp, model.time*v)-sum(p*_mp(mp, a[j]) for p, a in zip(weights, model.centers, strict=True))
                     for j, (x, v) in enumerate(zip(point, unique.domain.target, strict=True)))

    start = tuple(_mp(mp, model.time*v)+sum(_mp(mp, center[j]) for center in model.centers)/model.mode_count
                  for j, v in enumerate(unique.domain.target))
    root = mp.findroot(equation, start, tol=mp.mpf("1e-85"), maxsteps=100)
    return tuple(root) if hasattr(root, "rows") else (root,)


def _assert_diagnostics(result):
    assert result.evaluations == result.iterations+1
    assert 0 <= result.selected_iteration <= result.iterations <= result.max_iterations
    assert len(result.error_bounds) == result.evaluations
    assert result.residual.error_bound == result.error_bounds[result.selected_iteration]
    assert result.residual.error_bound == min(result.error_bounds)
    assert all(isfinite(value) and value >= 0 for value in result.error_bounds)
    assert result.status in {"tolerance", "iteration_budget", "stagnation"}
    if result.status == "tolerance":
        assert result.residual.error_bound <= result.tolerance
    assert verify_certificate_digest(result.residual.certificate)


@pytest.mark.parametrize("initial", [None, (1, 0), (0, 1), (F(2, 7), F(5, 7))])
def test_scalar_proposals_enclose_independent_true_inverse(initial):
    mp = pytest.importorskip("mpmath")
    unique = _unique()
    result = _solve(unique, initial_probabilities=initial, tolerance=1e-10, max_iterations=32)
    _assert_diagnostics(result)
    assert result.status == "tolerance"
    with mp.workdps(100):
        root = _unconstrained_root(mp, unique)[0]
        error = abs(_mp(mp, result.residual.point[0])-root)
        assert error <= _mp(mp, result.residual.error_bound)
        assert _mp(mp, result.residual.enclosure[0].lo) <= root <= _mp(mp, result.residual.enclosure[0].hi)
        minimum = _energy(mp, unique, (root,))
        sample = _energy(mp, unique, (_mp(mp, result.residual.point[0]),))
        assert _mp(mp, result.residual.minimum.lo) <= minimum <= _mp(mp, result.residual.minimum.hi)
        assert _mp(mp, result.residual.value.lo) <= sample <= _mp(mp, result.residual.value.hi)
        assert sample-minimum <= _mp(mp, result.residual.suboptimality_bound)


def test_coupled_noncollinear_true_root_obeys_euclidean_error_bound():
    mp = pytest.importorskip("mpmath")
    unique = _unique(centers=((0, 0), (1, 0), (0, 1)), biases=(0, F(1, 3), F(-1, 7)),
                     target=(F(1, 10), F(-1, 5)))
    result = _solve(unique, tolerance=1e-9, max_iterations=32)
    _assert_diagnostics(result)
    with mp.workdps(100):
        root = _unconstrained_root(mp, unique)
        norm = mp.sqrt(sum((_mp(mp, x)-value)**2 for x, value in zip(result.residual.point, root, strict=True)))
        assert norm <= _mp(mp, result.residual.error_bound)
        assert all(_mp(mp, iv.lo) <= value <= _mp(mp, iv.hi) for iv, value in zip(result.residual.enclosure, root, strict=True))


def test_zero_update_budget_still_certifies_initial_vertex_outside_posterior_domain():
    unique = _unique()
    assert unique.domain.parameter_box[0].hi < 1
    result = _solve(unique, initial_probabilities=(1, 0), tolerance=0, max_iterations=0)
    _assert_diagnostics(result)
    assert result.status == "iteration_budget"
    assert result.evaluations == 1 and result.iterations == result.selected_iteration == 0
    assert result.residual.probabilities == (F(1), F(0))
    assert result.residual.point == (F(1, 5),)
    assert result.residual.posterior[1].lo > 0  # Proposal weights are not asserted to be the true posterior.
    assert result.residual.error_bound > 0


def test_default_zero_update_candidate_is_exactly_uniform():
    result = _solve(max_iterations=0, tolerance=0)
    assert result.residual.probabilities == (F(1, 2), F(1, 2))
    assert result.iterations == 0 and result.evaluations == 1


def test_equal_logits_at_exact_uniform_root_certify_zero_error_without_updates():
    result = _solve(_unique(target=(0,)), tolerance=0, max_iterations=32)
    _assert_diagnostics(result)
    assert result.status == "tolerance"
    assert result.evaluations == 1 and result.iterations == 0
    assert result.residual.error_bound == 0
    assert result.residual.point == (F(1, 2),)


def test_unchanged_finite_precision_proposal_reports_stagnation_without_false_zero_error():
    result = _solve(tolerance=0, max_iterations=32)
    _assert_diagnostics(result)
    assert result.status == "stagnation"
    assert 0 < result.iterations < result.max_iterations
    assert result.residual.error_bound > 0


def test_numerical_iteration_budget_exit_retains_a_true_error_bound():
    result = _solve(initial_probabilities=(1, 0), tolerance=0, max_iterations=1)
    _assert_diagnostics(result)
    assert result.status == "iteration_budget"
    assert result.iterations == 1 and result.evaluations == 2
    assert result.residual.error_bound > 0


def test_selected_candidate_minimizes_exact_recorded_residual_not_last_iteration(monkeypatch):
    unique = _unique()
    module = import_module("omnibias.core.verified.gaussian_entropy_solver")
    original = module._ResidualOracle.evaluate
    evaluated = []

    def record(self, probabilities):
        data = original(self, probabilities)
        evaluated.append(data)
        return data

    monkeypatch.setattr(module._ResidualOracle, "evaluate", record)
    result = _solve(unique, tolerance=0, max_iterations=32)
    selected = min(range(len(evaluated)), key=lambda i: evaluated[i].residual_squared_bound)
    assert result.selected_iteration == selected
    assert result.residual.point == evaluated[selected].point
    assert result.residual.probabilities == evaluated[selected].probabilities
    assert result.residual.residual_squared_bound == evaluated[selected].residual_squared_bound


def test_nonconvex_global_q_four_can_be_solved_with_posterior_certificate():
    mp = pytest.importorskip("mpmath")
    unique = _unique(centers=((-2,), (2,)), biases=(0, -12), viscosity=F(1, 2), target=(0,))
    assert unique.domain.model.contraction_bound == 4
    result = _solve(unique, tolerance=1e-10, max_iterations=16)
    _assert_diagnostics(result)
    assert result.status == "tolerance"
    with mp.workdps(100):
        root = _unconstrained_root(mp, unique)[0]
        assert root < _mp(mp, F(-19, 10))
        assert abs(_mp(mp, result.residual.point[0])-root) <= _mp(mp, result.residual.error_bound)


def test_underflowed_posterior_intervals_produce_valid_exact_simplex_proposals():
    unique = _unique(centers=((0,), (10,)), biases=(0, -1000), target=(0,))
    assert not unique.domain.gradient_safe and unique.domain.model.contraction_bound > 1
    result = _solve(unique, tolerance=1e-8, max_iterations=8)
    _assert_diagnostics(result)
    assert result.status == "tolerance"
    assert all(value >= 0 for value in result.residual.probabilities)
    assert sum(result.residual.probabilities) == 1
    assert result.residual.error_bound > 0


@pytest.mark.parametrize("count", [1, 3])
def test_zero_variance_exact_affine_case_meets_zero_tolerance_without_updates(count):
    unique = _unique(centers=((F(1, 3), F(-2, 5)),)*count, biases=tuple(range(count)),
                     rows=((1, 0), (-1, 0)), bounds=(0, 0), target=(F(1, 7), F(1, 5)))
    assert unique.variance_upper_bound == 0
    result = _solve(unique, tolerance=0, max_iterations=0)
    assert result.status == "tolerance" and result.residual.error_bound == 0
    assert result.residual.point == (F(0), F(-1, 5))
    assert result.evaluations == 1 and result.iterations == 0


def test_constraint_projection_remains_exactly_feasible():
    unique = _unique(centers=((0, 0), (1, 0), (0, 1)), biases=(0, 0, 0),
                     rows=((1, 1), (-1, -1)), bounds=(F(1, 3), F(-1, 3)), target=(F(1, 8), F(-1, 9)))
    result = _solve(unique, tolerance=1e-7, max_iterations=24)
    _assert_diagnostics(result)
    assert sum(result.residual.point) == F(1, 3)


def test_factory_record_and_nested_certificate_snapshots():
    result = _solve(max_iterations=0)
    assert schema_errors_v1(result.certificate) == []
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    saved = deepcopy(result.certificate)
    with pytest.raises(TypeError):
        GaussianEntropySolve()
    with pytest.raises(FrozenInstanceError):
        result.iterations = 999
    result.unique.certificate["payload"]["type"] = "damaged"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert solve_gaussian_entropy(result.unique) is None


def test_initial_probability_list_and_replay_metadata_are_snapshotted():
    probabilities = [F(2, 7), F(5, 7)]
    result = _solve(initial_probabilities=probabilities, max_iterations=2)
    probabilities[:] = [0, 1]
    assert result.certificate["payload"]["initial_probabilities"] == [[2, 7], [5, 7]]
    assert verify_certificate_digest(result.certificate)


@pytest.mark.parametrize("nested", ["unique", "domain", "projector"])
def test_damaged_dependency_refuses_without_evaluating_candidates(nested):
    unique = deepcopy(_unique())
    dependency = unique if nested == "unique" else unique.domain if nested == "domain" else unique.domain.projector
    dependency.certificate["payload"]["type"] = "damaged"
    assert solve_gaussian_entropy(unique, max_iterations=0) is None


@pytest.mark.parametrize("strict", [False, True])
def test_strict_and_sticky_provenance_are_preserved(strict, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _solve(unique, max_iterations=0)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        solve_gaussian_entropy(unique, max_iterations=0)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_missing_rigorous_backend_is_explicit(monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        solve_gaussian_entropy(unique, max_iterations=0)


def test_conditional_uniqueness_proof_cannot_be_promoted_to_rigorous_solver():
    unique = deepcopy(_unique())
    unique.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(unique.certificate)
    unique.certificate.clear()
    unique.certificate.update(changed)
    assert not certificate_is_unconditional(unique.certificate)
    assert solve_gaussian_entropy(unique, max_iterations=0) is None


@pytest.mark.parametrize("keyword,value,error", [("max_iterations", True, TypeError), ("max_iterations", 1.5, TypeError),
                                                ("max_iterations", -1, ValueError), ("tolerance", True, TypeError),
                                                ("tolerance", "1", TypeError), ("tolerance", -1, ValueError),
                                                ("tolerance", float("inf"), ValueError), ("tolerance", float("nan"), ValueError)])
def test_invalid_controls_raise_before_budget(keyword, value, error, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid control reached budget preflight"))
    with pytest.raises(error):
        solve_gaussian_entropy(unique, **{keyword: value})


@pytest.mark.parametrize("probabilities,error", [((True, 0), TypeError), ((.5, .5), TypeError), ((1,), ValueError),
                                               ((1, 0, 0), ValueError), ((-1, 2), ValueError), ((0, 0), ValueError),
                                               ((1, 1), ValueError), ((F(1, 2), F(1, 3)), ValueError)])
def test_invalid_initial_simplex_raises(probabilities, error):
    with pytest.raises(error):
        solve_gaussian_entropy(_unique(), initial_probabilities=probabilities)


def test_bad_unique_type_raises():
    with pytest.raises(TypeError):
        solve_gaussian_entropy(object())


def test_overlarge_iteration_request_refuses_as_resource_failure():
    assert solve_gaussian_entropy(_unique(), max_iterations=10**1000) is None


def test_later_resource_failure_does_not_return_an_earlier_valid_candidate(monkeypatch):
    unique = _unique()
    module = import_module("omnibias.core.verified.gaussian_entropy_solver")
    original = module._ResidualOracle.evaluate
    calls = 0

    def bounded(self, probabilities):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("test cumulative candidate arithmetic budget exhaustion")
        return original(self, probabilities)

    monkeypatch.setattr(module._ResidualOracle, "evaluate", bounded)
    assert solve_gaussian_entropy(unique, initial_probabilities=(1, 0), tolerance=0, max_iterations=2) is None
    assert calls == 2


def test_solver_reuses_factors_and_seals_only_the_best_residual(monkeypatch):
    unique = _unique(rows=((-1,),), bounds=(0,))
    solver = import_module("omnibias.core.verified.gaussian_entropy_solver")
    residual = import_module("omnibias.core.verified.gaussian_entropy_residual")
    projector = import_module("omnibias.core.verified.polyhedral_projection")
    linear = import_module("omnibias.core._rational_linear")

    def fail(*args, **kwargs):
        pytest.fail("solver reset a query budget or compiled a new factorization")

    monkeypatch.setattr(residual, "certify_gaussian_entropy_residual", fail)
    monkeypatch.setattr(type(unique.domain.projector), "project", fail)
    for owner in (projector, linear):
        for name in ("solve_affine", "factor_positive_definite"):
            monkeypatch.setattr(owner, name, fail)
    original = solver._seal_residual
    seals = []

    def seal(source, data):
        seals.append(data)
        return original(source, data)

    monkeypatch.setattr(solver, "_seal_residual", seal)
    result = _solve(unique, max_iterations=3, tolerance=0)
    assert result.evaluations == 4 and len(seals) == 1
    assert result.residual.residual_squared_bound == seals[0].residual_squared_bound


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_aggregate_preflight_precedes_dependency_hashing(guard, monkeypatch):
    unique = _unique()
    module = import_module("omnibias.core.verified.gaussian_entropy_solver")

    def exhausted(*args):
        raise ValueError("test solver resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_unique_intact", lambda *_a: pytest.fail("unreserved source scan"))
    assert solve_gaussian_entropy(unique) is None
