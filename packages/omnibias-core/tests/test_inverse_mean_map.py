# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global inverse guarantees, independent roots, provenance and refusal cases."""

from __future__ import annotations

import importlib
import math
import subprocess
import sys
from collections.abc import Sequence
from copy import deepcopy
from fractions import Fraction
from typing import cast

import pytest
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.inverse_mean_map import certify_mean_inverse, inverse_mean_map
from omnibias.core.verified.kantorovich import krawczyk_certificate
from omnibias.core.verified.mean_map import MeanEvaluation, MeanMap


def _simplex() -> MeanMap:
    return MeanMap(((0, 0), (1, 0), (0, 1)), (0, 1, 2), (Fraction(1, 3),) * 2)


def _binary() -> MeanMap:
    return MeanMap(((0,), (1,)), (0, 0), (Fraction(1, 2),))


def test_exact_simplex_inverse_has_global_certificate_and_bound_source() -> None:
    pytest.importorskip("mpmath")
    model = _simplex()
    certificate = certify_mean_inverse(model, (-1.0, -2.0), ((6.0, 3.0), (3.0, 6.0)), 1e-5)
    assert certificate is not None
    assert certificate.affine_rank == 2
    assert certificate.kappa < 1.0
    assert certificate.point == (-1.0, -2.0)
    assert all(axis.contains(expected) for axis, expected in zip(certificate.enclosure, (-1.0, -2.0), strict=True))
    payload = certificate.certificate["payload"]
    assert payload["target"] == [[1, 3], [1, 3]]
    assert payload["biases"] == [[0, 1], [1, 1], [2, 1]]
    assert payload["affine_differences"] == [[[1, 1], [0, 1]], [[0, 1], [1, 1]]]
    assert payload["tested_box"] and payload["residual_at_center"] and payload["jacobian_on_box"]
    assert "theorem_prover_verified" not in certificate.certificate["honesty"]
    assert certificate_is_unconditional(certificate.certificate)
    assert schema_errors_v1(certificate.certificate) == []
    tampered = deepcopy(certificate.certificate)
    tampered["payload"]["target"][0][0] = 2
    assert not verify_certificate_digest(tampered)


def test_damped_solver_certifies_separable_analytic_inverse() -> None:
    mp = pytest.importorskip("mpmath")
    model = MeanMap(((-1, -1), (-1, 1), (1, -1), (1, 1)), (0, 0, 0, 0),
                    (Fraction(1, 2), Fraction(-1, 3)))
    result = inverse_mean_map(model, radii=(1e-8,))
    assert result.status == "certified" and result.certificate is not None
    assert 0 < result.iterations <= 64
    with mp.workdps(80):
        expected = (mp.atanh(mp.mpf(1) / 2), mp.atanh(-mp.mpf(1) / 3))
        for axis, exact in zip(result.certificate.enclosure, expected, strict=True):
            assert mp.mpf(axis.lo) <= exact <= mp.mpf(axis.hi)


def test_zero_iteration_limit_can_certify_initial_point() -> None:
    pytest.importorskip("mpmath")
    result = inverse_mean_map(_simplex(), initial=(-1.0, -2.0), max_iterations=0)
    assert result.status == "certified"
    assert result.iterations == 0


def test_zero_iteration_failure_is_inconclusive() -> None:
    pytest.importorskip("mpmath")
    result = inverse_mean_map(_simplex(), max_iterations=0)
    assert result.status == "not_certified"
    assert result.certificate is None
    assert result.point == (0.0, 0.0)


@pytest.mark.parametrize("target", [0, 1, 2, -1])
def test_boundary_and_exterior_targets_never_gain_inverse_claim(target: int) -> None:
    pytest.importorskip("mpmath")
    result = inverse_mean_map(MeanMap(((0,), (1,)), (0, 0), (target,)), max_iterations=16)
    assert result.status == "not_certified"
    assert result.certificate is None


def test_rank_deficiency_is_not_reported_as_infeasibility(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    model = MeanMap(((0, 0), (1, 1), (2, 2)), (0, 0, 0), (1, 1))
    result = inverse_mean_map(model)
    assert result.status == "rank_deficient"
    assert result.iterations == 0 and result.certificate is None
    assert certify_mean_inverse(model, (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)), 0.1) is None


@pytest.mark.parametrize("sticky", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_provenance_and_strict_state_survive_all_proof_outcomes(
    monkeypatch: pytest.MonkeyPatch, sticky: bool, strict: bool, success: bool,
) -> None:
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    result = certify_mean_inverse(_binary(), (0.0,), ((4.0 if success else 0.0,),), 1e-4)
    assert (result is not None) is success
    if result is not None:
        assert certificate_is_unconditional(result.certificate)
    assert transcend.libm_fallback_used() is sticky
    assert transcend.strict_backend() is strict


def test_explicit_krawczyk_meta_is_backward_compatible_and_does_not_clear_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("mpmath")
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)

    def function(box: list[Interval]) -> tuple[Interval, ...]:
        return (transcend.exp_iv(box[0]) - 1.0,)

    def jacobian(box: list[Interval]) -> tuple[tuple[Interval, ...], ...]:
        return ((transcend.exp_iv(box[0]),),)

    original = krawczyk_certificate(function, jacobian, (0.0,), ((1.0,),), 0.1)
    assert original is not None and verify_certificate_digest(original.certificate)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="libm_fallback|conditionally"):
        krawczyk_certificate(function, jacobian, (0.0,), ((1.0,),), 0.1)
    explicit = krawczyk_certificate(function, jacobian, (0.0,), ((1.0,),), 0.1,
                                    certificate_meta={"transcend_backend": "mpmath"})
    assert explicit is not None
    assert explicit.enclosure == original.enclosure
    assert certificate_is_unconditional(explicit.certificate)
    assert transcend.libm_fallback_used() and transcend.strict_backend()


def test_missing_rigorous_backend_is_explicit_and_does_not_mutate_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    with pytest.raises(RuntimeError, match="rigorous transcendental backend"):
        certify_mean_inverse(_binary(), (0.0,), ((4.0,),), 1e-4)
    assert transcend.libm_fallback_used()
    assert not transcend.strict_backend()


def test_bad_preconditioner_and_extreme_finite_products_cannot_certify() -> None:
    pytest.importorskip("mpmath")
    assert certify_mean_inverse(_binary(), (0.0,), ((0.0,),), 0.1) is None
    model = MeanMap(((0,), (10**150,)), (0, 0), (5 * 10**149,))
    assert certify_mean_inverse(model, (0.0,), ((1e100,),), 1e-155) is None


def test_degenerate_or_overflowed_test_box_is_inconclusive() -> None:
    model = _binary()
    assert certify_mean_inverse(model, (1e308,), ((4.0,),), 1e-8) is None
    assert certify_mean_inverse(model, (1e308,), ((4.0,),), 1e308) is None


def test_actual_tested_box_is_reported() -> None:
    pytest.importorskip("mpmath")
    point = (-1.0, -2.0)
    radius = 1e-5
    certificate = certify_mean_inverse(_simplex(), point, ((6.0, 3.0), (3.0, 6.0)), radius)
    assert certificate is not None
    assert certificate.box == tuple(Interval(x - radius, x + radius) for x in point)
    assert all(outer.lo < inner.lo <= inner.hi < outer.hi
               for outer, inner in zip(certificate.box, certificate.enclosure, strict=True))


@pytest.mark.parametrize("value", [True, Fraction(1, 3), 2**53 + 1, math.inf, math.nan])
def test_invalid_point_and_preconditioner_metadata_raise(value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        certify_mean_inverse(_binary(), (cast(float, value),), ((4.0,),), 0.1)
    with pytest.raises((ValueError, TypeError)):
        certify_mean_inverse(_binary(), (0.0,), ((cast(float, value),),), 0.1)


@pytest.mark.parametrize("radius", [0.0, -1.0, True, Fraction(1, 3), math.inf, math.nan])
def test_invalid_radius_metadata_raises(radius: object) -> None:
    with pytest.raises((ValueError, TypeError), match="radius"):
        certify_mean_inverse(_binary(), (0.0,), ((4.0,),), cast(float, radius))


def test_invalid_dimensions_and_model_raise() -> None:
    with pytest.raises(ValueError, match="dimension"):
        certify_mean_inverse(_binary(), (), ((4.0,),), 0.1)
    with pytest.raises(ValueError, match="preconditioner"):
        certify_mean_inverse(_binary(), (0.0,), ((4.0, 0.0),), 0.1)
    with pytest.raises(TypeError, match="MeanMap"):
        inverse_mean_map(cast(MeanMap, None))
    with pytest.raises(ValueError, match="dimension"):
        inverse_mean_map(_binary(), initial=())


@pytest.mark.parametrize("iterations", [-1, True, 1.5])
def test_invalid_iteration_limits_raise(iterations: object) -> None:
    with pytest.raises((ValueError, TypeError), match="max_iterations"):
        inverse_mean_map(_binary(), max_iterations=cast(int, iterations))


@pytest.mark.parametrize("radii", [(), (0.0,), (math.inf,)])
def test_invalid_radius_sequences_raise(radii: Sequence[float]) -> None:
    with pytest.raises(ValueError, match="radi"):
        inverse_mean_map(_binary(), radii=radii)


def test_metadata_search_limits_respect_resource_budget() -> None:
    model = _binary()
    with pytest.raises(ValueError, match="work"):
        inverse_mean_map(model, max_iterations=model.budget.max_elimination_work + 1)
    with pytest.raises(ValueError, match="entries"):
        inverse_mean_map(model, radii=(0.1,) * (model.budget.max_matrix_entries + 1))


def test_cumulative_proof_budget_failure_is_inconclusive() -> None:
    pytest.importorskip("mpmath")
    model = MeanMap(((0,), (1,)), (0, 0), (Fraction(1, 2),),
                    budget=RidgeBudget(max_elimination_work=800))
    assert certify_mean_inverse(model, (0.0,), ((4.0,),), 1e-4) is None
    result = inverse_mean_map(model, initial=(3.0,), radii=(1e-8,) * 100)
    assert result.status == "not_certified"
    assert result.iterations == 0


def test_exhausted_budget_stops_remaining_radius_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mpmath")
    module = importlib.import_module("omnibias.core.verified.inverse_mean_map")
    original = module._certify
    attempts = 0

    def counted(model: MeanMap, point: tuple[float, ...], inverse: tuple[tuple[float, ...], ...],
                radius: float, work: object) -> object:
        nonlocal attempts
        attempts += 1
        return original(model, point, inverse, radius, work)

    monkeypatch.setattr(module, "_certify", counted)
    model = MeanMap(((0,), (1,)), (0, 0), (Fraction(1, 2),),
                    budget=RidgeBudget(max_elimination_work=2500))
    result = inverse_mean_map(model, initial=(3.0,), radii=(1e-8,) * 20)
    assert result.status == "not_certified"
    assert attempts == 2
    assert result.iterations == 0


def test_oversized_numerical_trial_can_backtrack_and_later_certify(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("mpmath")
    original = MeanMap.float_evaluate
    failed = False

    def proposal(model: MeanMap, point: Sequence[int | float]) -> MeanEvaluation:
        nonlocal failed
        if tuple(point) != (3.0,) and not failed:
            failed = True
            raise ArithmeticError("trial exceeds numerical range")
        return original(model, point)

    monkeypatch.setattr(MeanMap, "float_evaluate", proposal)
    result = inverse_mean_map(_binary(), initial=(3.0,), radii=(1e-8,))
    assert failed
    assert result.status == "certified" and result.certificate is not None
    assert result.certificate.enclosure[0].contains(0.0)


def test_huge_common_bias_does_not_destroy_proposal_descent() -> None:
    pytest.importorskip("mpmath")
    model = MeanMap(((0,), (1,)), (10**200, 10**200 + 1), (Fraction(1, 2),))
    result = inverse_mean_map(model, radii=(1e-8,))
    assert result.status == "certified" and result.certificate is not None
    assert result.certificate.enclosure[0].contains(-1.0)


def test_fresh_import_order_does_not_create_proof_cycle() -> None:
    result = subprocess.run([sys.executable, "-c", "import omnibias.core.proof.certificate; "
                             "from omnibias.core.verified import inverse_mean_map; "
                             "assert callable(inverse_mean_map)"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
