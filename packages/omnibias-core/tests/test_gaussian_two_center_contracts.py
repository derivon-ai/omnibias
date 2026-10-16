# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact branch selection, complete brackets and certificate resource boundaries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

import pytest
from omnibias.core._rational_linear import RationalLDLT
from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.proof.certificate import (
    certificate_is_unconditional,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_two_center import (
    GaussianTwoCenterMinimum,
    certify_gaussian_two_center_minimum,
)


def _model(*, bias=0, shift=0, gauge=0, budget=None):
    return GaussianHeatMap(((shift-2,), (shift+2,)), (gauge, gauge+bias), budget=budget)


def _minimum(target=(0,), **kwargs):
    result = certify_gaussian_two_center_minimum(_model(), target, **kwargs)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _winner(mp, result):
    # Independent scalar minimizer equation; the upper starting point is beyond
    # the positive outer root, so findroot cannot select the unstable origin.
    inverse_q, field = _mp(mp, result.inverse_q), abs(_mp(mp, result.field))
    upper = (field+1)/inverse_q
    return mp.findroot(lambda y: inverse_q*y-mp.tanh(y)-field, (upper, upper+1))


def test_exact_geometry_tie_and_every_global_minimizer_are_retained():
    result = _minimum()
    assert result.model.contraction_bound == 2
    assert result.midpoint == (0,) and result.half_difference == (2,)
    assert result.orthogonal_projector == ((0,),)
    assert result.bias_contrast == result.field == 0 and result.inverse_q == F(1, 2)
    assert result.branch_signs == (-1, 1) and len(result.minimizer_boxes) == 2
    assert result.termination == "tolerance"
    lo, hi = result.positive_root_bounds
    assert 0 < lo <= hi and hi-lo <= result.tolerance
    assert result.minimizer_boxes[0][0].hi < 0 < result.minimizer_boxes[1][0].lo
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        root = _winner(mp, result)
        assert _mp(mp, lo) <= root <= _mp(mp, hi)
        for sign, box in zip(result.branch_signs, result.minimizer_boxes, strict=True):
            assert mp.mpf(box[0].lo) <= sign*root <= mp.mpf(box[0].hi)


@pytest.mark.parametrize("sign", [-1, 1])
def test_tiny_exact_nonzero_field_selects_one_branch_without_float_sign_guess(sign):
    epsilon = F(sign, 10**1000)
    result = certify_gaussian_two_center_minimum(_model(bias=4*epsilon), (0,))
    assert result is not None and result.field == epsilon
    assert result.branch_signs == (sign,) and len(result.minimizer_boxes) == 1
    box = result.minimizer_boxes[0][0]
    assert box.lo > 0 if sign > 0 else box.hi < 0
    assert result.positive_root_bounds[0] > 1


def test_exact_field_cancellation_distinguishes_a_tie_from_an_arbitrarily_small_perturbation():
    epsilon = F(1, 10**1000)
    model = _model(bias=1)
    tie = certify_gaussian_two_center_minimum(model, (F(-1, 2),))
    perturbed = certify_gaussian_two_center_minimum(model, (F(-1, 2)+2*epsilon,))
    assert tie is not None and perturbed is not None
    assert tie.field == 0 and tie.branch_signs == (-1, 1)
    assert perturbed.field == epsilon and perturbed.branch_signs == (1,)


def test_common_bias_gauge_and_translation_preserve_scalar_geometry():
    plain = certify_gaussian_two_center_minimum(_model(bias=F(1, 3)), (F(2, 7),))
    shifted = certify_gaussian_two_center_minimum(_model(bias=F(1, 3), gauge=2**2500, shift=7), (F(2, 7),))
    assert plain is not None and shifted is not None
    assert shifted.midpoint == (7,)
    for name in ("half_difference", "orthogonal_projector", "bias_contrast", "field", "inverse_q", "branch_signs", "positive_root_bounds"):
        assert getattr(plain, name) == getattr(shifted, name)
    assert shifted.model.biases[0] == 2**2500
    for left, right in zip(plain.minimizer_boxes[0], shifted.minimizer_boxes[0], strict=True):
        assert left.lo+7 <= right.hi and right.lo <= left.hi+7


def test_mode_permutation_changes_coordinates_but_preserves_physical_minimizer():
    model = _model(bias=F(1, 3))
    reverse = GaussianHeatMap(tuple(reversed(model.centers)), tuple(reversed(model.biases)))
    forward = certify_gaussian_two_center_minimum(model, (F(1, 5),))
    backward = certify_gaussian_two_center_minimum(reverse, (F(1, 5),))
    assert forward is not None and backward is not None
    assert forward.field == -backward.field and forward.bias_contrast == -backward.bias_contrast
    assert forward.branch_signs == (1,) and backward.branch_signs == (-1,)
    assert forward.positive_root_bounds == backward.positive_root_bounds
    assert forward.minimizer_boxes == backward.minimizer_boxes


def test_exact_transverse_geometry_is_retained_in_a_coupled_ambient_space():
    model = GaussianHeatMap(((2, -4), (4, 0)), (0, F(1, 7)), viscosity=F(1, 2), time=2)
    target = (F(2, 3), F(-1, 4))
    result = certify_gaussian_two_center_minimum(model, target)
    assert result is not None
    assert result.midpoint == (3, -2) and result.half_difference == (1, 2)
    assert result.orthogonal_projector == ((F(4, 5), F(-2, 5)), (F(-2, 5), F(1, 5)))
    assert result.inverse_q == F(2, 5)
    assert result.field == F(2, 21)
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        root = _winner(mp, result)
        physical = [2*_mp(mp, v)+_mp(mp, c)+_mp(mp, a)*mp.tanh(root)
                    for v, c, a in zip(target, result.midpoint, result.half_difference, strict=True)]
        for interval, coordinate in zip(result.minimizer_boxes[0], physical, strict=True):
            assert mp.mpf(interval.lo) <= coordinate <= mp.mpf(interval.hi)


@pytest.mark.parametrize("model", [GaussianHeatMap(((0,),), (0,)), GaussianHeatMap(((0,), (0,)), (0, 1)),
    GaussianHeatMap(((-1,), (1,)), (0, 0)), GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 2)),
    GaussianHeatMap(((-2,), (0,), (2,)), (0, 0, 0))])
def test_unsupported_models_refuse_instead_of_claiming_infeasibility(model):
    assert certify_gaussian_two_center_minimum(model, (0,)) is None


def test_original_metadata_is_snapshotted_and_the_record_is_factory_only():
    centers, biases, target = [[-2], [2]], [0, F(1, 3)], [F(1, 5)]
    model = GaussianHeatMap(centers, biases)
    result = certify_gaussian_two_center_minimum(model, target)
    assert result is not None and result.model is model
    saved = deepcopy(result.certificate)
    centers[0][0], biases[0], target[0] = 99, 99, 99
    assert result.model.centers == ((-2,), (2,)) and result.model.biases == (0, F(1, 3))
    assert result.target == (F(1, 5),) and result.certificate == saved
    with pytest.raises(TypeError):
        GaussianTwoCenterMinimum()
    with pytest.raises(FrozenInstanceError):
        result.field = F(0)
    assert verify_certificate_digest(result.certificate)
    assert schema_errors_v1(result.certificate) == [] and certificate_is_unconditional(result.certificate)
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    result.certificate["payload"]["field"] = [99, 1]
    assert not verify_certificate_digest(result.certificate) and verify_certificate_digest(saved)
    fresh = certify_gaussian_two_center_minimum(model, (F(1, 5),))
    assert fresh is not None and fresh.certificate == saved


@pytest.mark.parametrize("target,error", [(None, TypeError), ((), ValueError), ((0, 1), ValueError),
    ((True,), TypeError), ((0.0,), TypeError), ((complex(0),), TypeError)])
def test_target_metadata_validation_precedes_preflight(target, error, monkeypatch):
    model = _model()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid target reached preflight"))
    with pytest.raises(error):
        certify_gaussian_two_center_minimum(model, target)


@pytest.mark.parametrize("value,error", [(None, TypeError), (True, TypeError), (0.1, TypeError), (complex(1), TypeError), (0, ValueError), (-1, ValueError)])
def test_tolerance_metadata_validation_precedes_preflight(value, error, monkeypatch):
    model = _model()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid tolerance reached preflight"))
    with pytest.raises(error):
        certify_gaussian_two_center_minimum(model, (0,), tolerance=value)


@pytest.mark.parametrize("value,error", [(None, TypeError), (True, TypeError), (1.0, TypeError), (F(1), TypeError), (-1, ValueError)])
def test_iteration_metadata_validation_precedes_preflight(value, error, monkeypatch):
    model = _model()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_: pytest.fail("invalid step count reached preflight"))
    with pytest.raises(error):
        certify_gaussian_two_center_minimum(model, (0,), max_steps=value)


def test_wrong_model_type_is_rejected():
    with pytest.raises(TypeError):
        certify_gaussian_two_center_minimum(None, (0,))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_factory_resource_preflight_precedes_geometry_allocation(guard, monkeypatch):
    model = _model()

    def exhausted(*args):
        raise ValueError("resource exhausted")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(_Arithmetic, "add", lambda *_: pytest.fail("failed preflight reached arithmetic"))
    assert certify_gaussian_two_center_minimum(model, (0,)) is None


def test_exact_bit_growth_and_unrepresentable_physical_lifts_refuse_safely():
    assert certify_gaussian_two_center_minimum(_model(), (2**5000,)) is None
    assert certify_gaussian_two_center_minimum(_model(), (0,), tolerance=F(1, 2**5000)) is None
    assert certify_gaussian_two_center_minimum(_model(shift=2**1800), (0,)) is None


def test_scalar_certifier_uses_no_numerical_inverse_or_matrix_factorization(monkeypatch):
    model = _model()

    def fail(*args, **kwargs):
        pytest.fail("scalar two-center certificate attempted a matrix or inverse solver")

    monkeypatch.setattr(RationalLDLT, "solve", fail)
    monkeypatch.setattr(import_module("omnibias.core._rational_linear"), "factor_positive_definite", fail)
    module = import_module("omnibias.core.verified.inverse_gaussian_heat")
    monkeypatch.setattr(module, "inverse_gaussian_heat", fail)
    monkeypatch.setattr(module, "certify_gaussian_inverse", fail)
    assert certify_gaussian_two_center_minimum(model, (F(1, 4),)) is not None


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_iteration_termination_keeps_complete_two_branch_brackets(steps):
    result = _minimum(max_steps=steps)
    assert result.termination == "steps" and result.iterations == steps and not result.unique
    assert result.positive_root_bounds[1]-result.positive_root_bounds[0] > result.tolerance
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        root = _winner(mp, result)
        for sign, box in zip(result.branch_signs, result.minimizer_boxes, strict=True):
            assert mp.mpf(box[0].lo) <= sign*root <= mp.mpf(box[0].hi)


def test_sub_binary64_tolerance_returns_valid_precision_termination():
    result = _minimum(tolerance=F(1, 10**100))
    assert result.termination == "precision" and result.iterations < result.max_steps
    lo, hi = result.positive_root_bounds
    assert hi-lo > result.tolerance
    mp = pytest.importorskip("mpmath")
    with mp.workdps(120):
        root = _winner(mp, result)
        assert _mp(mp, lo) < root < _mp(mp, hi)
        for sign, box in zip(result.branch_signs, result.minimizer_boxes, strict=True):
            assert mp.mpf(box[0].lo) <= sign*root <= mp.mpf(box[0].hi)


@pytest.mark.parametrize("work", [2000, 4000, 10000])
def test_refinement_budget_preserves_brackets_when_final_output_fits(work):
    result = certify_gaussian_two_center_minimum(_model(budget=RidgeBudget(max_elimination_work=work)), (0,))
    assert result is not None and result.termination == "budget"
    assert result.iterations < result.max_steps and result.positive_root_bounds[1]-result.positive_root_bounds[0] > result.tolerance
    assert len(result.minimizer_boxes) == 2 and verify_certificate_digest(result.certificate)
    mp = pytest.importorskip("mpmath")
    with mp.workdps(90):
        root = _winner(mp, result)
        assert _mp(mp, result.positive_root_bounds[0]) <= root <= _mp(mp, result.positive_root_bounds[1])


def test_construction_refinement_and_final_lift_use_one_cumulative_meter(monkeypatch):
    model = _model()
    counts = []
    original = RidgeBudget._work

    def record(self, count):
        counts.append(count)
        return original(self, count)

    monkeypatch.setattr(RidgeBudget, "_work", record)
    result = certify_gaussian_two_center_minimum(model, (0,), max_steps=3)
    assert result is not None and result.iterations == 3
    assert len(counts) > 3 and all(a < b for a, b in zip(counts, counts[1:], strict=False))
    assert counts[-1] > counts[0]+3*256


def test_extreme_step_count_metadata_is_budget_checked_before_serialization():
    assert certify_gaussian_two_center_minimum(_model(), (0,), max_steps=2**5000) is None


def test_requested_tolerance_controls_scalar_bracket_not_physical_box_width():
    a = 2**200
    model = GaussianHeatMap(((-a,), (a,)), (0, 0), viscosity=2**398)
    result = certify_gaussian_two_center_minimum(model, (0,))
    assert result is not None and result.inverse_q == F(1, 2)
    assert result.termination == "tolerance"
    lo, hi = result.positive_root_bounds
    assert hi-lo <= result.tolerance
    assert result.minimizer_boxes[1][0].hi-result.minimizer_boxes[1][0].lo > 1


def test_exact_q_just_above_one_is_not_rounded_to_criticality():
    q = 1+F(1, 2**1200)
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=1/(2*q))
    result = certify_gaussian_two_center_minimum(model, (0,), max_steps=0)
    assert result is not None and result.inverse_q == 1/q < 1
    assert result.positive_root_bounds == (1-1/q, q)
    assert result.positive_root_bounds[0] > 0 and result.branch_signs == (-1, 1)
    # Outward physical rounding may include zero without confusing the theorem's
    # exact assertion that there are two distinct nonzero minima.
    assert result.minimizer_boxes[0][0].hi <= 0 <= result.minimizer_boxes[1][0].lo


@pytest.mark.parametrize("sign", [-1, 1])
def test_huge_field_signs_and_exact_initial_brackets_remain_finite(sign):
    result = _minimum((sign*10**300,), max_steps=0)
    assert result.branch_signs == (sign,) and result.unique
    assert result.field == sign*F(10**300, 2)
    assert result.positive_root_bounds == (F(1, 2), F(10**300+2))
    # The exact winner lies between target and target+2*sign; do not sample
    # tanh at 10**300 or round the exact selection test to machine precision.
    box = result.minimizer_boxes[0][0]
    assert F(box.lo) <= sign*(10**300+1) <= F(box.hi)


@pytest.mark.parametrize("sticky", [False, True])
def test_tiny_inverse_q_uses_rational_exponential_upper_without_a_backend(sticky, monkeypatch):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), viscosity=F(1, 2*10**400))
    module = import_module("omnibias.core.verified.gaussian_two_center")
    monkeypatch.setattr(module, "exp_iv", lambda *_: pytest.fail("huge root required an exponential"))
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    result = certify_gaussian_two_center_minimum(model, (0,), max_steps=16)
    assert result is not None and result.inverse_q == F(1, 10**400)
    assert result.termination == "steps" and result.iterations == 16
    assert result.certificate["meta"]["transcend_backend"] == "not_used"
    assert result.certificate["payload"]["lower_sign_evidence"]["kind"] == "rational_exponential_upper"
    assert result.minimizer_boxes[0][0].lo == -1 and result.minimizer_boxes[1][0].hi == 1
    assert result.minimizer_boxes[1][0].lo > 0.999
    assert transcend.libm_fallback_used() is sticky and not transcend.strict_backend()


@pytest.mark.parametrize("strict", [False, True])
def test_successful_refinement_preserves_strict_flag_and_records_actual_backend(strict, monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    result = certify_gaussian_two_center_minimum(model, (0,))
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "mpmath"
    assert certificate_is_unconditional(result.certificate)
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()


@pytest.mark.parametrize("strict", [False, True])
def test_refinement_refuses_sticky_conditional_history_without_clearing_it(strict, monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_two_center_minimum(model, (0,))
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_absent_rigorous_backend_refuses_refinement_but_allows_exact_endpoints(monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_two_center_minimum(model, (0,))
    result = certify_gaussian_two_center_minimum(model, (0,), max_steps=0)
    assert result is not None and result.certificate["meta"]["transcend_backend"] == "not_used"
    assert not transcend.strict_backend() and not transcend.libm_fallback_used()


def test_exact_endpoint_path_still_obeys_shared_strict_sealing_policy(monkeypatch):
    model = _model()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_two_center_minimum(model, (0,), max_steps=0)
    assert transcend.strict_backend()


def test_numerical_probe_failure_keeps_last_proved_bracket(monkeypatch):
    model = _model()
    module = import_module("omnibias.core.verified.gaussian_two_center")

    def fail(*args):
        raise ArithmeticError("simulated interval numerical range failure")

    monkeypatch.setattr(module, "exp_iv", fail)
    result = certify_gaussian_two_center_minimum(model, (0,))
    assert result is not None and result.termination == "precision"
    assert result.positive_root_bounds[0] < F(19, 10) < F(192, 100) <= result.positive_root_bounds[1]
    assert result.branch_signs == (-1, 1) and verify_certificate_digest(result.certificate)
