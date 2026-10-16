# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Sharp posterior tilts, uniform decision drift and bias-region contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from itertools import product
from math import isfinite
from random import Random

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
from omnibias.core.verified.gaussian_entropy_bias_region import (
    GaussianEntropyBiasRegion,
    certify_gaussian_entropy_bias_region,
)
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _unique(*, centers=((0,), (1,)), biases=(0, 0), viscosity=1, time=1,
            target=None, rows=(), bounds=(), reference=None, budget=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector,
                                              (0,)*model.dimension if target is None else target)
    assert domain is not None
    unique = certify_gaussian_entropy_unique(domain, reference=reference)
    assert unique is not None
    return unique


def _region(unique=None, oscillation=F(1), **kwargs):
    result = certify_gaussian_entropy_bias_region(_unique(**kwargs) if unique is None else unique, oscillation)
    assert result is not None
    return result


def _mp_exact(mp, rational):
    rational = F(rational)
    return mp.mpf(rational.numerator)/rational.denominator


def _posterior(mp, model, point, perturbation):
    unnormalized = tuple(mp.exp(_mp_exact(mp, bias+change)
                               -mp.fsum((_mp_exact(mp, x)-_mp_exact(mp, a))**2
                                        for x, a in zip(point, center, strict=True))
                               /(4*_mp_exact(mp, model.viscosity)*_mp_exact(mp, model.time)))
                         for center, bias, change in zip(model.centers, model.biases, perturbation, strict=True))
    normalizer = mp.fsum(unnormalized)
    return tuple(value/normalizer for value in unnormalized)


@pytest.mark.parametrize("oscillation", [F(0), F(1, 10), F(1), F(10)])
def test_recorded_bounds_replay_exact_rational_algebra(oscillation):
    result = _region(oscillation=oscillation, time=F(3, 2))
    unique, model = result.unique, result.unique.domain.model
    base, maximum = unique.variance_upper_bound, max(unique.squared_radii)
    threshold = 2*model.viscosity*model.time
    expected = maximum*base/(base+(maximum-base)*F(result.tilt_decay.lo))
    assert result.oscillation == oscillation
    assert result.max_squared_radius == maximum
    assert result.variance_upper_bound == expected
    assert result.uniqueness_margin == threshold-expected > 0
    assert result.contraction_bound == expected/threshold < 1
    assert result.strong_convexity_bound == (threshold-expected)/(threshold*model.time)
    assert result.mean_shift_squared_bound == model.diameter_squared*result.total_variation_bound**2
    assert result.displacement_squared_bound == result.mean_shift_squared_bound/(1-base/threshold)**2
    assert F(result.displacement_bound)**2 >= result.displacement_squared_bound
    assert isfinite(result.displacement_bound)


def test_two_point_tilt_attains_the_extremal_radius_formula():
    mp = pytest.importorskip("mpmath")
    # At the upper endpoint of K=[0,1], the original logit contrast is0;
    # this point maximizes the expected squared radius on the source domain.
    unique = _unique(centers=((0,), (1,)), biases=(0, F(-1, 4)))
    result = _region(unique, F(3, 2))
    with mp.workdps(100):
        posterior = _posterior(mp, unique.domain.model, (F(1),), (F(0), F(3, 2)))
        sharp = 1/(1+mp.exp(-mp.mpf(3)/2))
        assert mp.almosteq(posterior[1], sharp)
        assert _mp_exact(mp, result.variance_upper_bound) >= sharp
        assert _mp_exact(mp, result.variance_upper_bound)-sharp < mp.mpf("1e-12")


def test_total_variation_tanh_quarter_bound_has_a_sharp_two_point_example():
    mp = pytest.importorskip("mpmath")
    omega = F(3, 2)
    # Original logits differ by -omega/2; tilting the second by omega swaps the masses.
    unique = _unique(centers=((0,), (1,)), biases=(0, F(1, 4)-omega/2),
                     rows=((1,), (-1,)), bounds=(0, 0))
    result = _region(unique, omega)
    with mp.workdps(100):
        base = _posterior(mp, unique.domain.model, (F(0),), (F(0), F(0)))
        changed = _posterior(mp, unique.domain.model, (F(0),), (F(0), omega))
        total_variation = mp.fsum(abs(p-q) for p, q in zip(base, changed, strict=True))/2
        sharp = mp.tanh(_mp_exact(mp, omega)/4)
        assert mp.almosteq(total_variation, sharp)
        assert _mp_exact(mp, result.total_variation_bound) >= sharp
        assert _mp_exact(mp, result.total_variation_bound)-sharp < mp.mpf("1e-12")


def test_independent_random_tilts_over_physical_box_obey_all_moment_bounds():
    mp = pytest.importorskip("mpmath")
    unique = _unique(centers=((0, 0), (1, 0), (0, 1), (1, 1)), biases=(0, F(-1, 3), F(1, 7), 0),
                     viscosity=2, target=(F(1, 8), F(-1, 9)))
    result = _region(unique, F(2))
    rng = Random(812)
    changes = tuple(tuple(F(rng.randrange(-4, 5), 4) for _ in range(4)) for _ in range(12))
    points = tuple(product(*((lower, (lower+upper)/2, upper) for lower, upper in unique.domain.physical_bounds)))
    with mp.workdps(100):
        for point, change in zip(points*2, changes, strict=False):
            assert max(change)-min(change) <= result.oscillation
            base = _posterior(mp, unique.domain.model, point, (F(0),)*4)
            changed = _posterior(mp, unique.domain.model, point, change)
            expectation = mp.fsum(p*_mp_exact(mp, radius) for p, radius in zip(changed, unique.squared_radii, strict=True))
            variation = mp.fsum(abs(p-q) for p, q in zip(base, changed, strict=True))/2
            mean_shift = tuple(mp.fsum((p-q)*_mp_exact(mp, center[axis])
                                      for p, q, center in zip(changed, base, unique.domain.model.centers, strict=True))
                               for axis in range(2))
            assert expectation <= _mp_exact(mp, result.variance_upper_bound)
            assert variation <= _mp_exact(mp, result.total_variation_bound)
            assert mp.fsum(value**2 for value in mean_shift) <= _mp_exact(mp, result.mean_shift_squared_bound)


def test_threshold_with_maximum_radius_above_curvature_refuses_large_tilt():
    unique = _unique(centers=((0,), (2,)), biases=(0, -4))
    assert unique.variance_upper_bound < 2 < max(unique.squared_radii)
    assert _region(unique, F(2)).uniqueness_margin > 0
    assert certify_gaussian_entropy_bias_region(unique, F(3)) is None
    assert certify_gaussian_entropy_bias_region(unique, F(4)) is None


@pytest.mark.parametrize("oscillation", [F(1024), F(1025), F(10**1000)])
def test_radius_below_curvature_accepts_every_representable_finite_bias_radius(oscillation):
    result = _region(oscillation=oscillation)
    assert result.max_squared_radius < 2
    assert result.variance_upper_bound <= result.max_squared_radius
    assert result.uniqueness_margin > 0
    assert result.tilt_decay.lo == 0
    assert result.total_variation_bound == 1


def test_equal_maximum_radius_and_threshold_refuses_underflowed_strict_margin():
    unique = _unique(centers=((0,), (1,)), biases=(0, -4), viscosity=F(1, 2))
    assert unique.variance_upper_bound < 1 == max(unique.squared_radii)
    assert _region(unique, F(1)).uniqueness_margin > 0
    assert certify_gaussian_entropy_bias_region(unique, F(1025)) is None


def test_all_radius_values_equal_keeps_variance_bound_at_any_oscillation():
    unique = _unique(centers=((-1,), (1,)), reference=(0,))
    assert unique.variance_upper_bound == max(unique.squared_radii) == 1
    for omega in (F(0), F(1), F(10**1000)):
        result = _region(unique, omega)
        assert result.variance_upper_bound == 1 and result.uniqueness_margin == 1


@pytest.mark.parametrize("count", [1, 4])
def test_coincident_centers_have_zero_variance_and_exact_zero_drift(count):
    unique = _unique(centers=((F(1, 3), F(-2, 5)),)*count, biases=tuple(range(count)),
                     viscosity=F(3, 4), time=F(5, 3), rows=((1, 0), (-1, 0)), bounds=(0, 0))
    result = _region(unique, F(10**1000))
    assert result.max_squared_radius == result.variance_upper_bound == 0
    assert result.contraction_bound == 0
    assert result.mean_shift_squared_bound == result.displacement_squared_bound == result.displacement_bound == 0
    assert result.strong_convexity_bound == F(3, 5)


def test_zero_oscillation_is_exact_gauge_invariance_even_for_nonconstant_centers():
    unique = _unique(centers=((0,), (2,)), biases=(0, -4))
    result = _region(unique, F(0))
    assert result.variance_upper_bound == unique.variance_upper_bound
    assert result.tilt_decay.lo == result.tilt_decay.hi == 1
    assert result.total_variation_bound == result.mean_shift_squared_bound == 0
    assert result.displacement_squared_bound == result.displacement_bound == 0


def test_nonconvex_q_four_has_positive_region_and_original_rho_drift_bound():
    mp = pytest.importorskip("mpmath")
    unique = _unique(centers=((-2,), (2,)), biases=(0, -12), viscosity=F(1, 2), reference=(-2,))
    result = _region(unique, F(1))
    assert unique.domain.model.contraction_bound == 4
    assert result.uniqueness_margin > 0
    original_rho = unique.variance_upper_bound
    assert result.displacement_squared_bound == result.mean_shift_squared_bound/(1-original_rho)**2
    assert result.displacement_squared_bound < result.mean_shift_squared_bound/(1-result.contraction_bound)**2
    with mp.workdps(100):
        base = mp.findroot(lambda x: x-2*mp.tanh(2*x-6), (-2, mp.mpf("-1.9")))
        changed = mp.findroot(lambda x: x-2*mp.tanh(2*x-mp.mpf(11)/2), (-2, mp.mpf("-1.9")))
        assert 0 < (changed-base)**2 <= _mp_exact(mp, result.displacement_squared_bound)
        assert abs(changed-base) <= mp.mpf(result.displacement_bound)


def test_region_allows_active_face_switch_and_bounds_projection_input_motion():
    mp = pytest.importorskip("mpmath")
    unique = _unique(target=(F(-1, 2),), rows=((1,),), bounds=(0,))
    result = _region(unique, F(1))
    with mp.workdps(100):
        base = mp.findroot(lambda x: x+mp.mpf(1)/2-1/(1+mp.exp(mp.mpf(1)/4-x/2)), (-1, 0))
        # After increasing the second log weight by1, the original gradient
        # at x=0 is negative. The constrained optimum switches to that wall.
        changed_p = 1/(1+mp.exp(-mp.mpf(3)/4))
        assert base < 0 and changed_p > mp.mpf(1)/2
        changed = mp.mpf(0)
        old_input, new_input = base, changed_p-mp.mpf(1)/2
        assert abs(changed-base) <= mp.mpf(result.displacement_bound)
        assert abs(new_input-old_input) <= mp.mpf(result.displacement_bound)
        assert abs(new_input-old_input) > abs(changed-base)


def test_region_record_is_frozen_and_snapshots_original_unique_proof():
    result = _region()
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    saved = deepcopy(result.certificate)
    with pytest.raises(TypeError):
        GaussianEntropyBiasRegion()
    with pytest.raises(FrozenInstanceError):
        result.oscillation = F(0)
    result.unique.certificate["payload"]["type"] = "damaged"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_entropy_bias_region(result.unique, F(1)) is None


@pytest.mark.parametrize("level", ["unique", "domain", "projector"])
def test_deep_damage_refuses_region_even_at_zero_radius(level):
    unique = _unique()
    dependency = unique if level == "unique" else unique.domain if level == "domain" else unique.domain.projector
    dependency.certificate["payload"]["type"] = "damaged"
    assert certify_gaussian_entropy_bias_region(unique, F(0)) is None


@pytest.mark.parametrize("field", ["variance_upper_bound", "squared_radii", "reference"])
def test_source_field_tamper_is_detected_despite_valid_digest(field):
    unique = _unique()
    changed = F(0) if field == "variance_upper_bound" else (F(0), F(0)) if field == "squared_radii" else (F(100),)
    object.__setattr__(unique, field, changed)
    assert verify_certificate_digest(unique.certificate)
    assert certify_gaussian_entropy_bias_region(unique, F(0)) is None


def test_conditional_source_cannot_be_promoted_to_rigorous_region():
    unique = _unique()
    unique.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(unique.certificate)
    unique.certificate.clear()
    unique.certificate.update(changed)
    assert certify_gaussian_entropy_bias_region(unique, F(1)) is None


@pytest.mark.parametrize("oscillation,error", [(True, TypeError), (0.0, TypeError), ("1", TypeError), (F(-1), ValueError)])
def test_oscillation_validation_precedes_preflight(oscillation, error, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("malformed radius reached preflight"))
    with pytest.raises(error):
        certify_gaussian_entropy_bias_region(unique, oscillation)


def test_invalid_unique_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_bias_region(object(), 1)


def test_oversized_exact_metadata_refuses_before_transcendental_conversion():
    assert certify_gaussian_entropy_bias_region(_unique(), 10**1500) is None


@pytest.mark.parametrize("strict", [False, True])
def test_factory_strict_sticky_contract_preserves_history(strict, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _region(unique)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_entropy_bias_region(unique, 1)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


@pytest.mark.parametrize("oscillation", [F(0), F(1)])
def test_missing_backend_cannot_seal_a_new_region(oscillation, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_bias_region(unique, oscillation)


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_resource_preflight_precedes_source_digest_and_new_transcendentals(guard, monkeypatch):
    unique = _unique()
    module = import_module("omnibias.core.verified.gaussian_entropy_bias_region")

    def exhausted(*args):
        raise ValueError("test bias-region resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_unique_intact", lambda *_a: pytest.fail("unreserved source scan"))
    monkeypatch.setattr(transcend, "exp_iv", lambda *_a: pytest.fail("unreserved exponential"))
    assert certify_gaussian_entropy_bias_region(unique, F(1)) is None


@pytest.mark.parametrize("field", ["oscillation", "max_squared_radius", "variance_upper_bound",
    "uniqueness_margin", "contraction_bound", "strong_convexity_bound", "total_variation_bound",
    "mean_shift_squared_bound", "displacement_squared_bound", "displacement_bound", "tilt_decay"])
def test_intact_helper_replays_all_bound_record_fields(field):
    result = _region()
    module = import_module("omnibias.core.verified.gaussian_entropy_bias_region")
    assert module._region_intact(result)
    original = getattr(result, field)
    changed = Interval.point(0.0) if field == "tilt_decay" else original+1
    object.__setattr__(result, field, changed)
    assert verify_certificate_digest(result.certificate)
    assert not module._region_intact(result)


def test_intact_helper_rejects_conditional_own_proof():
    result = _region()
    result.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(result.certificate)
    result.certificate.clear()
    result.certificate.update(changed)
    assert verify_certificate_digest(result.certificate)
    module = import_module("omnibias.core.verified.gaussian_entropy_bias_region")
    assert not module._region_intact(result)


def test_source_common_bias_and_center_translation_cancel_exactly():
    original = _unique(centers=((0, 0), (1, 0), (0, 1)), biases=(0, F(1, 3), F(-1, 5)), viscosity=2)
    center_shift = (F(10**400), F(-10**400))
    translated = _unique(centers=tuple(tuple(a+offset for a, offset in zip(row, center_shift, strict=True))
                                      for row in original.domain.model.centers),
                         biases=tuple(value+10**1000 for value in original.domain.model.biases), viscosity=2)
    first, second = _region(original), _region(translated)
    for field in ("max_squared_radius", "tilt_decay", "variance_upper_bound", "uniqueness_margin",
                  "contraction_bound", "strong_convexity_bound", "total_variation_bound",
                  "mean_shift_squared_bound", "displacement_squared_bound", "displacement_bound"):
        assert getattr(first, field) == getattr(second, field)


def test_region_factory_uses_no_projection_posterior_rebuild_or_factorization(monkeypatch):
    unique = _unique()

    def fail(*args, **kwargs):
        pytest.fail("bias-region construction rebuilt a posterior or solved a decision")

    for name in ("gaussian_posterior_domain", "gaussian_entropy_residual", "gaussian_projection_minimum"):
        module = import_module("omnibias.core.verified."+name)
        monkeypatch.setattr(module, "_project_exact", fail)
    monkeypatch.setattr(type(unique.domain.projector), "project", fail)
    linear = import_module("omnibias.core._rational_linear")
    monkeypatch.setattr(linear, "factor_positive_definite", fail)
    monkeypatch.setattr(linear, "solve_affine", fail)
    domain_module = import_module("omnibias.core.verified.gaussian_posterior_domain")
    monkeypatch.setattr(domain_module, "_probabilities", fail)
    solver = import_module("omnibias.core.verified.gaussian_entropy_solver")
    monkeypatch.setattr(solver, "solve_gaussian_entropy", fail)
    assert _region(unique).uniqueness_margin > 0


def test_positive_tiny_oscillation_does_not_become_false_zero_drift():
    mp = pytest.importorskip("mpmath")
    omega = F(1, 2**400)
    result = _region(oscillation=omega)
    assert result.total_variation_bound > 0
    assert result.mean_shift_squared_bound > 0 and result.displacement_squared_bound > 0
    assert result.displacement_bound > 0
    with mp.workdps(160):
        assert _mp_exact(mp, result.total_variation_bound) >= mp.tanh(_mp_exact(mp, omega)/4)


def test_real_coefficient_bit_exhaustion_refuses_without_partial_region():
    unique = _unique(budget=RidgeBudget(max_coefficient_bits=128))
    assert _region(unique, F(0)).displacement_bound == 0
    assert certify_gaussian_entropy_bias_region(unique, F(1)) is None


def test_new_region_budget_reservation_covers_shared_actual_work(monkeypatch):
    unique = _unique(centers=((0, 0), (1, 0), (0, 1)), biases=(0, F(1, 3), F(-1, 5)), viscosity=2)
    original = _Arithmetic._work
    charges = []

    def observed(self, amount):
        charges.append(amount)
        return original(self, amount)

    monkeypatch.setattr(_Arithmetic, "_work", observed)
    result = _region(unique)
    assert charges and all(amount >= 0 for amount in charges)
    assert 0 < sum(charges) < unique.domain.model.budget.max_elimination_work
    assert verify_certificate_digest(result.certificate)
