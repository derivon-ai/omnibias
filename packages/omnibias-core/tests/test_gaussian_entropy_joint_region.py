# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Joint target/bias regions, exact Euclidean membership and proof contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module

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
from omnibias.core.verified.gaussian_entropy_bias_region import certify_gaussian_entropy_bias_region
from omnibias.core.verified.gaussian_entropy_joint_region import (
    GaussianEntropyJointRegion,
    certify_gaussian_entropy_joint_region,
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


def _region(unique=None, bias=F(1, 2), target=F(1, 4), **kwargs):
    result = certify_gaussian_entropy_joint_region(_unique(**kwargs) if unique is None else unique, bias, target)
    assert result is not None
    return result


def _mp_exact(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def test_nonunit_age_viscosity_record_replays_both_tilt_bounds_and_spatial_denominator():
    unique = _unique(centers=((0,), (2,)), viscosity=F(3, 2), time=F(5, 3))
    result = _region(unique)
    model, base = unique.domain.model, unique.variance_upper_bound
    threshold, maximum = 2*model.viscosity*model.time, max(unique.squared_radii)
    assert result.unique is unique and result.bias_oscillation == F(1, 2) and result.target_radius == F(1, 4)
    assert result.diameter_bound**2 >= model.diameter_squared
    assert result.expansion_radius == model.time*result.target_radius == F(5, 12)
    assert result.expanded_coordinate_bounds == tuple((lo-result.expansion_radius, hi+result.expansion_radius)
                                                       for lo, hi in unique.domain.physical_bounds)
    assert result.spatial_oscillation == result.diameter_bound*result.target_radius/(2*model.viscosity)
    assert result.total_oscillation == result.spatial_oscillation+result.bias_oscillation
    assert result.max_squared_radius == maximum
    spatial = maximum*base/(base+(maximum-base)*F(result.spatial_tilt_decay.lo))
    combined = maximum*base/(base+(maximum-base)*F(result.tilt_decay.lo))
    assert result.spatial_variance_upper_bound == spatial
    assert result.variance_upper_bound == combined
    assert result.spatial_contraction_bound == spatial/threshold
    assert result.contraction_bound == combined/threshold
    assert result.uniqueness_margin == threshold-combined > 0
    assert result.strong_convexity_bound == (threshold-combined)/(threshold*model.time)
    assert result.forcing_bound == result.expansion_radius+result.diameter_bound*result.total_variation_bound
    assert result.displacement_squared_bound == (result.forcing_bound/(1-result.spatial_contraction_bound))**2
    assert F(result.displacement_bound)**2 >= result.displacement_squared_bound
    assert base/threshold < result.spatial_contraction_bound < result.contraction_bound


@pytest.mark.parametrize("omega", [F(0), F(1, 10), F(1)])
def test_zero_target_radius_recovers_bias_only_guarantee(omega):
    unique = _unique()
    result = _region(unique, bias=omega, target=F(0))
    bias = certify_gaussian_entropy_bias_region(unique, omega)
    assert bias is not None
    assert result.expansion_radius == result.spatial_oscillation == 0
    assert result.spatial_tilt_decay.lo == result.spatial_tilt_decay.hi == 1
    assert result.spatial_variance_upper_bound == unique.variance_upper_bound
    assert result.variance_upper_bound == bias.variance_upper_bound
    assert result.uniqueness_margin == bias.uniqueness_margin
    assert result.total_variation_bound == bias.total_variation_bound
    assert result.displacement_squared_bound >= bias.displacement_squared_bound


def test_physical_membership_is_euclidean_expansion_not_its_coordinate_hull():
    unique = _unique(centers=((0, 0),), biases=(0,), time=2)
    result = _region(unique, bias=F(0), target=F(1))
    assert unique.domain.physical_bounds == ((F(0), F(0)), (F(0), F(0)))
    assert result.expanded_coordinate_bounds == ((F(-2), F(2)), (F(-2), F(2)))
    assert result.contains_physical((2, 0))
    assert result.contains_physical((F(6, 5), F(8, 5)))
    assert not result.contains_physical((2, 2))
    assert not result.contains_physical((F(6, 5), F(8, 5)+F(1, 2**100)))
    assert result.displacement_squared_bound == 4 and result.displacement_bound >= 2


def test_physical_membership_does_not_claim_polyhedral_feasibility():
    result = _region(_unique(centers=((0, 0),), biases=(0,), rows=((1, 0),), bounds=(0,)),
                     bias=F(0), target=F(1))
    assert result.contains_physical((F(1, 2), 0))  # Violates x<=0 but belongs to M.


def test_parameter_membership_uses_exact_l2_and_common_bias_gauge():
    result = _region(_unique(centers=((0, 0), (1, 0))), bias=F(2, 3), target=F(1))
    common = 2**2500
    assert result.contains_parameters((F(3, 5), F(4, 5)), (common, common+F(2, 3)))
    assert not result.contains_parameters((F(3, 5), F(4, 5)+F(1, 2**100)), (0, 0))
    assert not result.contains_parameters((0, 0), (common, common+F(2, 3)+F(1, 2**100)))
    assert result.contains_parameters((0, 0), (common, common))
    assert not result.contains_parameters((1, 1), (0, 0))


def test_zero_joint_radius_allows_only_target_zero_and_common_bias_changes():
    result = _region(bias=F(0), target=F(0))
    assert result.forcing_bound == result.displacement_squared_bound == result.displacement_bound == 0
    assert result.contains_parameters((0,), (2**2500, 2**2500))
    assert not result.contains_parameters((F(1, 2**100),), (0, 0))
    assert not result.contains_parameters((0,), (0, F(1, 2**100)))
    lower, upper = result.unique.domain.physical_bounds[0]
    assert result.contains_physical((lower,)) and result.contains_physical((upper,))
    assert not result.contains_physical((upper+F(1, 2**100),))


@pytest.mark.parametrize("count", [1, 4])
def test_coincident_modes_have_exact_affine_target_drift_even_for_huge_bias_radius(count):
    unique = _unique(centers=((F(1, 3), F(-2, 5)),)*count, biases=tuple(range(count)),
                     viscosity=F(3, 4), time=F(5, 3), target=(F(1, 7), F(-1, 9)))
    result = _region(unique, bias=F(10**1000), target=F(3, 5))
    assert result.diameter_bound == result.max_squared_radius == 0
    assert result.spatial_variance_upper_bound == result.variance_upper_bound == 0
    assert result.spatial_contraction_bound == result.contraction_bound == 0
    assert result.forcing_bound == result.expansion_radius == 1
    assert result.displacement_squared_bound == 1
    assert result.strong_convexity_bound == F(3, 5)


def test_q_four_spatial_tilts_are_sharp_and_changed_minimum_can_leave_old_box():
    mp = pytest.importorskip("mpmath")
    unique = _unique(centers=((-2,), (2,)), biases=(0, -12), viscosity=F(1, 2), reference=(-2,))
    result = _region(unique, bias=F(1, 2), target=F(1, 10))
    assert unique.domain.model.contraction_bound == 4
    assert result.spatial_oscillation == F(2, 5)
    assert result.total_oscillation == F(9, 10)
    assert result.uniqueness_margin > 0
    with mp.workdps(100):
        spatial = 16/(1+mp.exp(mp.mpf(18)/5))
        combined = 16/(1+mp.exp(mp.mpf(31)/10))
        assert 0 <= _mp_exact(mp, result.spatial_variance_upper_bound)-spatial < mp.mpf("1e-11")
        assert 0 <= _mp_exact(mp, result.variance_upper_bound)-combined < mp.mpf("1e-11")
        base = mp.findroot(lambda x: x-2*mp.tanh(2*x-6), (-2, mp.mpf("-1.9")))
        changed = mp.findroot(lambda x: x+mp.mpf(1)/10-2*mp.tanh(2*x-mp.mpf(23)/4),
                              (mp.mpf("-2.1"), -2))
        assert changed < _mp_exact(mp, unique.domain.physical_bounds[0][0])
        assert _mp_exact(mp, result.expanded_coordinate_bounds[0][0]) <= changed
        assert abs(changed-base) <= mp.mpf(result.displacement_bound)
    assert result.contains_parameters((F(-1, 10),), (F(0), F(1, 2)))
    rho_base = unique.variance_upper_bound
    assert result.displacement_squared_bound > (result.forcing_bound/(1-rho_base))**2
    assert result.displacement_squared_bound < (result.forcing_bound/(1-result.contraction_bound))**2


def test_joint_region_allows_constraint_switch_and_bounds_projection_input():
    mp = pytest.importorskip("mpmath")
    unique = _unique(target=(F(-1, 2),), rows=((1,),), bounds=(0,))
    result = _region(unique, bias=F(1, 2), target=F(1, 4))
    with mp.workdps(100):
        base = mp.findroot(lambda x: x+mp.mpf(1)/2-1/(1+mp.exp(mp.mpf(1)/4-x/2)), (-1, 0))
        # The target increases by1/4 and the second bias by1/2: x=0 is
        # now the constrained optimum and its projection input is positive.
        changed_p = 1/(1+mp.exp(-mp.mpf(1)/4))
        changed_input = -mp.mpf(1)/4+changed_p
        assert base < 0 < changed_input
        assert abs(base) <= mp.mpf(result.displacement_bound)
        assert abs(changed_input-base) <= mp.mpf(result.displacement_bound)


def test_large_spatial_oscillation_refuses_when_decay_underflow_loses_strict_margin():
    unique = _unique(biases=(0, -4), viscosity=F(1, 2))
    assert max(unique.squared_radii) == 2*unique.domain.model.viscosity == 1
    assert certify_gaussian_entropy_joint_region(unique, 0, 2049) is None


def test_radius_below_curvature_handles_large_bias_and_moderate_spatial_change():
    result = _region(bias=F(10**1000), target=F(2))
    assert result.variance_upper_bound <= result.max_squared_radius < 2
    assert result.uniqueness_margin > 0
    assert result.tilt_decay.lo == 0 and result.total_variation_bound == 1


def test_record_and_nested_snapshot_are_frozen():
    result = _region()
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    saved = deepcopy(result.certificate)
    with pytest.raises(TypeError):
        GaussianEntropyJointRegion()
    with pytest.raises(FrozenInstanceError):
        result.target_radius = F(0)
    result.unique.certificate["payload"]["type"] = "damaged"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_entropy_joint_region(result.unique, 0, 0) is None
    with pytest.raises(ValueError):
        result.contains_physical((0,))
    with pytest.raises(ValueError):
        result.contains_parameters((0,), (0, 0))


@pytest.mark.parametrize("strict,sticky", [(False, False), (False, True), (True, False), (True, True)])
def test_cached_membership_requires_no_backend_projection_or_seal(strict, sticky, monkeypatch):
    result = _region(_unique(centers=((0, 0), (1, 0))))

    def fail(*args, **kwargs):
        pytest.fail("cached joint-region membership performed new numerical certification")

    for path in ("gaussian_projection_minimum", "gaussian_posterior_domain", "polyhedral_projection"):
        monkeypatch.setattr(import_module("omnibias.core.verified."+path), "_project_exact", fail)
    proof = import_module("omnibias.core.proof.certificate")
    monkeypatch.setattr(proof, "make_certificate", fail)
    for name in ("exp_iv", "tanh_iv", "ln_iv"):
        monkeypatch.setattr(transcend, name, fail)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    assert result.contains_physical((F(1, 2), 0))
    assert not result.contains_physical((0, 1))
    assert result.contains_parameters((F(1, 4), 0), (0, F(1, 2)))
    assert not result.contains_parameters((F(1, 2), 0), (0, 0))
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


@pytest.mark.parametrize("level", ["unique", "domain", "projector"])
def test_nested_damage_refuses_even_zero_region_and_query(level):
    result = _region()
    unique = result.unique
    dependency = unique if level == "unique" else unique.domain if level == "domain" else unique.domain.projector
    dependency.certificate["payload"]["type"] = "damaged"
    assert certify_gaussian_entropy_joint_region(unique, 0, 0) is None
    with pytest.raises(ValueError):
        result.contains_parameters((0,), (0, 0))


@pytest.mark.parametrize("dependency", [False, True])
def test_conditional_proof_refuses_cached_membership_and_factory(dependency):
    result = _region()
    certificate = result.unique.certificate if dependency else result.certificate
    certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(certificate)
    certificate.clear()
    certificate.update(changed)
    if dependency:
        assert certify_gaussian_entropy_joint_region(result.unique, 0, 0) is None
    with pytest.raises(ValueError):
        result.contains_physical((0,))


@pytest.mark.parametrize("bias,target,error", [(True, 0, TypeError), (0, False, TypeError),
    (0.0, 0, TypeError), (0, 0.0, TypeError), ("0", 0, TypeError),
    (F(-1), 0, ValueError), (0, F(-1), ValueError)])
def test_invalid_radii_raise_before_resource_reservations(bias, target, error, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid radii reached preflight"))
    with pytest.raises(error):
        certify_gaussian_entropy_joint_region(unique, bias, target)


def test_invalid_source_type_raises():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_joint_region(object(), 0, 0)


@pytest.mark.parametrize("point,error", [((True,), TypeError), ((0.0,), TypeError), ((), ValueError), ((0, 0), ValueError)])
def test_invalid_physical_points_raise_before_query_preflight(point, error, monkeypatch):
    result = _region()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid point reached preflight"))
    with pytest.raises(error):
        result.contains_physical(point)


@pytest.mark.parametrize("target,bias,error", [((True,), (0, 0), TypeError), ((0,), (0.0, 0), TypeError),
    ((), (0, 0), ValueError), ((0,), (0,), ValueError)])
def test_invalid_parameter_vectors_raise_before_query_preflight(target, bias, error, monkeypatch):
    result = _region()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid parameters reached preflight"))
    with pytest.raises(error):
        result.contains_parameters(target, bias)


@pytest.mark.parametrize("strict", [False, True])
def test_factory_strict_and_sticky_policy_is_preserved(strict, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    _region(unique)
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_entropy_joint_region(unique, 0, 0)
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


def test_factory_missing_backend_refuses_even_zero_region(monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_joint_region(unique, 0, 0)


def test_overlarge_exact_query_metadata_respects_coefficient_budget():
    result = _region()
    assert certify_gaussian_entropy_joint_region(result.unique, 10**1500, 0) is None
    with pytest.raises(ValueError):
        result.contains_physical((10**1500,))
    with pytest.raises(ValueError):
        result.contains_parameters((0,), (10**1500, 10**1500))


@pytest.mark.parametrize("field", ["bias_oscillation", "target_radius", "diameter_bound", "expansion_radius",
    "expanded_coordinate_bounds", "spatial_oscillation", "total_oscillation", "spatial_tilt_decay", "tilt_decay",
    "max_squared_radius", "spatial_variance_upper_bound", "variance_upper_bound", "spatial_contraction_bound",
    "contraction_bound", "uniqueness_margin", "strong_convexity_bound", "total_variation_bound", "forcing_bound",
    "displacement_squared_bound", "displacement_bound"])
def test_all_runtime_region_fields_are_bound_to_cached_proof(field):
    result = _region()
    original = getattr(result, field)
    if field == "expanded_coordinate_bounds":
        changed = ((F(-100), F(100)),)
    elif field in ("spatial_tilt_decay", "tilt_decay"):
        changed = Interval.point(0.0)
    else:
        changed = original+1
    object.__setattr__(result, field, changed)
    assert verify_certificate_digest(result.certificate)
    with pytest.raises(ValueError):
        result.contains_physical((0,))
    with pytest.raises(ValueError):
        result.contains_parameters((0,), (0, 0))


@pytest.mark.parametrize("field", ["variance_upper_bound", "squared_radii", "reference"])
def test_exact_unique_runtime_field_damage_refuses_both_paths(field):
    result = _region()
    unique = result.unique
    changed = F(0) if field == "variance_upper_bound" else (F(0), F(0)) if field == "squared_radii" else (F(100),)
    object.__setattr__(unique, field, changed)
    assert verify_certificate_digest(unique.certificate)
    assert certify_gaussian_entropy_joint_region(unique, 0, 0) is None
    with pytest.raises(ValueError):
        result.contains_parameters((0,), (0, 0))


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_factory_reservation_precedes_source_digest(guard, monkeypatch):
    unique = _unique()
    module = import_module("omnibias.core.verified.gaussian_entropy_joint_region")

    def exhausted(*args):
        raise ValueError("test joint-region resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_unique_intact", lambda *_a: pytest.fail("unreserved unique proof scan"))
    assert certify_gaussian_entropy_joint_region(unique, 0, 0) is None


@pytest.mark.parametrize("guard", ["_entries", "_work"])
@pytest.mark.parametrize("method", ["physical", "parameters"])
def test_query_reservation_precedes_full_cached_proof_scan(guard, method, monkeypatch):
    result = _region()
    module = import_module("omnibias.core.verified.gaussian_entropy_joint_region")

    def exhausted(*args):
        raise ValueError("test joint-region resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_joint_region_intact", lambda *_a: pytest.fail("unreserved cached proof scan"))
    with pytest.raises(ValueError, match="resource"):
        result.contains_physical((0,)) if method == "physical" else result.contains_parameters((0,), (0, 0))


def test_small_positive_exact_radii_keep_nonzero_motion_bounds():
    radius = F(1, 2**200)
    result = _region(bias=radius, target=radius)
    assert result.expansion_radius > 0 and result.spatial_oscillation > 0
    assert result.total_variation_bound > 0 and result.forcing_bound > 0
    assert result.displacement_squared_bound > 0 and result.displacement_bound > 0
    assert result.contains_parameters((radius,), (0, radius))
    assert not result.contains_parameters((2*radius,), (0, 0))


def test_actual_exact_bit_limit_refuses_nontrivial_region_without_partial_proof():
    unique = _unique(budget=RidgeBudget(max_coefficient_bits=128))
    assert _region(unique, bias=F(0), target=F(0)).displacement_bound == 0
    assert certify_gaussian_entropy_joint_region(unique, F(1, 2), F(1, 4)) is None


def test_huge_common_center_and_bias_translation_preserve_joint_bounds():
    original = _unique(centers=((0, 0), (1, 0), (0, 1)), biases=(0, F(1, 3), F(-1, 5)), viscosity=2)
    shift = (F(10**400), F(-10**400))
    translated = _unique(centers=tuple(tuple(a+offset for a, offset in zip(center, shift, strict=True))
                                      for center in original.domain.model.centers),
                         biases=tuple(bias+10**1000 for bias in original.domain.model.biases), viscosity=2)
    first, second = _region(original), _region(translated)
    for field in ("diameter_bound", "expansion_radius", "spatial_oscillation", "total_oscillation",
                  "spatial_variance_upper_bound", "variance_upper_bound", "contraction_bound",
                  "strong_convexity_bound", "forcing_bound", "displacement_squared_bound", "displacement_bound"):
        assert getattr(first, field) == getattr(second, field)
    for old, new, offset in zip(first.expanded_coordinate_bounds, second.expanded_coordinate_bounds, shift, strict=True):
        assert (old[0]+offset, old[1]+offset) == new
    assert second.contains_physical(shift)


def test_factory_does_not_resolve_original_posterior_projection_or_optimizer(monkeypatch):
    unique = _unique()

    def fail(*args, **kwargs):
        pytest.fail("joint-region construction performed a new posterior or optimizer solve")

    for path in ("gaussian_projection_minimum", "gaussian_posterior_domain", "polyhedral_projection"):
        monkeypatch.setattr(import_module("omnibias.core.verified."+path), "_project_exact", fail)
    monkeypatch.setattr(type(unique.domain.projector), "project", fail)
    domain = import_module("omnibias.core.verified.gaussian_posterior_domain")
    monkeypatch.setattr(domain, "_probabilities", fail)
    linear = import_module("omnibias.core._rational_linear")
    monkeypatch.setattr(linear, "factor_positive_definite", fail)
    monkeypatch.setattr(linear, "solve_affine", fail)
    assert _region(unique).uniqueness_margin > 0
