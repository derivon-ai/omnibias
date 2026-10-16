# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact posterior-radius uniqueness bounds and cached-proof contracts."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from itertools import product

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
from omnibias.core.verified.gaussian_entropy_unique import (
    GaussianEntropyUnique,
    certify_gaussian_entropy_unique,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _domain(*, centers=((0,), (1,)), biases=(0, 0), viscosity=1, time=1,
            target=None, rows=(), bounds=(), budget=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time, budget=budget)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds, budget=budget)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, (0,)*model.dimension if target is None else target)
    assert domain is not None
    return domain


def _unique(domain=None, **kwargs):
    result = certify_gaussian_entropy_unique(_domain() if domain is None else domain, **kwargs)
    assert result is not None
    return result


def _corner_expectations(domain, radii):
    """Independent full simplex weights at every box corner, no tail recursion."""
    values = []
    for corner in product(*((F(iv.lo), F(iv.hi)) for iv in domain.parameter_box)):
        weights, remainder = [], F(1)
        for coordinate in corner[:domain.model.mode_count-1]:
            weights.append(coordinate*remainder)
            remainder *= 1-coordinate
        weights.append(remainder)
        values.append(sum(p*r for p, r in zip(weights, radii, strict=True)))
    return min(values), max(values)


@pytest.mark.parametrize("reference", [(0, 0), (F(1, 3), F(-1, 5)), (1, 1)])
def test_exact_expectation_extrema_match_independent_corner_enumeration(reference):
    domain = _domain(centers=((0, 0), (1, 0), (0, 1), (1, 1)), biases=(0, F(-1, 3), F(1, 7), 0),
                     viscosity=3, time=F(5, 4), target=(F(1, 8), F(-1, 9)))
    result = _unique(domain, reference=reference)
    radii = tuple(sum((a-c)**2 for a, c in zip(center, reference, strict=True)) for center in domain.model.centers)
    expected = _corner_expectations(domain, radii)
    assert result.domain is domain and result.reference == tuple(map(F, reference))
    assert result.squared_radii == radii
    assert result.expectation_bounds == expected
    assert result.variance_upper_bound == expected[1]
    assert result.uniqueness_margin == 2*domain.model.viscosity*domain.model.time-expected[1] > 0
    assert result.entropy_monotonicity_lower_bound == result.uniqueness_margin/domain.model.time > 0


def test_strict_threshold_refuses_equality_even_for_known_unique_critical_model():
    domain = _domain(centers=((-1,), (1,)), viscosity=F(1, 2))
    assert _corner_expectations(domain, (F(1), F(1))) == (1, 1)
    assert certify_gaussian_entropy_unique(domain, reference=(0,)) is None


def test_arbitrary_reference_can_improve_a_conservative_first_center_bound():
    domain = _domain(centers=((-1,), (1,)), viscosity=1)
    assert certify_gaussian_entropy_unique(domain) is None
    result = _unique(domain, reference=(0,))
    assert result.expectation_bounds == (1, 1)
    assert result.variance_upper_bound == 1
    assert result.uniqueness_margin == result.entropy_monotonicity_lower_bound == 1


def test_weak_or_negative_margin_is_inconclusive_not_nonuniqueness():
    domain = _domain(centers=((-1,), (1,)), viscosity=F(1, 4))
    assert certify_gaussian_entropy_unique(domain, reference=(0,)) is None
    assert certify_gaussian_entropy_unique(_domain(), reference=(100,)) is None


def test_boundary_inclusive_posterior_can_prove_uniqueness_beyond_global_q_bound():
    domain = _domain(centers=((0,), (10,)), biases=(0, -1000))
    assert domain.model.contraction_bound > 1 and not domain.gradient_safe
    result = _unique(domain)
    assert result.variance_upper_bound < F(1, 1000)
    assert result.uniqueness_margin > 1


def test_polyhedral_geometry_can_tighten_the_posterior_uniqueness_bound():
    domain = _domain(centers=((0,), (10,)), rows=((-1,),), bounds=(-20,))
    assert domain.model.contraction_bound > 1
    result = _unique(domain, reference=(10,))
    assert result.variance_upper_bound < 1 and result.uniqueness_margin > 1


@pytest.mark.parametrize("count", [1, 4])
def test_one_mode_and_coincident_modes_have_exact_zero_variance_bound(count):
    domain = _domain(centers=((F(1, 3), F(-2, 5)),)*count, biases=tuple(range(count)),
                     viscosity=F(3, 4), time=F(5, 3), rows=((1, 0), (-1, 0)), bounds=(0, 0))
    result = _unique(domain)
    assert result.reference == domain.model.centers[0]
    assert result.squared_radii == (F(0),)*count
    assert result.expectation_bounds == (F(0), F(0)) and result.variance_upper_bound == 0
    assert result.uniqueness_margin == F(5, 2)
    assert result.entropy_monotonicity_lower_bound == F(3, 2)


def test_huge_common_center_and_bias_cancel_before_squared_radius_arithmetic():
    original = _domain(centers=((0, 0), (1, 0), (0, 1)), biases=(0, 0, 0), viscosity=2)
    shift = (F(10**400), F(-10**400))
    shifted_centers = tuple(tuple(value+offset for value, offset in zip(center, shift, strict=True))
                            for center in original.model.centers)
    shifted = _domain(centers=shifted_centers, biases=(10**1000,)*3, viscosity=2)
    reference = (F(1, 3), F(1, 5))
    first = _unique(original, reference=reference)
    second = _unique(shifted, reference=tuple(value+offset for value, offset in zip(reference, shift, strict=True)))
    for name in ("squared_radii", "expectation_bounds", "variance_upper_bound", "uniqueness_margin",
                 "entropy_monotonicity_lower_bound"):
        assert getattr(first, name) == getattr(second, name)


def test_reference_input_is_snapshotted():
    reference = [0]
    result = _unique(reference=reference)
    reference[0] = 100
    reference.append(200)
    assert result.reference == (F(0),)
    assert verify_certificate_digest(result.certificate)


def test_exact_proof_record_is_frozen_and_snapshots_dependency():
    result = _unique()
    saved = deepcopy(result.certificate)
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    with pytest.raises(TypeError):
        GaussianEntropyUnique()
    with pytest.raises(FrozenInstanceError):
        result.uniqueness_margin = F(0)
    result.domain.certificate["payload"]["type"] = "damaged"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_entropy_unique(result.domain) is None


@pytest.mark.parametrize("dependency", ["domain", "projector", "snapshot"])
def test_damaged_or_broken_nested_dependency_refuses(dependency):
    domain = deepcopy(_domain())
    if dependency == "domain":
        domain.certificate["payload"]["type"] = "damaged"
    elif dependency == "projector":
        domain.projector.certificate["payload"]["type"] = "damaged"
    else:
        domain.certificate["payload"]["projector_certificate"]["payload"]["type"] = "damaged"
        changed = seal_certificate(domain.certificate)
        domain.certificate.clear()
        domain.certificate.update(changed)
        assert verify_certificate_digest(domain.certificate)
    assert certify_gaussian_entropy_unique(domain) is None


def test_conditional_source_cannot_be_promoted_to_unconditional_uniqueness():
    domain = deepcopy(_domain())
    domain.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(domain.certificate)
    domain.certificate.clear()
    domain.certificate.update(changed)
    assert not certificate_is_unconditional(domain.certificate)
    assert certify_gaussian_entropy_unique(domain) is None


@pytest.mark.parametrize("strict,sticky", [(False, False), (False, True), (True, False)])
def test_exact_postprocessing_runs_no_projection_transcendental_or_minimum_query(strict, sticky, monkeypatch):
    domain = _domain()

    def fail(*args, **kwargs):
        pytest.fail("exact uniqueness postprocessing performed a numerical query")

    for path in ("gaussian_entropy_minimum", "gaussian_projection_minimum"):
        module = import_module("omnibias.core.verified."+path)
        monkeypatch.setattr(module, "minimize_box", fail)
        monkeypatch.setattr(module, "_project_exact", fail)
    projection = import_module("omnibias.core.verified.polyhedral_projection")
    monkeypatch.setattr(projection, "_project_exact", fail)
    monkeypatch.setattr(type(domain.projector), "project", fail)
    for name in ("exp_iv", "ln_iv", "tanh_iv"):
        monkeypatch.setattr(transcend, name, fail)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", sticky)
    result = _unique(domain)
    assert result.certificate["meta"]["transcend_backend"] == domain.certificate["meta"]["transcend_backend"]
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used() is sticky


def test_missing_backend_preserves_cached_provenance_and_strict_sealing_policy(monkeypatch):
    domain = _domain()
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    assert _unique(domain).certificate["meta"]["transcend_backend"] == "mpmath"
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_unique(domain)


def test_strict_sticky_history_is_not_cleared_by_exact_postprocessing(monkeypatch):
    domain = _domain()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", True)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_entropy_unique(domain)
    assert transcend.strict_backend() and transcend.libm_fallback_used()


@pytest.mark.parametrize("reference,error", [((True,), TypeError), ((0.0,), TypeError), (("0",), TypeError),
                                            ((), ValueError), ((0, 0), ValueError), ("0", TypeError)])
def test_invalid_reference_raises_before_resource_preflight(reference, error, monkeypatch):
    domain = _domain()
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid reference reached resource preflight"))
    with pytest.raises(error):
        certify_gaussian_entropy_unique(domain, reference=reference)


@pytest.mark.parametrize("value", [None, (), {}, 1, True])
def test_invalid_record_raises(value, monkeypatch):
    monkeypatch.setattr(_Arithmetic, "_entries", lambda *_a: pytest.fail("invalid record reached resource preflight"))
    with pytest.raises(TypeError):
        certify_gaussian_entropy_unique(value)


@pytest.mark.parametrize("guard", ["_entries", "_work"])
def test_aggregate_resource_preflight_precedes_dependency_scan(guard, monkeypatch):
    domain = _domain()
    module = import_module("omnibias.core.verified.gaussian_entropy_unique")

    def exhausted(*args):
        raise ValueError("test uniqueness resource exhaustion")

    monkeypatch.setattr(RidgeBudget, guard, exhausted)
    monkeypatch.setattr(module, "_domain_intact", lambda *_a: pytest.fail("unreserved certificate scan"))
    assert certify_gaussian_entropy_unique(domain) is None


def test_oversized_exact_reference_refuses_as_resource_failure():
    assert certify_gaussian_entropy_unique(_domain(), reference=(10**1500,)) is None


def test_valid_source_can_refuse_on_exact_expectation_bit_growth():
    domain = _domain(centers=((0,), (1,), (2,)), biases=(0,)*3, viscosity=3,
                     budget=RidgeBudget(max_coefficient_bits=64))
    # The mathematical bound is at most4<6; refusal comes from exact products
    # of the stored binary64 coordinate endpoints exceeding the bit budget.
    assert domain.model.centers[-1][0]**2 < 2*domain.model.viscosity*domain.model.time
    assert certify_gaussian_entropy_unique(domain) is None


@pytest.mark.parametrize("limited_source", ["model", "projector"])
def test_both_source_work_budgets_apply_to_exact_postprocessing(limited_source):
    limits = RidgeBudget(max_elimination_work=100_000)
    model = GaussianHeatMap(((0,), (1,)), (0, 0), budget=limits if limited_source == "model" else None)
    projector = compile_polyhedral_projector(1, (), (), budget=limits if limited_source == "projector" else None)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, (0,))
    assert domain is not None
    assert certify_gaussian_entropy_unique(domain) is None
