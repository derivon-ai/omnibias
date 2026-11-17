# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Original-posterior coverage, exact gauges and boundary-safe metadata."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from importlib import import_module
from itertools import product

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.core.ridge_system import RidgeBudget
from omnibias.core.verified import transcend
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import (
    GaussianPosteriorDomain,
    _domain_intact,
    _domain_source_slots,
    certify_gaussian_posterior_domain,
)
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _domain(centers=((-1,), (1,)), biases=(0, 0), *, rows=(), bounds=(), target=None, viscosity=1, time=1):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, biases, viscosity=viscosity, time=time)
    projector = compile_polyhedral_projector(model.dimension, rows, bounds)
    assert projector is not None
    result = certify_gaussian_posterior_domain(model, projector, (0,)*model.dimension if target is None else target)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _sticks(mp, model, point):
    logits = [_mp(mp, bias-model.biases[0])-sum((_mp(mp, x)-_mp(mp, a))**2 for x, a in zip(point, center, strict=True))
              /(4*_mp(mp, model.viscosity)*_mp(mp, model.time)) for bias, center in zip(model.biases, model.centers, strict=True)]
    result = []
    for j in range(model.mode_count-1):
        peak = max(logits[j:])
        weights = [mp.exp(value-peak) for value in logits[j:]]
        result.append(weights[0]/sum(weights))
    return tuple(result)


def test_symmetric_two_mode_box_contains_exact_sigmoid_extremes():
    mp = pytest.importorskip("mpmath")
    domain = _domain()
    assert domain.physical_bounds == ((F(-1), F(1)),) and domain.gradient_safe
    with mp.workdps(90):
        for point in ((F(-1),), (F(0),), (F(1),)):
            truth = _sticks(mp, domain.model, point)[0]
            interval = domain.parameter_box[0]
            assert _mp(mp, interval.lo) <= truth <= _mp(mp, interval.hi)


@pytest.mark.parametrize("constrained", [False, True])
def test_noncollinear_original_conditionals_enclosed_on_whole_physical_box(constrained):
    mp = pytest.importorskip("mpmath")
    rows, bounds = (((-1, 0), (0, -1)), (0, 0)) if constrained else ((), ())
    domain = _domain(((-1, 0), (1, 1), (0, -2), (F(1, 3), F(2, 5))), (0, F(1, 4), F(-1, 3), 0),
                     rows=rows, bounds=bounds, target=(F(1, 5), F(-1, 7)), viscosity=F(3, 4), time=F(5, 4))
    grids = tuple((lo, (lo+hi)/2, hi) for lo, hi in domain.physical_bounds)
    with mp.workdps(100):
        for point in product(*grids):
            truth = _sticks(mp, domain.model, point)
            assert all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
                       for interval, value in zip(domain.parameter_box, truth, strict=True))
    assert len(domain.parameter_box) == 3
    assert domain.gradient_safe == all(0 < interval.lo <= interval.hi < 1 for interval in domain.parameter_box)


def test_exact_equality_projection_gives_singleton_physical_bound():
    domain = _domain(((-2,), (0,), (3,)), (0, 1, -1), rows=((1,), (-1,)), bounds=(F(1, 3), F(-1, 3)))
    assert F(domain.physical_bounds[0][0]) <= F(1, 3) <= F(domain.physical_bounds[0][1])
    # Nonexpansiveness gives a conservative box, not an exact-image oracle.
    assert domain.gradient_safe


def test_projected_hull_image_includes_projection_of_interior_not_only_vertices():
    domain = _domain(((-1, 1), (1, -1)), (0, 0), rows=((-1, 0), (0, -1)), bounds=(0, 0))
    assert all(lo <= 0 <= hi for lo, hi in domain.physical_bounds)
    assert domain.parameter_box[0].lo <= .5 <= domain.parameter_box[0].hi


@pytest.mark.parametrize("modes", [1, 3])
def test_coincident_modes_have_constant_posterior_and_exact_projected_image(modes):
    mp = pytest.importorskip("mpmath")
    domain = _domain(((2, -1),)*modes, tuple(range(modes)), rows=((-1, 0), (0, -1)), bounds=(0, 0), target=(0, 0))
    assert domain.physical_bounds == ((F(2), F(2)), (F(0), F(0)))
    assert domain.gradient_safe
    if modes == 1:
        assert len(domain.parameter_box) == 1 and domain.parameter_box[0].lo == domain.parameter_box[0].hi == 0
    else:
        with mp.workdps(100):
            values = _sticks(mp, domain.model, (2, 0))
            assert all(_mp(mp, interval.lo) <= value <= _mp(mp, interval.hi) for interval, value in zip(domain.parameter_box, values, strict=True))


@pytest.mark.parametrize("biases", [(0, -10000), (-10000, 0)])
def test_underflow_retains_valid_closed_bounds_and_disables_gradient(biases):
    mp = pytest.importorskip("mpmath")
    domain = _domain(biases=biases)
    assert not domain.gradient_safe
    assert domain.parameter_box[0].lo == 0 or domain.parameter_box[0].hi == 1
    with mp.workdps(100):
        for point in ((-1,), (0,), (1,)):
            probability = _sticks(mp, domain.model, point)[0]
            assert _mp(mp, domain.parameter_box[0].lo) <= probability <= _mp(mp, domain.parameter_box[0].hi)


def test_huge_common_bias_and_common_translation_cancel_before_conversion():
    first = _domain(((-1,), (0,), (2,)), (0, F(1, 3), F(-1, 5)))
    shift, gauge = 10**400, 10**400
    second = _domain(((shift-1,), (shift,), (shift+2,)), (gauge, gauge+F(1, 3), gauge-F(1, 5)))
    assert first.parameter_box == second.parameter_box and first.gradient_safe == second.gradient_safe
    assert tuple((lo+shift, hi+shift) for lo, hi in first.physical_bounds) == second.physical_bounds


def test_pairwise_contrasts_remove_huge_shared_logit_terms():
    mp = pytest.importorskip("mpmath")
    # Modes 1 and 2 have a huge common bias relative to mode0, but z1 uses
    # their small DIFFERENCE; it must not round their conditional to one half.
    domain = _domain(((0,), (0,), (0,)), (0, 10**100, 10**100+1))
    with mp.workdps(100):
        exact = 1/(1+mp.e)
        assert _mp(mp, domain.parameter_box[1].lo) <= exact <= _mp(mp, domain.parameter_box[1].hi)
    assert domain.parameter_box[1].hi < .3
    assert not domain.gradient_safe  # First stick probability underflows.


def test_nested_certificate_source_snapshot_and_immutable_metadata():
    domain = _domain()
    assert _domain_intact(domain) and verify_certificate_digest(domain.certificate)
    assert _domain_source_slots(domain) > 0
    domain.projector.certificate["payload"]["dimension"] = 4
    assert verify_certificate_digest(domain.certificate) and not _domain_intact(domain)
    with pytest.raises(FrozenInstanceError):
        domain.gradient_safe = False
    with pytest.raises(TypeError):
        GaussianPosteriorDomain()


@pytest.mark.parametrize("field", ["target", "centers", "biases", "parameter_box", "physical_bounds", "gradient_safe"])
def test_resealed_snapshot_mismatch_refuses_integrity(field):
    domain = deepcopy(_domain())
    payload = {**domain.certificate["payload"], field: []}
    domain.certificate.update(seal_certificate({**domain.certificate, "payload": payload}))
    assert not _domain_intact(domain)


def test_damaged_or_infeasible_projector_returns_none_without_posterior_calls(monkeypatch):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    damaged = compile_polyhedral_projector(1, (), ())
    infeasible = compile_polyhedral_projector(1, ((1,), (-1,)), (0, -1))
    assert damaged is not None and infeasible is not None
    damaged.certificate["payload"]["dimension"] = 5
    module = import_module("omnibias.core.verified.gaussian_posterior_domain")
    monkeypatch.setattr(module, "_probabilities", lambda *_: pytest.fail("invalid source reached probabilities"))
    assert certify_gaussian_posterior_domain(model, damaged, (0,)) is None
    assert certify_gaussian_posterior_domain(model, infeasible, (0,)) is None


@pytest.mark.parametrize("limited", ["model", "projector"])
def test_combined_budget_preflight_precedes_source_hash(limited, monkeypatch):
    limits = RidgeBudget(max_matrix_entries=10_000)
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), budget=limits if limited == "model" else None)
    projector = compile_polyhedral_projector(1, (), (), budget=limits if limited == "projector" else None)
    assert projector is not None
    module = import_module("omnibias.core.verified.gaussian_posterior_domain")
    monkeypatch.setattr(module, "_intact", lambda *_: pytest.fail("unreserved domain source scan"))
    assert certify_gaussian_posterior_domain(model, projector, (0,)) is None


@pytest.mark.parametrize("strict", [False, True])
def test_strict_provenance_restored_and_sticky_fallback_refused(strict, monkeypatch):
    source = _domain()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
    assert certify_gaussian_posterior_domain(source.model, source.projector, source.target) is not None
    assert transcend.strict_backend() is strict
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="libm"):
        certify_gaussian_posterior_domain(source.model, source.projector, source.target)
    assert transcend.libm_fallback_used() and transcend.strict_backend() is strict


def test_missing_backend_and_nonfinite_contrast_refuse_explicitly(monkeypatch):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    projector = compile_polyhedral_projector(1, (), ())
    assert projector is not None
    with monkeypatch.context() as patch:
        patch.setattr(transcend, "_mpmath", lambda: None)
        patch.setattr(transcend, "_LIBM_FALLBACK_USED", False)
        with pytest.raises(RuntimeError, match="mpmath"):
            certify_gaussian_posterior_domain(model, projector, (0,))
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((-1,), (1,)), (0, 10**400))
    assert certify_gaussian_posterior_domain(model, projector, (0,)) is None


def test_posterior_arithmetic_failure_restores_strict_state(monkeypatch):
    source = _domain()
    module = import_module("omnibias.core.verified.gaussian_posterior_domain")

    def fail(*_args):
        raise ArithmeticError("simulated nonfinite posterior arithmetic")

    monkeypatch.setattr(module, "_probabilities", fail)
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", False)
    assert certify_gaussian_posterior_domain(source.model, source.projector, source.target) is None
    assert not transcend.strict_backend()


def test_exact_query_bit_failure_returns_no_partial_domain(monkeypatch):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0), budget=RidgeBudget(max_coefficient_bits=16))
    projector = compile_polyhedral_projector(1, (), ())
    assert projector is not None
    module = import_module("omnibias.core.verified.gaussian_posterior_domain")
    monkeypatch.setattr(module, "_probabilities", lambda *_: pytest.fail("invalid exact query reached posterior"))
    assert certify_gaussian_posterior_domain(model, projector, (2**30,)) is None


@pytest.mark.parametrize("target,error", [((True,), TypeError), ((1.0,), TypeError), ((), ValueError), ((0, 1), ValueError)])
def test_invalid_target_raises(target, error):
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    projector = compile_polyhedral_projector(1, (), ())
    with pytest.raises(error):
        certify_gaussian_posterior_domain(model, projector, target)


def test_invalid_source_types_and_dimension_raise():
    model = GaussianHeatMap(((-1,), (1,)), (0, 0))
    projector = compile_polyhedral_projector(2, (), ())
    with pytest.raises(TypeError):
        certify_gaussian_posterior_domain(object(), projector, (0, 0))
    with pytest.raises(TypeError):
        certify_gaussian_posterior_domain(model, object(), (0,))
    with pytest.raises(ValueError):
        certify_gaussian_posterior_domain(model, projector, (0,))
