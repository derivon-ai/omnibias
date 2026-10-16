# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent residual, objective and projection contracts for global bounds."""

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
from omnibias.core.verified.gaussian_entropy_residual import (
    GaussianEntropyResidual,
    _residual_source_slots,
    _residual_work_bound,
    _ResidualOracle,
    certify_gaussian_entropy_residual,
)
from omnibias.core.verified.gaussian_entropy_unique import certify_gaussian_entropy_unique
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_posterior_domain import certify_gaussian_posterior_domain
from omnibias.core.verified.mean_map import _strict
from omnibias.core.verified.polyhedral_projection import compile_polyhedral_projector


def _unique(*, centers=((-1,), (1,)), biases=None, viscosity=1, time=1,
            target=None, rows=(), upper=(), reference=None):
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(centers, (0,)*len(centers) if biases is None else biases,
                            viscosity=viscosity, time=time)
    projector = compile_polyhedral_projector(model.dimension, rows, upper)
    assert projector is not None
    domain = certify_gaussian_posterior_domain(model, projector, (0,)*model.dimension if target is None else target)
    assert domain is not None
    result = certify_gaussian_entropy_unique(domain, reference=(0,)*model.dimension if reference is None else reference)
    assert result is not None
    return result


def _mp(mp, value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def _original(mp, unique, point):
    """Original squared-distance logits, with the declared common constant."""
    model = unique.domain.model
    logits = tuple(_mp(mp, b)-sum((x-_mp(mp, a))**2 for x, a in zip(point, row, strict=True))
                   /(4*_mp(mp, model.viscosity)*_mp(mp, model.time))
                   for b, row in zip(model.biases, model.centers, strict=True))
    shift = max(logits)
    weights = tuple(mp.exp(value-shift) for value in logits)
    total = sum(weights)
    posterior = tuple(value/total for value in weights)
    value = -2*_mp(mp, model.viscosity)*(shift+mp.log(total))
    value -= sum(_mp(mp, v)*x for v, x in zip(unique.domain.target, point, strict=True))
    value += 2*_mp(mp, model.viscosity)*_mp(mp, model.biases[0])
    value += sum(_mp(mp, v*a) for v, a in zip(unique.domain.target, model.centers[0], strict=True))
    return value, posterior


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


@pytest.mark.parametrize("weights", [(F(1), F(0)), (F(1, 2), F(1, 2)), (F(1, 7), F(6, 7))])
def test_independent_scalar_root_value_posterior_and_spatial_error(weights):
    mp = pytest.importorskip("mpmath")
    unique = _unique(biases=(0, F(1, 3)), viscosity=F(3, 4), target=(F(1, 5),))
    result = certify_gaussian_entropy_residual(unique, weights)
    assert result is not None
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-mp.tanh(x/mp.mpf("1.5")+mp.mpf(1)/6)-mp.mpf(1)/5, 1)
        point = _mp(mp, result.point[0])
        value, posterior = _original(mp, unique, (point,))
        optimum, _ = _original(mp, unique, (root,))
        residual = sum((2*j-1)*(_mp(mp, weights[j])-posterior[j]) for j in range(2))
        _contains(mp, result.residual[0], residual)
        for interval, probability in zip(result.posterior, posterior, strict=True):
            _contains(mp, interval, probability)
        _contains(mp, result.value, value)
        _contains(mp, result.minimum, optimum)
        _contains(mp, result.enclosure[0], root)
        assert abs(point-root) <= _mp(mp, result.error_bound)
        assert 0 <= value-optimum <= _mp(mp, result.suboptimality_bound)
        assert residual**2 <= _mp(mp, result.residual_squared_bound)
    assert result.suboptimality_bound == result.residual_squared_bound/(2*result.strong_convexity_bound)
    assert result.contraction_bound == unique.variance_upper_bound/(2*unique.domain.model.viscosity)


def test_noncollinear_nonconvex_original_family_and_weights_outside_posterior_domain():
    mp = pytest.importorskip("mpmath")
    unique = _unique(centers=((0, 0), (4, 0), (0, 4)), biases=(0, -8, -8))
    assert unique.domain.model.contraction_bound == 4
    weights = (F(0), F(1, 3), F(2, 3))
    assert unique.domain.parameter_box[0].lo > 0
    result = certify_gaussian_entropy_residual(unique, weights)
    assert result is not None and result.point == (F(4, 3), F(8, 3))
    with mp.workdps(100):
        root = mp.findroot(lambda x: x-4*mp.exp(-12+2*x)/(1+2*mp.exp(-12+2*x)), mp.mpf("0.000025"))
        optimum, _ = _original(mp, unique, (root, root))
        point = tuple(_mp(mp, v) for v in result.point)
        value, posterior = _original(mp, unique, point)
        negative_curvature = mp.diff(lambda x: _original(mp, unique, (x, mp.mpf(0)))[0], mp.mpf(6), 2)
        assert negative_curvature < 0
        _contains(mp, result.minimum, optimum)
        _contains(mp, result.value, value)
        assert sum((x-root)**2 for x in point) <= _mp(mp, result.error_bound)**2
        assert value-optimum <= _mp(mp, result.suboptimality_bound)
        for axis in result.enclosure:
            _contains(mp, axis, root)
        for axis, centers in zip(result.residual, zip(*unique.domain.model.centers, strict=True), strict=True):
            residual = sum(_mp(mp, a)*(_mp(mp, p)-pi) for a, p, pi in zip(centers, weights, posterior, strict=True))
            _contains(mp, axis, residual)


def test_constraint_normal_cancels_nonzero_original_gradient_exactly():
    unique = _unique(centers=((0, 0),), target=(-2, 3), rows=((-1, 0), (0, 1)), upper=(0, 1))
    result = certify_gaussian_entropy_residual(unique, (1,))
    assert result is not None
    assert result.projection_source == (-2, 3) and result.point == (0, 1)
    assert result.normal == (-2, 2)
    assert result.residual_squared_bound == result.error_bound == result.suboptimality_bound == 0
    assert all(axis.lo == axis.hi == 0 for axis in result.residual)
    assert tuple((axis.lo, axis.hi) for axis in result.enclosure) == ((0, 0), (1, 1))
    # The original gradient is (2,-2), while the normal is (-2,2).
    assert result.normal != (0, 0)
    assert result.minimum == result.value


@pytest.mark.parametrize("weights", [(0, 1), (F(1, 2), F(1, 2)), (1, 0)])
def test_polyhedral_boundary_root_is_enclosed_for_every_simplex_proposal(weights):
    unique = _unique(centers=((0,), (1,)), target=(-2,), rows=((-1,),), upper=(0,))
    result = certify_gaussian_entropy_residual(unique, weights)
    assert result is not None and result.point == (0,)
    assert result.normal == (-2+weights[1],)
    assert result.enclosure[0].lo <= 0 <= result.enclosure[0].hi
    assert result.minimum.lo <= result.value.lo <= result.minimum.hi


def test_equal_logit_groups_preserve_exact_symmetric_residual():
    unique = _unique(centers=((0, 0), (2, 0), (0, 2), (2, 2)), viscosity=2, reference=(1, 1))
    result = certify_gaussian_entropy_residual(unique, (F(1, 4),)*4)
    assert result is not None and result.point == (1, 1)
    assert result.residual_squared_bound == result.error_bound == 0
    assert all(axis.lo == axis.hi == .25 for axis in result.posterior)
    assert result.enclosure[0].lo == result.enclosure[0].hi == 1


def test_equal_logit_subgroups_preserve_cross_group_symmetry():
    unique = _unique(centers=((-1, 0), (1, 0), (0, -1), (0, 1)), biases=(0, 0, 1, 1))
    result = certify_gaussian_entropy_residual(unique, (F(1, 4),)*4)
    assert result is not None and result.point == (0, 0)
    assert result.posterior[0].hi < result.posterior[2].lo
    assert result.residual_squared_bound == result.error_bound == 0


@pytest.mark.parametrize("weights", [(1, 0, 0), (F(1, 2), F(1, 3), F(1, 6))])
def test_coincident_centers_underflow_and_nonbinary_exact_projection(weights):
    center = (F(1, 3), F(-2, 7))
    unique = _unique(centers=(center,)*3, biases=(0, -1000, -10000), reference=center,
                     time=F(5, 3), target=(F(2, 7), F(-1, 5)))
    assert not unique.domain.gradient_safe
    result = certify_gaussian_entropy_residual(unique, weights)
    assert result is not None and result.contraction_bound == 0
    assert result.point == (F(17, 21), F(-13, 21))
    assert result.residual_squared_bound == result.error_bound == result.suboptimality_bound == 0
    for coordinate, interval in zip(result.point, result.enclosure, strict=True):
        assert F(interval.lo) <= coordinate <= F(interval.hi)
    assert result.posterior[-1].lo == 0


def test_underflowed_positive_mode_keeps_an_outward_nonzero_residual_bound():
    unique = _unique(centers=((0,), (10,)), biases=(0, -1000))
    result = certify_gaussian_entropy_residual(unique, (1, 0))
    assert result is not None and result.posterior[1].lo == 0
    assert result.residual[0].lo < 0 <= result.residual[0].hi
    assert result.residual_squared_bound > 0 and result.error_bound > 0


def test_common_bias_and_translation_cancel_before_binary_conversion():
    unique = _unique(centers=((0, 0), (1, 0), (0, 1)), viscosity=2, biases=(0, -1, 1))
    translation = (10**100, -10**100)
    shifted = _unique(centers=tuple(tuple(a+b for a, b in zip(row, translation, strict=True))
                                     for row in unique.domain.model.centers),
                      viscosity=2, biases=(10**1000, 10**1000-1, 10**1000+1), reference=translation)
    weights = (F(1, 7), F(2, 7), F(4, 7))
    first, second = (certify_gaussian_entropy_residual(item, weights) for item in (unique, shifted))
    assert first is not None and second is not None
    assert second.point == tuple(a+b for a, b in zip(first.point, translation, strict=True))
    for name in ("posterior", "residual", "residual_squared_bound", "contraction_bound", "strong_convexity_bound",
                 "error_bound", "value", "minimum", "suboptimality_bound"):
        assert getattr(first, name) == getattr(second, name)


def test_record_is_frozen_and_snapshots_unique_dependency_and_weights():
    unique = _unique()
    weights = [F(1, 3), F(2, 3)]
    result = certify_gaussian_entropy_residual(unique, weights)
    assert result is not None
    assert verify_certificate_digest(result.certificate) and certificate_is_unconditional(result.certificate)
    assert schema_errors_v1(result.certificate) == []
    assert "theorem_prover_verified" not in result.certificate["honesty"]
    saved = deepcopy(result.certificate)
    weights[0] = 0
    assert result.probabilities == (F(1, 3), F(2, 3))
    with pytest.raises(TypeError):
        GaussianEntropyResidual()
    with pytest.raises(FrozenInstanceError):
        result.error_bound = 0
    unique.certificate["payload"]["type"] = "broken"
    assert result.certificate == saved and verify_certificate_digest(result.certificate)
    assert certify_gaussian_entropy_residual(unique, (1, 0)) is None


@pytest.mark.parametrize("weights,error", [(None, TypeError), ([1], ValueError), ([True, 0], TypeError),
                                          ([.5, .5], TypeError), ([F(-1, 3), F(4, 3)], ValueError),
                                          ([0, 0], ValueError), ([1, 1], ValueError), (["1", 0], TypeError)])
def test_invalid_probability_metadata_raises(weights, error):
    with pytest.raises(error):
        certify_gaussian_entropy_residual(_unique(), weights)


def test_invalid_unique_type_raises_without_a_numerical_backend():
    with pytest.raises(TypeError):
        certify_gaussian_entropy_residual(None, (1,))


@pytest.mark.parametrize("damage", ["unique", "domain", "projector", "snapshot", "payload"])
def test_broken_nested_dependencies_refuse(damage):
    unique = deepcopy(_unique())
    if damage == "domain":
        unique.domain.certificate["payload"]["type"] = "broken"
    elif damage == "projector":
        unique.domain.projector.certificate["payload"]["type"] = "broken"
    elif damage == "snapshot":
        unique.certificate["payload"]["domain_certificate"]["payload"]["type"] = "broken"
        changed = seal_certificate(unique.certificate)
        unique.certificate.clear()
        unique.certificate.update(changed)
    elif damage == "payload":
        unique.certificate["payload"] = []
        changed = seal_certificate(unique.certificate)
        unique.certificate.clear()
        unique.certificate.update(changed)
    else:
        unique.certificate["payload"]["type"] = "broken"
    assert certify_gaussian_entropy_residual(unique, (1, 0)) is None


def test_conditional_source_refuses_even_when_digest_is_valid():
    unique = deepcopy(_unique())
    unique.certificate["meta"]["transcend_backend"] = "libm"
    changed = seal_certificate(unique.certificate)
    unique.certificate.clear()
    unique.certificate.update(changed)
    assert certify_gaussian_entropy_residual(unique, (1, 0)) is None


@pytest.mark.parametrize("strict", [False, True])
def test_prior_libm_history_is_never_cleared(strict, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_LIBM_FALLBACK_USED", True)
    with pytest.raises(RuntimeError, match="history"):
        certify_gaussian_entropy_residual(unique, (1, 0))
    assert transcend.strict_backend() is strict and transcend.libm_fallback_used()


@pytest.mark.parametrize("strict", [False, True])
def test_missing_rigorous_backend_restores_prior_strict_state(strict, monkeypatch):
    unique = _unique()
    monkeypatch.setattr(transcend, "_STRICT_BACKEND", strict)
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    with pytest.raises(RuntimeError, match="mpmath"):
        certify_gaussian_entropy_residual(unique, (1, 0))
    assert transcend.strict_backend() is strict and not transcend.libm_fallback_used()


@pytest.mark.parametrize("kind", ["storage", "work", "coefficient"])
def test_resource_refusal_and_metadata_precedence(kind, monkeypatch):
    unique = _unique()
    limits = {"storage": RidgeBudget(max_matrix_entries=1), "work": RidgeBudget(max_elimination_work=1),
              "coefficient": RidgeBudget(max_coefficient_bits=8)}[kind]
    object.__setattr__(unique.domain.model, "budget", limits)
    module = import_module("omnibias.core.verified.gaussian_entropy_residual")
    calls = []
    original = module._project_exact

    def tracked(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(module, "_project_exact", tracked)
    assert certify_gaussian_entropy_residual(unique, (F(1, 257), F(256, 257))) is None
    assert not calls
    with pytest.raises(TypeError):
        certify_gaussian_entropy_residual(unique, (.5, .5))
    with pytest.raises(ValueError, match="nonnegative"):
        certify_gaussian_entropy_residual(unique, (-1, 2))


def test_shared_kernel_uses_cumulative_meter_and_no_per_query_certificate_copy(monkeypatch):
    unique = _unique()
    limits = unique.domain.model.budget
    arithmetic = _Arithmetic(limits)
    allowance = _residual_work_bound(unique)
    module = import_module("omnibias.core.proof.certificate")

    def forbidden(*args, **kwargs):
        pytest.fail("the private residual oracle sealed or replayed a certificate")

    monkeypatch.setattr(module, "make_certificate", forbidden)
    monkeypatch.setattr(module, "verify_certificate_digest", forbidden)
    with _strict():
        oracle = _ResidualOracle(unique, arithmetic)
        first = oracle.evaluate((F(1), F(0)))
        assert arithmetic.used < allowance
        before = arithmetic.used
        second = oracle.evaluate((F(1, 2), F(1, 2)))
    assert 0 < arithmetic.used-before < allowance
    assert first.error_bound > second.error_bound == 0
    assert _residual_source_slots(unique) > 0


def test_nonfinite_physical_enclosure_refuses_and_restores_strict_state():
    center = (10**400,)
    unique = _unique(centers=(center,), reference=center)
    prior = transcend.strict_backend()
    assert certify_gaussian_entropy_residual(unique, (1,)) is None
    assert transcend.strict_backend() is prior
