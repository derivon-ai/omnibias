# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Einselection collapse: a sound coherence enclosure decides a distribution.

Not a wave-function-collapse claim, not a measurement-problem resolution,
not a single-outcome claim, not a Born-rule derivation. The global state
stays pure and entangled; ``rho(t)`` is an improper mixture.
"""

from __future__ import annotations

import inspect
import math
import random

import pytest
from omnibias.core.collapse import (
    EINSELECTION_SPEC,
    CollapseSpec,
    DephasingModel,
    are_distinct,
    coherence_enclosure,
    commutator_enclosure,
    einselected_distribution,
    einselection_collapse,
    get_collapse,
    pointer_basis_verdict,
    propose_pointer_basis,
    reduced_density_matrix,
    register_collapse,
    reset_collapse_registry,
)
from omnibias.core.collapse import einselection as einselection_module
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval

_ALWAYS_FALSE_KEYS = (
    "wave_function_collapse_claim",
    "measurement_problem_resolved",
    "single_outcome_claim",
    "born_rule_derived",
    "continuum_parent_inferred",
    "float_residual_is_proof",
)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def _two_level_model(*, amplitude: float, gamma: float) -> DephasingModel:
    return DephasingModel(
        amplitudes=(ComplexInterval.point(amplitude), ComplexInterval.point(amplitude)),
        rates=(
            (Interval.point(0.0), Interval.point(gamma)),
            (Interval.point(gamma), Interval.point(0.0)),
        ),
    )


# --------------------------------------------------------------------------
# G1 -- registry distinctness and founding-object refusal
# --------------------------------------------------------------------------


def test_einselection_is_registered_and_distinct_from_all_priors() -> None:
    assert get_collapse("einselection") == EINSELECTION_SPEC
    for name in (
        "bias",
        "temperature",
        "enclosure",
        "verdict",
        "identity",
        "winding",
        "pairing",
        "rank",
        "relaxation",
    ):
        report = are_distinct(EINSELECTION_SPEC, get_collapse(name))
        assert report.distinct, (name, report.reasons)


def test_same_axis_rebrand_is_refused() -> None:
    clone = CollapseSpec(
        name="decoherence",
        parameter="decoherence_rate",
        limit="inf",
        surviving_object="einselected_distribution",
        failure="residual_coherence_above_budget",
        home="omnibias.core.collapse.schema",
        register="measure",
    )
    with pytest.raises(ValueError, match="rebrand of 'einselection'"):
        register_collapse(clone)


def test_founding_surviving_object_is_refused() -> None:
    clone = CollapseSpec(
        name="wave_function_collapse",
        parameter="decoherence_rate",
        limit="inf",
        surviving_object="indicator",
        failure="residual_coherence_above_budget",
        home="omnibias.core.collapse.schema",
        register="measure",
    )
    with pytest.raises(ValueError, match="founding surviving object"):
        register_collapse(clone)


# --------------------------------------------------------------------------
# DephasingModel validation
# --------------------------------------------------------------------------


def test_model_accepts_a_normalized_two_level_state() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    populations = einselected_distribution(model)
    assert len(populations) == 2
    for box in populations:
        assert box.contains(0.5)


def test_model_rejects_a_nonzero_diagonal_rate() -> None:
    with pytest.raises(ValueError, match="must be exactly zero"):
        DephasingModel(
            amplitudes=(ComplexInterval.point(1.0),),
            rates=((Interval.point(0.1),),),
        )


def test_model_rejects_a_negative_rate() -> None:
    with pytest.raises(ValueError, match="must be nonnegative"):
        DephasingModel(
            amplitudes=(ComplexInterval.point(1.0 / math.sqrt(2.0)),) * 2,
            rates=(
                (Interval.point(0.0), Interval.point(-1.0)),
                (Interval.point(-1.0), Interval.point(0.0)),
            ),
        )


def test_model_rejects_an_asymmetric_rate_matrix() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        DephasingModel(
            amplitudes=(ComplexInterval.point(1.0 / math.sqrt(2.0)),) * 2,
            rates=(
                (Interval.point(0.0), Interval.point(1.0)),
                (Interval.point(2.0), Interval.point(0.0)),
            ),
        )


def test_model_rejects_an_unnormalized_amplitude() -> None:
    with pytest.raises(ValueError, match="excludes 1.0"):
        DephasingModel(
            amplitudes=(ComplexInterval.point(0.1),),
            rates=((Interval.point(0.0),),),
        )


def test_negative_time_and_nonpositive_budget_are_refused() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        einselection_collapse(model, time=-1.0, coherence_budget=1e-6)
    with pytest.raises(ValueError, match="positive finite"):
        einselection_collapse(model, time=1.0, coherence_budget=0.0)
    with pytest.raises(ValueError, match="positive finite"):
        einselection_collapse(model, time=1.0, coherence_budget=float("inf"))


# --------------------------------------------------------------------------
# G2 -- grid-plus-random enclosure soundness
# --------------------------------------------------------------------------


def _float_rho_offdiag(c_i: complex, c_j: complex, gamma: float, t: float) -> complex:
    return c_i * c_j.conjugate() * math.exp(-gamma * t)


def test_reduced_density_matrix_soundly_encloses_the_float_truth_grid() -> None:
    amplitude = 1.0 / math.sqrt(2.0)
    gammas = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    times = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for gamma in gammas:
        model = _two_level_model(amplitude=amplitude, gamma=gamma)
        for t in times:
            rho = reduced_density_matrix(model, t)
            entry = rho[0][1]
            truth = _float_rho_offdiag(complex(amplitude), complex(amplitude), gamma, t)
            assert entry.re.contains(truth.real)
            assert entry.im.contains(truth.imag)
            # diagonal is the exact population, time-independent
            assert rho[0][0].re.contains(amplitude * amplitude)
            assert rho[0][0].im.contains(0.0)


def test_reduced_density_matrix_soundly_encloses_the_float_truth_random() -> None:
    rng = random.Random(20260908)
    for _ in range(64):
        amplitude = 0.1 + 0.8 * rng.random()
        other = math.sqrt(max(0.0, 1.0 - amplitude * amplitude))
        gamma = rng.random() * 5.0
        t = rng.random() * 20.0
        model = DephasingModel(
            amplitudes=(ComplexInterval.point(amplitude), ComplexInterval.point(other)),
            rates=(
                (Interval.point(0.0), Interval.point(gamma)),
                (Interval.point(gamma), Interval.point(0.0)),
            ),
        )
        rho = reduced_density_matrix(model, t)
        entry = rho[0][1]
        truth = _float_rho_offdiag(complex(amplitude), complex(other), gamma, t)
        assert entry.re.contains(truth.real)
        assert entry.im.contains(truth.imag)
        coherence = coherence_enclosure(rho)
        assert coherence.contains(abs(truth))


# --------------------------------------------------------------------------
# G3 -- monotone verdict transitions
# --------------------------------------------------------------------------


def test_verdict_moves_disproved_blocked_proved_and_never_regresses() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    eps = 1e-6
    times = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
    order = {"DISPROVED": 0, "BLOCKED": 1, "PROVED": 2}
    statuses = [einselection_collapse(model, time=t, coherence_budget=eps).status for t in times]
    ranks = [order[status] for status in statuses]
    assert ranks == sorted(ranks), (times, statuses)
    assert "DISPROVED" in statuses
    assert "PROVED" in statuses


def test_zero_rate_never_proves_below_the_initial_coherence() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=0.0)
    for t in (0.0, 1.0, 10.0, 1000.0):
        verdict = einselection_collapse(model, time=t, coherence_budget=0.4)
        assert not verdict.proved


def test_worked_example_from_the_theory_spec() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    at_5 = einselection_collapse(model, time=5.0, coherence_budget=1e-6)
    assert at_5.disproved
    at_20 = einselection_collapse(model, time=20.0, coherence_budget=1e-6)
    assert at_20.proved
    assert at_20.outcome.surviving == "einselected_distribution"


# --------------------------------------------------------------------------
# G4 -- pointer-basis commutator
# --------------------------------------------------------------------------


def test_commuting_diagonal_pair_proves() -> None:
    a = (
        (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(-1.0)),
    )
    h = (
        (ComplexInterval.point(2.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(3.0)),
    )
    verdict = pointer_basis_verdict(a, h)
    assert verdict.proved
    assert verdict.outcome.surviving == "commuting_pointer_basis"


def test_noncommuting_pauli_pair_disproves() -> None:
    sigma_x = (
        (ComplexInterval.point(0.0), ComplexInterval.point(1.0)),
        (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
    )
    sigma_z = (
        (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(-1.0)),
    )
    verdict = pointer_basis_verdict(sigma_x, sigma_z)
    assert verdict.disproved


def test_genuine_interval_uncertainty_never_forges_a_proof() -> None:
    a = (
        (ComplexInterval.point(1.0), ComplexInterval(Interval(-0.1, 0.1), Interval.point(0.0))),
        (ComplexInterval(Interval(-0.1, 0.1), Interval.point(0.0)), ComplexInterval.point(-1.0)),
    )
    h = (
        (ComplexInterval.point(2.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(3.0)),
    )
    verdict = pointer_basis_verdict(a, h)
    assert verdict.blocked


def test_commutator_enclosure_requires_matching_square_matrices() -> None:
    a = ((ComplexInterval.point(1.0),),)
    h = (
        (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(1.0)),
    )
    with pytest.raises(ValueError, match="dimension"):
        commutator_enclosure(a, h)


def test_propose_pointer_basis_never_gates_either_verdict() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    before = einselection_collapse(model, time=5.0, coherence_budget=1e-6)
    ranking = propose_pointer_basis(model)
    assert set(ranking) == {0, 1}
    after = einselection_collapse(model, time=5.0, coherence_budget=1e-6)
    assert before.status == after.status
    assert before.outcome.surviving == after.outcome.surviving

    a = (
        (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(-1.0)),
    )
    h = (
        (ComplexInterval.point(2.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(3.0)),
    )
    verdict_before = pointer_basis_verdict(a, h)
    propose_pointer_basis(model)
    verdict_after = pointer_basis_verdict(a, h)
    assert verdict_before.status == verdict_after.status


def test_propose_pointer_basis_ranks_larger_population_first() -> None:
    skewed = DephasingModel(
        amplitudes=(ComplexInterval.point(0.9), ComplexInterval.point(math.sqrt(1 - 0.81))),
        rates=(
            (Interval.point(0.0), Interval.point(1.0)),
            (Interval.point(1.0), Interval.point(0.0)),
        ),
    )
    assert propose_pointer_basis(skewed) == (0, 1)


# --------------------------------------------------------------------------
# G5 -- honesty non-vacuity
# --------------------------------------------------------------------------


def test_einselection_collapse_honesty_never_forges_the_six_keys() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    for t in (1.0, 5.0, 20.0):
        verdict = einselection_collapse(model, time=t, coherence_budget=1e-6)
        for key in _ALWAYS_FALSE_KEYS:
            assert verdict.outcome.honesty[key] is False, (t, key)
        assert verdict.outcome.honesty["einselection_collapse"] is True


def test_pointer_basis_verdict_honesty_never_forges_the_six_keys() -> None:
    a = (
        (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(-1.0)),
    )
    h = (
        (ComplexInterval.point(2.0), ComplexInterval.point(0.0)),
        (ComplexInterval.point(0.0), ComplexInterval.point(3.0)),
    )
    for candidate_h in (h, a):  # a vs a commutes trivially too
        verdict = pointer_basis_verdict(a, candidate_h)
        for key in _ALWAYS_FALSE_KEYS:
            assert verdict.outcome.honesty[key] is False


def test_module_source_never_assigns_true_to_a_permanently_false_key() -> None:
    source = inspect.getsource(einselection_module)
    for key in _ALWAYS_FALSE_KEYS:
        assert f'"{key}"] = True' not in source
        assert f"{key}=True" not in source.replace(" ", "")


def test_einselected_distribution_sums_to_an_enclosure_containing_one() -> None:
    model = _two_level_model(amplitude=1.0 / math.sqrt(2.0), gamma=1.0)
    populations = einselected_distribution(model)
    lo = sum(box.lo for box in populations)
    hi = sum(box.hi for box in populations)
    assert lo <= 1.0 <= hi
