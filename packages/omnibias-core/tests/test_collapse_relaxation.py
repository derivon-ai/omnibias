# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Relaxation collapse: unique steady state, disagreement with einselection."""

from __future__ import annotations

import inspect
import math

import pytest
from omnibias.core.collapse import (
    EINSELECTION_SPEC,
    RELAXATION_SPEC,
    are_distinct,
    einselection_collapse,
    get_collapse,
    relaxation_collapse,
    reset_collapse_registry,
)
from omnibias.core.collapse import relaxation as relaxation_module
from omnibias.core.collapse.einselection import DephasingModel
from omnibias.core.lindblad import qubit_pure_dephasing, qubit_thermal
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval

_ALWAYS_FALSE = (
    "wave_function_collapse_claim",
    "measurement_problem_resolved",
    "single_outcome_claim",
    "born_rule_derived",
    "non_markovian_claim",
    "general_closed_form_claim",
    "thermodynamic_limit_taken",
    "continuum_limit_taken",
    "quantum_advantage_claim",
    "float_residual_is_proof",
)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_g6_registry_distinctness() -> None:
    assert get_collapse("relaxation") == RELAXATION_SPEC
    report = are_distinct(RELAXATION_SPEC, EINSELECTION_SPEC)
    assert report.distinct is True
    assert "parameter" in report.reasons
    assert "surviving_object" in report.reasons


def test_g6_disagreement_on_pure_dephasing() -> None:
    amplitude = ComplexInterval.point(1.0 / math.sqrt(2.0))
    dephasing = DephasingModel(
        amplitudes=(amplitude, amplitude),
        rates=(
            (Interval.point(0.0), Interval.point(1.0)),
            (Interval.point(1.0), Interval.point(0.0)),
        ),
    )
    ein = einselection_collapse(dephasing, time=20.0, coherence_budget=1e-6)
    assert ein.status == "PROVED"
    model = qubit_pure_dephasing(rate=1.0)
    relax = relaxation_collapse(model, time=20.0, distance_budget=1e-6)
    assert relax.status == "BLOCKED"
    assert relax.outcome.surviving is None


def test_distance_budget_is_never_defaulted() -> None:
    sig = inspect.signature(relaxation_collapse)
    assert sig.parameters["distance_budget"].default is inspect.Parameter.empty
    model = qubit_thermal(omega=1.0, beta=1.0, gamma_down=1.0)
    with pytest.raises(TypeError):
        relaxation_collapse(model, time=1.0)  # type: ignore[call-arg]


def test_honesty_keys() -> None:
    model = qubit_thermal(omega=1.0, beta=0.7, gamma_down=1.2)
    verdict = relaxation_collapse(model, time=0.5, distance_budget=0.2)
    assert verdict.outcome.honesty["relaxation_collapse"] is True
    assert verdict.outcome.honesty["markovian_model_declared_not_derived"] is True
    for key in _ALWAYS_FALSE:
        assert verdict.outcome.honesty[key] is False, key


def test_module_source_never_assigns_true_to_permanently_false_keys() -> None:
    source = inspect.getsource(relaxation_module)
    for key in _ALWAYS_FALSE:
        assert f'"{key}"] = True' not in source
        assert f"{key}=True" not in source.replace(" ", "")
