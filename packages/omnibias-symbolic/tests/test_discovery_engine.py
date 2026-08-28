# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact operator-span families and the soft→exact lift."""

from __future__ import annotations

from fractions import Fraction

from omnibias.core.proof import Conjecture, run_discovery
from omnibias.symbolic.families import (
    ActivationIdentityFamily,
    PdeOperatorSpanFamily,
    RecurrenceSpanFamily,
    bell_samples,
    catalan_samples,
)
from omnibias.symbolic.lift import (
    planted_heat_rational,
    snap_sparse_equation,
    sparse_from_coeffs,
)
from omnibias.symbolic.proofmachine import (
    ACTIVATION_IDENTITY_EXACT,
    FAMILY_CATALOG,
    PDE_OPERATOR_SPAN,
    RECURRENCE_SPAN,
    build_symbolic_machine,
)
from omnibias.symbolic.propose import propose_jet_condition


def test_catalan_recurrence_span_hits() -> None:
    family = RecurrenceSpanFamily(samples=catalan_samples(), max_order=2, max_index_degree=1)
    result = run_discovery(family.statement, family, "score_guided", budget=6)
    assert result.status == "PROVED"
    assert result.equation is not None
    assert "a[n]" in result.equation.pretty
    assert result.check is not None
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True
    assert result.check.payload["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_bell_recurrence_span_miss_is_blocked() -> None:
    family = RecurrenceSpanFamily(samples=bell_samples(), max_order=2, max_index_degree=1)
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is False
    assert result.detail == "no witness in enumerated family"


def test_tanh_activation_identity_exact() -> None:
    family = ActivationIdentityFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=32)
    assert result.status == "PROVED"
    assert result.equation is not None
    pretty = result.equation.pretty.replace(" ", "")
    assert "yp" in pretty
    assert result.check is not None
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True


def test_snap_heat_succeeds_and_perturbed_fails() -> None:
    design, target, names = planted_heat_rational(diffusivity=Fraction(1, 8))
    good = snap_sparse_equation(
        sparse_from_coeffs((0.0, 0.0, 0.125), names),
        design,
        target,
        denom_bound=16,
    )
    assert good is not None
    assert "1/8" in good.pretty or "0.125" in good.pretty
    bad = snap_sparse_equation(
        sparse_from_coeffs((0.0, 0.0, 0.13), names),
        design,
        target,
        denom_bound=16,
    )
    assert bad is None


def test_pde_operator_span_family() -> None:
    family = PdeOperatorSpanFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=1)
    assert result.status == "PROVED"
    assert result.equation is not None


def test_neuraljet_shaped_snap_becomes_hypothesis() -> None:
    design, target, names = planted_heat_rational(diffusivity=Fraction(1, 8))
    payload = propose_jet_condition(
        sparse_from_coeffs((0.0, 0.0, 0.125), names),
        design,
        target,
        sort="pde_operator",
        kind="pde_span",
    )
    assert payload["mode"] == "exact_search"
    assert payload["hypothesis"].sort == "pde_operator"
    assert payload["statement"] is not None


def test_symbolic_machine_kinds() -> None:
    machine = build_symbolic_machine()
    assert set(machine.kinds()) == set(FAMILY_CATALOG)
    for kind in (RECURRENCE_SPAN, ACTIVATION_IDENTITY_EXACT, PDE_OPERATOR_SPAN):
        verdict = machine.evaluate(Conjecture(name=kind, kind=kind))
        assert verdict.status == "PROVED", kind
        assert verdict.certificate is not None
        assert verdict.certificate["honesty"]["navier_stokes_proof_claim"] is False
