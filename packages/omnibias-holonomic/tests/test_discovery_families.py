# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomic guess families on the discovery loop."""

from __future__ import annotations

from omnibias.core.proof import Conjecture, run_discovery
from omnibias.holonomic.families import (
    HolonomicAlgebraicGuessFamily,
    HolonomicDFiniteGuessFamily,
    HolonomicRecurrenceGuessFamily,
    exp_series,
    fibonacci_samples,
    linear_series,
)
from omnibias.holonomic.proofmachine import (
    HOLONOMIC_ALGEBRAIC_GUESS,
    HOLONOMIC_DFINITE_GUESS,
    HOLONOMIC_RECURRENCE_GUESS,
    build_holonomic_machine,
)


def test_recurrence_guess_fibonacci() -> None:
    family = HolonomicRecurrenceGuessFamily(samples=fibonacci_samples())
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["prefix_verified_only"] is True
    assert result.check.payload["honesty"]["jacobian_conjecture_proof_claim"] is False


def test_dfinite_guess_exp() -> None:
    family = HolonomicDFiniteGuessFamily(series=exp_series())
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "PROVED"


def test_algebraic_guess_linear() -> None:
    family = HolonomicAlgebraicGuessFamily(series=linear_series())
    result = run_discovery(family.statement, family, "score_guided", budget=6)
    assert result.status == "PROVED"


def test_holonomic_machine_guess_kinds() -> None:
    machine = build_holonomic_machine()
    for kind in (
        HOLONOMIC_RECURRENCE_GUESS,
        HOLONOMIC_DFINITE_GUESS,
        HOLONOMIC_ALGEBRAIC_GUESS,
    ):
        verdict = machine.evaluate(Conjecture(name=kind, kind=kind))
        assert verdict.status == "PROVED", kind
