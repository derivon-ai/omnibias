# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Holonomic rank-collapse consumer: integerize Q, then exact syzygy."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.proof import Conjecture, build_engine_machine
from omnibias.holonomic import (
    HOLONOMIC_SYZYGY,
    certify_holonomic_syzygy,
    integerize_matrix,
)
from omnibias.holonomic.proofmachine import build_holonomic_machine, holonomic_provers


def test_integerize_clears_denominators() -> None:
    got = integerize_matrix(((Fraction(1, 2), Fraction(1)), (Fraction(1), Fraction(2))))
    assert got == [[1, 2], [2, 4]]


def test_dependent_q_matrix_proves() -> None:
    report = certify_holonomic_syzygy(((Fraction(1, 2), 1), (1, 2)))
    assert report.proved
    assert report.kernel
    assert report.verdict.outcome.honesty["float_svd_is_proof"] is False
    assert report.verdict.outcome.honesty["holonomic_special_function_claim"] is False


def test_full_rank_disproves() -> None:
    report = certify_holonomic_syzygy(((1, 0), (0, 1)))
    assert report.disproved
    assert report.kernel == ()


def test_float_matrix_is_refused() -> None:
    with pytest.raises(TypeError, match="float singular value"):
        integerize_matrix(((1.0, 2.0), (2.0, 4.0)))


def test_holonomic_machine_kind() -> None:
    machine = build_holonomic_machine()
    proved = machine.evaluate(
        Conjecture("dep", HOLONOMIC_SYZYGY, {"matrix": [[1, 2], [2, 4]]})
    )
    assert proved.proved
    honesty = (proved.certificate or {})["honesty"]
    assert honesty["jacobian_conjecture_proof_claim"] is False
    assert honesty["holonomic_special_function_claim"] is False
    disproved = machine.evaluate(
        Conjecture("full", HOLONOMIC_SYZYGY, {"matrix": [[1, 0], [0, 1]]})
    )
    assert disproved.disproved


def test_compose_with_engine_machine() -> None:
    machine = build_engine_machine()
    for prover in holonomic_provers():
        machine.register(prover)
    verdict = machine.evaluate(
        Conjecture("dep", HOLONOMIC_SYZYGY, {"matrix": [[Fraction(1, 2), 1], [1, 2]]})
    )
    assert verdict.proved
