# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 07-05 G5: arrangement-adapted SOS bases reduce degree on a named set."""

from __future__ import annotations

import pytest
from omnibias.core.proof.certificate import make_certificate
from omnibias.sos.certify import certify_sos, degree_reduction_report, named_adapted_problems
from omnibias.sos.monomials import arrangement_adapted_basis


def test_g5_degree_drop_on_at_least_half() -> None:
    rows = degree_reduction_report()
    assert len(rows) == len(named_adapted_problems())
    wins = sum(1 for r in rows if r["win"])
    assert wins >= (len(rows) + 1) // 2, rows
    # Failures stay in the report.
    assert all("name" in r and "drop" in r for r in rows)


def test_adapted_basis_is_sparser_than_total_degree() -> None:
    spec = named_adapted_problems()[0]
    poly = spec["polynomial"]
    adapted = arrangement_adapted_basis(poly, spec["arrangement"], degree=1)
    assert certify_sos(poly, basis=adapted).certified
    assert not certify_sos(poly, half_degree=1).certified


def test_g6_sos_kernel_flag_cannot_be_forged() -> None:
    with pytest.raises(ValueError, match="theorem_prover_verified"):
        make_certificate(
            claim="forged",
            payload={"ok": True},
            honesty={"theorem_prover_verified": True},
        )
