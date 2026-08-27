# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-30: core inequality protocol (no owning backends)."""

from __future__ import annotations

import pytest
from omnibias.core.proof import Conjecture
from omnibias.core.proof.discovery import ExactCheck
from omnibias.core.proof.inequality import (
    INEQUALITY_KIND,
    InequalitySystem,
    Proposal,
    RationalWitness,
    _reset_inequality_backends_for_tests,
    default_inequality_honesty,
    locked_catalog,
    prove_inequality,
    register_inequality_backend,
    run_inequality_pipeline,
)


def test_unknown_sort_raises() -> None:
    with pytest.raises(ValueError, match="unknown inequality sort"):
        InequalitySystem.from_mapping({"sort": "cad", "data": {}})


def test_missing_backend_is_blocked() -> None:
    _reset_inequality_backends_for_tests()
    system = InequalitySystem(sort="linear", existential=True, data={})
    status, payload, honesty = run_inequality_pipeline(system)
    assert status == "BLOCKED"
    assert payload["detail"] == "backend_unavailable"
    assert honesty["p_equals_np_claim"] is False
    attempt = prove_inequality(
        Conjecture("missing", INEQUALITY_KIND, system.as_dict())
    )
    assert attempt.status == "BLOCKED"


def test_honesty_defaults_forbid_parent_claims() -> None:
    honesty = default_inequality_honesty()
    assert honesty["complete_solver"] is False
    assert honesty["new_lp_algorithm_claim"] is False
    assert honesty["unsat_from_float_infeasible"] is False
    assert honesty["soft_residual_is_exact_check"] is False
    assert honesty["jacobian_conjecture_proof_claim"] is False


def test_soft_residual_cannot_become_exact_check() -> None:
    _reset_inequality_backends_for_tests()

    class _Soft:
        sort = "boolean"

        def propose(self, system: InequalitySystem) -> Proposal:
            return Proposal(method="soft", skipped=True)

        def rationalize(
            self, system: InequalitySystem, proposal: Proposal
        ) -> RationalWitness:
            return RationalWitness(method="none")

        def check(
            self, system: InequalitySystem, witness: RationalWitness
        ) -> ExactCheck:
            return ExactCheck(
                ok=True,
                payload={
                    "role": "witness",
                    "soft_residual_is_exact_check": True,
                },
            )

    register_inequality_backend(_Soft())
    with pytest.raises(ValueError, match="soft residual"):
        run_inequality_pipeline(
            InequalitySystem(sort="boolean", existential=True, data={})
        )
    _reset_inequality_backends_for_tests()


def test_locked_catalog_has_twelve() -> None:
    rows = locked_catalog()
    assert len(rows) == 12
    sorts = {str(row["sort"]) for row in rows}
    assert sorts == {"linear", "polynomial", "boolean", "csp"}
    expected = {str(row["expected"]) for row in rows}
    assert expected == {"PROVED", "DISPROVED", "BLOCKED"}
