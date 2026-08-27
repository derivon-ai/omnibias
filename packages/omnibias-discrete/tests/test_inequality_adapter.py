# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 09-30: CSP inequality adapter."""

from __future__ import annotations

from omnibias.core.proof.inequality import (
    InequalitySystem,
    register_inequality_backend,
    run_inequality_pipeline,
)
from omnibias.discrete.csp.inequality import CspInequalityBackend


def _csp(allowed: list[list[int]], *, oracle: bool) -> InequalitySystem:
    return InequalitySystem(
        sort="csp",
        existential=True,
        data={
            "variables": [
                {"name": "x0", "domain": ["a", "b"]},
                {"name": "x1", "domain": ["a", "b"]},
            ],
            "relations": [{"scope": [0, 1], "allowed": allowed}],
            "oracle": oracle,
        },
    )


def test_csp_three_verdicts() -> None:
    register_inequality_backend(CspInequalityBackend())
    assert run_inequality_pipeline(_csp([[0, 0], [1, 1]], oracle=False))[0] == "PROVED"
    assert run_inequality_pipeline(_csp([], oracle=True))[0] == "DISPROVED"
    assert run_inequality_pipeline(_csp([], oracle=False))[0] == "BLOCKED"
