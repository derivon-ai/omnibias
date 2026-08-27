# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-30: Boolean inequality adapter."""

from __future__ import annotations

from omnibias.boolean.inequality import BooleanInequalityBackend
from omnibias.core.proof.inequality import (
    InequalitySystem,
    register_inequality_backend,
    run_inequality_pipeline,
)


def test_boolean_three_verdicts() -> None:
    register_inequality_backend(BooleanInequalityBackend())
    proved = InequalitySystem(
        sort="boolean",
        existential=True,
        data={"tables": [[0, 0, 0, 1]]},
    )
    disproved = InequalitySystem(
        sort="boolean",
        existential=True,
        data={"tables": [[1, 1]]},
    )
    blocked = InequalitySystem(
        sort="boolean",
        existential=True,
        data={"tables": [[0, 0, 0, 1]], "budget": 0},
    )
    assert run_inequality_pipeline(proved)[0] == "PROVED"
    assert run_inequality_pipeline(disproved)[0] == "DISPROVED"
    assert run_inequality_pipeline(blocked)[0] == "BLOCKED"
