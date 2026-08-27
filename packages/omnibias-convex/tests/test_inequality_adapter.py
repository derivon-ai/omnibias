# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 09-30: linear inequality adapter."""

from __future__ import annotations

from omnibias.convex.inequality import LinearInequalityBackend
from omnibias.core.proof.inequality import (
    InequalitySystem,
    register_inequality_backend,
    run_inequality_pipeline,
)


def test_linear_three_verdicts() -> None:
    register_inequality_backend(LinearInequalityBackend())
    proved = InequalitySystem(
        sort="linear",
        existential=True,
        data={"A": [["1"], ["-1"]], "b": ["1", "0"]},
    )
    disproved = InequalitySystem(
        sort="linear",
        existential=True,
        data={"A": [["1"], ["-1"]], "b": ["-1", "-1"]},
    )
    high_a = []
    high_b = []
    for i in range(5):
        pos = ["0"] * 5
        neg = ["0"] * 5
        pos[i] = "1"
        neg[i] = "-1"
        high_a.extend([pos, neg])
        high_b.extend(["-1", "-1"])
    blocked = InequalitySystem(
        sort="linear",
        existential=True,
        data={"A": high_a, "b": high_b},
    )
    assert run_inequality_pipeline(proved)[0] == "PROVED"
    assert run_inequality_pipeline(disproved)[0] == "DISPROVED"
    assert run_inequality_pipeline(blocked)[0] == "BLOCKED"
