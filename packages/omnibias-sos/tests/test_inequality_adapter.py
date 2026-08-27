# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 09-30: polynomial inequality adapter."""

from __future__ import annotations

from omnibias.core.proof.inequality import (
    InequalitySystem,
    register_inequality_backend,
    run_inequality_pipeline,
)
from omnibias.sos.inequality import PolynomialInequalityBackend


def test_polynomial_three_verdicts() -> None:
    register_inequality_backend(PolynomialInequalityBackend())
    proved = InequalitySystem(
        sort="polynomial",
        existential=False,
        data={"poly_n_vars": 1, "poly_terms": [[[0], "1"]]},
    )
    disproved = InequalitySystem(
        sort="polynomial",
        existential=False,
        data={"poly_n_vars": 1, "poly_terms": [[[0], "-1"]]},
    )
    blocked = InequalitySystem(
        sort="polynomial",
        existential=False,
        data={"poly_n_vars": 2, "poly_terms": [[[2, 0], "1"], [[0, 2], "1"]]},
    )
    assert run_inequality_pipeline(proved)[0] == "PROVED"
    assert run_inequality_pipeline(disproved)[0] == "DISPROVED"
    assert run_inequality_pipeline(blocked)[0] == "BLOCKED"
