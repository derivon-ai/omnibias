# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Fisher assembly must not evaluate an unused, overflow-prone score array."""

import numpy as np
from omnibias.curvature.information._core import _assemble_g


def test_fisher_assembly_avoids_unused_score_overflow() -> None:
    # The weighted Fisher integrand is finite. Forming jac/p first would
    # overflow, although that intermediate is unnecessary for the result.
    p = np.asarray([1e-250, 1e-250])
    jac = np.asarray([[1e60, 0.0], [0.0, 2e60]])
    weights = np.asarray([1e-250, 1e-250])
    with np.errstate(over="raise", invalid="raise"):
        actual = _assemble_g(p, jac, weights)
    np.testing.assert_allclose(actual, np.diag([1e120, 4e120]), rtol=2e-15, atol=0.0)
