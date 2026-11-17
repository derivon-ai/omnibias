# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Convergence diagnostics must describe the returned Newton point."""

import math

from omnibias.core.locus import EqualitySystem, UnitTerm, newton_project, residual


def test_last_allowed_step_can_reach_tolerance():
    system = EqualitySystem((UnitTerm(0, 1.0, (1.0,), 0.0),
                             UnitTerm(0, 1.0, (0.0,), 0.0)), "tanh")
    result = newton_project(system, (0.01,), max_iter=1, tol=1e-6)
    expected = math.hypot(*residual(system, result.point))
    assert result.residual_norm == expected
    assert expected < 1e-6
    assert result.converged
    assert result.transversal
