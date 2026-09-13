# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""D-optimal adapter feeds the existing solver and exact small-set oracle."""

import math

import numpy as np
import pytest
from omnibias.submodular._core.greedy import brute_force_max, greedy_maximize
from omnibias.submodular.design import InformationLogDet, information_design_problem


def test_information_design_against_exhaustive_oracle():
    rng = np.random.default_rng(317)
    factors = [rng.normal(size=(2, 3)) for _ in range(9)]
    problem = information_design_problem(factors, np.eye(3), budget=3)
    x, v = greedy_maximize(problem.function, problem.matroid)
    _, opt = brute_force_max(problem.function, problem.matroid)
    assert (1 - 1 / math.e) * opt <= v <= opt + 1e-12
    assert sum(x) <= 3
    assert problem.function.value(np.zeros(9)) == pytest.approx(0.0)
    for _ in range(15):
        small = np.zeros(9)
        small[rng.choice(9, 2, replace=False)] = 1
        big = small.copy()
        big[rng.choice(9, 3, replace=False)] = 1
        i = next((i for i in range(9) if not big[i]), None)
        if i is not None:
            assert (
                problem.function.marginal_gains(small)[i]
                >= problem.function.marginal_gains(big)[i] - 1e-12
            )


def test_design_adapter_refuses_unlicensed_objective_shapes():
    with pytest.raises(ValueError, match="positive definite"):
        InformationLogDet([np.eye(2)], np.zeros((2, 2)))
    f = InformationLogDet([np.eye(2)], np.eye(2))
    with pytest.raises(NotImplementedError, match="multilinear"):
        f.multilinear(np.array([0.5]))
    with pytest.raises(ValueError, match="binary"):
        f.value([0.5])
