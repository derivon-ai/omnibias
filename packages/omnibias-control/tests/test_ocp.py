# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""DifferentiableEnvironment seam + OCPSpec / Rollout containers (theory 10-02)."""

from __future__ import annotations

import pytest
from omnibias.control.ocp import OCPSpec, Rollout, rollout_generic


class _NumpyLinearEnv:
    """A minimal numpy DifferentiableEnvironment for testing rollout_generic."""

    state_dim = 2
    action_dim = 2

    def step(self, y, u):
        import numpy as np

        return np.asarray(y) + np.asarray(u)

    def cost(self, y, u):
        import numpy as np

        return float(np.sum(np.asarray(y) ** 2))

    def terminal_cost(self, y):
        import numpy as np

        return float(np.sum(np.asarray(y) ** 2)) * 2.0


def test_ocp_spec_validates_dims():
    OCPSpec(state_dim=2, action_dim=1, horizon=5)
    with pytest.raises(ValueError):
        OCPSpec(state_dim=0, action_dim=1, horizon=5)
    with pytest.raises(ValueError):
        OCPSpec(state_dim=2, action_dim=1, horizon=0)
    with pytest.raises(ValueError):
        OCPSpec(state_dim=2, action_dim=1, horizon=5, dt=0.0)


def test_rollout_generic_matches_manual_unroll():
    import numpy as np

    env = _NumpyLinearEnv()
    y0 = np.array([1.0, -1.0])

    def policy(y):
        return -0.1 * y

    result = rollout_generic(env, policy, y0, horizon=4)
    assert isinstance(result, Rollout)
    assert len(result.states) == 5
    assert len(result.actions) == 4
    assert len(result.costs) == 4

    y = y0
    expected_costs = []
    for _ in range(4):
        u = policy(y)
        expected_costs.append(env.cost(y, u))
        y = env.step(y, u)
    assert np.allclose(result.costs, expected_costs)
    assert np.allclose(result.states[-1], y)
    assert result.terminal_cost == pytest.approx(env.terminal_cost(y))


def test_rollout_generic_rejects_bad_horizon():
    env = _NumpyLinearEnv()
    with pytest.raises(ValueError):
        rollout_generic(env, lambda y: y, [0.0, 0.0], horizon=0)


def test_differentiable_environment_protocol_runtime_checkable():
    from omnibias.control.ocp import DifferentiableEnvironment

    env = _NumpyLinearEnv()
    assert isinstance(env, DifferentiableEnvironment)

    class _NotAnEnv:
        pass

    assert not isinstance(_NotAnEnv(), DifferentiableEnvironment)
