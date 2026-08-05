# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Acyclic (deterministic) MDP: hard value iteration + trajectory oracle (pure numpy).

A finite-horizon deterministic MDP whose state graph is a DAG (actions always move to a
higher-index state). The optimal return from the start is hard value iteration
``V(s) = max_a [ r(s, a) + V(next(s, a)) ]`` (``0`` at a terminal state). Its differentiable
relaxation is the **soft-Bellman** backup ``V_beta(s) = lse_beta_a [ r + V_beta(next) ]`` --
the entropy-regularised value -- which anneals to the hard optimum as ``beta -> inf`` with a
certified ``log(N)/beta`` suboptimality (``N`` = number of trajectories). This is the same
``lse_beta`` substrate in the RL register; :func:`brute_force_optimal_return` enumerates
every trajectory as the oracle.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AcyclicMDP:
    r"""A deterministic acyclic MDP: ``actions[i] = (state, next_state)`` with ``state < next``.

    Action ``i`` earns reward ``rewards[i]`` (supplied separately, so it can be a learnable
    backend tensor). A state with no outgoing action is terminal; trajectories run from
    :attr:`start` to any terminal.
    """

    num_states: int
    actions: tuple[tuple[int, int], ...]
    start: int = 0
    _by_state: dict[int, list[int]] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ValueError(f"num_states must be >= 1, got {self.num_states}")
        by_state: dict[int, list[int]] = {}
        for idx, (u, v) in enumerate(self.actions):
            if not 0 <= u < self.num_states or not 0 <= v < self.num_states:
                raise ValueError(f"action {idx} ({u}->{v}) out of range")
            if u >= v:
                raise ValueError(f"action {idx} ({u}->{v}) violates topological order (u < v)")
            by_state.setdefault(u, []).append(idx)
        object.__setattr__(self, "_by_state", by_state)

    def actions_of(self, state: int) -> list[int]:
        """Action ids available in ``state`` (empty iff terminal)."""
        return self._by_state.get(state, [])

    def is_terminal(self, state: int) -> bool:
        """Whether ``state`` has no outgoing action."""
        return not self._by_state.get(state)

    def enumerate_trajectories(self) -> Iterator[tuple[int, ...]]:
        """Yield every ``start -> terminal`` trajectory as a tuple of action ids."""

        def dfs(state: int, acc: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
            acts = self.actions_of(state)
            if not acts:
                yield acc
                return
            for i in acts:
                yield from dfs(self.actions[i][1], (*acc, i))

        yield from dfs(self.start, ())

    def trajectory_return(self, action_ids: object, rewards: FloatArray) -> float:
        """Total reward of a trajectory (sequence of action ids)."""
        r = np.asarray(rewards, dtype=float)
        return float(sum(float(r[int(i)]) for i in np.asarray(action_ids, dtype=int).reshape(-1)))

    def count_trajectories(self) -> int:
        """Number of ``start -> terminal`` trajectories (topological-order DP)."""
        cnt = [0] * self.num_states
        for s in range(self.num_states - 1, -1, -1):
            acts = self.actions_of(s)
            cnt[s] = 1 if not acts else sum(cnt[self.actions[i][1]] for i in acts)
        return int(cnt[self.start])


def hard_value_iteration(mdp: AcyclicMDP, rewards: FloatArray) -> float:
    r"""Optimal expected return from ``mdp.start`` (deterministic hard value iteration)."""
    r = np.asarray(rewards, dtype=float)
    value = np.zeros(mdp.num_states)
    for s in range(mdp.num_states - 1, -1, -1):
        acts = mdp.actions_of(s)
        if acts:
            value[s] = max(r[i] + value[mdp.actions[i][1]] for i in acts)
    return float(value[mdp.start])


def brute_force_optimal_return(mdp: AcyclicMDP, rewards: FloatArray) -> float:
    r"""Optimal return by enumerating every trajectory (oracle for tiny MDPs)."""
    return max(mdp.trajectory_return(t, rewards) for t in mdp.enumerate_trajectories())


__all__ = [
    "AcyclicMDP",
    "brute_force_optimal_return",
    "hard_value_iteration",
]
