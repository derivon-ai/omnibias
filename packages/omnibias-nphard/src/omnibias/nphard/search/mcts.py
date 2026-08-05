# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""A from-scratch UCT / PUCT Monte-Carlo tree search over a construction MDP.

The "Go-like" search track. We phrase an NP-hard family as a *construction MDP*:

* **state** = the tuple of decisions made so far (a partial assignment);
* **action** = fix the next undecided variable to a legal value;
* **terminal** = every variable fixed (a full solution);
* **reward** = ``-energy`` of the completed solution.

:func:`mcts_search` is a small, self-contained PUCT search (AlphaZero-style: a prior over
actions guides the tree, a value estimate scores leaves). The default
:func:`relaxation_prior` reads the **differentiable relaxation** heatmap as the action
prior, and the leaf value is a heatmap-guided greedy completion -- so the differentiable
layer becomes the neural prior of the search. It is a **heuristic**: the best solution
found is still handed to :func:`omnibias.nphard.certify_gap` for a *sound* optimality gap,
never an optimality guarantee. Deterministic given ``seed`` and CPU-tiny.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omnibias.nphard._core.qap import QAPProblem, perm_to_x
from omnibias.nphard._core.scheduling import SchedulingProblem, assignment_to_x

FloatArray = NDArray[np.float64]
State = tuple[int, ...]
PriorFn = Callable[[State, Sequence[int]], Sequence[float]]
RolloutFn = Callable[[State], "tuple[FloatArray, float]"]


@dataclass(frozen=True)
class ConstructionMDP:
    r"""A generic sequential-construction MDP for an NP-hard family.

    Attributes
    ----------
    n_steps:
        The number of decisions (variables) to fix (a terminal state has this length).
    n_choices:
        The size of the per-step action alphabet (columns of the relaxation heatmap).
    legal_actions:
        ``partial -> list of legal next actions`` (encodes feasibility, e.g. QAP's
        no-repeated-location permutation constraint).
    terminal_x:
        ``complete partial -> binary x in {0, 1}^n`` in the QUBO variable space.
    energy:
        ``x -> energy`` (the family's ``problem.energy``).
    name:
        Optional label.
    """

    n_steps: int
    n_choices: int
    legal_actions: Callable[[State], list[int]]
    terminal_x: Callable[[State], FloatArray]
    energy: Callable[[FloatArray], float]
    name: str = "construction"

    def is_terminal(self, state: State) -> bool:
        return len(state) >= self.n_steps


def qap_mdp(problem: QAPProblem) -> ConstructionMDP:
    r"""Construction MDP for QAP: step ``t`` assigns facility ``t`` to an unused location."""
    dim = problem.dim

    def legal(state: State) -> list[int]:
        used = set(state)
        return [j for j in range(dim) if j not in used]

    return ConstructionMDP(
        n_steps=dim,
        n_choices=dim,
        legal_actions=legal,
        terminal_x=lambda s: perm_to_x(s, dim),
        energy=lambda x: float(problem.energy(x)),
        name="qap",
    )


def scheduling_mdp(problem: SchedulingProblem) -> ConstructionMDP:
    r"""Construction MDP for scheduling: step ``t`` assigns job ``t`` to any machine."""
    m = problem.machines

    return ConstructionMDP(
        n_steps=problem.n_jobs,
        n_choices=m,
        legal_actions=lambda state: list(range(m)),
        terminal_x=lambda s: assignment_to_x(s, m),
        energy=lambda x: float(problem.energy(x)),
        name="scheduling",
    )


# --------------------------------------------------------------------------------------
# priors + rollouts
# --------------------------------------------------------------------------------------


def uniform_prior(state: State, legal: Sequence[int]) -> list[float]:
    r"""A flat prior over legal actions (the uninformed baseline)."""
    k = len(legal)
    return [1.0 / k] * k if k else []


def relaxation_prior(heat: object, *, temperature: float = 0.5) -> PriorFn:
    r"""AlphaZero-style prior from the differentiable relaxation heatmap.

    ``heat`` is the ``(n_steps, n_choices)`` soft assignment (e.g.
    ``relax(problem).reshape(dim, dim)``). At step ``t = len(state)`` the prior over legal
    actions is the tempered softmax ``softmax(heat[t, legal] / temperature)``. The
    relaxation heatmap is near one-hot, so a bare readout gives a degenerate (near-greedy)
    prior that starves MCTS of exploration; ``temperature`` (default ``0.5``) softens it
    into an informative-but-exploratory prior (higher = flatter).
    """
    heat_arr = np.asarray(heat, dtype=float)

    def prior(state: State, legal: Sequence[int]) -> list[float]:
        step = len(state)
        row = heat_arr[step, list(legal)] / max(temperature, 1e-6)
        row = row - float(np.max(row))  # stable softmax
        exp = np.exp(row)
        total = float(exp.sum())
        if total <= 0:
            return uniform_prior(state, legal)
        weights: list[float] = (exp / total).tolist()
        return weights

    return prior


def heatmap_rollout(mdp: ConstructionMDP, heat: object) -> RolloutFn:
    r"""Greedy leaf completion following the relaxation heatmap (the default value estimate)."""
    heat_arr = np.asarray(heat, dtype=float)

    def rollout(state: State) -> tuple[FloatArray, float]:
        s = list(state)
        while len(s) < mdp.n_steps:
            legal = mdp.legal_actions(tuple(s))
            # deterministic argmax (ties -> smallest action index)
            s.append(max(legal, key=lambda a: (heat_arr[len(s), a], -a)))
        x = mdp.terminal_x(tuple(s))
        return x, mdp.energy(x)

    return rollout


def hungarian_rollout(mdp: ConstructionMDP, heat: object) -> RolloutFn:
    r"""Heatmap-guided *Hungarian* leaf completion for a permutation-construction MDP (QAP).

    Where :func:`heatmap_rollout` fills the remaining variables greedily (argmax per step),
    this completes the partial permutation *optimally with respect to the heatmap*: it solves
    a linear-assignment problem matching the remaining facilities to the remaining locations
    to maximise total soft-assignment mass. That globally coherent completion is a much
    stronger QAP leaf value, which is what lets the differentiable relaxation act as an
    informative AlphaZero-style prior for placement (a greedy readout starves the search).

    Assumes a permutation MDP (``n_steps == n_choices``, each location used once), i.e.
    :func:`qap_mdp`; for the non-exclusive scheduling MDP use :func:`heatmap_rollout` /
    :func:`random_rollout`.
    """
    from scipy.optimize import linear_sum_assignment

    heat_arr = np.asarray(heat, dtype=float)

    def rollout(state: State) -> tuple[FloatArray, float]:
        s = list(state)
        remaining_fac = list(range(len(s), mdp.n_steps))
        used = set(s)
        remaining_loc = [j for j in range(mdp.n_choices) if j not in used]
        if remaining_fac and remaining_loc:
            sub = heat_arr[np.ix_(remaining_fac, remaining_loc)]
            rows, cols = linear_sum_assignment(-sub)  # maximise soft-assignment mass
            match = {remaining_fac[int(r)]: remaining_loc[int(c)] for r, c in zip(rows, cols, strict=True)}
            for facility in remaining_fac:
                s.append(int(match[facility]))
        x = mdp.terminal_x(tuple(s))
        return x, mdp.energy(x)

    return rollout


def random_rollout(mdp: ConstructionMDP, rng: np.random.Generator) -> RolloutFn:
    r"""Uniform-random leaf completion (the uninformed baseline value estimate)."""

    def rollout(state: State) -> tuple[FloatArray, float]:
        s = list(state)
        while len(s) < mdp.n_steps:
            legal = mdp.legal_actions(tuple(s))
            s.append(int(rng.choice(legal)))
        x = mdp.terminal_x(tuple(s))
        return x, mdp.energy(x)

    return rollout


# --------------------------------------------------------------------------------------
# the tree search
# --------------------------------------------------------------------------------------


@dataclass
class _Node:
    state: State
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[int, _Node] = field(default_factory=dict)

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class MCTSResult:
    r"""The outcome of :func:`mcts_search`.

    Attributes
    ----------
    assignment:
        The best complete solution found, as a binary tuple ``x in {0, 1}^n`` (feeds
        directly into :func:`omnibias.nphard.decode` / :func:`omnibias.nphard.certify_gap`).
    energy:
        The energy of ``assignment`` (a heuristic *upper* bound on the optimum).
    actions:
        The construction path (per-step action) of the best solution.
    iterations:
        The number of search iterations run.
    root_visits:
        The visit count accumulated at the root.
    """

    assignment: tuple[int, ...]
    energy: float
    actions: tuple[int, ...]
    iterations: int
    root_visits: int


def _expand(
    node: _Node,
    mdp: ConstructionMDP,
    prior_fn: PriorFn,
    *,
    noise: NDArray[np.float64] | None = None,
    noise_frac: float = 0.0,
) -> None:
    legal = mdp.legal_actions(node.state)
    priors = list(prior_fn(node.state, legal))
    if noise is not None and priors:  # AlphaZero root Dirichlet-noise exploration
        priors = [(1.0 - noise_frac) * p + noise_frac * float(nz) for p, nz in zip(priors, noise, strict=False)]
    for action, p in zip(legal, priors, strict=False):
        node.children[action] = _Node(state=(*node.state, action), prior=float(p))
    node.expanded = True


def _select(node: _Node, c_puct: float) -> _Node:
    sqrt_n = math.sqrt(max(node.visits, 1))
    best_score, best_child = -math.inf, None
    for action in sorted(node.children):  # deterministic tie-break by action index
        child = node.children[action]
        u = c_puct * child.prior * sqrt_n / (1.0 + child.visits)
        score = child.q + u
        if score > best_score:
            best_score, best_child = score, child
    assert best_child is not None
    return best_child


def mcts_search(
    mdp: ConstructionMDP,
    *,
    prior_fn: PriorFn,
    rollout_fn: RolloutFn,
    iterations: int = 200,
    c_puct: float = 1.5,
    root_noise_frac: float = 0.25,
    dirichlet_alpha: float = 0.6,
    seed: int = 0,
) -> MCTSResult:
    r"""PUCT search over the construction MDP; returns the best solution found.

    ``prior_fn`` guides selection (use :func:`relaxation_prior` for the differentiable
    AlphaZero-style prior, :func:`uniform_prior` for the uninformed baseline);
    ``rollout_fn`` scores leaves (use :func:`heatmap_rollout` or :func:`random_rollout`).
    Values are normalised against the root rollout so PUCT is scale-stable. Root
    Dirichlet noise (``root_noise_frac`` of a ``Dirichlet(dirichlet_alpha)`` draw, the
    AlphaZero exploration trick) keeps a peaked relaxation prior from starving
    exploration. Deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    root = _Node(state=())
    best_x, best_e = rollout_fn(())  # root reference completion
    best_actions: State = ()
    best_e_ref = best_e
    scale = max(abs(best_e), 1.0)

    def value_of(energy: float) -> float:
        # map lower energy -> higher value, centred on the reference (stationary, in ~[-1, 1])
        return float(np.clip((best_e_ref - energy) / scale, -1.0, 1.0))

    root_legal = mdp.legal_actions(())
    noise = (
        rng.dirichlet([dirichlet_alpha] * len(root_legal))
        if root_noise_frac > 0.0 and root_legal
        else None
    )
    _expand(root, mdp, prior_fn, noise=noise, noise_frac=root_noise_frac)
    for _ in range(iterations):
        node = root
        path = [root]
        while node.expanded and not mdp.is_terminal(node.state) and node.children:
            node = _select(node, c_puct)
            path.append(node)
        if mdp.is_terminal(node.state):
            x = mdp.terminal_x(node.state)
            energy = mdp.energy(x)
            actions = node.state
        else:
            _expand(node, mdp, prior_fn)
            x, energy = rollout_fn(node.state)
            actions = None  # rollout path not tracked; the reached state is a prefix
        if energy < best_e:
            best_x, best_e = x, energy
            if actions is not None:
                best_actions = actions
        value = value_of(energy)
        for nd in path:
            nd.visits += 1
            nd.value_sum += value
    return MCTSResult(
        assignment=tuple(int(v) for v in best_x),
        energy=float(best_e),
        actions=tuple(int(a) for a in best_actions),
        iterations=iterations,
        root_visits=root.visits,
    )


def mdp_for(problem: Any) -> ConstructionMDP:
    r"""The construction MDP for a supported family (QAP or scheduling)."""
    if isinstance(problem, QAPProblem):
        return qap_mdp(problem)
    if isinstance(problem, SchedulingProblem):
        return scheduling_mdp(problem)
    raise TypeError(f"no construction MDP for {type(problem).__name__} (use QAP or scheduling)")


__all__ = [
    "ConstructionMDP",
    "MCTSResult",
    "heatmap_rollout",
    "hungarian_rollout",
    "mcts_search",
    "mdp_for",
    "qap_mdp",
    "random_rollout",
    "relaxation_prior",
    "scheduling_mdp",
    "uniform_prior",
]
