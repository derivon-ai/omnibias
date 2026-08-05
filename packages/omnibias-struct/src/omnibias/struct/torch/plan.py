# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable soft value iteration on an acyclic MDP (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.plan` (float64). The soft-Bellman backup
``V_beta(s) = lse_beta_a [ r(s, a) + V_beta(next) ]`` is the entropy-regularised value; it
reuses the shared ``lse_beta`` and anneals to hard value iteration as ``beta -> inf`` (the
temperature axis), with a certified ``log(N)/beta`` suboptimality. Differentiable in the
reward vector. Do not conflate the ``beta -> inf`` relaxation with the ``delta -> 0`` tower.
"""

from __future__ import annotations

import torch
from omnibias.struct._core.plan import AcyclicMDP
from omnibias.struct.torch._logsumexp import logsumexp_beta
from torch import Tensor


def soft_value_iteration(rewards: Tensor, mdp: AcyclicMDP, beta: float = 1.0) -> Tensor:
    r"""Soft-Bellman value of ``mdp.start`` for a reward vector ``rewards`` ``(num_actions,)``.

    ``V_beta(s) = lse_beta_a [ r(s, a) + V_beta(next(s, a)) ]`` (``0`` at terminals);
    ``-> hard value iteration`` as ``beta -> inf``. Differentiable in ``rewards`` (its
    gradient is the soft action-visitation).
    """
    values: list[Tensor | None] = [None] * mdp.num_states
    for s in range(mdp.num_states - 1, -1, -1):
        acts = mdp.actions_of(s)
        if not acts:
            values[s] = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
            continue
        contribs = []
        for i in acts:
            nxt = values[mdp.actions[i][1]]
            assert nxt is not None  # next state has a higher index -> already computed
            contribs.append(rewards[i] + nxt)
        values[s] = logsumexp_beta(torch.stack(contribs), beta, axis=-1)
    start_value = values[mdp.start]
    if start_value is None:  # pragma: no cover - start is always in range
        raise ValueError("start state value was not computed")
    return start_value


def soft_value_iteration_batched(rewards: Tensor, mdp: AcyclicMDP, beta: float = 1.0) -> Tensor:
    r"""Batched :func:`soft_value_iteration` -> ``(B,)`` for ``rewards`` ``(B, num_actions)``.

    Maps the soft-Bellman backup over the leading batch axis with ``torch.func.vmap`` (shared
    ``mdp`` structure) -- bit-identical to looping the per-example layer.
    """
    from torch.func import vmap

    def fwd(r: Tensor) -> Tensor:
        return soft_value_iteration(r, mdp, beta)

    out: Tensor = vmap(fwd)(rewards)
    return out


__all__ = ["soft_value_iteration", "soft_value_iteration_batched"]
