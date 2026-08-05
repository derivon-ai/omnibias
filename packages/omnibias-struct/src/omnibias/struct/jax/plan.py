# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable soft value iteration on an acyclic MDP (jax).

Bit-identical twin of :mod:`omnibias.struct.torch.plan` (float64; needs ``jax_enable_x64``).
The soft-Bellman backup ``V_beta(s) = lse_beta_a [ r(s, a) + V_beta(next) ]`` is the
entropy-regularised value; it reuses the shared ``lse_beta`` and anneals to hard value
iteration as ``beta -> inf``, with a certified ``log(N)/beta`` suboptimality. Differentiable
in the reward vector. Do not conflate the ``beta -> inf`` relaxation with the ``delta -> 0`` tower.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from omnibias.struct._core.plan import AcyclicMDP
from omnibias.struct.jax._logsumexp import logsumexp_beta


def soft_value_iteration(rewards: Array, mdp: AcyclicMDP, beta: float = 1.0) -> Array:
    r"""Soft-Bellman value of ``mdp.start`` for a reward vector ``rewards`` ``(num_actions,)``.

    ``V_beta(s) = lse_beta_a [ r(s, a) + V_beta(next(s, a)) ]`` (``0`` at terminals);
    ``-> hard value iteration`` as ``beta -> inf``. Differentiable in ``rewards`` (its
    gradient is the soft action-visitation).
    """
    values: list[Array | None] = [None] * mdp.num_states
    for s in range(mdp.num_states - 1, -1, -1):
        acts = mdp.actions_of(s)
        if not acts:
            values[s] = jnp.zeros((), dtype=rewards.dtype)
            continue
        contribs = []
        for i in acts:
            nxt = values[mdp.actions[i][1]]
            assert nxt is not None  # next state has a higher index -> already computed
            contribs.append(rewards[i] + nxt)
        values[s] = logsumexp_beta(jnp.stack(contribs), beta, axis=-1)
    start_value = values[mdp.start]
    if start_value is None:  # pragma: no cover - start is always in range
        raise ValueError("start state value was not computed")
    return start_value


def soft_value_iteration_batched(rewards: Array, mdp: AcyclicMDP, beta: float = 1.0) -> Array:
    r"""Batched :func:`soft_value_iteration` -> ``(B,)`` for ``rewards`` ``(B, num_actions)`` (``jax.vmap``)."""
    import jax

    out: Array = jax.vmap(lambda r: soft_value_iteration(r, mdp, beta))(rewards)
    return out


__all__ = ["soft_value_iteration", "soft_value_iteration_batched"]
