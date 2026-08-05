# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Planning (soft value iteration), MAS, and structured attention on the shared substrate.

Each reuses the one ``lse_beta`` / marginal machinery: soft value iteration anneals to hard
value iteration with a certified suboptimality; soft-MAS anneals to hard MAS with closed-form
alignment marginals == autodiff; structured attention is the linear-chain marginal and
collapses to plain softmax when transitions vanish. All pinned to a hard oracle + torch/jax
parity.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    AcyclicMDP,
    brute_force_mas,
    brute_force_optimal_return,
    brute_force_soft_mas,
    certify_soft_dp,
    hard_mas,
    hard_value_iteration,
)
from omnibias.struct.jax import monotonic as jmono  # noqa: E402
from omnibias.struct.jax import plan as jplan  # noqa: E402
from omnibias.struct.jax import soft_viterbi_marginals as jsvm  # noqa: E402
from omnibias.struct.jax import structured_attention as jattn  # noqa: E402
from omnibias.struct.jax._logsumexp import softmax_beta as jsoftmax  # noqa: E402
from omnibias.struct.torch import monotonic as tmono  # noqa: E402
from omnibias.struct.torch import plan as tplan  # noqa: E402
from omnibias.struct.torch import structured_attention as tattn  # noqa: E402
from omnibias.struct.torch._logsumexp import softmax_beta as tsoftmax  # noqa: E402

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- planning
def _diamond_mdp() -> AcyclicMDP:
    # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3, 3 -> 4 (terminal 4)
    return AcyclicMDP(5, ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)))


def test_soft_value_iteration_anneals_to_hard_with_certified_gap() -> None:
    mdp = _diamond_mdp()
    rng = np.random.default_rng(0)
    rewards = rng.standard_normal(len(mdp.actions))
    rt = torch.tensor(rewards)
    hard = hard_value_iteration(mdp, rewards)
    assert abs(hard - brute_force_optimal_return(mdp, rewards)) < 1e-9
    num_traj = mdp.count_trajectories()
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(tplan.soft_value_iteration(rt, mdp, beta))
        cert = certify_soft_dp(hard, soft, num_traj, beta, brute_force_value=brute_force_optimal_return(mdp, rewards))
        assert cert.is_sound and cert.agrees_with_bruteforce  # V* <= V_beta <= V* + log(N)/beta
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap


def test_soft_value_iteration_gradient_is_action_visitation() -> None:
    mdp = _diamond_mdp()
    rewards = torch.tensor([0.5, -0.2, 0.3, 0.1, 0.7], requires_grad=True)
    tplan.soft_value_iteration(rewards, mdp, 3.0).backward()
    grad = rewards.grad
    assert grad is not None
    # Every trajectory takes the single terminal action 3->4, so its visitation is 1.
    assert abs(float(grad[4]) - 1.0) < 1e-9
    # Sibling actions out of the start share the whole visitation mass.
    assert abs(float(grad[0] + grad[1]) - 1.0) < 1e-9


def test_soft_value_iteration_parity() -> None:
    mdp = _diamond_mdp()
    rng = np.random.default_rng(1)
    rewards = rng.standard_normal(len(mdp.actions))
    for beta in (1.0, 8.0):
        v_t = float(tplan.soft_value_iteration(torch.tensor(rewards), mdp, beta))
        v_j = float(jplan.soft_value_iteration(jnp.asarray(rewards), mdp, beta))
        assert abs(v_t - v_j) < 1e-9


# --------------------------------------------------------------------------- MAS
def test_soft_mas_matches_oracle_and_marginals_equal_autograd() -> None:
    rng = np.random.default_rng(2)
    score = rng.standard_normal((3, 5))
    st = torch.tensor(score)
    hard = hard_mas(score)
    assert abs(hard - brute_force_mas(score)) < 1e-9
    for beta in (0.5, 2.0, 8.0):
        assert abs(float(tmono.soft_mas(st, beta)) - brute_force_soft_mas(score, beta)) < 1e-9
    stg = torch.tensor(score, requires_grad=True)
    tmono.soft_mas(stg, 3.0).backward()
    marg = tmono.soft_mas_marginals(torch.tensor(score), 3.0)
    assert torch.max(torch.abs(stg.grad - marg)).item() < 1e-9
    assert torch.allclose(marg.sum(dim=0), torch.ones(5), atol=1e-9)  # each frame assigned once


def test_soft_mas_certified_gap_and_parity() -> None:
    from omnibias.struct import count_alignments  # noqa: PLC0415

    rng = np.random.default_rng(3)
    score = rng.standard_normal((3, 5))
    st, sj = torch.tensor(score), jnp.asarray(score)
    hard = hard_mas(score)
    n = count_alignments(3, 5)
    for beta in (1.0, 4.0, 16.0):
        soft = float(tmono.soft_mas(st, beta))
        cert = certify_soft_dp(hard, soft, n, beta, brute_force_value=brute_force_mas(score))
        assert cert.is_sound
        assert abs(soft - float(jmono.soft_mas(sj, beta))) < 1e-9
    m_t = tmono.soft_mas_marginals(st, 4.0).numpy()
    m_j = np.asarray(jmono.soft_mas_marginals(sj, 4.0))
    assert np.max(np.abs(m_t - m_j)) < 1e-9


# --------------------------------------------------------------------------- attention
def test_structured_attention_collapses_to_softmax_without_transitions() -> None:
    rng = np.random.default_rng(4)
    scores = rng.standard_normal((5, 4))
    zero = np.zeros((4, 4))
    beta = 2.0
    attn = tattn(torch.tensor(scores), torch.tensor(zero), beta)
    plain = tsoftmax(torch.tensor(scores), beta, axis=1)
    assert torch.max(torch.abs(attn - plain)).item() < 1e-9  # zero transitions -> plain softmax
    assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-9)


def test_structured_attention_parity_and_matches_marginals() -> None:
    rng = np.random.default_rng(5)
    scores = rng.standard_normal((5, 3))
    trans = rng.standard_normal((3, 3))
    beta = 1.5
    a_t = tattn(torch.tensor(scores), torch.tensor(trans), beta).numpy()
    a_j = np.asarray(jattn(jnp.asarray(scores), jnp.asarray(trans), beta))
    assert np.max(np.abs(a_t - a_j)) < 1e-9
    # It *is* the linear-chain marginal, by construction.
    marg = np.asarray(jsvm(jnp.asarray(scores), jnp.asarray(trans), beta))
    assert np.max(np.abs(a_j - marg)) < 1e-9
    _ = jsoftmax  # jax softmax import kept for symmetry with the torch oracle
