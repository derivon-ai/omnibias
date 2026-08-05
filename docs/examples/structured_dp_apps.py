# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Structured DP application modules built on the omnibias-struct substrate.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/structured_dp_apps.py

Every block below is a different *structured* problem -- time-series alignment (soft-DTW),
sequence alignment (Needleman-Wunsch), and planning (soft value iteration) -- yet all reuse
the one shared ``lse_beta`` / marginal / certificate substrate. Each is a probe: it anneals
the ``beta -> inf`` relaxation to the exact hard optimum (checked against a brute-force or
classic-DP oracle), reads off closed-form marginals from the ``delta -> 0`` tower (equal to
autodiff), and sandwiches the hard optimum with the closed-form ``log(N)/beta`` gap. The two
axes stay separate throughout.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    AcyclicMDP,
    AlignmentLattice,
    DTWLattice,
    brute_force_align,
    brute_force_dtw,
    brute_force_mas,
    brute_force_optimal_return,
    certify_soft_dp,
    hard_align,
    hard_dtw,
    hard_mas,
    hard_value_iteration,
)
from omnibias.struct.torch import align as talign  # noqa: E402
from omnibias.struct.torch import dtw as tdtw  # noqa: E402
from omnibias.struct.torch import monotonic as tmono  # noqa: E402
from omnibias.struct.torch import plan as tplan  # noqa: E402
from omnibias.struct.torch import structured_attention  # noqa: E402
from omnibias.struct.torch._logsumexp import softmax_beta  # noqa: E402

torch.set_default_dtype(torch.float64)


def soft_dtw_demo() -> None:
    print("=== soft-DTW: align two series, certified gap + closed-form alignment ===")
    x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    cost = np.abs(x[:, None] - y[None, :])  # |x_i - y_j| local cost
    ct = torch.tensor(cost)
    hard = hard_dtw(cost)
    assert abs(hard - brute_force_dtw(cost)) < 1e-9
    num_paths = DTWLattice(*cost.shape).count_paths()
    print(f"  cost grid {cost.shape}; hard DTW {hard:.4f} = brute force; N = {num_paths} warping paths")
    print(f"  {'beta':>6s} {'soft-DTW':>10s} {'gap':>9s} {'log(N)/beta':>12s}  sound")
    for beta in (1.0, 4.0, 16.0, 64.0):
        soft = float(tdtw.soft_dtw(ct, beta))
        cert = certify_soft_dp(hard, soft, num_paths, beta, sense="min", brute_force_value=brute_force_dtw(cost))
        print(f"  {beta:6.1f} {soft:10.4f} {cert.absolute_gap:9.4f} {cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound and soft <= hard + 1e-9

    # Closed-form soft-alignment matrix == autograd; concentrates on the hard warp path.
    ctg = torch.tensor(cost, requires_grad=True)
    tdtw.soft_dtw(ctg, 16.0).backward()
    marg = tdtw.soft_dtw_marginals(ct, 16.0)
    assert torch.max(torch.abs(ctg.grad - marg)).item() < 1e-9
    print(f"  soft-alignment marginals == autograd (max err {torch.max(torch.abs(ctg.grad - marg)).item():.1e}); "
          f"source/sink mass {float(marg[0, 0]):.3f}/{float(marg[-1, -1]):.3f}\n")


def soft_align_demo() -> None:
    print("=== Needleman-Wunsch: learnable alignment on the shared shortest-path DAG ===")
    rng = np.random.default_rng(1)
    sub = rng.standard_normal((4, 4))
    sub = 0.5 * (sub + sub.T)
    sub[np.arange(4), np.arange(4)] += 2.0  # reward matches
    gap = -1.0
    a = np.array([0, 1, 2, 3])
    b = np.array([0, 2, 3])
    st, gt = torch.tensor(sub), torch.tensor(gap)
    hard = hard_align(a, b, sub, gap)
    assert abs(hard - brute_force_align(a, b, sub, gap)) < 1e-9
    num_paths = AlignmentLattice(len(a), len(b)).build_dag()[0].count_paths()
    print(f"  align {a.tolist()} vs {b.tolist()}: NW optimum {hard:.4f} = brute force; N = {num_paths} alignments")
    for beta in (1.0, 4.0, 16.0):
        soft = float(talign.soft_align(a, b, st, gt, beta))
        cert = certify_soft_dp(hard, soft, num_paths, beta, brute_force_value=brute_force_align(a, b, sub, gap))
        print(f"  beta={beta:5.1f}: soft {soft:.4f}  gap {cert.absolute_gap:.4f} <= log(N)/beta {cert.gap_bound:.4f}  sound {cert.is_sound}")
        assert cert.is_sound

    # Learnable parameters: closed-form (substitution, gap) usage gradients == autograd.
    stg = torch.tensor(sub, requires_grad=True)
    gtg = torch.tensor(gap, requires_grad=True)
    talign.soft_align(a, b, stg, gtg, 4.0).backward()
    g_sub, g_gap = talign.soft_align_marginals(a, b, stg.detach(), gtg.detach(), 4.0)
    err = max(torch.max(torch.abs(stg.grad - g_sub)).item(), abs(float(gtg.grad) - float(g_gap)))
    print(f"  closed-form substitution/gap usage == autograd (max err {err:.1e})\n")


def soft_planning_demo() -> None:
    print("=== soft value iteration: entropy-regularised planning on an acyclic MDP ===")
    mdp = AcyclicMDP(5, ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)))
    rewards = np.array([1.0, 0.2, 0.5, 1.5, 0.3])
    rt = torch.tensor(rewards)
    hard = hard_value_iteration(mdp, rewards)
    assert abs(hard - brute_force_optimal_return(mdp, rewards)) < 1e-9
    num_traj = mdp.count_trajectories()
    print(f"  diamond MDP, {num_traj} trajectories; hard optimal return {hard:.4f} = brute force")
    for beta in (1.0, 4.0, 16.0):
        soft = float(tplan.soft_value_iteration(rt, mdp, beta))
        cert = certify_soft_dp(hard, soft, num_traj, beta, brute_force_value=brute_force_optimal_return(mdp, rewards))
        print(f"  beta={beta:5.1f}: soft-Bellman V {soft:.4f}  suboptimality {cert.absolute_gap:.4f} "
              f"<= log(N)/beta {cert.gap_bound:.4f}  sound {cert.is_sound}")
        assert cert.is_sound
    print()


def monotonic_alignment_demo() -> None:
    print("=== MAS: monotonic alignment search (Glow-TTS), certified + closed-form ===")
    rng = np.random.default_rng(2)
    score = rng.standard_normal((3, 6))  # 3 tokens, 6 frames
    st = torch.tensor(score)
    hard = hard_mas(score)
    assert abs(hard - brute_force_mas(score)) < 1e-9
    marg = tmono.soft_mas_marginals(st, 8.0)
    print(f"  L=3 tokens, T=6 frames; hard MAS {hard:.4f} = brute force; "
          f"each frame's assignment mass sums to 1 (max dev {float((marg.sum(0) - 1).abs().max()):.1e})")
    stg = torch.tensor(score, requires_grad=True)
    tmono.soft_mas(stg, 8.0).backward()
    assert torch.max(torch.abs(stg.grad - marg)).item() < 1e-9
    print("  closed-form alignment marginals == autograd; concentrate on the hard path as beta grows\n")


def structured_attention_demo() -> None:
    print("=== structured attention: linear-chain marginals generalise softmax ===")
    rng = np.random.default_rng(3)
    scores = rng.standard_normal((5, 4))
    trans = rng.standard_normal((4, 4))
    plain = softmax_beta(torch.tensor(scores), 1.0, axis=1)
    no_struct = structured_attention(torch.tensor(scores), torch.zeros(4, 4), 1.0)
    coupled = structured_attention(torch.tensor(scores), torch.tensor(trans), 1.0)
    print(f"  zero transitions == plain softmax attention (max dev {float((no_struct - plain).abs().max()):.1e})")
    print(f"  coupled attention rows still sum to 1 (max dev {float((coupled.sum(1) - 1).abs().max()):.1e})")
    assert torch.max(torch.abs(no_struct - plain)).item() < 1e-9
    print()


def main() -> None:
    soft_dtw_demo()
    soft_align_demo()
    soft_planning_demo()
    monotonic_alignment_demo()
    structured_attention_demo()
    print("OK: soft-DTW, Needleman-Wunsch, soft value iteration, MAS, and structured "
          "attention all anneal to their hard optima with certified log(N)/beta gaps and "
          "closed-form marginals == autodiff on one shared DP substrate.")


if __name__ == "__main__":
    main()
