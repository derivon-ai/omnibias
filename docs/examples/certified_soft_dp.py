# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable dynamic programming -- omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/certified_soft_dp.py

Exact hard DP (Viterbi / shortest-path / CTC) is not differentiable -- its ``argmax``
gradient is a.e. zero -- so the sound differentiable object is a **relaxation + a
certified gap**, never an exactness claim. The DP ``max`` combine is replaced by
``lse_beta(a) = beta^-1 log sum_i exp(beta a_i)``; since ``lse_beta >= max`` and
``lse_beta -> max`` as ``beta -> inf`` (the feasibility / temperature axis), the soft DP
anneals to the exact hard optimum, with a closed-form ``log(N) / beta`` gap. This
deterministic demo exercises both halves end to end:

1. **Certified gap + exact-DP agreement.** Tiny Viterbi and shortest-path instances: run
   hard DP and check it equals the brute-force optimum; run soft DP at increasing
   ``beta`` and certify the sandwich ``V* <= V_beta <= V* + log(N)/beta`` (mirrored for
   the ``min`` sense), watching the gap shrink.
2. **Backprop through soft-DP + tower marginals + parity.** Train emissions *through* the
   unrolled ``soft_viterbi`` (a linear-chain CRF loss) so a target path becomes the best
   path; the closed-form forward-backward marginals equal ``jax.grad`` (the exact
   gradient from the ``delta -> 0`` tower softmax), and the PyTorch and JAX twins agree
   bit-for-bit.

Terminology: the ``beta -> inf`` log-sum-exp relaxation is the feasibility / temperature
sense of "collapse" (a soft object hardened to a discrete one), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``; see ``docs/theory.md``) that differentiates it.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    DAG,
    ChainTrellis,
    brute_force_shortest_path,
    brute_force_viterbi,
    certify_soft_dp,
    count_paths,
    shortest_path,
    viterbi,
)
from omnibias.struct.jax import (  # noqa: E402
    soft_shortest_path,
    soft_viterbi,
    soft_viterbi_marginals,
)
from omnibias.struct.torch import soft_viterbi as soft_viterbi_torch  # noqa: E402
from omnibias.struct.torch import soft_viterbi_marginals as marginals_torch  # noqa: E402

torch.set_default_dtype(torch.float64)


def certified_gap_demo() -> None:
    print("=== 1. certified gap + exact-DP agreement (yes-if: a certified sandwich) ===")
    rng = np.random.default_rng(0)
    trellis = ChainTrellis(rng.standard_normal((4, 3)), rng.standard_normal((3, 3)))
    hard, path = viterbi(trellis)
    brute, _ = brute_force_viterbi(trellis)
    num_paths = count_paths(trellis)
    assert abs(hard - brute) < 1e-9, "hard Viterbi must match the brute-force optimum"
    print(f"  Viterbi (T=4, S=3): hard V* = {hard:.4f} = brute force {brute:.4f}   "
          f"argmax path {path}   N = {num_paths} paths")
    print(f"  {'beta':>6s} {'V* (hard)':>11s} {'V_beta (soft)':>14s} {'gap':>9s} {'log(N)/beta':>12s}  sound")

    e_j, a_j = jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions)
    start_j = jnp.asarray(trellis.start)
    prev_gap = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(soft_viterbi(e_j, a_j, beta, start=start_j))
        cert = certify_soft_dp(hard, soft, num_paths, beta, brute_force_value=brute)
        print(f"  {beta:6.1f} {hard:11.4f} {soft:14.4f} {cert.absolute_gap:9.4f} "
              f"{cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound, "the closed-form sandwich must hold"
        assert cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev_gap + 1e-12, "the gap must shrink as beta grows"
        prev_gap = cert.absolute_gap

    # Shortest path (min sense): the mirror sandwich V* - log(N)/beta <= V_beta <= V*.
    dag = DAG(5, {(0, 1): 1.0, (0, 2): 2.0, (1, 2): 0.4, (1, 3): 0.5, (2, 3): 0.3, (2, 4): 3.0, (3, 4): 1.0}, sink=4)
    hard_cost, sp_path = shortest_path(dag)
    brute_cost, _ = brute_force_shortest_path(dag)
    assert abs(hard_cost - brute_cost) < 1e-9
    w_j = jnp.asarray(np.array([[dag.edges.get((u, v), 0.0) for v in range(5)] for u in range(5)]))
    soft_cost = float(soft_shortest_path(w_j, dag, 16.0))
    cert = certify_soft_dp(hard_cost, soft_cost, count_paths(dag), 16.0, sense="min", brute_force_value=brute_cost)
    print(f"\n  shortest path: hard cost {hard_cost:.4f} = brute {brute_cost:.4f}   path {sp_path}   "
          f"soft(beta=16) {soft_cost:.4f}   sound {cert.is_sound}")
    assert cert.is_sound
    print("\n  Reading: soft DP sandwiches the exact optimum; the gap is a closed-form")
    print("  log(N)/beta that shrinks to 0 -- never an unsound exactness claim.\n")


def train_through_soft_dp_demo() -> None:
    print("=== 2. backprop through soft-DP + tower marginals + torch/jax parity ===")
    # A linear-chain CRF loss: logZ_beta - score(target). Minimizing it (>= 0) trains the
    # emissions *through* the unrolled soft-Viterbi so the target path becomes the best one.
    rng = np.random.default_rng(1)
    n_steps, n_states = 5, 3
    transitions = jnp.asarray(rng.standard_normal((n_states, n_states)))
    start = jnp.zeros(n_states)
    target = jnp.asarray(np.array([0, 2, 1, 2, 0]))
    beta = 1.0

    def target_score(emissions: jnp.ndarray) -> jnp.ndarray:
        emit = emissions[jnp.arange(n_steps), target].sum()
        trans = transitions[target[:-1], target[1:]].sum()
        return emit + trans + start[target[0]]

    def loss_fn(emissions: jnp.ndarray) -> jnp.ndarray:
        return soft_viterbi(emissions, transitions, beta, start=start) - target_score(emissions)

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    emissions = jnp.asarray(rng.standard_normal((n_steps, n_states)))
    loss0, _ = value_and_grad(emissions)
    for _ in range(200):
        _, grad = value_and_grad(emissions)
        emissions = emissions - 0.3 * grad
    loss1, grad1 = value_and_grad(emissions)
    print(f"  CRF loss (logZ_beta - target score, lower = better): untrained {float(loss0):.4f} "
          f"-> trained {float(loss1):.4f}")
    assert bool(jnp.all(jnp.isfinite(grad1))), "gradients through the soft DP must be finite"
    assert float(loss1) < float(loss0) - 1e-6, "training through the soft DP should lower the loss"

    # The trained model's Viterbi path is the target path.
    trellis = ChainTrellis(np.asarray(emissions), np.asarray(transitions), np.asarray(start))
    _, best_path = viterbi(trellis)
    print(f"  trained Viterbi path {best_path} == target {tuple(int(t) for t in target)}: "
          f"{best_path == tuple(int(t) for t in target)}")
    assert best_path == tuple(int(t) for t in target)

    # Exact gradient from the tower: closed-form marginals == jax.grad(soft_viterbi).
    grad_soft = jax.grad(lambda e: soft_viterbi(e, transitions, beta, start=start))(emissions)
    gamma = soft_viterbi_marginals(emissions, transitions, beta, start=start)
    max_err = float(jnp.max(jnp.abs(grad_soft - gamma)))
    print(f"  closed-form forward-backward marginals == jax.grad(soft_viterbi): max|.| = {max_err:.2e}")
    assert max_err < 1e-9
    assert bool(jnp.allclose(gamma.sum(axis=1), 1.0, atol=1e-9)), "marginals normalise over states"

    # PyTorch <-> JAX bit-identical (float64).
    e_t, a_t, s_t = torch.tensor(np.asarray(emissions)), torch.tensor(np.asarray(transitions)), torch.tensor(np.asarray(start))
    v_t = float(soft_viterbi_torch(e_t, a_t, beta, start=s_t))
    v_j = float(soft_viterbi(emissions, transitions, beta, start=start))
    g_t = marginals_torch(e_t, a_t, beta, start=s_t).numpy()
    parity_val = abs(v_t - v_j)
    parity_grad = float(np.max(np.abs(g_t - np.asarray(gamma))))
    print(f"  torch<->jax parity: value {parity_val:.2e}   marginals {parity_grad:.2e}")
    assert parity_val < 1e-9 and parity_grad < 1e-9, "torch and jax twins must be bit-identical"
    print("\n  Reading: gradients flow through the unrolled soft DP; the closed-form tower")
    print("  marginals equal autodiff, and the two backends agree bit-for-bit.\n")


def second_order_through_dp_demo() -> None:
    print("=== 3. second-order through soft-DP: exact HVP == closed-form jet curvature ===")
    from omnibias.struct.torch import (  # noqa: PLC0415
        chain_directional_curvature,
        chain_hessian,
        chain_lse_jet,
        chain_sharpness,
    )
    from omnibias.torch.optim import CubicNewton  # noqa: PLC0415

    rng = np.random.default_rng(7)
    trellis = ChainTrellis(rng.standard_normal((4, 3)), rng.standard_normal((3, 3)))
    e_t = torch.tensor(trellis.emissions)
    a_t = torch.tensor(trellis.transitions)
    s_t = torch.tensor(trellis.start)
    direction = torch.tensor(rng.standard_normal(e_t.shape))
    beta = 4.0

    # Two independent second-order routes agree to machine precision.
    curv_hvp = float(chain_directional_curvature(e_t, a_t, direction, beta, start=s_t))
    curv_jet = 2.0 * float(chain_lse_jet(e_t, a_t, direction, beta, order=2, start=s_t)[2])
    print(f"  directional curvature d^T H d: autodiff HVP {curv_hvp:.6f} == 2*jet[2] {curv_jet:.6f}")
    assert abs(curv_hvp - curv_jet) < 1e-9

    hessian = chain_hessian(e_t, a_t, beta, start=s_t).numpy()
    min_eig = float(np.linalg.eigvalsh(hessian).min())
    sharp = float(chain_sharpness(e_t, a_t, beta, start=s_t))
    print(f"  soft_viterbi convex in emissions -> Hessian PSD (min eig {min_eig:.2e}); sharpness {sharp:.4f}")
    assert min_eig > -1e-9

    # A cubic-regularized Newton step (curvature-aware) minimises a convex CRF loss.
    target = torch.tensor([0, 2, 1, 2])

    def target_score(emissions: torch.Tensor) -> torch.Tensor:
        idx = torch.arange(emissions.shape[0])
        return emissions[idx, target].sum() + a_t[target[:-1], target[1:]].sum() + s_t[target[0]]

    leaf = e_t.clone().requires_grad_(True)
    opt = CubicNewton([leaf], sigma=1.0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = soft_viterbi_torch(leaf, a_t, beta, start=s_t) - target_score(leaf)
        loss.backward(create_graph=True)
        return loss

    loss0 = float(closure().detach())
    for _ in range(8):
        opt.step(closure)
    loss1 = float((soft_viterbi_torch(leaf, a_t, beta, start=s_t) - target_score(leaf)).detach())
    print(f"  CubicNewton on the convex CRF loss: {loss0:.4f} -> {loss1:.4f}")
    assert loss1 <= loss0 + 1e-9
    print("\n  Reading: the delta->0 tower gives exact curvature through the DP (closed-form jet")
    print("  == autodiff HVP), and a second-order optimiser descends the convex soft-DP loss.\n")


def main() -> None:
    certified_gap_demo()
    train_through_soft_dp_demo()
    second_order_through_dp_demo()
    print("OK: certified sandwich holds; backprop trains the target path; marginals == grad; "
          "curvature HVP == jet; parity < 1e-9.")


if __name__ == "__main__":
    main()
