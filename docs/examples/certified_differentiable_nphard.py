# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable heuristics for NP-hard families -- omnibias-nphard.

Run:

    pip install "omnibias-nphard[jax,sos,convex]"
    python docs/examples/certified_differentiable_nphard.py

The quadratic assignment problem (QAP) is NP-hard, so there is no poly-time
differentiable map to the *exact* global optimum (that would imply P = NP). Like
``omnibias-qubo`` / ``omnibias-routing`` -- and unlike the P-class
``omnibias-combinatorics`` -- the sound answer is a **yes-if**: a differentiable annealed
*relaxation* + a structure-preserving *decoder* + a rigorous but **honestly non-tight**
*optimality-gap certificate*. This deterministic demo exercises both halves end to end:

1. **Certified gap.** For a small QAP, relax -> decode a valid permutation (Hungarian +
   2-opt) -> certify ``lower <= optimum <= decoded`` with two lower bounds (``spectral``
   box-QP and the tighter ``sos`` Lasserre bound), with the exact brute-force optimum
   sandwiched in between as a self-check, and the decoded solution matching the named
   classical baseline ``scipy.optimize.quadratic_assignment`` (FAQ + 2-opt). Because QAP
   is NP-hard the gap is generally **non-zero** -- a weaker bound only widens it, never
   asserted zero.
2. **Backprop through the solver.** A *predicted* flow matrix is trained by gradient
   descent *through* the unrolled annealed relaxation (``jax.value_and_grad``) to yield a
   lower-cost *decision* (under the true flow) than the untrained baseline -- so a model
   predicting the QAP weights is trainable end to end.

Terminology: the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` is the feasibility /
temperature sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from
the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omnibias.nphard import brute_force_min, certify_gap, decode, qap  # noqa: E402
from omnibias.nphard._core.qap import qap_classical, qap_round  # noqa: E402
from omnibias.nphard.jax import qap_decision_cost  # noqa: E402
from omnibias.nphard.jax import relax as relax_jax  # noqa: E402
from omnibias.qubo.problem import AnnealSchedule  # noqa: E402


def _symmetric(rng: np.random.Generator, dim: int) -> np.ndarray:
    m = rng.integers(0, 9, size=(dim, dim)).astype(float)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m, 0.0)
    return m


def certified_gap_demo() -> None:
    print("=== 1. certified optimality gap (yes-if: a certified gap, never a P=NP claim) ===")
    rng = np.random.default_rng(7)
    dim = 3  # tiny so the SOS/Lasserre bound is cheap and brute force is exact
    flow, dist = _symmetric(rng, dim), _symmetric(rng, dim)
    prob = qap(flow, dist)

    _, e_opt = brute_force_min(prob)  # exact optimum (brute force over dim! perms)
    heat = np.asarray(relax_jax(prob)).reshape(dim, dim)  # differentiable relaxation
    x_dec, e_dec = decode(prob, relaxed=heat)  # Hungarian + 2-opt -> permutation
    _, e_scipy = qap_classical(prob)  # named baseline: scipy FAQ / 2-opt
    print(f"  QAP dim={dim}: decoded {e_dec:.1f}   scipy FAQ/2-opt {e_scipy:.1f}   optimum {e_opt:.1f}\n")
    print(f"  {'bound':>9s} {'lower':>10s} {'optimum':>10s} {'decoded':>10s} {'rel gap':>10s}  sandwich")

    certs = {}
    for kind in ("spectral", "sos"):
        cert = certify_gap(prob, x_dec, kind=kind, level=1, bisection_steps=16)
        certs[kind] = cert
        ok = cert.lower_bound <= e_opt + 1e-6 <= cert.energy + 1e-6
        print(f"  {kind:>9s} {cert.lower_bound:10.1f} {e_opt:10.1f} {cert.energy:10.1f} "
              f"{cert.relative_gap:10.1%}  {ok}")
        assert cert.lower_bound <= e_opt + 1e-6, "lower bound must never exceed the true optimum"
        assert cert.is_sound

    sos, spectral = certs["sos"], certs["spectral"]
    assert sos.lower_bound >= spectral.lower_bound - 1e-6, "SOS should be at least as tight"
    assert e_dec <= e_scipy + 1e-9, "decoded should match / beat the classical baseline"
    print("\n  Reading: both bounds sandwich the true optimum; the SOS (Lasserre) bound is")
    print("  tighter. The gap is NP-hard-honest -- generally non-zero, never asserted zero.\n")


def train_through_relaxation_demo() -> None:
    print("=== 2. train *through* the QAP relaxation (backprop-through-opt) ===")
    # A predicted flow F_pred induces a relaxed assignment; we are graded by that
    # decision's cost under the *true* flow F_true. Starting from an uninformed F_pred = 0
    # the decoded permutation is poor; training F_pred *through* the unrolled relaxation
    # recovers a near-optimal decision. We read the raw Hungarian decision (``qap_round``,
    # no local search) so the heatmap's improvement is visible, not masked by 2-opt.
    rng = np.random.default_rng(0)
    dim = 4
    dist, flow_true = _symmetric(rng, dim), _symmetric(rng, dim)
    train_sched = AnnealSchedule(beta0=0.4, beta_growth=1.3, stages=6, steps=40)
    eval_sched = AnnealSchedule()

    def decoded_true_cost(flow_pred: np.ndarray) -> float:
        heat = np.asarray(relax_jax(qap(flow_pred, dist), schedule=eval_sched)).reshape(dim, dim)
        return float(qap(flow_true, dist).objective(qap_round(heat, dim)))

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        return qap_decision_cost(theta, dist, flow_true, schedule=train_sched)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    theta = jnp.zeros((dim, dim))
    loss0, _ = value_and_grad(theta)
    for _ in range(120):
        _, grad = value_and_grad(theta)
        theta = theta - 0.5 * grad / (jnp.linalg.norm(grad) + 1e-12)  # normalized step
    loss1, grad1 = value_and_grad(theta)

    e_untrained = decoded_true_cost(np.zeros((dim, dim)))
    e_trained = decoded_true_cost(np.asarray(theta))
    _, e_opt = brute_force_min(qap(flow_true, dist))
    print("  differentiable decision cost (training loss, lower = better):")
    print(f"    untrained (flow = 0)       {float(loss0):8.1f}")
    print(f"    trained (through the opt)  {float(loss1):8.1f}")
    print("  hard-decoded decision cost under the TRUE flow (Hungarian readout):")
    print(f"    optimum (brute force)      {e_opt:8.1f}")
    print(f"    untrained (flow = 0)       {e_untrained:8.1f}")
    print(f"    trained (through the opt)  {e_trained:8.1f}")
    assert bool(jnp.all(jnp.isfinite(grad1))), "gradients through the relaxation must be finite"
    assert float(loss1) < float(loss0) - 1e-9, "backprop through the relaxation should lower the loss"
    assert e_trained < e_untrained - 1e-6, "the trained decision must be strictly better"
    assert (e_trained - e_opt) < 0.1 * (e_untrained - e_opt), "training closes most of the gap"
    print("\n  Reading: gradients flow through the (soft) unrolled relaxation, so a predicted")
    print("  QAP weight is trainable through the solver -- here recovering a near-optimal decision.\n")


def main() -> None:
    certified_gap_demo()
    train_through_relaxation_demo()
    print("OK: certified sandwich holds (spectral & SOS); backprop lowers the decoded decision.")


if __name__ == "__main__":
    main()
