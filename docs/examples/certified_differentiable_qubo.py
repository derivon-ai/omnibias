# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable quadratic Boolean optimization -- omnibias-qubo.

Run:

    pip install "omnibias-qubo[jax,sos,convex]"
    python docs/examples/certified_differentiable_qubo.py

Minimizing ``E(x) = x^T Q x + c^T x`` over ``x in {0, 1}^n`` (QUBO / Ising) is NP-hard,
so there is no poly-time differentiable map to the *exact* global optimum (that would
imply P = NP). The sound "differentiable QUBO" is a **yes-if**: a differentiable
annealed *relaxation* + a *decoder* + a rigorous *optimality-gap certificate*. This
deterministic demo exercises both halves end to end:

1. **Certified gap.** For a random weighted max-cut instance, relax -> decode a valid
   binary point -> certify ``lower <= optimum <= energy`` with two lower bounds
   (``spectral`` box-QP and the tighter ``sos`` Lasserre bound), with the exact
   brute-force optimum sandwiched in between as a self-check. A weaker bound only
   widens the certified gap; it is never unsound.
2. **Backprop through the optimizer.** A predicted linear term is trained by gradient
   descent *through* the unrolled annealed relaxation (``jax.value_and_grad`` + ``jit``)
   to yield a lower-energy *decision* than the untrained baseline -- gradients flow
   through the solver, so a model predicting ``Q`` / ``c`` is trainable end to end.

Terminology: the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` is the feasibility /
temperature sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct
from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to the
closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omnibias.qubo import (  # noqa: E402
    AnnealSchedule,
    QUBOProblem,
    brute_force_min,
    certify_qubo_gap,
    decode_qubo,
    max_cut,
    round_relaxed,
)
from omnibias.qubo.jax import qubo_relaxation  # noqa: E402


def certified_gap_demo() -> None:
    print("=== 1. certified optimality gap (yes-if: a certified gap, never a P=NP claim) ===")
    n = 6
    rng = np.random.default_rng(0)
    w = rng.random((n, n))
    w = 0.5 * (w + w.T)
    np.fill_diagonal(w, 0.0)
    prob = max_cut(w)

    _, e_opt = brute_force_min(prob)  # exact optimum (brute force), the ground truth
    x_soft = np.asarray(qubo_relaxation(prob))  # differentiable relaxation -> soft assignment
    assignment, energy = decode_qubo(prob, relaxed=x_soft)  # round + 1-flip -> upper bound
    print(f"  weighted max-cut, n={n}: decoded cut {-energy:.4f}   optimal cut {-e_opt:.4f}   (E = -cut)\n")
    print(f"  {'bound':>9s} {'lower':>10s} {'optimum':>10s} {'energy':>10s} {'rel gap':>9s}  sandwich")

    certs = {}
    for kind in ("spectral", "sos"):
        cert = certify_qubo_gap(prob, assignment, kind=kind, level=1)
        certs[kind] = cert
        ok = cert.lower_bound <= e_opt + 1e-6 <= cert.energy + 1e-6
        print(f"  {kind:>9s} {cert.lower_bound:10.4f} {e_opt:10.4f} {cert.energy:10.4f} "
              f"{cert.relative_gap:9.1%}  {ok}")
        assert cert.lower_bound <= e_opt + 1e-6, "lower bound must never exceed the true optimum"
        assert cert.is_sound

    sos, spectral = certs["sos"], certs["spectral"]
    assert sos.lower_bound >= spectral.lower_bound - 1e-6, "SOS should be at least as tight"
    assert sos.method == "sos" and sos.certified and sos.sealed is not None, "SOS bound must be sealed"
    print("\n  Reading: both bounds sandwich the true optimum; the SOS (Lasserre) bound is")
    print("  tighter and hash-sealed. A weaker bound only widens the gap -- never unsound.\n")


def train_through_relaxation_demo() -> None:
    print("=== 2. train *through* the QUBO relaxation (backprop-through-opt) ===")
    # A weak coupling Q dominated by a decisive linear term c_true: the solver optimizes
    # x^T Q x + bias^T x, but we are graded on the *true* energy x^T Q x + c_true^T x. With
    # the uninformed bias = 0 the decision ignores c_true (so it is clearly suboptimal);
    # training the bias *through* the solver recovers a low-true-energy decision.
    n = 6
    rng = np.random.default_rng(2)
    m = rng.standard_normal((n, n))
    q = 0.1 * (m + m.T)
    c_true = 3.0 * rng.standard_normal(n)
    prob = QUBOProblem(q, c_true)
    _, e_opt = brute_force_min(prob)

    q_j, c_j = jnp.asarray(q), jnp.asarray(c_true)
    # Train through a *soft* (low-beta) schedule so gradients flow -- a fully annealed
    # solver saturates the sigmoid and dx/dbias -> 0 (the "backprop through a hard solver"
    # problem); decode the learned bias with the default hard schedule at eval time.
    train_sched = AnnealSchedule(beta0=0.4, beta_growth=1.3, stages=6, steps=40)
    eval_sched = AnnealSchedule()

    def decision_energy(bias: jnp.ndarray) -> jnp.ndarray:
        # x(bias): the differentiable decision from the unrolled annealed relaxation;
        # the loss is that decision's energy under the *true* objective (Q, c_true).
        x = qubo_relaxation(q_j, bias, schedule=train_sched)
        return x @ (q_j @ x) + c_j @ x

    value_and_grad = jax.jit(jax.value_and_grad(decision_energy))
    bias = jnp.zeros(n)
    loss0, _ = value_and_grad(bias)
    for _ in range(150):
        _, grad = value_and_grad(bias)
        bias = bias - 0.2 * grad
    loss1, grad1 = value_and_grad(bias)

    e0 = prob.energy(round_relaxed(np.asarray(qubo_relaxation(q_j, jnp.zeros(n), schedule=eval_sched))))
    e1 = prob.energy(round_relaxed(np.asarray(qubo_relaxation(q_j, bias, schedule=eval_sched))))
    print("  soft-relaxed decision energy (the differentiable training loss, lower = better):")
    print(f"    untrained (bias = 0)       {float(loss0):8.4f}")
    print(f"    trained (through the opt)  {float(loss1):8.4f}")
    print("  hard-decoded decision energy (eval schedule; brute-force optimum for reference):")
    print(f"    optimum (brute force)      {e_opt:8.4f}")
    print(f"    untrained (bias = 0)       {e0:8.4f}")
    print(f"    trained (through the opt)  {e1:8.4f}")
    assert bool(jnp.all(jnp.isfinite(grad1))), "gradients through the relaxation must be finite"
    assert float(loss1) < float(loss0) - 1e-9, "backprop through the relaxation should lower the loss"
    assert e1 < e0 - 1e-6, "the trained decision should be strictly lower-energy than the untrained one"
    print("\n  Reading: gradients flow through the (soft) unrolled relaxation, so a predicted")
    print("  parameter is trainable through the solver -- here recovering the optimal decision.\n")


def main() -> None:
    certified_gap_demo()
    train_through_relaxation_demo()
    print("OK: certified sandwich holds (spectral & SOS); backprop lowers the decision energy.")


if __name__ == "__main__":
    main()
