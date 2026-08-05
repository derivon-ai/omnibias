# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable matching / flow / matroid -- omnibias-combinatorics.

Run:

    pip install "omnibias-combinatorics[jax,graph,convex]"
    python docs/examples/certified_differentiable_matching.py

Assignment (min-cost bipartite matching) is in **P** -- the Hungarian algorithm solves
it exactly in poly time -- so unlike TSP / QUBO there is no ``P = NP`` subtext here. The
honest "differentiable matching" is still a **yes-if**: the exact argmin is
piecewise-constant, so its gradient is a.e. zero and useless for learning; the fix is a
differentiable *entropic / Sinkhorn relaxation* + a *decoder* + an *optimality-gap
certificate* -- and because the Birkhoff polytope is **integral**, that certificate is
*tight* (``gap ~ 0``), not merely valid. This deterministic demo exercises both halves:

1. **Tight certified sandwich (across seeds).** For several random assignment instances,
   relax -> decode a permutation -> certify ``lower <= optimum <= objective`` with the
   Neumaier-Shcherbina verified LP dual, with the exact ``brute_force_min`` optimum and
   the Hungarian ``classical_optimum`` sandwiched in between as self-checks. The decoded
   objective equals the Hungarian optimum and the certified gap is ``~0`` -- tight because
   the polytope is integral.
2. **Backprop through the Sinkhorn.** A predicted cost (an untrained, randomly-initialized
   cost head) is trained by gradient descent *through* the unrolled Sinkhorn relaxation
   (``jax.value_and_grad`` + ``jit``), graded on the true cost of the resulting soft
   matching, to yield a strictly lower-cost *decision* than the untrained baseline --
   gradients flow through the solver, so a model predicting the cost is trainable end to
   end.

Terminology: the relaxation's ``beta -> inf`` hardening (a soft doubly-stochastic point
collapsing onto a permutation) is the feasibility / temperature sense of "collapse",
distinct from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to the
closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omnibias.combinatorics import (  # noqa: E402
    AnnealSchedule,
    AssignmentProblem,
    brute_force_min,
    certify_gap,
    classical_optimum,
    decode,
)
from omnibias.combinatorics.jax import assignment_relaxation  # noqa: E402


def _has_convex() -> bool:
    try:
        import omnibias.convex  # noqa: F401
    except ImportError:
        return False
    return True


def certified_gap_demo() -> None:
    print("=== 1. tight certified optimality gap (yes-if: a certified gap, integral -> ~0) ===")
    n = 5
    sealed = _has_convex()
    print(f"  assignment (n={n}), interval-sealed LP dual: {sealed}\n")
    print(f"  {'seed':>4s} {'lower':>10s} {'brute':>10s} {'hungarian':>10s} "
          f"{'decoded':>10s} {'rel gap':>9s}  sandwich")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        cost = rng.random((n, n))
        prob = AssignmentProblem(cost)

        _, e_bf = brute_force_min(prob)  # exact optimum (exponential vertex enumeration)
        _, e_h = classical_optimum(prob)  # Hungarian -- exact, poly-time, best-in-class
        p_soft = np.asarray(assignment_relaxation(cost))  # differentiable relaxation (jax)
        x_dec, e_dec = decode(prob, relaxed=p_soft)  # round -> permutation (upper bound)
        cert = certify_gap(prob, x_dec)  # verified LP-dual lower bound (tight, integral)

        ok = cert.lower_bound <= e_bf + 1e-6 and e_bf <= e_dec + 1e-6
        print(f"  {seed:>4d} {cert.lower_bound:10.4f} {e_bf:10.4f} {e_h:10.4f} "
              f"{e_dec:10.4f} {cert.relative_gap:9.1e}  {ok}")

        assert cert.lower_bound <= e_bf + 1e-9, "lower bound must never exceed the true optimum"
        assert e_bf <= e_dec + 1e-9, "brute-force optimum must not exceed the decoded objective"
        assert cert.is_sound, "the certified sandwich must hold"
        assert np.isclose(e_dec, e_h), "decoded objective must equal the Hungarian optimum"
        assert np.isclose(e_dec, e_bf), "decoded objective must equal the brute-force optimum"
        assert cert.relative_gap < 1e-6, "the gap is tight (~0) because the polytope is integral"
        if sealed:
            assert cert.certified and cert.method == "lp_dual", "convex present -> interval-sealed"

    print("\n  Reading: the verified LP dual sandwiches the true optimum and the decoded")
    print("  permutation equals Hungarian, with gap ~0 -- tight because Birkhoff is integral.\n")


def train_through_relaxation_demo() -> None:
    print("=== 2. train *through* the Sinkhorn relaxation (backprop-through-opt) ===")
    # A "cost head" predicts the assignment cost. The true costs have a unique, well-
    # separated optimum -- the identity matching (diagonal 0, every off-diagonal >= 1).
    n = 5
    rng = np.random.default_rng(1)
    true_cost = rng.random((n, n)) + 1.0
    np.fill_diagonal(true_cost, 0.0)
    prob = AssignmentProblem(true_cost)
    _, e_opt = brute_force_min(prob)  # optimum = identity matching, cost 0

    c_true = jnp.asarray(true_cost)
    # The untrained head is *miscalibrated*: it confidently predicts a low cost for the
    # wrong (cyclic-shift) matching, so its decoded matching is far from optimal. Training
    # the predicted cost *through* the Sinkhorn must overcome this bad initialization.
    shift = (np.arange(n) + 1) % n
    theta0_np = np.ones((n, n))
    theta0_np[np.arange(n), shift] = -1.0
    theta0 = jnp.asarray(theta0_np)

    # Train through a *soft* (low-beta) schedule so gradients flow -- a fully annealed
    # Sinkhorn saturates and dP/dtheta -> 0 (the "backprop through a hard solver" problem);
    # decode the learned cost with the default sharp schedule at eval time.
    train_sched = AnnealSchedule(beta0=0.4, beta_growth=1.3, stages=6, steps=60)
    eval_sched = AnnealSchedule()

    def soft_true_cost(theta: jnp.ndarray) -> jnp.ndarray:
        # P(theta): the differentiable soft matching from the unrolled Sinkhorn relaxation;
        # the loss is that decision's cost under the *true* objective.
        p = assignment_relaxation(theta, schedule=train_sched)
        return jnp.sum(c_true * p)

    value_and_grad = jax.jit(jax.value_and_grad(soft_true_cost))
    theta = theta0
    loss0, _ = value_and_grad(theta)
    for _ in range(300):
        _, grad = value_and_grad(theta)
        theta = theta - 1.0 * grad
    loss1, grad1 = value_and_grad(theta)

    # Hard-decoded decisions (sharp eval schedule): decode the soft matching -> permutation.
    p0 = np.asarray(assignment_relaxation(theta0, schedule=eval_sched))
    p1 = np.asarray(assignment_relaxation(theta, schedule=eval_sched))
    _, e0 = decode(prob, relaxed=p0)
    _, e1 = decode(prob, relaxed=p1)

    print("  soft-matching true cost (the differentiable training loss, lower = better):")
    print(f"    untrained (miscalibrated)  {float(loss0):8.4f}")
    print(f"    trained (through the opt)  {float(loss1):8.4f}")
    print("  hard-decoded matching cost (eval schedule; brute-force optimum for reference):")
    print(f"    optimum (brute force)      {e_opt:8.4f}")
    print(f"    untrained (miscalibrated)  {e0:8.4f}")
    print(f"    trained (through the opt)  {e1:8.4f}")
    assert bool(jnp.all(jnp.isfinite(grad1))), "gradients through the relaxation must be finite"
    assert float(loss1) < float(loss0) - 1e-9, "backprop through the relaxation should lower the loss"
    assert e0 > e_opt + 1e-6, "the untrained (miscalibrated) decision should be suboptimal"
    assert e1 < e0 - 1e-6, "the trained decision should be strictly lower-cost than the untrained one"
    assert np.isclose(e1, e_opt), "the trained decision should recover the optimal matching"
    print("\n  Reading: gradients flow through the unrolled Sinkhorn, so a predicted cost is")
    print("  trainable through the solver -- here correcting a miscalibrated head to optimal.\n")


def main() -> None:
    certified_gap_demo()
    train_through_relaxation_demo()
    print("OK: tight certified sandwich holds across seeds; backprop lowers the decoded cost.")


if __name__ == "__main__":
    main()
