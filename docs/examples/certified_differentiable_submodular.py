# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable submodular maximization -- omnibias-submodular.

Run:

    pip install "omnibias-submodular[jax,sos]"
    python docs/examples/certified_differentiable_submodular.py

Maximizing a monotone submodular ``f: 2^[n] -> R>=0`` over the independent sets of a
matroid (here a cardinality / max-coverage constraint) is NP-hard, so there is no
poly-time differentiable map to the *exact* optimum (that would imply P = NP). The sound
"differentiable submodular maximization" is a **yes-if**: the multilinear-extension
relaxation ``F(p) = E_{x~p}[f(x)]``, continuous greedy (Frank-Wolfe over the matroid
polytope), pipage / swap rounding with the a-priori ``1 - 1/e`` guarantee, and a rigorous
optimality-gap certificate. This deterministic demo exercises both halves end to end:

1. **Certified (1 - 1/e) sandwich.** For a small weighted max-coverage instance, run
   continuous greedy -> pipage-round a feasible set ``S`` -> certify
   ``f(S) <= OPT <= U(S)`` with the marginal-gain upper bound ``U(S)``, the exact
   brute-force optimum sandwiched in between as a self-check, and the a-priori
   ``f(S) >= (1 - 1/e) OPT`` guarantee (matching / beating the greedy baseline). A looser
   ``U`` only widens the certified gap; it is never unsound.
2. **Backprop through the optimizer.** A solver picks ``k`` sets under *predicted*
   element weights; we grade the decoded choice under the *true* weights. Training the
   predicted weights by gradient descent *through* the unrolled soft continuous-greedy
   relaxation (``jax.value_and_grad`` + ``jit``) recovers a higher-true-value decision
   than the uninformed baseline -- gradients flow through the solver, so a model
   predicting the coverage data is trainable end to end.

Terminology: the multilinear extension relaxing ``{0,1}^n -> [0,1]^n`` and the
Frank-Wolfe oracle ``sigmoid(beta (g - tau))``, ``beta -> inf``, hardening onto a ``0/1``
matroid-basis vertex is the **feasibility** / temperature sense of "collapse" (a soft
indicator becoming a step), distinct from the **founding bias collapse** (the multi-bias
``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omnibias.submodular import (  # noqa: E402
    ContinuousGreedySchedule,
    Coverage,
    UniformMatroid,
    brute_force_max,
    certify_submodular_gap,
    greedy_maximize,
    max_coverage,
    maximize,
    verify_guarantee,
)
from omnibias.submodular.jax import coverage_multilinear, coverage_relaxation  # noqa: E402


def _support(selection: object) -> tuple[int, ...]:
    """The indices of the chosen sets in a ``0/1`` selection."""
    return tuple(int(i) for i, v in enumerate(np.asarray(selection).reshape(-1)) if v)


def certified_gap_demo() -> None:
    print("=== 1. certified (1 - 1/e) sandwich (yes-if: a certified gap, never a P=NP claim) ===")
    # A small weighted max-coverage instance: pick k of n candidate sets to cover the most
    # weighted universe. Submodular maximization is NP-hard, so we certify a gap, never
    # exactness.
    n_sets, universe, k = 6, 8, 3
    rng = np.random.default_rng(26)
    sets = [
        sorted(rng.choice(universe, size=int(rng.integers(2, 5)), replace=False).tolist())
        for _ in range(n_sets)
    ]
    weights = rng.uniform(0.5, 2.0, size=universe)
    prob = max_coverage(sets, universe=universe, k=k, weights=weights)

    sol = maximize(prob, rounding="pipage")  # continuous greedy -> pipage -> polish
    cert = certify_submodular_gap(prob, sol.selection, fractional=sol.fractional)
    _, opt = brute_force_max(prob.function, prob.matroid)  # exact OPT (small-n oracle)
    _, greedy_val = greedy_maximize(prob.function, prob.matroid)  # best-in-class baseline

    ratio = sol.value / opt
    ok = (cert.value <= opt + 1e-9) and (opt <= cert.upper_bound + 1e-9)
    frac = 0.0 if sol.fractional_value is None else sol.fractional_value
    print(f"  weighted max-coverage: n={n_sets} sets, universe={universe}, k={k}")
    print(f"  chosen sets {_support(sol.selection)}   f(S)={sol.value:.4f}   F(p*)={frac:.4f}\n")
    print(
        f"  {'lower f(S)':>12s} {'OPT':>10s} {'U(S)':>10s} "
        f"{'f/OPT':>8s} {'(1-1/e)':>9s}  sandwich"
    )
    print(
        f"  {cert.value:12.4f} {opt:10.4f} {cert.upper_bound:10.4f} "
        f"{ratio:8.1%} {cert.approx_ratio:9.1%}  {ok}"
    )

    assert cert.internal_consistent, "the sandwich f(S) <= U(S) must hold"
    assert cert.value <= opt + 1e-9, "the decoded value is a valid lower bound on OPT"
    assert opt <= cert.upper_bound + 1e-9, "the marginal bound U(S) must upper-bound OPT"
    assert sol.value >= cert.approx_ratio * opt - 1e-9, "the (1 - 1/e) guarantee must hold"
    assert cert.certified_ratio <= ratio + 1e-9, "the certified ratio is a sound lower bound"
    assert verify_guarantee(prob, sol.selection), "the brute-force self-check must pass"
    assert sol.value >= greedy_val - 1e-9, "must match or beat the greedy baseline"
    print("\n  Reading: f(S) <= OPT <= U(S) is a certified gap; f(S) clears (1 - 1/e) OPT and")
    print(f"  matches/beats greedy ({greedy_val:.4f}). A looser U only widens the gap.\n")


def train_through_relaxation_demo() -> None:
    print("=== 2. train *through* the soft continuous-greedy relaxation (backprop-through-opt) ===")
    # The solver picks k sets to maximize coverage under *predicted* element weights; we
    # grade the decoded choice under the *true* weights. The high-value elements {0,1,2,3}
    # are covered only by small sets, while a big set covers the low-value elements
    # {4,5,6,7}: with uniform predicted weights the solver prefers the big (many-element)
    # set, so training the predicted weights *through* continuous greedy is what recovers
    # the high-true-value choice.
    universe = 8
    high, low = (0, 1, 2, 3), (4, 5, 6, 7)
    sets = [
        [0, 1],  # set 0: two high-value elements
        [2, 3],  # set 1: two high-value elements
        [4, 5, 6],  # set 2: three low-value elements
        [5, 6, 7],  # set 3: three low-value elements
        [4, 5, 6, 7],  # set 4: all four low-value elements
        [0, 4],  # set 5: mixed
    ]
    n_sets, k = len(sets), 2
    membership = np.zeros((universe, n_sets))
    for i, elements in enumerate(sets):
        for e in elements:
            membership[e, i] = 1.0
    w_true = np.ones(universe)
    for e in high:
        w_true[e] = 5.0
    for e in low:
        w_true[e] = 0.5
    matroid = UniformMatroid(n_sets, k)
    f_true = Coverage(membership, w_true)

    c_j = jnp.asarray(membership)
    w_true_j = jnp.asarray(w_true)
    # Train through a *soft* (low-beta) schedule so gradients flow; decode with a harder
    # schedule at eval time (a fully hardened oracle saturates the sigmoid and dp/dw -> 0).
    train_sched = ContinuousGreedySchedule.fast()
    eval_sched = ContinuousGreedySchedule(steps=40, beta=200.0)

    def soft_true_coverage(log_w: jnp.ndarray) -> jnp.ndarray:
        # p(log_w): the differentiable continuous-greedy decision computed with *predicted*
        # weights exp(log_w); the loss is that decision's coverage under the *true* weights.
        w_pred = jnp.exp(log_w)
        p = coverage_relaxation(c_j, w_pred, matroid, train_sched)
        return coverage_multilinear(p, c_j, w_true_j)

    value_and_grad = jax.jit(jax.value_and_grad(lambda lw: -soft_true_coverage(lw)))
    log_w = jnp.zeros(universe)  # uniform predicted weights (uninformed baseline)
    loss0, _ = value_and_grad(log_w)
    for _ in range(200):
        _, grad = value_and_grad(log_w)
        log_w = log_w - 0.5 * grad
    loss1, grad1 = value_and_grad(log_w)

    def decode_true_value(log_w_vec: jnp.ndarray) -> tuple[float, tuple[int, ...]]:
        w_pred = jnp.exp(log_w_vec)
        p = np.asarray(coverage_relaxation(c_j, w_pred, matroid, eval_sched))
        sel = matroid.max_weight_basis(p)  # top-k by fractional mass (no re-optimization)
        return float(f_true.value(sel)), _support(sel)

    v0, sel0 = decode_true_value(jnp.zeros(universe))
    v1, sel1 = decode_true_value(log_w)
    _, opt = brute_force_max(f_true, matroid)

    print("  soft-relaxed true coverage (the differentiable training signal, higher = better):")
    print(f"    untrained (uniform weights)  {float(-loss0):8.4f}")
    print(f"    trained (through the opt)    {float(-loss1):8.4f}")
    print("  hard-decoded true coverage (eval schedule; brute-force optimum for reference):")
    print(f"    optimum (brute force)        {opt:8.4f}")
    print(f"    untrained (uniform weights)  {v0:8.4f}   sets {sel0}")
    print(f"    trained (through the opt)    {v1:8.4f}   sets {sel1}")

    assert bool(jnp.all(jnp.isfinite(grad1))), "gradients through the relaxation must be finite"
    assert float(-loss1) > float(-loss0) + 1e-9, "training should raise the soft true coverage"
    assert v1 > v0 + 1e-9, "the trained decision should cover strictly more true value"
    print("\n  Reading: gradients flow through the soft continuous-greedy oracle, so a model")
    print("  predicting the coverage data is trainable through the solver.\n")


def main() -> None:
    certified_gap_demo()
    train_through_relaxation_demo()
    print("OK: certified sandwich holds (f(S) <= OPT <= U(S), (1 - 1/e)); backprop lifts coverage.")


if __name__ == "__main__":
    main()
