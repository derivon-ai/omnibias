# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified chip placement -- differentiable + certified block floorplanning (QAP).

Run:

    pip install "omnibias-nphard[jax,sos,convex]"
    python docs/examples/certified_chip_placement.py

VLSI block / macro placement is the Koopmans-Beckmann *quadratic assignment problem* (QAP):
place ``N`` modules on ``N`` grid slots to minimise the connectivity-weighted total
wirelength ``sum_{i,k} F[i,k] * D[slot(i), slot(k)]`` -- ``F`` the netlist connectivity,
``D`` the slot Manhattan distance. QAP is NP-hard, so there is no poly-time map to the exact
optimum (that would be ``P = NP``). What omnibias adds -- the *category-of-one* claim -- is a
differentiable placement layer that emits a **sound optimality-gap certificate**, and, via
the **Gilmore-Lawler bound**, one that stays non-trivial at realistic block counts
(``N ~ 12-25``) where the Lasserre / SOS SDP is intractable and the spectral bound is useless.

Honest scope: this is *block-level* floorplanning (tens of modules). Industrial placers win
on million-cell designs; we do not claim otherwise. The defensible claim is the *certificate*
``GLB <= optimum <= decoded wirelength`` -- which no other differentiable placer provides. The
gap is NP-hard-honest: generally **non-zero**, never asserted zero.

Three acts:

1. **Certified floorplan.** relax -> decode a valid placement -> ``certify_gap(kind="glb")``;
   GLB sandwiches the brute-force optimum on a tiny grid, dominates the spectral bound, beats
   even the (level-1) SOS bound here, and still certifies a larger grid where brute force and
   SOS are both infeasible; the decode ties / beats ``scipy.optimize.quadratic_assignment``.
2. **Learned search (rescoped, honest).** A relaxation-guided MCTS whose found placement is
   handed to ``certify_gap`` for a sound gap. We *measured* that the QAP relaxation heatmap is
   a **weak** construction prior (it does not reliably beat a uniform-prior search), so the
   value here is the sound certificate, not beating uninformed search -- reported transparently.
3. **Place *through* the solver.** Backprop a predicted connectivity through the unrolled
   relaxation to a placement that reaches the optimum, strictly below the untrained baseline.

Terminology: the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` is the feasibility /
temperature sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from the
**founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form derivative
``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from omnibias.nphard import (  # noqa: E402
    brute_force_min,
    certify_gap,
    decode,
    gilmore_lawler_bound,
    placement_qap,
)
from omnibias.nphard._core.qap import qap_classical, qap_round  # noqa: E402
from omnibias.nphard.jax import qap_decision_cost  # noqa: E402
from omnibias.nphard.jax import relax as relax_jax  # noqa: E402
from omnibias.nphard.search import (  # noqa: E402
    hungarian_rollout,
    mcts_search,
    mdp_for,
    random_rollout,
    relaxation_prior,
    uniform_prior,
)
from omnibias.qubo.problem import AnnealSchedule  # noqa: E402


def _connectivity(rng: np.random.Generator, n: int, hi: int = 5) -> np.ndarray:
    """A symmetric, integer netlist connectivity (zero self-connection)."""
    m = rng.integers(0, hi, size=(n, n)).astype(float)
    m = m + m.T
    np.fill_diagonal(m, 0.0)
    return m


def _manhattan(grid: tuple[int, int]) -> np.ndarray:
    r, c = grid
    coords = np.array([(s // c, s % c) for s in range(r * c)])
    return np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1).astype(float)


def certified_floorplan_demo() -> None:
    print("=== 1. certified floorplan: GLB <= optimum <= decoded wirelength (a sound gap) ===")

    # (a) A tiny 2x3 floorplan where the brute-force optimum is a cheap self-check.
    rng = np.random.default_rng(0)
    grid = (2, 3)
    dim = grid[0] * grid[1]  # 6 modules on 6 slots
    prob = placement_qap(_connectivity(rng, dim), grid)

    heat = np.asarray(relax_jax(prob)).reshape(dim, dim)  # differentiable soft placement
    x_dec, wire_dec = decode(prob, relaxed=heat)  # Hungarian + 2-opt -> a valid placement
    _, wire_opt = brute_force_min(prob)  # exact optimum (dim! placements) -- self-check only
    _, wire_scipy = qap_classical(prob)  # named baseline: scipy FAQ / 2-opt

    glb, sound = gilmore_lawler_bound(prob)
    cert = certify_gap(prob, x_dec, kind="glb")  # GLB <= optimum <= decoded wirelength
    spec = certify_gap(prob, x_dec, kind="spectral", bisection_steps=16)  # generic QUBO bound

    print(f"  {grid} grid, {dim} modules:  decoded {wire_dec:.0f}   scipy FAQ {wire_scipy:.0f}   optimum {wire_opt:.0f}")
    print(f"  {'bound':>9s} {'lower':>12s} {'optimum':>9s} {'decoded':>9s} {'rel gap':>9s}   sound")
    print(f"  {'Gilmore-Lawler':>9s} {cert.lower_bound:12.1f} {wire_opt:9.0f} {cert.energy:9.0f} {cert.relative_gap:9.1%}   {cert.is_sound}")
    print(f"  {'spectral':>9s} {spec.lower_bound:12.1f} {wire_opt:9.0f} {spec.energy:9.0f} {spec.relative_gap:9.1%}   {spec.is_sound}")

    # Soundness (hard): the lower bound provably never exceeds the true optimum.
    assert sound and cert.method == "gilmore_lawler" and cert.is_sound and cert.certified
    assert cert.lower_bound <= wire_opt + 1e-9, "GLB must not exceed the true optimum"
    assert spec.lower_bound <= wire_opt + 1e-9, "spectral bound must not exceed the optimum"
    # GLB is dramatically tighter than the generic spectral / box-QP bound for placement.
    assert cert.lower_bound > spec.lower_bound, "GLB should be far tighter than spectral"
    assert cert.relative_gap < 0.20, "GLB certifies a genuinely useful (single/low-double-digit) gap"
    # The decode is competitive with the named classical baseline.
    assert wire_dec <= wire_scipy + 1e-9, "the decode should tie / beat scipy quadratic_assignment"
    print(f"  -> GLB gap {cert.relative_gap:.1%} vs spectral {spec.relative_gap:.0%}: the certificate is *useful*, not vacuous.\n")

    # (b) A tiny 1x3 line where SOS is affordable -- GLB beats even the Lasserre bound here.
    rng = np.random.default_rng(1)
    line = (1, 3)
    prob3 = placement_qap(_connectivity(rng, 3), line)
    x3, _ = decode(prob3, relaxed=np.asarray(relax_jax(prob3)).reshape(3, 3))
    _, opt3 = brute_force_min(prob3)
    cg3 = certify_gap(prob3, x3, kind="glb")
    cs3 = certify_gap(prob3, x3, kind="sos", level=1, bisection_steps=16)
    print("  N=3 (SOS still affordable): all three bounds sandwich the optimum --")
    print(f"    GLB {cg3.lower_bound:.1f} ({cg3.relative_gap:.1%})   SOS/Lasserre {cs3.lower_bound:.1f} ({cs3.relative_gap:.1%})   optimum {opt3:.0f}")
    assert cg3.lower_bound <= opt3 + 1e-9 and cs3.lower_bound <= opt3 + 1e-9, "both bounds sound"
    assert cg3.lower_bound >= cs3.lower_bound - 1e-9, "GLB is at least as tight as level-1 SOS here"
    print("    -> even where SOS is affordable, GLB is at least as tight -- and it *scales*.\n")

    # (c) A 3x4 floorplan where brute force (12! ~ 5e8) and the SOS SDP (144 vars) are both
    #     infeasible, yet GLB still certifies a non-trivial gap in milliseconds.
    rng = np.random.default_rng(2)
    big = (3, 4)
    ndim = big[0] * big[1]  # 12 modules
    probB = placement_qap(_connectivity(rng, ndim), big)
    xB, wireB = decode(probB, relaxed=np.asarray(relax_jax(probB)).reshape(ndim, ndim))
    _, wireB_scipy = qap_classical(probB)
    certB = certify_gap(probB, xB, kind="glb")
    print(f"  {big} grid, {ndim} modules (brute force & SOS both infeasible):")
    print(f"    GLB {certB.lower_bound:.0f} <= optimum <= decoded {certB.energy:.0f}   (scipy {wireB_scipy:.0f})   certified gap {certB.relative_gap:.1%}")
    assert certB.is_sound and certB.certified, "GLB certificate is sound at scale"
    assert certB.lower_bound <= wireB + 1e-9, "GLB <= decoded wirelength (a valid sandwich)"
    print("    -> this is the flagship: a *sound* placement certificate at a size no exact")
    print("       method reaches. NP-hard-honest -- the gap is non-zero, never asserted zero.\n")


def learned_search_demo() -> None:
    print("=== 2. learned-prior MCTS whose placement is certified (search + sound gap) ===")
    rng = np.random.default_rng(0)
    grid = (2, 3)
    dim = grid[0] * grid[1]
    prob = placement_qap(_connectivity(rng, dim), grid)
    mdp = mdp_for(prob)
    heat = np.asarray(relax_jax(prob)).reshape(dim, dim)
    _, wire_opt = brute_force_min(prob)

    # AlphaZero-style: the differentiable relaxation supplies the prior and a Hungarian
    # completion as the leaf value. The uninformed baseline is a uniform prior + random rollout.
    guided = mcts_search(
        mdp,
        prior_fn=relaxation_prior(heat, temperature=1.0),
        rollout_fn=hungarian_rollout(mdp, heat),
        iterations=150,
        seed=0,
    )
    uninformed = mcts_search(
        mdp,
        prior_fn=uniform_prior,
        rollout_fn=random_rollout(mdp, np.random.default_rng(0)),
        iterations=150,
        seed=0,
    )
    cert = certify_gap(prob, guided.assignment, kind="glb")
    print(f"  guided (relaxation prior + Hungarian rollout) wirelength {guided.energy:.0f}")
    print(f"  uninformed (uniform prior + random rollout)     wirelength {uninformed.energy:.0f}")
    print(f"  optimum {wire_opt:.0f};  certified gap on the *searched* placement: "
          f"GLB {cert.lower_bound:.0f} <= optimum <= {cert.energy:.0f} ({cert.relative_gap:.1%})")

    # Sound invariants (what we DO assert): the search returns a valid placement and its gap
    # is soundly certified. (``assignment`` is the full solution; ``certify_gap(kind="glb")``
    # would itself reject a non-permutation, so a clean certificate implies a valid placement.)
    placement = np.asarray(guided.assignment, dtype=float).reshape(dim, dim)
    assert np.all(placement.sum(0) == 1) and np.all(placement.sum(1) == 1), "MCTS returns a valid placement"
    assert cert.is_sound and cert.lower_bound <= guided.energy + 1e-9, "searched placement is certified"
    assert cert.lower_bound <= wire_opt + 1e-9, "and the GLB lower bound is sound vs the true optimum"
    # What we do NOT assert: guided < uninformed. We measured (in the separate
    # omnibias_experiments project, omnibias-nphard/placement.py) that the QAP relaxation
    # heatmap is a *weak* construction prior -- with the
    # completion held fixed it gives no reliable edge over a uniform prior, and at larger N the
    # guided search can lose. So the certificate, not beating uninformed search, is the value.
    print("  Note: the QAP relaxation heatmap is a *weak* MCTS prior (measured), so we do NOT")
    print("  claim guided beats uninformed search -- we certify whatever placement the search finds.\n")


def place_through_the_solver_demo() -> None:
    print("=== 3. place *through* the solver: predict connectivity, backprop the wirelength ===")
    # A 1x4 linear track (e.g. a row of standard cells). A predicted connectivity F_pred
    # induces a relaxed placement; we are graded by that decision's wirelength under the *true*
    # connectivity. Untrained (F_pred = 0) ignores the netlist; training F_pred *through* the
    # unrolled relaxation recovers the optimal placement. We read the raw Hungarian decision
    # (``qap_round``, no 2-opt) so the heatmap's improvement is visible, not masked by local search.
    dim, grid, seed = 4, (1, 4), 2
    dist = _manhattan(grid)
    rng = np.random.default_rng(seed)
    m = rng.integers(0, 9, size=(dim, dim)).astype(float)
    conn_true = (m + m.T) / 2.0
    np.fill_diagonal(conn_true, 0.0)
    train_sched = AnnealSchedule(beta0=0.4, beta_growth=1.3, stages=6, steps=40)
    eval_sched = AnnealSchedule()

    def decoded_true_wire(conn_pred: np.ndarray) -> float:
        heat = np.asarray(
            relax_jax(placement_qap(conn_pred, grid), schedule=eval_sched)
        ).reshape(dim, dim)
        return float(placement_qap(conn_true, grid).objective(qap_round(heat, dim)))

    def loss(theta: jnp.ndarray) -> jnp.ndarray:
        return qap_decision_cost(theta, dist, conn_true, schedule=train_sched)

    value_and_grad = jax.jit(jax.value_and_grad(loss))
    theta = jnp.zeros((dim, dim))
    loss0, _ = value_and_grad(theta)
    for _ in range(140):
        _, grad = value_and_grad(theta)
        theta = theta - 0.5 * grad / (jnp.linalg.norm(grad) + 1e-12)  # normalized step
    loss1, grad1 = value_and_grad(theta)

    wire_untrained = decoded_true_wire(np.zeros((dim, dim)))
    wire_trained = decoded_true_wire(np.asarray(theta))
    _, wire_opt = brute_force_min(placement_qap(conn_true, grid))
    print("  differentiable decision cost (training loss, lower = better):")
    print(f"    untrained (predict 0)      {float(loss0):8.1f}")
    print(f"    trained (through the opt)  {float(loss1):8.1f}")
    print("  hard-decoded wirelength under the TRUE connectivity (Hungarian readout):")
    print(f"    optimum (brute force)      {wire_opt:8.0f}")
    print(f"    untrained (predict 0)      {wire_untrained:8.0f}")
    print(f"    trained (through the opt)  {wire_trained:8.0f}")
    assert bool(jnp.all(jnp.isfinite(grad1))), "gradients through the relaxation must be finite"
    assert float(loss1) < float(loss0) - 1e-9, "backprop through the relaxation should lower the loss"
    assert wire_trained < wire_untrained - 1e-6, "the trained placement must be strictly better"
    assert (wire_trained - wire_opt) < 0.1 * (wire_untrained - wire_opt), "training closes most of the gap"
    print("\n  Reading: a predicted netlist is placeable end to end -- gradients flow through the")
    print("  (soft) unrolled relaxation, here recovering the optimal placement.\n")


def main() -> None:
    certified_floorplan_demo()
    learned_search_demo()
    place_through_the_solver_demo()
    print("OK: sound GLB certificate (tiny self-check + scales past brute/SOS); searched")
    print("placement certified; and a predicted netlist placed through the solver.")


if __name__ == "__main__":
    main()
