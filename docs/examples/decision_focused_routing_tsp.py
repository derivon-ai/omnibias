# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable decision-focused TSP -- omnibias-routing.

Run:

    pip install "omnibias-routing[jax,convex]" scipy
    python docs/examples/decision_focused_routing_tsp.py

The travelling-salesman tour is NP-hard, so there is no poly-time differentiable
map to the *exact* optimum (that would imply P = NP). The sound "differentiable
TSP" is a **yes-if**: a differentiable convex *relaxation* + a *decoder* + a rigorous
*optimality-gap certificate*. This trimmed demo exercises both halves end to end:

1. **Certified gap.** For a random Euclidean instance, solve each relaxation
   (assignment < flow < Held-Karp), decode a valid tour, and certify
   ``lower <= optimum <= tour_cost`` -- with the exact Held-Karp DP optimum sandwiched
   in between as a self-check. Tighter relaxation => tighter certified gap.
2. **Decision-focused learning.** Per-arc costs are unknown and predicted from
   features by a *misspecified* linear head (the truth is a nonlinear MLP). Training
   the head by backprop *through* the differentiable relaxation ("ours") yields lower
   test **regret** than a two-stage MSE fit -- better decisions, not better MSE.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after x64 config)
import numpy as np  # noqa: E402
from omnibias.routing import (  # noqa: E402
    RelaxationSchedule,
    RoutingProblem,
    certify_tour_gap,
    decode_tour,
    held_karp_dp,
    normalized_regret,
    optimal_tour_costs,
)
from omnibias.routing.jax import decision_cost  # noqa: E402

EPS = 0.05
P = 4  # per-arc feature dimension
HID = 12  # hidden width of the (hidden) nonlinear ground-truth cost map


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _zero_diag(c: np.ndarray) -> np.ndarray:
    c = np.array(c)
    for b in range(c.shape[0]):
        np.fill_diagonal(c[b], 0.0)
    return c


def _true_cost(phi: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return _zero_diag(_softplus(1.5 * (np.tanh(phi @ w1.T) @ w2)) + EPS)


def _predict(phi: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return _zero_diag(_softplus(phi @ w + b) + EPS)


def certified_gap_demo() -> None:
    print("=== 1. certified optimality gap (yes-if: a gap, never a P=NP claim) ===")
    prob = RoutingProblem.from_coords(np.random.default_rng(3).random((7, 2)))
    _, opt = held_karp_dp(prob.cost)
    tour, tour_cost = decode_tour(prob.cost)
    print(f"  decoded tour {tour}  cost {tour_cost:.4f}   exact optimum {opt:.4f}\n")
    print(f"  {'relaxation':>11s} {'lower':>9s} {'optimum':>9s} {'tour':>9s} "
          f"{'rel gap':>8s}  sandwich")
    for kind in ("assignment", "flow", "held_karp"):
        cert = certify_tour_gap(prob, tour, kind=kind)
        ok = cert.lower_bound <= opt + 1e-6 <= cert.tour_cost + 1e-6
        print(f"  {kind:>11s} {cert.lower_bound:9.4f} {opt:9.4f} {cert.tour_cost:9.4f} "
              f"{cert.relative_gap:8.1%}  {ok}")
        assert cert.lower_bound <= opt + 1e-6, "lower bound must be sound"
        assert cert.is_sound
    print("\n  Reading: tighter relaxation -> tighter certified gap; Held-Karp is exact")
    print("  here, so its gap collapses to the (heuristic) decoder's residual.\n")


def decision_focused_demo() -> None:
    print("=== 2. decision-focused routing: backprop through the relaxation ===")
    rng_w = np.random.default_rng(500)
    w1 = rng_w.standard_normal((HID, P)) / np.sqrt(P)
    w2 = rng_w.standard_normal(HID) / np.sqrt(HID)
    phi_tr = np.random.default_rng(1).standard_normal((40, 6, 6, P))
    phi_te = np.random.default_rng(2).standard_normal((60, 6, 6, P))
    c_tr, c_te = _true_cost(phi_tr, w1, w2), _true_cost(phi_te, w1, w2)
    opt_te = optimal_tour_costs(c_te)

    # two-stage: ridge-fit the (inverse-softplus) cost, then decode.
    x = phi_tr.reshape(-1, P)
    y = np.log(np.expm1(np.clip(c_tr.reshape(-1) - EPS, 1e-6, None)))
    aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    theta = np.linalg.solve(aug.T @ aug + 1e-2 * np.eye(P + 1), aug.T @ y)
    w0, b0 = theta[:P], float(theta[P])
    r_two = normalized_regret(_predict(phi_te, w0, b0), c_te, opt_te)

    # ours: minimise the true cost of the *relaxed decision* by plain SGD.
    phij, cj = jnp.asarray(phi_tr), jnp.asarray(c_tr)
    sched = RelaxationSchedule.fast()

    def loss(params: tuple) -> jnp.ndarray:
        w, b = params
        cpred = jax.nn.softplus(phij @ w + b) + EPS
        return decision_cost(cpred, cj, kind="assignment", schedule=sched)

    vg = jax.jit(jax.value_and_grad(loss))
    w, b = jnp.asarray(w0), jnp.asarray(b0)
    for _ in range(25):
        _, (gw, gb) = vg((w, b))
        w, b = w - 0.1 * gw, b - 0.1 * gb
    r_ours = normalized_regret(_predict(phi_te, np.asarray(w), float(b)), c_te, opt_te)

    rng = np.random.default_rng(0)
    rand = _zero_diag(np.abs(rng.standard_normal(c_te.shape)) + EPS)
    r_rand = normalized_regret(rand, c_te, opt_te)
    print("  normalised test regret (lower = better decisions):")
    print(f"    perfect (oracle)  {0.0:.4f}")
    print(f"    ours (through opt){r_ours:8.4f}")
    print(f"    two-stage MSE     {r_two:8.4f}")
    print(f"    random            {r_rand:8.4f}")
    assert r_ours <= r_two + 1e-9, "decision-focused should not be worse than two-stage"
    print("\n  Reading: 'ours' turns a misspecified linear cost head into better routing")
    print("  *decisions* than the two-stage fit -- exactly what backprop-through-opt buys.")


def main() -> None:
    certified_gap_demo()
    decision_focused_demo()
    print("\nOK: certified sandwich holds for every relaxation; decision-focused wins.")


if __name__ == "__main__":
    main()
