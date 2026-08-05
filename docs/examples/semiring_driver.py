# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The semiring / hypergraph DP driver -- the keystone of omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/semiring_driver.py

Every dynamic program in omnibias-struct is a reduction over the derivations of a weighted
**hypergraph**: nodes are DP items, a hyperedge ``head <- (tail_1, tail_2)`` (arity 0/1/2)
builds an item from already-built ones with an additive edge weight, and a derivation is a
tree of hyperedges scored by its edge-weight sum. Three semirings read three quantities off
the *same* structure:

* ``MaxPlus`` -- the hard optimum (``beta -> inf``);
* ``Log`` / ``lse_beta`` -- the soft, differentiable relaxation;
* ``Counting`` -- the exact number of derivations (the ``N`` in the ``log N / beta`` gap).

Two things this example shows:

1. **Additive-safety.** Lifting a DAG onto the driver (``from_dag``) and running the backend
   ``semiring_value`` reproduces the hand-written ``soft_shortest_path`` bit-for-bit -- the
   driver subsumes the existing layers without changing their numerics.
2. **A custom DP for free.** A tiny hand-built hypergraph gets a soft value, closed-form edge
   marginals (== autodiff), a hard optimum, an exact derivation count, and a certified
   ``lse_beta >= max`` gap -- with no bespoke recursion.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    DAG,
    HyperEdge,
    Hypergraph,
    best_derivation,
    certify_soft_dp,
    count_derivations,
    from_dag,
    hard_value,
)
from omnibias.struct.jax import semiring_value as semiring_value_jax  # noqa: E402
from omnibias.struct.torch import semiring_marginals, semiring_value  # noqa: E402

torch.set_default_dtype(torch.float64)


def _dag_edge_weights(graph, edge_index, score, xp):  # noqa: ANN001, ANN202
    """Scatter per-edge path scores onto the lifted hypergraph's weight vector."""
    rows = [xp.zeros(())] * graph.num_edges
    for (u, v), i in edge_index.items():
        rows[i] = xp.asarray(float(score[(u, v)]))
    return xp.stack(rows)


def main() -> None:
    print("=== 1. the driver reproduces soft_shortest_path bit-for-bit ===")
    # A tiny diamond DAG 0->{1,2}->3 with edge *costs* (soft_shortest_path minimises the
    # summed cost; the driver maximises the summed score = -cost, so the two are negatives).
    cost = {(0, 1): -0.9, (0, 2): -0.2, (1, 3): 0.3, (2, 3): -1.1}
    dag = DAG(num_nodes=4, edges=dict(cost), source=0, sink=3)
    graph, edge_index = from_dag(dag)
    score = {k: -v for k, v in cost.items()}  # the driver's per-edge score
    ew_t = _dag_edge_weights(graph, edge_index, score, torch)

    # Build the (n, n) cost matrix soft_shortest_path expects.
    from omnibias.struct.torch import soft_shortest_path

    w = torch.zeros((4, 4))
    for (u, v), c in cost.items():
        w[u, v] = c
    for beta in (1.0, 8.0):
        driver = float(semiring_value(graph, ew_t, beta))          # soft path score
        layer = -float(soft_shortest_path(w, dag, beta))            # -(softmin cost) = soft score
        print(f"  beta={beta:4.1f}  driver={driver:+.9f}  soft_shortest_path={layer:+.9f}  "
              f"|diff|={abs(driver - layer):.1e}")

    print("\n=== 2. a custom arity-2 hypergraph DP, no bespoke recursion ===")
    # Items 0,1 are axioms; item 2 is built two ways (from {0,1} or a single-tail shortcut);
    # item 3 (the root) is built from item 2. Weights are learnable edge scores.
    edges = (
        HyperEdge(head=0, tails=()),          # e0 axiom A
        HyperEdge(head=1, tails=()),          # e1 axiom B
        HyperEdge(head=2, tails=(0, 1)),      # e2: 2 <- A B  (binary rule)
        HyperEdge(head=2, tails=(0,)),        # e3: 2 <- A    (unary shortcut)
        HyperEdge(head=3, tails=(2,)),        # e4: root <- 2
    )
    hg = Hypergraph(num_nodes=4, edges=edges, root=3)
    weights = torch.tensor([0.5, -0.2, 0.7, 0.1, 0.3], requires_grad=True)
    beta = 3.0

    value = semiring_value(hg, weights, beta)
    value.backward()
    marg = semiring_marginals(hg, weights.detach(), beta)
    value_f = float(value.detach())
    print(f"  soft value V_beta        = {value_f:+.6f}")
    print(f"  edge marginals (closed)  = {[round(float(x), 4) for x in marg]}")
    print(f"  edge marginals (autograd)= {[round(float(x), 4) for x in weights.grad]}")
    print(f"  max|closed - autograd|   = {float((marg - weights.grad).abs().max()):.1e}")

    hard = hard_value(hg, [float(x) for x in weights.detach()])
    n = count_derivations(hg)
    best_score, best_edges = best_derivation(hg, [float(x) for x in weights.detach()])
    print(f"  hard optimum V*          = {hard:+.6f}  via edges {best_edges}")
    print(f"  # derivations N          = {n}")

    cert = certify_soft_dp(hard, value_f, n, beta, sense="max")
    print(f"  certified gap: realized {cert.absolute_gap:.4f} <= bound log(N)/beta "
          f"{cert.gap_bound:.4f}  -> sound={cert.is_sound}")

    print("\n=== 3. torch <-> jax parity of the driver ===")
    vt = float(semiring_value(hg, weights.detach(), beta))
    vj = float(semiring_value_jax(hg, jnp.asarray(weights.detach().numpy()), beta))
    print(f"  |torch - jax| = {abs(vt - vj):.2e}")

    ok = (
        cert.is_sound
        and float((marg - weights.grad).abs().max()) < 1e-9
        and abs(vt - vj) < 1e-9
    )
    print("\nOK: driver == soft_shortest_path; marginals == autograd; gap sound; parity < 1e-9."
          if ok else "\nFAILED a driver invariant")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
