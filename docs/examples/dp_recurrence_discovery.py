# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Discover the recurrence a dynamic program obeys, from its own path counts.

Run:

    pip install omnibias-struct omnibias-symbolic
    python docs/examples/dp_recurrence_discovery.py

A DP is *defined* by a transition/recurrence; here we run the inverse problem as a probe --
observe a sequence produced by an omnibias-struct counting DP and let omnibias-symbolic
recover the exact P-recursive law ``sum_j p_j(n) a_{n-j} = 0`` it satisfies (exact rational
null-space search, not a float fit). Three DPs, three laws:

* a linear-chain trellis with ``S`` states -> ``S**T`` paths -> ``a_n = S a_{n-1}``;
* a ``{+1, +2}`` DAG -> Fibonacci path counts -> ``a_n = a_{n-1} + a_{n-2}``;
* the DTW / alignment warping grid -> central Delannoy numbers ->
  ``n a_n = 3(2n-1) a_{n-1} - (n-1) a_{n-2}`` (a genuinely P-recursive, non-monic law).

Recovering the third exactly is the real test: it is the transition law of the alignment
lattice itself, read back out of nothing but the observed counts.
"""

from __future__ import annotations

import numpy as np
from omnibias.struct import DAG, ChainTrellis, DTWLattice
from omnibias.symbolic import discover_recurrence


def _fib_dag_count(n: int) -> int:
    edges = {}
    for i in range(n):
        if i + 1 < n:
            edges[(i, i + 1)] = 1.0
        if i + 2 < n:
            edges[(i, i + 2)] = 1.0
    return DAG(n, edges, source=0, sink=n - 1).count_paths()


def main() -> None:
    print("=== recover the recurrence a struct DP obeys (omnibias-symbolic) ===\n")

    # 1. Linear-chain path counts S**T -> geometric recurrence.
    geo = [ChainTrellis(np.zeros((t, 3)), np.zeros((3, 3))).count_paths() for t in range(1, 9)]
    rel_geo = discover_recurrence(geo, max_order=1, max_index_degree=0)
    print(f"  chain (S=3) path counts {geo}")
    print(f"    -> {rel_geo}  (expect a_n = 3 a_(n-1))\n")
    assert rel_geo is not None and rel_geo.order == 1

    # 2. {+1,+2} DAG path counts -> Fibonacci.
    fib = [_fib_dag_count(n) for n in range(2, 13)]
    rel_fib = discover_recurrence(fib, max_order=2, max_index_degree=0)
    print(f"  Fibonacci DAG path counts {fib}")
    print(f"    -> {rel_fib}  (expect a_n = a_(n-1) + a_(n-2))\n")
    assert rel_fib is not None and rel_fib.order == 2

    # 3. DTW / alignment warping-grid path counts -> central Delannoy numbers.
    delannoy = [DTWLattice(k + 1, k + 1).count_paths() for k in range(12)]
    rel_del = discover_recurrence(delannoy, max_order=2, max_index_degree=1)
    print(f"  DTW warping-grid (central Delannoy) counts {delannoy}")
    print(f"    -> {rel_del}")
    print("    (expect n a_n - 3(2n-1) a_(n-1) + (n-1) a_(n-2) = 0)\n")
    assert rel_del is not None and rel_del.order == 2 and rel_del.index_degree == 1

    print("OK: the exact transition law of each DP is recovered from its observed counts "
          "-- including the non-monic P-recursive Delannoy law of the alignment lattice.")


if __name__ == "__main__":
    main()
