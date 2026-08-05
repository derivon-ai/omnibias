# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable Boolean logic -- omnibias-logic.

Run:

    pip install "omnibias-logic[jax,sos]"
    python docs/examples/certified_logic_maxsat_sharpsat.py

Two classic hard problems on a CNF, each a **yes-if** (a certified object, never a `P = NP`
/ `#P` exactness claim). This deterministic, CPU-tiny demo exercises both halves end to end:

1. **Weighted MaxSAT -- a certified optimality gap.** Relax -> decode a min-violation
   assignment -> `certify_gap` a Lasserre / SOS lower bound, with the brute-force optimum
   sandwiched in between as a self-check: `lower <= optimum <= energy`.
2. **(Weighted) #SAT -- a certified count enclosure.** Truncated inclusion-exclusion gives a
   rigorous `lower <= #models <= upper` sandwich that tightens with the Bonferroni order and
   contains the exact `O(2^n)` count; the `beta -> inf` annealed model-finder relaxation plus
   multi-start local search collects distinct model witnesses that strengthen the lower bound
   over a no-search baseline (the differentiable optimizer improving the certified decision).

Terminology: every relaxation's `sigmoid(beta z)`, `beta -> inf` is the feasibility /
temperature sense of "collapse" (a soft indicator hardening to a 0/1 step), distinct from the
**founding bias collapse** (the multi-bias `delta -> 0` limit to the closed-form derivative
`sigma^(K-1)`; see `docs/theory.md`).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from omnibias.logic import (  # noqa: E402
    brute_force_min,
    certify_gap,
    count_enclosure,
    decode,
    exact_model_count,
    max_sat,
    model_count,
)
from omnibias.logic.jax import maxsat_relaxation, sat_relaxation  # noqa: E402


def maxsat_demo() -> None:
    print("=== 1. weighted MaxSAT: certified optimality gap (yes-if, never a P=NP claim) ===")
    # An unsatisfiable core over 2 vars: every assignment falsifies exactly one clause, so the
    # minimum violated weight is positive -- a genuine gap to certify.
    prob = max_sat([[1, 2], [-1, -2], [1, -2], [-1, 2]], weights=[1.5, 1.0, 2.0, 1.0])
    _, e_opt = brute_force_min(prob)  # exact optimum (brute force), the ground truth
    x_soft = np.asarray(maxsat_relaxation(prob))  # differentiable relaxation -> soft assignment
    z, energy = decode(prob, relaxed=x_soft, n_starts=16)  # round + 1-flip -> upper bound
    cert = certify_gap(prob, z, level=2)
    ok = cert.lower_bound <= e_opt + 1e-6 <= cert.energy + 1e-6
    print(f"  min violated weight: decoded {energy:.4f}   optimal {e_opt:.4f}")
    print(f"  gap: lower {cert.lower_bound:.4f} <= optimum {e_opt:.4f} <= energy "
          f"{cert.energy:.4f}   sandwich={ok}   method={cert.method} certified={cert.certified}")
    assert cert.lower_bound <= e_opt + 1e-6, "lower bound must never exceed the true optimum"
    assert cert.energy >= e_opt - 1e-9, "decoded energy is an upper bound"
    assert cert.is_sound
    if cert.certified:
        assert cert.sealed is not None, "a certified SOS bound must be sealed"
    print("  Reading: the certified gap sandwiches the exact optimum; a weaker bound only\n"
          "  widens it -- never unsound.\n")


def sharpsat_demo() -> None:
    print("=== 2. (weighted) #SAT: certified count enclosure sandwiching the exact count ===")
    clauses = [[1, 2], [2, 3], [3, 4], [-1, -4], [1, -3]]  # n = 4
    mc = model_count(clauses)
    exact = exact_model_count(mc)  # exact O(2^n) oracle (small n), the ground truth
    print(f"  unweighted #SAT, n=4, {len(clauses)} clauses: exact model count = {exact:.0f}")
    print(f"  {'order':>5s} {'lower':>8s} {'upper':>8s} {'width':>8s}  tight  contains_exact")
    prev_width = np.inf
    for order in range(0, len(clauses) + 1):
        enc = count_enclosure(mc, order=order)
        assert enc.is_sound and enc.contains(exact), (order, enc.lower, exact, enc.upper)
        assert enc.width <= prev_width + 1e-12, "a higher order must not widen the enclosure"
        prev_width = enc.width
        print(f"  {order:5d} {enc.lower:8.2f} {enc.upper:8.2f} {enc.width:8.2f}"
              f"  {str(enc.tight):>5s}  {enc.contains(exact)}")
    full = count_enclosure(mc, order=len(clauses) + 1)
    assert full.tight and abs(full.lower - exact) < 1e-9, "full order counts exactly"

    print("\n  weighted model counting (per-variable literal weights [w0, w1]):")
    weights = np.array([[1.0, 2.0], [1.0, 1.0], [2.0, 1.0], [1.0, 3.0]])
    mcw = model_count(clauses, weights=weights)
    exact_w = exact_model_count(mcw)
    encw = count_enclosure(mcw, order=len(clauses) + 1)
    print(f"    exact weighted count {exact_w:.4f} in [{encw.lower:.4f}, {encw.upper:.4f}]"
          f"  tight={encw.tight}")
    assert encw.contains(exact_w) and encw.tight

    print("\n  differentiable model finder tightens the certified lower bound:")
    # No-optimizer baseline: the violation of a naive rounded guess (no annealing, no search).
    naive = np.zeros(mc.n)
    baseline_energy = float(mc.energy(naive))
    # beta -> inf annealed relaxation + local search: decodes to a satisfying witness.
    x_soft = np.asarray(sat_relaxation(mc))
    z, annealed_energy = decode(mc, relaxed=x_soft, n_starts=16)
    base_lower = count_enclosure(mc, order=1)  # Bonferroni order-1 only
    with_witness = count_enclosure(mc, order=1, witnesses=np.array([z], dtype=float))
    print(f"    decoded violation: naive guess {baseline_energy:.0f}   "
          f"annealed + local search {annealed_energy:.0f}  (0 == a model)")
    print(f"    order-1 certified lower bound: without witness {base_lower.lower:.0f}   "
          f"with the found model {with_witness.lower:.0f}  (exact {exact:.0f})")
    assert annealed_energy <= baseline_energy, "annealing must not worsen the decoded decision"
    assert annealed_energy == 0.0 and mc.is_model(np.asarray(z, dtype=float)), "found a model"
    assert with_witness.lower >= max(1.0, base_lower.lower), "the witness strengthens the bound"
    assert with_witness.contains(exact)
    print("\n  Reading: the enclosure is a rigorous sandwich (exact-arithmetic inclusion-\n"
          "  exclusion); the differentiable finder supplies a sound witness that raises it.\n")


def main() -> None:
    maxsat_demo()
    sharpsat_demo()
    print("OK: MaxSAT gap and #SAT count enclosure both sandwich the exact oracle; the\n"
          "differentiable finder tightens the certified count lower bound.")


if __name__ == "__main__":
    main()
