# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Submodular function families & constraints -- omnibias-submodular (numpy-only).

Run:

    pip install omnibias-submodular
    python docs/examples/submodular_families_and_constraints.py

A tour of the package *beyond* the coverage / continuous-greedy headline, all keeping the
same honest **yes-if** framing -- a certified approximation with an a-priori ratio, never a
``P = NP`` exactness claim -- except the one problem that genuinely *is* poly-time exact:
unconstrained submodular **minimization** (a P-class result, not a P=NP claim).

1. **Accelerated greedy on a log-det DPP.** ``LogDeterminant`` (``f(S)=logdet(I+K_S)``) is a
   *greedy-path* function: closed-form value + Schur-complement marginals, but no closed-form
   multilinear extension. ``lazy_greedy`` (Minoux / CELF) returns the *identical* set as the
   textbook greedy with far fewer oracle calls; ``stochastic_greedy`` trades an ``epsilon`` for
   speed at ``(1 - 1/e - epsilon)``. Both are certified with the marginal-gain gap.
2. **Budgeted (knapsack) coverage.** A knapsack budget is *not* a matroid; Sviridenko's
   ``knapsack_maximize`` keeps the ``(1 - 1/e)`` guarantee and ``certify_knapsack_gap`` seals
   it with a fractional-knapsack upper bound.
3. **Non-monotone max-cut via double greedy.** ``GraphCut`` is submodular but non-monotone,
   so monotone greedy carries no guarantee; ``double_greedy`` gives the unconstrained
   ``1/3`` (deterministic) / ``1/2`` (randomized) approximation, certified against the sound
   singleton bound.
4. **Exact P-class submodular minimization.** ``submodular_minimize`` (Fujishige-Wolfe
   min-norm-point) returns the *exact* minimizer in polynomial time -- matching brute force.
   This is honest: minimization is P-class, distinct from the NP-hard maximization above.
"""

from __future__ import annotations

import numpy as np
from omnibias.submodular import (
    ONE_MINUS_INV_E,
    Coverage,
    GraphCut,
    KnapsackConstraint,
    LogDeterminant,
    SubmodularProblem,
    UniformMatroid,
    brute_force_max,
    brute_force_max_knapsack,
    budgeted,
    certify_knapsack_gap,
    certify_nonmonotone_gap,
    certify_submodular_gap,
    double_greedy,
    greedy_maximize,
    lazy_greedy,
    lovasz_extension,
    stochastic_greedy,
    submodular_minimize,
)
from omnibias.submodular.functions import SubmodularFunction


def _support(selection: object) -> tuple[int, ...]:
    return tuple(int(i) for i, v in enumerate(np.asarray(selection).reshape(-1)) if v)


def dpp_accelerated_greedy_demo() -> None:
    print("=== 1. accelerated greedy on a log-det DPP (greedy-path; certified gap) ===")
    n, k = 8, 3
    rng = np.random.default_rng(0)
    a = rng.random((n, n))
    kernel = a @ a.T + 0.5 * np.eye(n)  # symmetric positive definite DPP kernel
    f = LogDeterminant(kernel)
    matroid = UniformMatroid(n, k)
    prob = SubmodularProblem(f, matroid)

    greedy_set, greedy_val = greedy_maximize(f, matroid)
    lazy_set, lazy_val = lazy_greedy(f, matroid)
    _, opt = brute_force_max(f, matroid)

    # stochastic greedy is a (1 - 1/e - eps) guarantee *in expectation*; average over seeds.
    eps = 0.1
    stoch_vals = [stochastic_greedy(f, matroid, epsilon=eps, seed=s)[1] for s in range(8)]
    stoch_mean = float(np.mean(stoch_vals))

    cert = certify_submodular_gap(prob, lazy_set)
    print(f"  DPP: n={n}, k={k}")
    print(f"  greedy      set {_support(greedy_set)}   f(S)={greedy_val:.4f}")
    print(f"  lazy (CELF) set {_support(lazy_set)}   f(S)={lazy_val:.4f}  (identical to greedy)")
    print(f"  stochastic  mean f(S)={stoch_mean:.4f} over 8 seeds (eps={eps})")
    print(f"  brute-force OPT={opt:.4f}   certified gap  f(S)={cert.value:.4f} <= OPT <= U(S)={cert.upper_bound:.4f}")

    assert abs(lazy_val - greedy_val) <= 1e-9, "lazy_greedy must reproduce the greedy value"
    assert _support(lazy_set) == _support(greedy_set), "lazy_greedy must reproduce the greedy set"
    assert cert.value <= opt + 1e-9 and opt <= cert.upper_bound + 1e-9, "sandwich must hold"
    assert stoch_mean >= (ONE_MINUS_INV_E - eps) * opt - 1e-9, "(1 - 1/e - eps) must hold in mean"
    print("  Reading: CELF == greedy (bit-identical), stochastic clears (1 - 1/e - eps) OPT.\n")


def budgeted_knapsack_demo() -> None:
    print("=== 2. budgeted (knapsack) coverage -- a knapsack is NOT a matroid ===")
    universe, n_sets = 9, 7
    rng = np.random.default_rng(3)
    membership = (rng.random((universe, n_sets)) < 0.4).astype(float)
    weights = rng.uniform(0.5, 2.0, size=universe)
    f = Coverage(membership, weights)
    costs = rng.uniform(1.0, 3.0, size=n_sets)
    budget = 6.0

    sol = budgeted(f, costs, budget)  # Sviridenko partial-enumeration greedy, (1 - 1/e)
    constraint = KnapsackConstraint(costs, budget)
    cert = certify_knapsack_gap(f, constraint, sol.selection)
    _, opt = brute_force_max_knapsack(f, constraint)
    spent = float(costs @ np.asarray(sol.selection, dtype=float))

    print(f"  weighted coverage: universe={universe}, n_sets={n_sets}, budget={budget}")
    print(f"  chosen sets {_support(sol.selection)}   cost={spent:.2f} <= {budget}   f(S)={sol.value:.4f}")
    print(f"  brute-force OPT={opt:.4f}   certified gap  {cert.value:.4f} <= OPT <= {cert.upper_bound:.4f}")

    assert constraint.is_feasible(sol.selection), "the decoded set must be budget-feasible"
    assert cert.value <= opt + 1e-9 and opt <= cert.upper_bound + 1e-9, "sandwich must hold"
    assert sol.value >= ONE_MINUS_INV_E * opt - 1e-9, "the (1 - 1/e) knapsack guarantee must hold"
    print("  Reading: budget respected; f(S) clears (1 - 1/e) OPT with a certified gap.\n")


def nonmonotone_max_cut_demo() -> None:
    print("=== 3. non-monotone max-cut via double greedy (unconstrained) ===")
    n = 8
    rng = np.random.default_rng(1)
    w = rng.random((n, n))
    f = GraphCut(w + w.T)  # symmetric nonnegative adjacency

    det_set, det_val = double_greedy(f, randomized=False)  # deterministic 1/3
    rnd_vals = [double_greedy(f, randomized=True, seed=s)[1] for s in range(8)]
    rnd_mean = float(np.mean(rnd_vals))
    _, opt = brute_force_max(f, UniformMatroid(n, n))  # unconstrained max over all 2^n subsets
    cert = certify_nonmonotone_gap(f, det_set)  # unconstrained -> a-priori ratio 1/2

    print(f"  max-cut: n={n} (non-monotone: f(empty)=f(V)=0)")
    print(f"  double greedy (det)  cut(S)={det_val:.4f}   set {_support(det_set)}")
    print(f"  double greedy (rand) mean cut={rnd_mean:.4f} over 8 seeds")
    print(f"  brute-force OPT={opt:.4f}   certified gap  {cert.value:.4f} <= OPT <= {cert.upper_bound:.4f}")

    assert det_val >= (1.0 / 3.0) * opt - 1e-9, "deterministic double greedy is a 1/3 approximation"
    assert cert.value <= opt + 1e-9 and opt <= cert.upper_bound + 1e-9, "sandwich must hold"
    assert abs(cert.approx_ratio - 0.5) < 1e-12, "unconstrained double greedy records the 1/2 ratio"
    print("  Reading: monotone greedy has no guarantee here; double greedy clears 1/3 OPT.\n")


class _CutPlusModular(SubmodularFunction):
    """``f(S) = cut_W(S) + a . 1_S`` -- submodular (cut) + modular; a nontrivial minimizer."""

    def __init__(self, weights: np.ndarray, linear: np.ndarray) -> None:
        w = 0.5 * (np.asarray(weights, float) + np.asarray(weights, float).T)
        np.fill_diagonal(w, 0.0)
        self._w = w
        self._a = np.asarray(linear, float).reshape(-1)

    @property
    def n(self) -> int:
        return int(self._w.shape[0])

    def value(self, x: object) -> float:
        xv = np.asarray(x, dtype=float).reshape(-1)
        return float(xv @ self._w @ (1.0 - xv) + self._a @ xv)

    def multilinear(self, p: object) -> float:  # greedy/minimization-path
        raise NotImplementedError

    def to_polynomial(self):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def exact_submodular_minimization_demo() -> None:
    print("=== 4. exact P-class submodular minimization (Fujishige-Wolfe) ===")
    # Three elements pulled in (a=-5), three pushed out (a=+5), weak cut coupling -> min {0,1,2}.
    n = 6
    w = np.full((n, n), 0.1)
    np.fill_diagonal(w, 0.0)
    a = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])
    f = _CutPlusModular(w, a)

    result = submodular_minimize(f)
    # independent brute-force minimum over all 2^n subsets (small n)
    best_val, best_set = np.inf, ()
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        v = float(f.value(x))
        if v < best_val:
            best_val, best_set = v, _support(x)

    # the Lovasz extension agrees with f on the cube (a convex closure of a set function)
    on_vertex = lovasz_extension(f, np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))

    print(f"  min-norm-point minimizer  {result.support}   f={result.value:.4f}")
    print(f"  brute-force minimizer     {best_set}   f={best_val:.4f}")
    print(f"  Lovasz extension on 1_(0,1,2) = {on_vertex:.4f}  (== f on the cube)")

    assert result.value == float(np.round(best_val, 12)) or abs(result.value - best_val) <= 1e-7
    assert result.support == best_set, "the exact minimizer must match brute force"
    assert abs(on_vertex - f.value(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))) <= 1e-9
    print("  Reading: minimization is exact in poly time (P-class) -- not a P=NP claim.\n")


def main() -> None:
    dpp_accelerated_greedy_demo()
    budgeted_knapsack_demo()
    nonmonotone_max_cut_demo()
    exact_submodular_minimization_demo()
    print("OK: accelerated/knapsack/non-monotone maximizers certified; minimization exact.")


if __name__ == "__main__":
    main()
