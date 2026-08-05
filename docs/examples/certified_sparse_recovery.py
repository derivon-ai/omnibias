# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified sparse recovery -- omnibias.discrete.sparse.

Run:

    pip install "omnibias-discrete[jax,sos]" omnibias-convex
    python docs/examples/certified_sparse_recovery.py

Best-subset selection (choose the support that minimises a least-squares fit plus an
``l_0`` cardinality penalty) is NP-hard, so no poly-time map yields the *exact* optimum
(that would imply P = NP). The sound object is a **yes-if**: relax -> decode a support ->
certify ``lower <= optimum <= energy``. This deterministic demo exercises three layers,
each explicit about which object its certificate seals:

1. **Fork A -- pseudo-Boolean surrogate (SOS certificate).** The binary ``z in {0,1}^n``
   *is* the coefficient vector, so ``E(z) = 1/2||A z - b||^2 + lambda 1^T z`` is a QUBO
   certified directly by the Lasserre / SOS bound. On a **binary** ground truth this is
   exact best-subset coding: the decoder recovers the true support and the certified gap
   is tight (``lower = optimum = energy``), sandwiching the brute-force optimum.
2. **Fork B -- continuous best-subset (convex certificate).** ``min_w ||A_S w - b||^2 +
   lambda|z|`` fits continuous coefficients on the selected columns; it is *not*
   pseudo-Boolean, so it is certified by a sound convex box-QP bound (sealed by
   ``omnibias-convex``), back-stopped by the always-valid full-OLS-residual floor. The
   bound is loose but rigorous -- a weaker bound only widens the certified gap.
3. **Fork C -- hybrid.** Seal the pseudo-Boolean **surrogate** and ship an OLS refit on
   the decoded support for continuous coefficients.

The headline, and the property that makes this *omnibias* rather than one more heuristic,
is the **certificate**: OMP / Lasso return a support with no optimality guarantee, while
every fork here returns a rigorous sandwich around the true optimum. The ``l_p -> l_0``
penalty-exponent homotopy is a *relaxation* knob (swept below); the exact energy and its
certificate do not involve ``p``, so the seal is ``p``-independent and always sound.

Terminology: both the relaxation's ``sigmoid(beta z)``, ``beta -> inf`` hardening and the
``l_p -> l_0`` penalty-exponent homotopy are the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from omnibias.discrete import (  # noqa: E402
    brute_force_min,
    certify_gap,
    decode,
)
from omnibias.discrete.sparse import (  # noqa: E402
    BestSubsetProblem,
    SupportSelectionProblem,
    certified_sparse_fit,
    certify_best_subset_gap,
    sparse_least_squares,
)
from omnibias.discrete.sparse.jax import sparse_relaxation  # noqa: E402


def _binary_instance(seed: int, n: int = 8, k: int = 3, m: int = 20, noise: float = 0.05):
    """A sparse instance with a **binary** ground truth (the exact-surrogate regime)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n)) / np.sqrt(m)
    support = sorted(int(i) for i in rng.choice(n, k, replace=False))
    z = np.zeros(n)
    z[support] = 1.0
    b = A @ z + noise * rng.standard_normal(m)
    return A, b, set(support)


def _continuous_instance(seed: int, n: int = 7, k: int = 2, m: int = 20, noise: float = 0.03):
    """A sparse instance with **continuous** ground-truth coefficients (Fork B / C)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    support = sorted(int(i) for i in rng.choice(n, k, replace=False))
    x = np.zeros(n)
    x[support] = rng.choice([-1.0, 1.0], k) * rng.uniform(1.5, 3.0, k)
    b = A @ x + noise * rng.standard_normal(m)
    return A, b, set(support), x


def _f1(pred: set[int], true: set[int]) -> float:
    tp = len(pred & true)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(true) if true else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _omp_support(A: np.ndarray, b: np.ndarray, k: int) -> set[int]:
    """Orthogonal matching pursuit -- a named greedy baseline (no optimality guarantee)."""
    residual = b.copy()
    chosen: list[int] = []
    for _ in range(k):
        j = int(np.argmax(np.abs(A.T @ residual)))
        if j in chosen:
            break
        chosen.append(j)
        w, *_ = np.linalg.lstsq(A[:, chosen], b, rcond=None)
        residual = b - A[:, chosen] @ w
    return set(chosen)


def _lasso_support(A: np.ndarray, b: np.ndarray, lam: float, iters: int = 800) -> set[int]:
    """ISTA soft-thresholding for the l1 (Lasso) surrogate -- a named baseline."""
    n = A.shape[1]
    lip = float(np.linalg.norm(A, 2)) ** 2 + 1e-9
    x = np.zeros(n)
    for _ in range(iters):
        x = x - (A.T @ (A @ x - b)) / lip
        thresh = lam / lip
        x = np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)
    return {int(i) for i in np.nonzero(np.abs(x) > 1e-3)[0]}


def certified_sandwich_demo() -> None:
    print("=== 1. certified optimality-gap sandwich (yes-if: a certified gap, never P=NP) ===")
    A, b, _true = _binary_instance(0, n=6, k=2)
    lam = 0.15

    prob = SupportSelectionProblem(A=A, b=b, lam=lam)
    x_soft = np.asarray(sparse_relaxation(prob, p=0.5))  # differentiable l_p relaxation
    assignment, energy = decode(prob, relaxed=x_soft, n_starts=32)  # round + 1-flip -> upper
    _, e_opt = brute_force_min(prob)  # exact optimum (brute force) -- the ground truth
    cert_a = certify_gap(prob, np.array(assignment, float), level=1,
                         claim_label="sparse support-selection energy")
    print("  Fork A (pseudo-Boolean surrogate, SOS lower bound):")
    print(f"    lower {cert_a.lower_bound:8.4f}  <=  optimum {e_opt:8.4f}  <=  energy {cert_a.energy:8.4f}"
          f"   (rel gap {cert_a.relative_gap:.1%}, method={cert_a.method}, sealed={cert_a.sealed is not None})")
    assert cert_a.lower_bound <= e_opt + 1e-6, "SOS lower bound must not exceed the true optimum"
    assert cert_a.energy >= e_opt - 1e-9 and cert_a.is_sound
    assert abs(energy - e_opt) < 1e-9, "the decoder is exact on this small binary instance (tight gap)"

    A2, b2, _t2, _x2 = _continuous_instance(0, n=6, k=2)
    probB = BestSubsetProblem(A=A2, b=b2, lam=lam)
    assignB, energyB = decode(probB, n_starts=32)
    _, e_optB = brute_force_min(probB)
    cert_b = certify_best_subset_gap(probB, np.array(assignB, float))
    print("  Fork B (continuous best-subset, convex lower bound):")
    print(f"    lower {cert_b.lower_bound:8.4f}  <=  optimum {e_optB:8.4f}  <=  energy {cert_b.energy:8.4f}"
          f"   (method={cert_b.method}, certified={cert_b.certified})")
    assert cert_b.lower_bound <= e_optB + 1e-6, "convex lower bound must not exceed the true optimum"
    assert cert_b.is_sound
    print("\n  Reading: both forks sandwich the brute-force optimum. Fork A's SOS bound is tight")
    print("  and hash-sealed; Fork B's convex bound is sound but loose -- a weaker bound only")
    print("  widens the gap, never invalidates it.\n")


def support_recovery_demo() -> None:
    print("=== 2. support recovery on a binary ground truth, vs named baselines ===")
    lam = 0.15
    seeds = range(6)
    ours: list[float] = []
    omp: list[float] = []
    lasso: list[float] = []
    exact_gap = 0
    for seed in seeds:
        A, b, true = _binary_instance(seed)
        prob = sparse_least_squares(A, b, lam)
        x_soft = np.asarray(sparse_relaxation(prob, p=0.5))
        assignment, energy = decode(prob, relaxed=x_soft, n_starts=32)
        _, e_opt = brute_force_min(prob)  # the certified oracle
        support = {i for i, v in enumerate(assignment) if v}
        ours.append(_f1(support, true))
        omp.append(_f1(_omp_support(A, b, len(true)), true))
        lasso.append(_f1(_lasso_support(A, b, lam), true))
        if abs(energy - e_opt) < 1e-9:
            exact_gap += 1
    print(f"  mean support-F1 over {len(list(seeds))} seeds:")
    print(f"    omnibias (certified QUBO)  {np.mean(ours):.3f}")
    print(f"    OMP (greedy)               {np.mean(omp):.3f}")
    print(f"    Lasso (ISTA / l1)          {np.mean(lasso):.3f}")
    print(f"  decoder matched the brute-force optimum (tight certified gap) in {exact_gap}/{len(list(seeds))} seeds.")
    assert np.mean(ours) >= 0.75, "the certified QUBO should recover the binary support well"
    assert exact_gap == len(list(seeds)), "on small n the decoder is exact -> the gap is certified tight"
    print("\n  Reading: recovery is competitive with the greedy/convex baselines, but only")
    print("  omnibias returns a *certificate* -- OMP / Lasso give no optimality guarantee.\n")


def hybrid_fit_demo() -> None:
    print("=== 3. Fork C -- certified surrogate seal + continuous OLS refit ===")
    A, b, true, x_true = _continuous_instance(1, n=6, k=2)
    result = certified_sparse_fit(A, b, lam=0.15, level=1, n_starts=32)
    print(f"  selected support {result.support}   (true support {sorted(true)})")
    print(f"  refit coefficients {np.round(result.coefficients, 3)}")
    print(f"  surrogate certificate: method={result.certificate.method}, sound={result.certificate.is_sound}, "
          f"rel gap {result.certificate.relative_gap:.1%}")
    print(f"  note: {result.note}")
    nz = {int(i) for i in np.nonzero(result.coefficients)[0]}
    assert nz == set(result.support), "coefficients are supported exactly on the decoded support"
    assert result.certificate.is_sound
    print()


def lp_knob_demo() -> None:
    print("=== 4. the l_p -> l_0 relaxation knob (p-independent certificate) ===")
    A, b, _true = _binary_instance(2, n=6, k=2)
    prob = sparse_least_squares(A, b, 0.15)
    supports = []
    print(f"  {'p':>5s}  {'|selected|':>10s}   support   soft assignment stays in [0, 1]")
    for p in (1.0, 0.5, 0.1):  # concave l_p, p -> 0: the penalty-exponent homotopy
        x_soft = np.asarray(sparse_relaxation(prob, p=p))
        assignment, _ = decode(prob, relaxed=x_soft, n_starts=32)
        support = tuple(i for i, v in enumerate(assignment) if v)
        supports.append(support)
        assert np.all(x_soft >= 0.0) and np.all(x_soft <= 1.0), "the soft relaxation lives on the unit box"
        print(f"  {p:5.2f}  {int(sum(assignment)):10d}   {str(support):9s} {bool(np.all((x_soft >= 0) & (x_soft <= 1)))}")
    print("\n  Reading: p is a pure relaxation knob (concave l_p, p -> 0). ``certify_gap`` takes no")
    print("  p argument -- the exact energy and its SOS seal never see p -- so the certified")
    print("  lower bound is p-independent: sweeping p can only change the decoded upper bound.\n")


def main() -> None:
    certified_sandwich_demo()
    support_recovery_demo()
    hybrid_fit_demo()
    lp_knob_demo()
    print("OK: certified sandwiches hold (SOS + convex); decoder exact on small n; refit consistent.")


if __name__ == "__main__":
    main()
