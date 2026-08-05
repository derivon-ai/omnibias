# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact, differentiable non-projective dependency marginals (Matrix-Tree) -- omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/matrix_tree_nonprojective.py

Non-projective dependency parsing drops the no-crossing constraint, so the parses are *all*
spanning arborescences of the dense arc graph -- exponentially many. The key difference from
every other DP in this package: the partition is **exact and closed form**. By Tutte's
directed Matrix-Tree Theorem the weighted sum over all arborescences is a single determinant,

    Z(beta) = det L(beta),   L = Kirchhoff Laplacian of exp(beta * arc),

so ``soft_matrix_tree = log Z / beta`` is exact at finite beta (no ``lse_beta`` relaxation of
a ``max``), and the arc marginals come from ``L^{-1}``:

* **``beta -> inf`` (relaxation).** ``soft_matrix_tree`` decreases to the Chu-Liu/Edmonds
  maximum-arborescence score; the ``log(N) / beta`` gap is taken against that maximum
  (``N = (n + 1)^(n - 1)`` arborescences, Cayley). Here the soft value is exact, so the
  sandwich is only the beta-annealing gap.
* **``delta -> 0`` (founding tower).** It differentiates ``soft_matrix_tree`` exactly: the
  arc marginals equal ``autograd``, each modifier's head-column sums to ``1``, and the
  torch / jax twins are bit-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    brute_force_arborescence,
    certify_soft_dp,
    count_arborescences,
    hard_matrix_tree,
    matrix_tree_partition,
    max_arborescence,
)
from omnibias.struct.jax import matrix_tree_marginals as marginals_jax  # noqa: E402
from omnibias.struct.jax import soft_matrix_tree as soft_matrix_tree_jax  # noqa: E402
from omnibias.struct.torch import matrix_tree_marginals, soft_matrix_tree  # noqa: E402

torch.set_default_dtype(torch.float64)


def _arc_scores(seed: int, n_words: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_words + 1, n_words + 1))
    a[:, 0] = 0.0  # ROOT is never a modifier
    return a


def exact_partition_demo() -> None:
    print("=== 1. EXACT determinant partition (== brute force) + max arborescence ===")
    n_words = 5
    arc = _arc_scores(0, n_words)
    n_trees = count_arborescences(n_words)
    hard = hard_matrix_tree(arc)
    brute_max = brute_force_arborescence(arc, None)
    score, heads = max_arborescence(arc)
    assert abs(hard - brute_max) < 1e-9, "Chu-Liu/Edmonds must equal the brute-force maximum"
    print(f"  sentence: {n_words} words + ROOT;  N = {n_trees} spanning arborescences = (n+1)^(n-1)")
    print(f"  max arborescence score V* = {hard:.4f} = brute force {brute_max:.4f};  heads {heads}")
    print(f"  {'beta':>6s} {'Z=det L (soft)':>15s} {'brute-force lse':>16s} {'|diff|':>10s}")
    at = torch.tensor(arc)
    for beta in (1.0, 2.0, 4.0, 8.0):
        soft = float(soft_matrix_tree(at, beta))
        ref = brute_force_arborescence(arc, beta)
        core = matrix_tree_partition(arc, beta)
        print(f"  {beta:6.1f} {soft:15.6f} {ref:16.6f} {abs(soft - ref):10.2e}")
        assert abs(soft - ref) < 1e-9 and abs(core - ref) < 1e-9
    print("\n  Reading: unlike the lse_beta DPs, det L is the EXACT sum over arborescences --")
    print("  soft == brute force to machine precision. No relaxation error at finite beta.\n")


def certified_gap_demo() -> None:
    print("=== 2. beta->inf gap against the maximum arborescence (log(N)/beta) ===")
    n_words = 5
    arc = _arc_scores(2, n_words)
    n_trees = count_arborescences(n_words)
    hard = hard_matrix_tree(arc)
    brute_max = brute_force_arborescence(arc, None)
    print(f"  {'beta':>6s} {'V* (max)':>10s} {'V_beta':>12s} {'gap':>9s} {'log(N)/beta':>12s}  sound")
    at = torch.tensor(arc)
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(soft_matrix_tree(at, beta))
        cert = certify_soft_dp(hard, soft, n_trees, beta, sense="max", brute_force_value=brute_max)
        print(f"  {beta:6.1f} {hard:10.4f} {soft:12.4f} {cert.absolute_gap:9.4f} {cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap
    print()


def arc_marginals_demo() -> None:
    print("=== 3. closed-form L^-1 arc marginals == autograd + torch/jax parity ===")
    n_words = 5
    arc = _arc_scores(1, n_words)
    beta = 4.0
    at = torch.tensor(arc, requires_grad=True)
    value = soft_matrix_tree(at, beta)
    mu = matrix_tree_marginals(at, beta)
    (g,) = torch.autograd.grad(value, at)
    err = float((mu.detach() - g).abs().max())
    print(f"  arc marginals == autograd: max|.| = {err:.2e}")
    assert err < 1e-9

    cols = mu.detach().numpy()[:, 1:].sum(axis=0)
    print(f"  every modifier's head-column sums to 1: {np.array2string(cols, precision=6)}")
    assert np.allclose(cols, 1.0, atol=1e-9)

    aj = jax.numpy.asarray(arc)
    v_j = float(soft_matrix_tree_jax(aj, beta))
    mj = np.asarray(marginals_jax(aj, beta))
    parity_v = abs(float(value.detach()) - v_j)
    parity_m = float(np.max(np.abs(mu.detach().numpy() - mj)))
    print(f"  torch<->jax parity: value {parity_v:.2e}   marginals {parity_m:.2e}")
    assert parity_v < 1e-9 and parity_m < 1e-9
    print("\n  Reading: the delta->0 tower differentiates the determinant partition exactly")
    print("  (L^-1 marginals == autograd), and the two backends agree bit-for-bit.\n")


def main() -> None:
    exact_partition_demo()
    certified_gap_demo()
    arc_marginals_demo()
    print("OK: matrix-tree partition is exact (det == brute force); Chu-Liu/Edmonds max holds; "
          "L^-1 marginals == autograd; parity < 1e-9.")


if __name__ == "__main__":
    main()
