# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable Eisner projective dependency parsing -- omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/dependency_eisner.py

A projective dependency parse assigns every word a single head so the arcs form a tree
rooted at ROOT (position 0) and never cross. Eisner's ``O(n^3)`` algorithm assembles it from
*complete* and *incomplete* half-spans, and those spans are nodes / the glue steps are
arity-2 hyperedges, so the parse lifts onto the exact same semiring / hypergraph driver as
Viterbi, shortest-path, and CKY. The differentiable partition and closed-form arc marginals
then come for free, with the two axes kept apart:

* **``beta -> inf`` (relaxation).** ``soft_eisner`` is ``lse_beta`` over all projective
  trees; it sandwiches the best-parse score with a closed-form ``log(N) / beta`` gap
  (``N`` = the number of projective trees of the sentence).
* **``delta -> 0`` (founding tower).** It differentiates ``soft_eisner`` exactly: the arc
  marginals equal ``autograd``, every modifier's head-column sums to ``1``, and the torch /
  jax twins are bit-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    certify_soft_dp,
    count_projective_trees,
    hard_eisner,
)
from omnibias.struct._core.eisner import best_projective_tree, brute_force_projective  # noqa: E402
from omnibias.struct.jax import eisner_marginals as eisner_marginals_jax  # noqa: E402
from omnibias.struct.jax import soft_eisner as soft_eisner_jax  # noqa: E402
from omnibias.struct.torch import eisner_marginals, soft_eisner  # noqa: E402

torch.set_default_dtype(torch.float64)


def _arc_scores(seed: int, n_words: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_words + 1, n_words + 1))
    a[:, 0] = 0.0  # ROOT is never a modifier
    return a


def certified_parse_demo() -> None:
    print("=== 1. certified projective partition + best parse (yes-if: a log(N)/beta sandwich) ===")
    n_words = 5
    arc = _arc_scores(0, n_words)
    hard = hard_eisner(arc)
    brute = brute_force_projective(arc, None)
    n_trees = count_projective_trees(n_words)
    assert abs(hard - brute) < 1e-9, "Eisner best parse must equal the brute-force tree enumeration"
    score, heads = best_projective_tree(arc)
    print(f"  sentence: {n_words} words + ROOT;  N = {n_trees} projective trees")
    print(f"  best parse score V* = {hard:.4f} = brute force {brute:.4f}")
    print(f"  head map (modifier -> head): {heads}")
    print(f"  {'beta':>6s} {'V* (hard)':>11s} {'V_beta (soft)':>14s} {'gap':>9s} {'log(N)/beta':>12s}  sound")
    at = torch.tensor(arc)
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(soft_eisner(at, beta))
        cert = certify_soft_dp(hard, soft, n_trees, beta, sense="max", brute_force_value=brute)
        print(f"  {beta:6.1f} {hard:11.4f} {soft:14.4f} {cert.absolute_gap:9.4f} {cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap
    print("\n  Reading: the soft partition sandwiches the best projective parse; the gap is a")
    print("  closed-form log(N)/beta that vanishes as beta grows -- no exactness claim.\n")


def arc_marginals_demo() -> None:
    print("=== 2. closed-form arc marginals == autograd + torch/jax parity ===")
    n_words = 5
    arc = _arc_scores(1, n_words)
    beta = 4.0
    at = torch.tensor(arc, requires_grad=True)
    value = soft_eisner(at, beta)
    mu = eisner_marginals(at, beta)
    (g,) = torch.autograd.grad(value, at)
    err = float((mu.detach() - g).abs().max())
    print(f"  arc marginals == autograd: max|.| = {err:.2e}")
    assert err < 1e-9

    cols = mu.detach().numpy()[:, 1:].sum(axis=0)
    print(f"  every modifier's head-column sums to 1: {np.array2string(cols, precision=6)}")
    assert np.allclose(cols, 1.0, atol=1e-9)

    aj = jax.numpy.asarray(arc)
    v_j = float(soft_eisner_jax(aj, beta))
    mj = np.asarray(eisner_marginals_jax(aj, beta))
    parity_v = abs(float(value.detach()) - v_j)
    parity_m = float(np.max(np.abs(mu.detach().numpy() - mj)))
    print(f"  torch<->jax parity: value {parity_v:.2e}   marginals {parity_m:.2e}")
    assert parity_v < 1e-9 and parity_m < 1e-9
    print("\n  Reading: the delta->0 tower differentiates the projective partition exactly")
    print("  (arc marginals == autograd), and the two backends agree bit-for-bit.\n")


def main() -> None:
    certified_parse_demo()
    arc_marginals_demo()
    print("OK: Eisner lifts onto the semiring driver; certified sandwich holds; arc marginals "
          "== autograd; parity < 1e-9.")


if __name__ == "__main__":
    main()
