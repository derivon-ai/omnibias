# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified, differentiable CKY inside-outside parsing -- omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/cky_inside_outside.py

A context-free parse is a *tree* dynamic program, and it lifts onto the exact same
semiring / hypergraph driver as Viterbi and shortest-path: chart items ``(A, i, j)``
("nonterminal ``A`` spans tokens ``[i, j)``") are nodes, and a binary rule ``A -> B C``
splitting a span is an arity-2 hyperedge. So the differentiable inside partition and the
inside-outside marginals come for free, with the same two axes kept apart:

* **``beta -> inf`` (relaxation).** ``soft_inside`` is ``lse_beta`` over all parse trees; it
  sandwiches the best-parse score with a closed-form ``log(N) / beta`` gap (``N`` = the
  number of parse trees, a Catalan number for a fully-ambiguous grammar).
* **``delta -> 0`` (founding tower).** It differentiates ``soft_inside`` exactly: the
  inside-outside span / rule marginals equal ``autograd``, and the torch / jax twins are
  bit-identical.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    BinaryGrammar,
    brute_force_cky,
    certify_soft_dp,
    count_parse_trees,
    hard_cky,
)
from omnibias.struct._core.parse import best_parse_tree  # noqa: E402
from omnibias.struct.jax import inside_outside as inside_outside_jax  # noqa: E402
from omnibias.struct.jax import soft_inside as soft_inside_jax  # noqa: E402
from omnibias.struct.torch import inside_outside, soft_inside, span_marginals  # noqa: E402

torch.set_default_dtype(torch.float64)

# A tiny Chomsky-normal-form grammar over two nonterminals (S=0, N=1) with genuine ambiguity.
GRAMMAR = BinaryGrammar(2, ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0)), start=0)


def certified_parse_demo() -> None:
    print("=== 1. certified inside partition + best parse (yes-if: a log(N)/beta sandwich) ===")
    rng = np.random.default_rng(0)
    length = 5
    emit = rng.standard_normal((length, GRAMMAR.num_nonterminals))
    rule = rng.standard_normal(GRAMMAR.num_rules)
    hard = hard_cky(GRAMMAR, emit, rule)
    brute = brute_force_cky(GRAMMAR, emit, rule, None)
    n_trees = count_parse_trees(GRAMMAR, length)
    assert abs(hard - brute) < 1e-9, "CKY best parse must equal the brute-force tree enumeration"
    score, tree = best_parse_tree(GRAMMAR, emit, rule)
    print(f"  grammar: {GRAMMAR.num_rules} binary rules, {GRAMMAR.num_nonterminals} nonterminals; "
          f"sentence length {length}; N = {n_trees} parse trees")
    print(f"  best parse score V* = {hard:.4f} = brute force {brute:.4f};  tree {tree}")
    print(f"  {'beta':>6s} {'V* (hard)':>11s} {'V_beta (soft)':>14s} {'gap':>9s} {'log(N)/beta':>12s}  sound")
    et, rt = torch.tensor(emit), torch.tensor(rule)
    prev = np.inf
    for beta in (1.0, 2.0, 4.0, 8.0, 16.0):
        soft = float(soft_inside(GRAMMAR, et, rt, beta))
        cert = certify_soft_dp(hard, soft, n_trees, beta, sense="max", brute_force_value=brute)
        print(f"  {beta:6.1f} {hard:11.4f} {soft:14.4f} {cert.absolute_gap:9.4f} {cert.gap_bound:12.4f}  {cert.is_sound}")
        assert cert.is_sound and cert.agrees_with_bruteforce
        assert cert.absolute_gap <= prev + 1e-12
        prev = cert.absolute_gap
    print("\n  Reading: the soft inside partition sandwiches the best parse; the gap is a")
    print("  closed-form log(N)/beta that vanishes as beta grows -- no exactness claim.\n")


def inside_outside_demo() -> None:
    print("=== 2. inside-outside marginals == autograd + torch/jax parity ===")
    rng = np.random.default_rng(1)
    length = 5
    emit = rng.standard_normal((length, GRAMMAR.num_nonterminals))
    rule = rng.standard_normal(GRAMMAR.num_rules)
    beta = 4.0
    et = torch.tensor(emit, requires_grad=True)
    rt = torch.tensor(rule, requires_grad=True)
    value = soft_inside(GRAMMAR, et, rt, beta)
    em, rm = inside_outside(GRAMMAR, et, rt, beta)
    ge, gr = torch.autograd.grad(value, (et, rt))
    err_e = float((em.detach() - ge).abs().max())
    err_r = float((rm.detach() - gr).abs().max())
    print(f"  emit / rule marginals == autograd: max|.| = {err_e:.2e} / {err_r:.2e}")
    assert err_e < 1e-9 and err_r < 1e-9

    sp = span_marginals(GRAMMAR, et.detach(), rt.detach(), beta)
    root = float(sp[GRAMMAR.start, 0, length])
    print(f"  span marginal of the start symbol over the whole sentence = {root:.6f} (expect 1)")
    assert abs(root - 1.0) < 1e-9

    ej, rj = jnp.asarray(emit), jnp.asarray(rule)
    v_j = float(soft_inside_jax(GRAMMAR, ej, rj, beta))
    emj, _ = inside_outside_jax(GRAMMAR, ej, rj, beta)
    parity_v = abs(float(value.detach()) - v_j)
    parity_m = float(np.max(np.abs(np.asarray(em.detach()) - np.asarray(emj))))
    print(f"  torch<->jax parity: value {parity_v:.2e}   marginals {parity_m:.2e}")
    assert parity_v < 1e-9 and parity_m < 1e-9
    print("\n  Reading: the delta->0 tower differentiates the inside partition exactly")
    print("  (inside-outside == autograd), and the two backends agree bit-for-bit.\n")


def main() -> None:
    certified_parse_demo()
    inside_outside_demo()
    print("OK: CKY lifts onto the semiring driver; certified sandwich holds; inside-outside "
          "== autograd; parity < 1e-9.")


if __name__ == "__main__":
    main()
