# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""CKY inside-outside parsing on the semiring driver.

Oracles: the derivation count is the Catalan number for the fully-ambiguous grammar; the
classic CKY ``hard_cky`` equals the driver's ``hard_value`` and the flat brute-force tree
enumeration. Differentiable: ``soft_inside`` equals the flat soft oracle to ``< 1e-12``,
``inside_outside`` equals ``autograd``, span marginals of the root are ``1``, and the
torch <-> jax twins are bit-identical. The soft value is certified within ``log(N) / beta``
of the best parse.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest
from omnibias.struct import (
    BinaryGrammar,
    brute_force_cky,
    build_chart,
    certify_soft_dp,
    count_parse_trees,
    hard_cky,
)
from omnibias.struct._core.parse import best_parse_tree, chart_edge_weights
from omnibias.struct._core.semiring import hard_value

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.struct.jax as sj  # noqa: E402
import omnibias.struct.torch as st  # noqa: E402

torch.set_default_dtype(torch.float64)

SEEDS = range(5)
TOL = 1e-12
# A 2-nonterminal CNF grammar with genuine ambiguity.
GRAMMAR = BinaryGrammar(2, ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, 0)), start=0)


def _scores(seed: int, length: int):  # noqa: ANN202
    rng = np.random.default_rng(seed)
    return rng.standard_normal((length, GRAMMAR.num_nonterminals)), rng.standard_normal(GRAMMAR.num_rules)


@pytest.mark.parametrize("length", range(1, 8))
def test_derivation_count_is_catalan(length: int) -> None:
    # S -> S S with a single nonterminal: #trees of length L is the Catalan number C_{L-1}.
    grammar = BinaryGrammar(1, ((0, 0, 0),), start=0)
    catalan = comb(2 * (length - 1), length - 1) // length
    assert count_parse_trees(grammar, length) == catalan


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_cky_matches_driver_and_bruteforce(seed: int) -> None:
    emit, rule = _scores(seed, 4)
    spec = build_chart(GRAMMAR, 4)
    w = chart_edge_weights(spec, emit, rule)
    hc = hard_cky(GRAMMAR, emit, rule)
    assert abs(hc - hard_value(spec.graph, w)) < TOL
    assert abs(hc - brute_force_cky(GRAMMAR, emit, rule, None)) < 1e-11


@pytest.mark.parametrize("seed", SEEDS)
def test_best_parse_tree_score_and_coverage(seed: int) -> None:
    emit, rule = _scores(seed, 5)
    score, tree = best_parse_tree(GRAMMAR, emit, rule)
    assert abs(score - hard_cky(GRAMMAR, emit, rule)) < TOL

    leaves: list[int] = []

    def walk(node: tuple) -> None:
        if len(node) == 2:
            leaves.append(node[1])
        else:
            walk(node[3])
            walk(node[4])

    walk(tree)
    assert leaves == list(range(5))  # the tree covers every token exactly once, in order


@pytest.mark.parametrize("seed", SEEDS)
def test_soft_inside_matches_bruteforce(seed: int) -> None:
    emit, rule = _scores(seed, 4)
    et, rt = torch.tensor(emit), torch.tensor(rule)
    ej, rj = jnp.asarray(emit), jnp.asarray(rule)
    for beta in (1.0, 8.0, 64.0):
        ref = brute_force_cky(GRAMMAR, emit, rule, beta)
        assert abs(float(st.soft_inside(GRAMMAR, et, rt, beta)) - ref) < 1e-11
        assert abs(float(sj.soft_inside(GRAMMAR, ej, rj, beta)) - ref) < 1e-11


@pytest.mark.parametrize("seed", SEEDS)
def test_inside_outside_equals_autograd(seed: int) -> None:
    emit, rule = _scores(seed, 4)
    et = torch.tensor(emit, requires_grad=True)
    rt = torch.tensor(rule, requires_grad=True)
    beta = 6.0
    value = st.soft_inside(GRAMMAR, et, rt, beta)
    em, rm = st.inside_outside(GRAMMAR, et, rt, beta)
    ge, gr = torch.autograd.grad(value, (et, rt))
    assert float((em.detach() - ge).abs().max()) < 1e-9
    assert float((rm.detach() - gr).abs().max()) < 1e-9


def test_span_marginals_root_is_one() -> None:
    emit, rule = _scores(0, 5)
    sp = st.span_marginals(GRAMMAR, torch.tensor(emit), torch.tensor(rule), 4.0)
    assert abs(float(sp[GRAMMAR.start, 0, 5]) - 1.0) < 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_certified_gap_is_sound(seed: int) -> None:
    emit, rule = _scores(seed, 4)
    et, rt = torch.tensor(emit), torch.tensor(rule)
    for beta in (2.0, 32.0):
        hard = hard_cky(GRAMMAR, emit, rule)
        soft = float(st.soft_inside(GRAMMAR, et, rt, beta))
        cert = certify_soft_dp(hard, soft, count_parse_trees(GRAMMAR, 4), beta, sense="max")
        assert cert.is_sound
        assert cert.absolute_gap <= cert.gap_bound + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_torch_jax_parity(seed: int) -> None:
    emit, rule = _scores(seed, 4)
    et, rt = torch.tensor(emit), torch.tensor(rule)
    ej, rj = jnp.asarray(emit), jnp.asarray(rule)
    for beta in (1.0, 8.0):
        assert abs(float(st.soft_inside(GRAMMAR, et, rt, beta)) - float(sj.soft_inside(GRAMMAR, ej, rj, beta))) < 1e-11
        emt, rmt = st.inside_outside(GRAMMAR, et, rt, beta)
        emj, rmj = sj.inside_outside(GRAMMAR, ej, rj, beta)
        assert float(np.max(np.abs(np.asarray(emt) - np.asarray(emj)))) < 1e-11
        assert float(np.max(np.abs(np.asarray(rmt) - np.asarray(rmj)))) < 1e-11
