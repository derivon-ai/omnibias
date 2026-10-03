# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Combinatorics tests for :mod:`omnibias.core.contraction` (pure Python).

These pin the pure-Python bookkeeping the ``deep_field_laplacian`` /
``deep_field_polylaplacian`` kernels in ``omnibias.jax.laplacian`` /
``omnibias.torch.laplacian`` build on, independent of either backend.
"""

from __future__ import annotations

from itertools import product
from math import comb, factorial

import pytest
from omnibias.core.contraction import (
    DEFAULT_SUPPORT_BUDGET,
    support_jet_count,
    polylaplacian_support_terms,
    polylaplacian_multinomial_terms,
    polylaplacian_normalizer,
    select_mode,
)
from omnibias.core.multi_index import multi_index_factorial


# -- polylaplacian_normalizer ---------------------------------------------- #


def test_normalizer_k1_collapses_to_dimension() -> None:
    """``Delta f = D * E_v[partial_v^2 f]`` at k=1, for every D."""
    for dim in (1, 2, 3, 8, 30):
        assert polylaplacian_normalizer(dim, 1) == pytest.approx(float(dim))


def test_normalizer_matches_hand_derivation_k2() -> None:
    """``Delta^2 f = D(D+2) E_v[partial_v^4 f] / 3`` (k=2 hand derivation)."""
    for dim in (1, 2, 3, 5, 10):
        want = dim * (dim + 2) / 3.0
        assert polylaplacian_normalizer(dim, 2) == pytest.approx(want)


def test_normalizer_matches_hand_derivation_k3() -> None:
    """``Delta^3 f`` normaliser at k=3: ``D(D+2)(D+4) / 15``."""
    for dim in (1, 2, 3, 5):
        want = dim * (dim + 2) * (dim + 4) / 15.0
        assert polylaplacian_normalizer(dim, 3) == pytest.approx(want)


def test_normalizer_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        polylaplacian_normalizer(0, 1)
    with pytest.raises(ValueError, match="k must be"):
        polylaplacian_normalizer(3, 0)


# -- polylaplacian_multinomial_terms ---------------------------------------- #


def test_multinomial_terms_sum_of_coeffs_matches_multinomial_theorem() -> None:
    """sum_beta (k! / beta!) == n_spatial ** k (the multinomial theorem at x_i=1)."""
    for n_spatial in (1, 2, 3, 5):
        for k in (1, 2, 3, 4):
            terms = polylaplacian_multinomial_terms(n_spatial, k)
            total = sum(coeff for _, coeff in terms)
            assert total == n_spatial**k


def test_multinomial_terms_every_beta_sums_to_k() -> None:
    for n_spatial in (1, 3, 4):
        for k in (1, 2, 3):
            for beta, coeff in polylaplacian_multinomial_terms(n_spatial, k):
                assert len(beta) == n_spatial
                assert sum(beta) == k
                assert coeff == factorial(k) // multi_index_factorial(beta)


def test_multinomial_terms_k1_is_identity_expansion() -> None:
    """k=1: Delta f = sum_i d_i^2 f, one term per axis, coefficient 1."""
    terms = polylaplacian_multinomial_terms(4, 1)
    assert len(terms) == 4
    for beta, coeff in terms:
        assert coeff == 1
        assert sum(beta) == 1


def test_multinomial_terms_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_spatial must be"):
        polylaplacian_multinomial_terms(0, 1)
    with pytest.raises(ValueError, match="k must be"):
        polylaplacian_multinomial_terms(2, 0)


# -- support_jet_count / polylaplacian_support_terms --------------------------- #


def test_support_jet_count_matches_definition() -> None:
    for dim in (1, 2, 3, 5, 8):
        for k in (1, 2, 3):
            want = sum(comb(dim, s) for s in range(1, min(k, dim) + 1))
            assert support_jet_count(dim, k) == want


def test_support_jet_count_grows_polynomially_not_combinatorially() -> None:
    """At fixed k, support_jet_count(dim, k) is O(dim^k) -- unlike comb(dim+2k, dim)."""
    k = 2
    small = support_jet_count(10, k)
    large = support_jet_count(100, k)
    # O(dim^2): ratio should track (100/10)^2 = 100, not blow up combinatorially.
    assert large / small < 200


def test_support_jet_count_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        support_jet_count(0, 1)
    with pytest.raises(ValueError, match="k must be"):
        support_jet_count(3, 0)


def test_support_terms_reconstructs_the_multinomial_expansion() -> None:
    """Summing the grouped (support, terms) pairs must reproduce Delta^k exactly.

    Uses a toy quadratic-form stand-in: for a diagonal "Hessian-of-Hessian"
    tensor where ``D^(2 alpha) f`` is replaced by a simple product
    ``prod_i r_i^(2 a_i)`` (as if ``f`` were separable, ``f = prod_i g(x_i)``
    with ``g^{(2a_i)} = r_i^{a_i}``), the grouped support sum must equal the
    flat multinomial sum over all ``|a| = k`` multi-indices exactly.
    """
    import random

    rng = random.Random(0)
    for dim in (2, 3, 4):
        for k in (1, 2, 3):
            r = [rng.uniform(0.5, 2.0) for _ in range(dim)]

            # Flat oracle: sum over every multi-index a with |a|=k.
            flat_total = 0.0
            for a in product(range(k + 1), repeat=dim):
                if sum(a) != k:
                    continue
                coeff = factorial(k) // multi_index_factorial(a)
                term = coeff
                for i, ai in enumerate(a):
                    term *= r[i] ** ai
                flat_total += term

            # Grouped total via polylaplacian_support_terms.
            grouped_total = 0.0
            for support, terms in polylaplacian_support_terms(dim, k):
                for a_local, coeff in terms:
                    term = coeff
                    for axis, ai in zip(support, a_local, strict=True):
                        term *= r[axis] ** ai
                    grouped_total += term

            assert grouped_total == pytest.approx(flat_total, rel=1e-12), (
                f"dim={dim} k={k}"
            )


def test_support_terms_support_sets_are_disjoint_and_bounded() -> None:
    for dim in (3, 5):
        for k in (1, 2, 3):
            groups = polylaplacian_support_terms(dim, k)
            seen_supports = set()
            for support, terms in groups:
                assert support not in seen_supports
                seen_supports.add(support)
                assert len(support) <= min(k, dim)
                assert list(support) == sorted(support)
                for a_local, _coeff in terms:
                    assert len(a_local) == len(support)
                    assert all(v >= 1 for v in a_local)
                    assert sum(a_local) == k


def test_support_terms_is_cached() -> None:
    a = polylaplacian_support_terms(4, 2)
    b = polylaplacian_support_terms(4, 2)
    assert a is b


def test_support_terms_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        polylaplacian_support_terms(0, 1)
    with pytest.raises(ValueError, match="k must be"):
        polylaplacian_support_terms(3, 0)


# -- select_mode ------------------------------------------------------------- #


def test_select_mode_k1_is_always_forward() -> None:
    for dim in (1, 2, 100, 5000):
        assert select_mode(dim, 1) == "forward"


def test_select_mode_picks_support_within_budget() -> None:
    # dim=4, k=2: support_jet_count = comb(4,1) + comb(4,2) = 4 + 6 = 10, tiny.
    assert select_mode(4, 2, budget=DEFAULT_SUPPORT_BUDGET) == "support"


def test_select_mode_picks_estimator_beyond_budget() -> None:
    # A large dim/k combination whose support_jet_count exceeds a small budget.
    assert select_mode(200, 3, budget=100) == "estimator"


def test_select_mode_boundary_is_inclusive() -> None:
    dim, k = 6, 2
    size = support_jet_count(dim, k)
    assert select_mode(dim, k, budget=size) == "support"
    assert select_mode(dim, k, budget=size - 1) == "estimator"


def test_select_mode_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="dim must be"):
        select_mode(0, 1)
    with pytest.raises(ValueError, match="k must be"):
        select_mode(3, 0)
    with pytest.raises(ValueError, match="budget must be"):
        select_mode(3, 2, budget=0)
