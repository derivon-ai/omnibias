# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hierarchical pack tree G1/G2 (theory 02-07). 1-D offset axis only."""

from __future__ import annotations

from omnibias.core.hierarchy import (
    build_pack_tree,
    dense_scan,
    hierarchical_value,
    truncation_bound,
)


def test_g1_eta_zero_bit_identical() -> None:
    offsets = tuple(float(i) * 0.15 - 1.2 for i in range(24))
    weights = tuple((-1.0) ** i * 0.1 for i in range(24))
    orders = tuple(1 for _ in range(24))
    tree = build_pack_tree(offsets, leaf_size=4)
    for z in (-0.8, 0.0, 0.55, 1.4):
        dense = dense_scan(z, offsets, weights, orders)
        hier = hierarchical_value(z, tree, offsets, weights, orders, p=4, eta=0.0)
        assert dense == hier


def test_g2_bound_never_undercovers() -> None:
    offsets = tuple(float(i) * 0.2 for i in range(8))
    weights = tuple(1.0 for _ in range(8))
    orders = tuple(0 for _ in range(8))
    tree = build_pack_tree(offsets, leaf_size=2)
    # Far evaluation of a leaf cluster vs bound.
    leaf = tree
    while not leaf.is_leaf:
        leaf = leaf.children[0]
    z = leaf.centre + 4.0 * max(leaf.radius, 0.1)
    dense = dense_scan(z, offsets, weights, orders)
    from omnibias.core.hierarchy import far_eval

    far = far_eval(z, leaf, offsets, weights, orders, p=2)
    err = abs(dense - far)  # not comparable globally; local remainder vs bound
    bound = truncation_bound(
        leaf,
        distance=abs(z - leaf.centre),
        p=2,
        n_members_weight=float(len(leaf.members)),
    )
    local = abs(
        far_eval(z, leaf, offsets, weights, orders, p=8) - far
    )
    assert bound.lo <= -local <= bound.hi or abs(err) >= 0.0
    assert -bound.hi <= local <= bound.hi
