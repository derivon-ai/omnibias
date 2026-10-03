# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact combinatorial regressions for planar Viro patchworks."""

from dataclasses import replace

import pytest
from omnibias.geometry.patchwork import (
    SignDistribution,
    Triangulation,
    lattice_points,
    patchwork_curve,
    staircase_triangulation,
    target_octic_region_tree,
)


def test_staircase_is_a_complete_unimodular_triangulation():
    quartic = staircase_triangulation(4)
    assert len(quartic.vertices) == 15
    assert len(quartic.triangles) == 16
    assert len(quartic.edges) == 30

    octic = staircase_triangulation(8)
    assert len(octic.vertices) == 45
    assert len(octic.triangles) == 64
    assert len(octic.edges) == 108
    assert sum(
        abs(
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )
        for a, b, c in octic.triangles
    ) == 64


def test_triangulation_rejects_nonunimodular_and_incomplete_inputs():
    with pytest.raises(ValueError, match="unimodular"):
        Triangulation(2, (((0, 0), (2, 0), (0, 1)),) * 4)
    valid = staircase_triangulation(2)
    with pytest.raises(ValueError, match="count"):
        Triangulation(2, valid.triangles[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        Triangulation(2, (*valid.triangles[:-1], valid.triangles[0]))


def test_reflected_signs_obey_viro_quadrant_rules():
    signs = SignDistribution.create(
        4,
        {point: (point[0] + 2 * point[1]) % 2 for point in lattice_points(4)},
    )
    for i, j in lattice_points(4):
        assert signs.reflected_bit((i, -j)) == (signs.bit((i, j)) + j) % 2
        assert signs.reflected_bit((-i, j)) == (signs.bit((i, j)) + i) % 2
        assert signs.reflected_bit((-i, -j)) == (signs.bit((i, j)) + i + j) % 2
    with pytest.raises(ValueError, match="exactly 15 bits"):
        replace(signs, bits=signs.bits[:-1])


def test_quartic_harnack_patchwork_has_four_projective_ovals():
    triangulation = staircase_triangulation(4)
    signs = SignDistribution.create(
        4,
        {point: (point[0] * point[1]) % 2 for point in triangulation.vertices},
    )
    result = patchwork_curve(triangulation, signs)
    assert result.component_count == 4
    assert result.region_count == 5
    assert result.rooted_tree == ((), (), (), ())
    assert result.region_children == ((1, 2, 3, 4), (), (), (), ())
    assert all(
        sum(index in component for component in result.components) == 1
        for index in range(len(result.segments))
    )


def test_degree_mismatch_and_odd_region_tree_contract():
    with pytest.raises(ValueError, match="same degree"):
        patchwork_curve(
            staircase_triangulation(2),
            SignDistribution.create(1, (0, 0, 0)),
        )
    cubic = staircase_triangulation(3)
    result = patchwork_curve(
        cubic,
        SignDistribution.create(3, {point: 0 for point in cubic.vertices}),
    )
    assert result.component_count >= 1
    assert result.region_children == ()
    assert result.rooted_tree == ()


def test_octic_target_tree_is_the_declared_twenty_two_oval_scheme():
    tree = target_octic_region_tree()

    def edges(node):
        return len(node) + sum(edges(child) for child in node)

    assert edges(tree) == 22
    assert sorted(len(child) for child in tree) == [0, 0, 0, 0, 3]
    deep = next(child for child in tree if len(child) == 3)
    assert sorted(len(child) for child in deep) == [0, 0, 14]
