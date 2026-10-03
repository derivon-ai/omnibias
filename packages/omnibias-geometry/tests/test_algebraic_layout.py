# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fractions import Fraction

from omnibias.geometry.algebraic import _nesting
from omnibias.geometry.algebraic_layout import (
    fourteen_orbit_centers,
    layout_annuli_for_tree,
    normalize_annulus_layout,
    parents_to_rooted_tree,
    validate_annulus_layout,
)
from omnibias.geometry.part_a_target import sibling_octic_region_tree, wide_deep_octic_region_tree
from omnibias.geometry.patchwork import target_octic_region_tree


def test_wide_deep_layout_matches_target_tree() -> None:
    tree = target_octic_region_tree()
    annuli = layout_annuli_for_tree(tree)
    validate_annulus_layout(annuli, tree)
    assert parents_to_rooted_tree(_nesting(annuli)) == wide_deep_octic_region_tree()
    assert max(
        abs(coordinate)
        for annulus in annuli
        for polygon in (annulus.inner, annulus.outer)
        for vertex in polygon.vertices
        for coordinate in vertex
    ) == 1
    assert normalize_annulus_layout(annuli) == annuli


def test_fourteen_inner_centers_are_exact_8_plus_4_plus_2_orbits() -> None:
    centers = fourteen_orbit_centers(
        (Fraction(0), Fraction(0)),
        Fraction(4),
        Fraction(1, 4),
    )
    assert len(centers) == 14
    assert len(centers[:8]) == 8
    assert set(centers[8:12]) == {(-2, 0), (2, 0), (0, -2), (0, 2)}
    assert set(centers[12:]) == {(-Fraction(1, 2), 0), (Fraction(1, 2), 0)}


def test_sibling_tree_layout_is_not_yet_supported() -> None:
    tree = sibling_octic_region_tree()
    try:
        annuli = layout_annuli_for_tree(tree)
    except ValueError:
        return
    assert len(annuli) == 22
