# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from omnibias.geometry.part_a_target import (
    PARTA_SELECTED_TARGET,
    count_octic_ovals,
    sibling_octic_region_tree,
    wide_deep_octic_region_tree,
)


def test_both_open_targets_encode_exactly_22_ovals() -> None:
    assert count_octic_ovals(wide_deep_octic_region_tree()) == 22
    assert count_octic_ovals(sibling_octic_region_tree()) == 22


def test_sibling_tree_differs_from_wide_deep_target() -> None:
    assert sibling_octic_region_tree() != wide_deep_octic_region_tree()


def test_selected_target_is_sibling_narrow() -> None:
    assert PARTA_SELECTED_TARGET.name == "sibling_narrow"
    assert PARTA_SELECTED_TARGET.tree == sibling_octic_region_tree()
    assert PARTA_SELECTED_TARGET.oval_count == 22
