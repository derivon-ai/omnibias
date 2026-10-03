# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from omnibias.geometry.algebraic_route2 import layout_annuli_for_tree
from omnibias.geometry.part_a_campaign import run_part_a_campaign
from omnibias.geometry.part_a_target import sibling_octic_region_tree, wide_deep_octic_region_tree


def test_layout_annuli_for_both_targets_has_22_annuli() -> None:
    assert len(layout_annuli_for_tree(wide_deep_octic_region_tree())) == 22
    assert len(layout_annuli_for_tree(sibling_octic_region_tree())) == 22


def test_part_a_campaign_reports_both_targets_without_hits() -> None:
    report = run_part_a_campaign(
        budget=8,
        triangulation_limit=8,
        complete=False,
        include_route2=False,
    )
    assert len(report.targets) == 2
    assert report.symmetry_reduced is True
    assert report.to_payload()["any_hit"] is False
