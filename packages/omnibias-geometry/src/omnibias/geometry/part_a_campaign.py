# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Symmetry-reduced Part-A patchwork and Route-2 campaigns for both (19,3) trees."""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.geometry.algebraic_route2 import (
    Route2LpReport,
    Route2SearchReport,
    layout_annuli_for_tree,
    search_route2_coefficients,
    search_route2_full_coefficients,
    search_route2_lp,
    signs_by_nesting_depth,
)
from omnibias.geometry.part_a_target import PartATargetName, count_octic_ovals, target_tree
from omnibias.geometry.patchwork import RootedTree
from omnibias.geometry.patchwork_search import run_patchwork_discovery

__all__ = [
    "PartACampaignReport",
    "PartATargetCampaignResult",
    "run_part_a_campaign",
]


@dataclass(frozen=True)
class PartATargetCampaignResult:
    target_name: PartATargetName
    scheme: str
    oval_count: int
    patchwork_hit: bool
    route2_sparse_hit: bool
    route2_full_hit: bool
    route2_lp_hit: bool
    patchwork_detail: str
    route2_detail: str
    route2_lp_detail: str

    def to_payload(self) -> dict[str, object]:
        return {
            "target_name": self.target_name,
            "scheme": self.scheme,
            "oval_count": self.oval_count,
            "patchwork_hit": self.patchwork_hit,
            "route2_sparse_hit": self.route2_sparse_hit,
            "route2_full_hit": self.route2_full_hit,
            "route2_lp_hit": self.route2_lp_hit,
            "patchwork_detail": self.patchwork_detail,
            "route2_detail": self.route2_detail,
            "route2_lp_detail": self.route2_lp_detail,
        }


@dataclass(frozen=True)
class PartACampaignReport:
    complete: bool
    targets: tuple[PartATargetCampaignResult, ...]
    symmetry_reduced: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "symmetry_reduced": self.symmetry_reduced,
            "targets": [target.to_payload() for target in self.targets],
            "any_hit": any(
                target.patchwork_hit
                or target.route2_sparse_hit
                or target.route2_full_hit
                or target.route2_lp_hit
                for target in self.targets
            ),
        }


def _run_route2(tree: RootedTree) -> tuple[Route2SearchReport, Route2SearchReport, Route2LpReport]:
    try:
        annuli = layout_annuli_for_tree(tree)
    except ValueError as error:
        message = f"layout failed: {error}"
        skipped = Route2SearchReport(0, 0, False, None, False, message)
        skipped_lp = Route2LpReport(0, 0, 0.0, False, None, False, message)
        return skipped, skipped, skipped_lp
    signs = signs_by_nesting_depth(annuli)
    sparse = search_route2_coefficients(annuli, signs)
    full = search_route2_full_coefficients(annuli, signs, max_support=3, support_pool=15)
    lp = search_route2_lp(annuli, signs)
    return sparse, full, lp


def run_part_a_campaign(
    *,
    budget: int = 64,
    flip_depth: int = 1,
    triangulation_limit: int = 32,
    targets: tuple[PartATargetName, ...] = ("sibling_narrow", "wide_deep"),
    complete: bool = False,
    include_route2: bool = True,
) -> PartACampaignReport:
    """Run bounded patchwork and Route-2 searches on both corrected (19,3) trees."""
    results: list[PartATargetCampaignResult] = []
    for name in targets:
        tree = target_tree(name)
        ovals = count_octic_ovals(tree)
        discovery = run_patchwork_discovery(
            degree=8,
            budget=budget,
            flip_depth=flip_depth,
            triangulation_limit=triangulation_limit,
            target=tree,
        )
        result = discovery["result"]
        patchwork_hit = getattr(result, "status", None) == "PROVED"
        if include_route2:
            sparse, full, lp = _run_route2(tree)
        else:
            sparse = Route2SearchReport(0, 0, False, None, False, "route2 skipped")
            full = sparse
            lp = Route2LpReport(0, 0, 0.0, False, None, False, "route2 lp skipped")
        scheme = "14 + 1<2 + 1<4>>" if name == "sibling_narrow" else "4 + 1<2 + 1<14>>"
        results.append(
            PartATargetCampaignResult(
                name,
                scheme,
                ovals,
                patchwork_hit,
                sparse.candidate_found,
                full.candidate_found,
                lp.candidate_found and lp.curve_certificate_passed,
                getattr(result, "detail", "patchwork search finished"),
                full.detail,
                lp.detail,
            )
        )
    return PartACampaignReport(complete, tuple(results), True)
