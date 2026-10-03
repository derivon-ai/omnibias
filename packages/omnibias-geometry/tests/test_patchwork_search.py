# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable proposal and exact patchwork-acceptance regressions."""

from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.catalog import catalog_entry
from omnibias.core.proof.discovery import OneHotAnneal, run_discovery
from omnibias.geometry.algebraic import RationalRectangle, RectangularAnnulus
from omnibias.geometry.part_a_target import (
    sibling_octic_region_tree,
    wide_deep_octic_region_tree,
)
from omnibias.geometry.patchwork import (
    SignDistribution,
    flip_neighbors,
    staircase_triangulation,
)
from omnibias.geometry.patchwork_height_lp import certify_regular_heights
from omnibias.geometry.patchwork_search import (
    PATCHWORK_OCTIC_KIND,
    PatchworkSearchFamily,
    annealed_sign_seed,
    certify_patchwork_realization,
    csp_sign_seed,
    patchwork_polynomial,
    run_patchwork_discovery,
    statement_for_patchwork_target,
    triangulation_bank,
    verify_patchwork_realization,
)


def _regularity(triangulation):
    return certify_regular_heights(
        triangulation,
        {
            point: point[0] ** 2 + point[1] ** 2 + point[0] * point[1]
            for point in triangulation.vertices
        },
    )


def test_flip_bank_contains_only_complete_unimodular_triangulations():
    quartic = staircase_triangulation(4)
    neighbors = flip_neighbors(quartic)
    assert len(neighbors) == 18
    assert all(
        item.degree == 4
        and len(item.triangles) == 16
        and len(item.edges) == 30
        for item in neighbors
    )
    bank = triangulation_bank(4, flip_depth=1)
    assert len(bank) == 19
    assert bank[0] == quartic


def test_quartic_target_is_an_exact_discovery_hit_not_a_float_claim():
    triangulation = staircase_triangulation(4)
    bits = tuple(
        (point[0] * point[1]) % 2
        for point in triangulation.vertices
    )
    target = ((), (), (), ())
    family = PatchworkSearchFamily((triangulation,), bits, target)
    assert family.cardinality() == 1 << len(bits)
    result = run_discovery(
        family.statement,
        family,
        OneHotAnneal(seed=0),
        budget=1,
    )
    assert result.status == "PROVED"
    assert result.check is not None and result.check.ok
    assert result.check.payload["component_count"] == 4
    assert result.check.payload["honesty"]["target_region_tree_exact"]
    assert result.check.payload["honesty"]["regular_height_exact_q"]
    assert not result.check.payload["honesty"]["algebraic_polynomial_directly_certified"]


def test_both_differentiable_sign_proposers_return_hard_vertices():
    triangulation = staircase_triangulation(2)
    annealed = annealed_sign_seed(triangulation)
    csp = csp_sign_seed(triangulation, seed=0)
    assert len(annealed) == len(csp) == 6
    assert set(annealed) <= {0, 1}
    assert set(csp) <= {0, 1}


def test_generated_conic_passes_direct_coefficient_acceptance():
    triangulation = staircase_triangulation(2)
    signs = SignDistribution.create(
        2,
        {
            point: (point[0] * point[1]) % 2
            for point in triangulation.vertices
        },
    )
    regularity = _regularity(triangulation)
    center = Q(-32, 9)
    inner = RationalRectangle(
        center - Q(1, 10),
        center + Q(1, 10),
        center - Q(1, 10),
        center + Q(1, 10),
    )
    outer = RationalRectangle(center - 4, center + 4, center - 4, center + 4)
    annulus = RectangularAnnulus(inner, outer)
    certificate = certify_patchwork_realization(
        triangulation,
        signs,
        regularity,
        [annulus],
        t=Q(3, 4),
        target=((),),
        max_multiplier_degree=2,
    )
    assert certificate.curve_certificate.complete_real_scheme
    assert verify_patchwork_realization(certificate)
    assert not verify_patchwork_realization(
        replace(certificate, target=((), ())),
    )
    with pytest.raises(TypeError, match="not a float"):
        patchwork_polynomial(triangulation, signs, regularity, t=0.75)


def test_octic_search_is_registered_as_incomplete_exact_search():
    entry = catalog_entry(PATCHWORK_OCTIC_KIND)
    assert entry is not None
    assert entry.mode == "exact_search"
    assert entry.complete is False
    assert entry.parent_status == "open"


def test_patchwork_family_stays_incomplete_and_names_its_tree():
    triangulation = staircase_triangulation(2)
    bits = tuple(0 for _ in triangulation.vertices)
    family = PatchworkSearchFamily((triangulation,), bits, ((),))
    assert family.complete is False
    assert "declared rooted region tree" in family.statement.obligation
    sibling = statement_for_patchwork_target(sibling_octic_region_tree())
    wide = statement_for_patchwork_target(wide_deep_octic_region_tree())
    assert "14 + 1<2 + 1<4>>" in sibling.obligation
    assert "4 + 1<2 + 1<14>>" in wide.obligation
    assert sibling.obligation != wide.obligation
    with pytest.raises(ValueError, match="complete stays false"):
        PatchworkSearchFamily((triangulation,), bits, ((),), complete=True)


def test_discovery_statement_follows_the_supplied_tree_and_bank():
    triangulation = staircase_triangulation(2)
    report = run_patchwork_discovery(
        degree=2,
        budget=1,
        flip_depth=3,
        triangulation_limit=128,
        seed=3,
        target=((),),
        triangulations=(triangulation,),
    )
    result = report["result"]
    assert report["triangulation_count"] == 1
    assert "declared rooted region tree" in result.statement.obligation
    assert "1<14>" not in result.statement.obligation
    assert "1<4>" not in result.statement.obligation
    assert result.characterization is not None
    assert result.characterization.family_complete is False
    if result.status == "BLOCKED":
        assert result.search_incomplete is True
        assert result.detail == "search_incomplete"
    else:
        assert result.status == "PROVED"
        assert result.search_incomplete is False
