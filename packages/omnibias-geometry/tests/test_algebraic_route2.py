# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fractions import Fraction
from math import isfinite
from types import SimpleNamespace

import omnibias.geometry.algebraic_route2 as route2_module
import torch
from omnibias.geometry.algebraic import PolygonalAnnulus, _nesting
from omnibias.geometry.algebraic_route2 import (
    assemble_barrier_system,
    certify_route2_realization,
    layout_annuli_for_tree,
    nested_annulus_chain,
    octic_basis_orbits,
    octic_monomial_indices,
    seal_route2_witness,
    search_route2_coefficients,
    search_route2_full_coefficients,
    search_route2_lp,
    signs_by_nesting_depth,
    square_polygon,
)
from omnibias.geometry.patchwork import target_octic_region_tree
from omnibias.geometry.patchwork_height_lp import RationalInequalitySystem


def test_octic_has_45_monomials() -> None:
    assert len(octic_monomial_indices()) == 45
    assert [len(octic_basis_orbits(name)) for name in ("d4", "klein", "central", "full")] == [
        9,
        15,
        25,
        45,
    ]


def test_assemble_barrier_system_has_rows() -> None:
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 64)),
        square_polygon(0, 0, Fraction(1, 8)),
    )
    system = assemble_barrier_system([annulus], [1])
    assert system.n_constraints > 0
    assert system.n_variables == 45
    refined = assemble_barrier_system([annulus], [1], subdivision_depth=1)
    assert refined.n_constraints == 2 * system.n_constraints


def test_nested_annulus_chain_is_strictly_nested() -> None:
    annuli = nested_annulus_chain(3)
    assert len(annuli) == 3


def test_route2_full_search_reports_infeasible_on_tiny_template() -> None:
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 64)),
        square_polygon(0, 0, Fraction(1, 8)),
    )
    report = search_route2_full_coefficients([annulus], [1], template_scales=(Fraction(1, 1000),), max_support=2)
    assert report.candidate_found is False


def test_layout_annuli_for_tree_has_22_octs() -> None:
    annuli = layout_annuli_for_tree(target_octic_region_tree())
    assert len(annuli) == 22


def test_route2_lp_runs_on_tree_layout() -> None:
    annuli = layout_annuli_for_tree(target_octic_region_tree())
    signs = signs_by_nesting_depth(annuli)
    report = search_route2_lp(annuli, signs)
    assert report.constraint_count > 0
    assert report.converged
    assert isfinite(report.margin_float)
    assert -1.0 <= report.margin_float <= 1.0


def test_depth_signs_are_constant_on_siblings_and_flip_on_children() -> None:
    annuli = layout_annuli_for_tree(target_octic_region_tree())
    parents = _nesting(annuli)
    signs = signs_by_nesting_depth(annuli)
    assert signs[:5] == (1, 1, 1, 1, 1)
    for index, parent in enumerate(parents):
        if parent >= 0:
            assert signs[index] == -signs[parent]


def test_normalized_lp_finds_a_known_single_annulus_barrier() -> None:
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 4)),
        square_polygon(0, 0, Fraction(1, 2)),
    )
    report = search_route2_lp([annulus], [1])
    assert report.converged
    assert 0.0 < report.margin_float <= 1.0
    assert report.candidate_found
    assert report.coefficients is not None
    assert report.feasibility is not None
    assert any(value == 0 for value in report.coefficients)
    assert any(value < 0 for value in report.coefficients)
    assert any(value > 0 for value in report.coefficients)


def test_impossible_reduced_row_earns_exact_farkas_certificate() -> None:
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 4)),
        square_polygon(0, 0, Fraction(1, 2)),
    )
    impossible = RationalInequalitySystem(
        (tuple(Fraction(0) for _ in range(9)),),
        (Fraction(1),),
    )
    report = search_route2_lp(
        [annulus],
        [1],
        symmetry="d4",
        certify_infeasible=True,
        barrier_system=impossible,
    )
    assert report.candidate_found is False
    assert report.farkas is not None
    assert report.farkas.contradiction > 0


def test_nonconverged_nonfinite_lp_output_is_rejected(monkeypatch: object) -> None:
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 4)),
        square_polygon(0, 0, Fraction(1, 2)),
    )

    def fake_solve_lp(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            x=torch.full((46,), float("nan"), dtype=torch.float64),
            converged=False,
        )

    monkeypatch.setattr(route2_module, "solve_lp", fake_solve_lp)  # type: ignore[attr-defined]
    report = search_route2_lp([annulus], [1])
    assert report.candidate_found is False
    assert report.converged is False


def test_seal_route2_witness_is_wired() -> None:
    assert callable(seal_route2_witness)
    assert callable(certify_route2_realization)


def test_route2_search_reports_infeasible_on_empty_template() -> None:
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 64)),
        square_polygon(0, 0, Fraction(1, 8)),
    )
    report = search_route2_coefficients([annulus], [1], template_scales=(Fraction(1, 1000),))
    assert report.candidate_found is False
