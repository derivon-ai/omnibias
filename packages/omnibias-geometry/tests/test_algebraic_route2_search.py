# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from fractions import Fraction

from omnibias.core.proof.discovery import run_discovery
from omnibias.geometry.algebraic import PolygonalAnnulus
from omnibias.geometry.algebraic_route2 import Route2LpReport, square_polygon
from omnibias.geometry.algebraic_route2_search import (
    FINE_GEOMETRY_LEVELS,
    GEOMETRY_LEVELS,
    Route2Candidate,
    Route2FiniteFamily,
    route2_layout_params,
)
from omnibias.geometry.patchwork_height_lp import RationalInequalitySystem


def test_route2_finite_family_has_declared_cardinality_and_connected_neighbors() -> None:
    family = Route2FiniteFamily(symmetry="klein", max_subdivision_depth=2)
    assert family.complete is False
    expected = 3 ** len(GEOMETRY_LEVELS) * 3 * 30
    assert family.cardinality() == expected
    origin = family.origin()
    assert len(origin.coordinates) == len(GEOMETRY_LEVELS)
    assert origin.symmetry == "klein"
    assert family.neighbors(origin)


def test_route2_candidate_maps_to_exact_layout_parameters() -> None:
    candidate = Route2Candidate((1,) * len(GEOMETRY_LEVELS), 0)
    params = route2_layout_params(candidate)
    assert params.center_outer == 1
    assert params.exterior_center > params.center_outer
    assert params.deep_leaf_half < params.leaf_half


def test_fine_family_starts_from_measured_coarse_boundary() -> None:
    family = Route2FiniteFamily(symmetry="d4", max_subdivision_depth=0, fine=True)
    origin = family.origin()
    assert len(origin.coordinates) == len(FINE_GEOMETRY_LEVELS)
    params = route2_layout_params(origin, levels=FINE_GEOMETRY_LEVELS)
    assert params.exterior_center == 3
    assert params.exterior_half == Fraction(1, 10)
    assert params.deep_branch_scale == Fraction(2, 3)
    assert params.branch_offset == 0


def test_route2_family_is_compatible_with_discovery_machine() -> None:
    family = Route2FiniteFamily(symmetry="d4", max_subdivision_depth=0)
    result = run_discovery(family.statement, family, budget=0)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True


def test_farkas_honesty_excludes_only_the_bernstein_system(
    monkeypatch: object,
) -> None:
    family = Route2FiniteFamily(
        symmetry="d4",
        max_subdivision_depth=0,
        certify_infeasible=True,
    )
    candidate = family.origin()
    annulus = PolygonalAnnulus(
        square_polygon(0, 0, Fraction(1, 4)),
        square_polygon(0, 0, Fraction(1, 2)),
    )
    impossible = RationalInequalitySystem(
        (tuple(Fraction(0) for _ in range(9)),),
        (Fraction(1),),
    )
    report = Route2LpReport(
        1,
        1,
        -1.0,
        False,
        None,
        False,
        "miss",
        converged=True,
        symmetry="d4",
    )
    monkeypatch.setattr(family, "_layout", lambda _candidate: (annulus,))  # type: ignore[attr-defined]
    monkeypatch.setattr(family, "report", lambda _candidate: report)  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        family,
        "_system",
        lambda _candidate, _annuli: impossible,
    )
    checked = family.check(candidate)
    assert checked is not None
    honesty = checked.payload["honesty"]
    assert honesty["bernstein_system_infeasible"] is True
    assert honesty["anchor_chart_excluded"] is False
    assert honesty["fixed_layout_excluded"] is False
