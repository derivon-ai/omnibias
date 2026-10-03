# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite, best-first Route-2 search for the wide/deep 22-oval octic."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from omnibias.core.proof.discovery import ExactCheck, Statement
from omnibias.geometry.algebraic import PolygonalAnnulus
from omnibias.geometry.algebraic_layout import (
    LayoutParams,
    layout_annuli_for_tree,
    parents_to_rooted_tree,
)
from omnibias.geometry.algebraic_route2 import (
    OcticSymmetry,
    Route2LpReport,
    assemble_barrier_system,
    certify_route2_realization,
    octic_basis_orbits,
    seal_route2_witness,
    search_route2_lp,
    signs_by_nesting_depth,
)
from omnibias.geometry.patchwork import target_octic_region_tree
from omnibias.geometry.patchwork_height_lp import (
    RationalFarkasCertificate,
    RationalInequalitySystem,
    propose_farkas_infeasibility,
    verify_rational_infeasibility,
)

__all__ = [
    "FINE_GEOMETRY_LEVELS",
    "GEOMETRY_LEVELS",
    "Route2Candidate",
    "Route2FiniteFamily",
    "route2_layout_params",
]


GEOMETRY_LEVELS: tuple[tuple[Fraction, ...], ...] = (
    (Fraction(5, 2), Fraction(11, 4), Fraction(3)),  # outer-center ratio
    (Fraction(1, 10), Fraction(3, 20), Fraction(1, 5)),  # outer oval half
    (Fraction(3, 4), Fraction(4, 5), Fraction(17, 20)),  # collar inner ratio
    (Fraction(1, 32), Fraction(1, 24), Fraction(1, 20)),  # leaf half
    (Fraction(1, 4), Fraction(1, 3), Fraction(2, 5)),  # branch ratio
    (Fraction(1, 2), Fraction(2, 3), Fraction(3, 4)),  # deep branch scale
    (Fraction(3, 5), Fraction(7, 10), Fraction(4, 5)),  # leaf fill
    (Fraction(-1, 10), Fraction(0), Fraction(1, 10)),  # branch offset
)

FINE_GEOMETRY_LEVELS: tuple[tuple[Fraction, ...], ...] = (
    (Fraction(3), Fraction(13, 4), Fraction(7, 2), Fraction(4)),
    (Fraction(1, 20), Fraction(1, 16), Fraction(1, 12), Fraction(1, 10)),
    (Fraction(2, 3), Fraction(7, 10), Fraction(29, 40), Fraction(3, 4)),
    (Fraction(1, 64), Fraction(1, 48), Fraction(1, 40), Fraction(1, 32)),
    (Fraction(2, 5), Fraction(5, 12), Fraction(9, 20), Fraction(1, 2)),
    (Fraction(9, 20), Fraction(1, 2), Fraction(11, 20), Fraction(3, 5)),
    (
        Fraction(-1, 8),
        Fraction(-1, 16),
        Fraction(0),
        Fraction(1, 16),
        Fraction(1, 8),
    ),
)

GEOMETRY_AXES = (
    "outer_center_ratio",
    "outer_oval_half",
    "collar_inner_ratio",
    "leaf_half",
    "branch_ratio",
    "deep_branch_scale",
    "leaf_fill",
    "branch_offset",
)
FINE_GEOMETRY_AXES = tuple(
    axis for axis in GEOMETRY_AXES if axis != "deep_branch_scale"
)


@dataclass(frozen=True)
class Route2Candidate:
    """One member of the declared rational geometry/depth lattice."""

    coordinates: tuple[int, ...]
    subdivision_depth: int
    anchor: int = 0
    symmetry: OcticSymmetry = "d4"

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinates": list(self.coordinates),
            "subdivision_depth": self.subdivision_depth,
            "anchor": self.anchor,
            "symmetry": self.symmetry,
        }


def route2_layout_params(
    candidate: Route2Candidate,
    *,
    levels: tuple[tuple[Fraction, ...], ...] = GEOMETRY_LEVELS,
) -> LayoutParams:
    if len(candidate.coordinates) != len(levels):
        raise ValueError("Route-2 candidate has the wrong coordinate dimension")
    values = tuple(
        levels[index]
        for levels, index in zip(levels, candidate.coordinates, strict=True)
    )
    if len(levels) == len(FINE_GEOMETRY_LEVELS):
        (
            exterior_center,
            exterior_half,
            inner_ratio,
            leaf_half,
            branch_ratio,
            leaf_fill,
            branch_offset,
        ) = values
        deep_scale = Fraction(2, 3)
    else:
        (
            exterior_center,
            exterior_half,
            inner_ratio,
            leaf_half,
            branch_ratio,
            deep_scale,
            leaf_fill,
            branch_offset,
        ) = values
    return LayoutParams(
        exterior_center=exterior_center,
        exterior_half=exterior_half,
        exterior_inner_ratio=Fraction(5, 6),
        center_outer=Fraction(1),
        inner_ratio=inner_ratio,
        leaf_half=leaf_half,
        nested_leaf_half=leaf_half * Fraction(4, 5),
        deep_leaf_half=leaf_half / 4,
        branch_outer_ratio=branch_ratio,
        branch_offset=branch_offset,
        deep_branch_scale=deep_scale,
        leaf_fill=leaf_fill,
    )


class Route2FiniteFamily:
    """Connected finite lattice of deterministic LP proposals.

    The coordinate lattice is finite and connected, but the deterministic LP
    proposer is not exhaustive over rational coefficient vectors.  Therefore
    the discovery family is deliberately incomplete.
    """

    complete = False
    empty_miss_detail = "no sealed LP-proposed witness in the declared Route-2 lattice"

    def __init__(
        self,
        *,
        symmetry: OcticSymmetry,
        max_subdivision_depth: int = 2,
        certify_infeasible: bool = False,
        fine: bool = False,
        origin_candidate: Route2Candidate | None = None,
    ) -> None:
        if type(max_subdivision_depth) is not int or not 0 <= max_subdivision_depth <= 8:
            raise ValueError("max_subdivision_depth must lie from zero to eight")
        self.symmetry = symmetry
        self.max_subdivision_depth = max_subdivision_depth
        self.certify_infeasible = certify_infeasible
        self.levels = FINE_GEOMETRY_LEVELS if fine else GEOMETRY_LEVELS
        self.fine = fine
        self._origin_candidate = origin_candidate
        resolution = "fine_v2" if fine else "coarse_v2"
        self.name = (
            f"route2_wide_deep_{symmetry}_{resolution}_depth{max_subdivision_depth}"
        )
        self.statement = Statement(
            name="wide_deep_22_oval_octic",
            obligation="find and seal an octic realizing 4 + 1<2 + 1<14>>",
            parent="Hilbert 16 Part A degree 8",
            parent_status="open",
            existential=True,
        )
        self._reports: dict[Route2Candidate, Route2LpReport] = {}
        self._layouts: dict[Route2Candidate, tuple[PolygonalAnnulus, ...] | None] = {}
        self._errors: dict[Route2Candidate, str] = {}
        self._scores: dict[Route2Candidate, Fraction] = {}
        self._farkas: dict[
            tuple[tuple[int, ...], int], RationalFarkasCertificate | None
        ] = {}
        self._systems: dict[
            tuple[tuple[int, ...], int], RationalInequalitySystem
        ] = {}
        if origin_candidate is not None:
            if (
                len(origin_candidate.coordinates) != len(self.levels)
                or not 0 <= origin_candidate.subdivision_depth <= max_subdivision_depth
                or not 0 <= origin_candidate.anchor < self.anchor_count
                or origin_candidate.symmetry != self.symmetry
                or any(
                    not 0 <= value < len(axis_levels)
                    for value, axis_levels in zip(
                        origin_candidate.coordinates, self.levels, strict=True
                    )
                )
            ):
                raise ValueError("origin_candidate lies outside the declared finite family")

    def origin(self) -> Route2Candidate:
        if self._origin_candidate is not None:
            return self._origin_candidate
        coordinates = (
            (0, 3, 3, 3, 0, 3, 2)
            if self.fine
            else (1,) * len(GEOMETRY_LEVELS)
        )
        return Route2Candidate(coordinates, 0, 0, self.symmetry)

    def neighbors(self, candidate: Route2Candidate) -> tuple[Route2Candidate, ...]:
        neighbors: list[Route2Candidate] = []
        for axis, levels in enumerate(self.levels):
            for delta in (-1, 1):
                value = candidate.coordinates[axis] + delta
                if 0 <= value < len(levels):
                    coordinates = list(candidate.coordinates)
                    coordinates[axis] = value
                    neighbors.append(
                        Route2Candidate(
                            tuple(coordinates),
                            candidate.subdivision_depth,
                            candidate.anchor,
                            candidate.symmetry,
                        )
                    )
        for delta in (-1, 1):
            depth = candidate.subdivision_depth + delta
            if 0 <= depth <= self.max_subdivision_depth:
                neighbors.append(
                    Route2Candidate(
                        candidate.coordinates,
                        depth,
                        candidate.anchor,
                        candidate.symmetry,
                    )
                )
        anchor_count = 2 * len(octic_basis_orbits(self.symmetry))
        for delta in (-1, 1):
            anchor = candidate.anchor + delta
            if 0 <= anchor < anchor_count:
                neighbors.append(
                    Route2Candidate(
                        candidate.coordinates,
                        candidate.subdivision_depth,
                        anchor,
                        candidate.symmetry,
                    )
                )
        return tuple(neighbors)

    def cardinality(self) -> int:
        geometry_count = 1
        for levels in self.levels:
            geometry_count *= len(levels)
        return (
            geometry_count
            * (self.max_subdivision_depth + 1)
            * self.anchor_count
        )

    def lattice_spec(self) -> dict[str, list[str]]:
        axes = FINE_GEOMETRY_AXES if self.fine else GEOMETRY_AXES
        return {
            axis: [str(value) for value in levels]
            for axis, levels in zip(axes, self.levels, strict=True)
        }

    @property
    def anchor_count(self) -> int:
        return 2 * len(octic_basis_orbits(self.symmetry))

    def _layout(self, candidate: Route2Candidate) -> tuple[PolygonalAnnulus, ...] | None:
        if candidate not in self._layouts:
            try:
                self._layouts[candidate] = layout_annuli_for_tree(
                    target_octic_region_tree(),
                    params=route2_layout_params(candidate, levels=self.levels),
                    normalize=True,
                    shrink_to_fit=False,
                )
            except ValueError as error:
                self._layouts[candidate] = None
                self._errors[candidate] = str(error)
        return self._layouts[candidate]

    def report(self, candidate: Route2Candidate) -> Route2LpReport | None:
        if candidate in self._reports:
            return self._reports[candidate]
        annuli = self._layout(candidate)
        if annuli is None:
            return None
        system = self._system(candidate, annuli)
        report = search_route2_lp(
            annuli,
            signs_by_nesting_depth(annuli),
            symmetry=self.symmetry,
            subdivision_depth=candidate.subdivision_depth,
            certify_infeasible=False,
            anchor=(candidate.anchor // 2, 1 if candidate.anchor % 2 == 0 else -1),
            barrier_system=system,
        )
        self._reports[candidate] = report
        return report

    def _system(
        self,
        candidate: Route2Candidate,
        annuli: tuple[PolygonalAnnulus, ...],
    ) -> RationalInequalitySystem:
        key = (candidate.coordinates, candidate.subdivision_depth)
        if key not in self._systems:
            self._systems[key] = assemble_barrier_system(
                annuli,
                signs_by_nesting_depth(annuli),
                symmetry=self.symmetry,
                subdivision_depth=candidate.subdivision_depth,
            )
        return self._systems[key]

    def score(self, candidate: Route2Candidate) -> Fraction:
        if candidate in self._scores:
            return self._scores[candidate]
        report = self.report(candidate)
        if report is None or not report.converged:
            score = Fraction(-1)
        else:
            score = Fraction(str(report.margin_float)).limit_denominator(10**12)
        self._scores[candidate] = score
        return score

    def score_checkpoint(self) -> list[dict[str, object]]:
        """Return deterministic JSON-able scores for restartable best-first search."""
        return [
            {
                "candidate": candidate.as_dict(),
                "score": str(score),
            }
            for candidate, score in sorted(
                self._scores.items(),
                key=lambda item: (
                    item[0].subdivision_depth,
                    item[0].coordinates,
                    item[0].anchor,
                ),
            )
        ]

    def top_scores(self, limit: int = 10) -> list[dict[str, object]]:
        """Return the strongest evaluated LP charts in descending margin order."""
        return [
            {"candidate": candidate.as_dict(), "score": str(score)}
            for candidate, score in sorted(
                self._scores.items(),
                key=lambda item: (item[1], item[0].coordinates, item[0].anchor),
                reverse=True,
            )[:limit]
        ]

    def load_score_checkpoint(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            raw = row.get("candidate")
            if not isinstance(raw, dict):
                continue
            coordinates = raw.get("coordinates")
            depth = raw.get("subdivision_depth")
            anchor = raw.get("anchor", 0)
            symmetry = raw.get("symmetry", self.symmetry)
            score = row.get("score")
            if (
                isinstance(coordinates, list)
                and all(type(value) is int for value in coordinates)
                and type(depth) is int
                and type(anchor) is int
                and symmetry == self.symmetry
                and isinstance(score, str)
            ):
                self._scores[
                    Route2Candidate(tuple(coordinates), depth, anchor, self.symmetry)
                ] = Fraction(score)

    def check(self, candidate: Route2Candidate) -> ExactCheck | None:
        report = self.report(candidate)
        annuli = self._layout(candidate)
        if report is None or annuli is None:
            return ExactCheck(
                False,
                {
                    "detail": self._errors.get(candidate, "invalid layout"),
                    "honesty": {"search_incomplete": True, "full_hilbert16_solved": False},
                },
            )
        evidence: dict[str, Any] = {
            "margin_float": report.margin_float,
            "converged": report.converged,
            "symmetry": self.symmetry,
            "constraint_count": report.constraint_count,
            "anchor": candidate.anchor,
            "detail": report.detail,
            "farkas_digest": report.farkas.seal["digest"] if report.farkas is not None else None,
            "farkas_attempted": self.certify_infeasible,
            "exact_lift": (
                [str(value) for value in report.coefficients]
                if report.coefficients is not None
                else None
            ),
            "exact_lift_digest": (
                report.feasibility.seal["digest"]
                if report.feasibility is not None
                else None
            ),
            "evidence_scope": (
                f"fixed_layout_depth_{candidate.subdivision_depth}_{self.symmetry}_basis"
            ),
            "farkas_problem": "unanchored_exact_bernstein_system",
        }
        if not report.candidate_found or report.coefficients is None:
            farkas = report.farkas
            if self.certify_infeasible and farkas is None:
                key = (candidate.coordinates, candidate.subdivision_depth)
                if key not in self._farkas:
                    system = self._system(candidate, annuli)
                    try:
                        proposed = propose_farkas_infeasibility(system)
                        self._farkas[key] = (
                            proposed
                            if verify_rational_infeasibility(system, proposed)
                            else None
                        )
                    except (RuntimeError, ValueError):
                        self._farkas[key] = None
                farkas = self._farkas[key]
            evidence["farkas_digest"] = (
                farkas.seal["digest"] if farkas is not None else None
            )
            return ExactCheck(
                False,
                {
                    **evidence,
                    "honesty": {
                        "bernstein_system_infeasible": farkas is not None,
                        "anchor_chart_excluded": False,
                        "fixed_layout_excluded": False,
                        "symmetry_tier_only": self.symmetry != "full",
                        "global_nonrealizability_claim": False,
                        "full_hilbert16_solved": False,
                    },
                },
            )
        try:
            certificate = certify_route2_realization(
                annuli,
                signs_by_nesting_depth(annuli),
                report.coefficients,
                symmetry=self.symmetry,
                barrier_subdivision_depth=candidate.subdivision_depth,
            )
            formal = seal_route2_witness(certificate)
        except (ArithmeticError, TypeError, ValueError) as error:
            return ExactCheck(
                False,
                {
                    **evidence,
                    "detail": f"realization sealing failed: {error}",
                    "honesty": {
                        "search_incomplete": True,
                        "full_hilbert16_solved": False,
                    },
                },
            )
        target_ok = (
            len(certificate.annuli) == 22
            and certificate.component_lower_bound == 22
            and certificate.complete_real_scheme
            and parents_to_rooted_tree(certificate.barrier_parents)
            == target_octic_region_tree()
            and certificate.formal_seal["payload"]["type"] == "polynomial_identity_q"
            and formal.theorem_prover_verified
        )
        return ExactCheck(
            target_ok,
            {
                **evidence,
                "curve_digest": certificate.curve_digest,
                "formal_digest": certificate.formal_seal["digest"],
                "coefficients": [str(value) for value in report.coefficients],
                "honesty": {
                    "wide_deep_22_oval_witness": target_ok,
                    "topological_implication_formally_verified": False,
                    "full_hilbert16_solved": False,
                },
            },
        )
