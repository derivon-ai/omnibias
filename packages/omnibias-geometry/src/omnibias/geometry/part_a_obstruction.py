# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""H7 audit: polygonal octic barriers and the SOS obstruction route.

The exact geometry layer already supports rational polygonal annuli, including
a validated 22-annulus layout for the wide/deep open ``(19,3)`` octic tree.
Thus rectangular barriers are no longer the blocker.  The remaining Route-2
problem is coefficient feasibility and projective smoothness.

A Positivstellensatz can refute a *specified finite basic semialgebraic set*.
The statement "no octic realizes this isotopy scheme" is not supplied here as
such a set in the 45 coefficients: it quantifies over embeddings, the whole
real locus, and complex nonsingularity.  A certificate for one polygon layout
or one symmetry-reduced ansatz excludes only that ansatz unless a complete
reduction from every realization is proved.

This module checks both sides.  A one-variable empty-set example earns an
actual certified Putinar witness, while the open octic has no complete
semialgebraic encoding or symmetry reduction.  It also records the elementary
Gram-size explosion before any topological constraints are added.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import comb

from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.geometry.algebraic_layout import (
    layout_annuli_for_tree,
    validate_annulus_layout,
)
from omnibias.geometry.algebraic_route2 import (
    assemble_barrier_system,
    octic_basis_orbits,
    search_route2_lp,
    signs_by_nesting_depth,
)
from omnibias.geometry.part_a_target import wide_deep_octic_region_tree
from omnibias.sos.positivstellensatz import (
    certify_nonneg_on_set,
    seal_positivstellensatz_certificate,
)
from omnibias.sos.problem import Polynomial

__all__ = [
    "PartAObstructionAudit",
    "audit_part_a_obstruction_route",
    "sos_basis_size",
    "symmetric_gram_entries",
]


def sos_basis_size(n_variables: int, half_degree: int) -> int:
    """Number of monomials of total degree at most ``half_degree``."""
    if (
        type(n_variables) is not int
        or type(half_degree) is not int
        or n_variables < 1
        or half_degree < 0
    ):
        raise ValueError("positive variable count and nonnegative half-degree required")
    return comb(n_variables + half_degree, half_degree)


def symmetric_gram_entries(n_variables: int, half_degree: int) -> int:
    """Independent entries in the corresponding dense SOS Gram matrix."""
    size = sos_basis_size(n_variables, half_degree)
    return size * (size + 1) // 2


def _toy_empty_set_certified() -> bool:
    x = Polynomial.variable(0, 1)
    one = Polynomial.constant(1.0, 1)
    # x - 1 >= 0 and -x - 1 >= 0 are inconsistent.  Certifying -1 >= 0
    # on this set is a Positivstellensatz emptiness witness.
    certificate = certify_nonneg_on_set(
        -one,
        (x - one, -x - one),
        half_degree=0,
    )
    if not certificate.certified:
        return False
    sealed = seal_positivstellensatz_certificate(
        certificate,
        claim="the declared one-variable basic closed set is empty",
    )
    return verify_certificate_digest(sealed)


@dataclass(frozen=True)
class PartAObstructionAudit:
    """Outcome of the widened-barrier and reduced-SOS H7 test."""

    target_annulus_count: int
    polygonal_layout_valid: bool
    route2_constraint_count: int
    coefficient_dimensions: Mapping[str, int]
    route2_lp_ran: bool
    route2_lp_candidate_found: bool | None
    route2_exact_infeasibility_certified: bool
    toy_positivstellensatz_empty_set_certified: bool
    full_octic_half_degree_two_basis_size: int
    full_octic_half_degree_two_gram_entries: int
    scheme_basic_closed_encoding_supplied: bool
    symmetry_reduction_complete_for_all_realizations: bool
    octic_scheme_obstructed: bool
    part_a_22_oval_realized: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "hilbert16-part-a-polygon-sos-audit-v1",
            "target_annulus_count": self.target_annulus_count,
            "polygonal_layout_valid": self.polygonal_layout_valid,
            "route2_constraint_count": self.route2_constraint_count,
            "coefficient_dimensions": dict(self.coefficient_dimensions),
            "route2_lp_ran": self.route2_lp_ran,
            "route2_lp_candidate_found": self.route2_lp_candidate_found,
            "route2_exact_infeasibility_certified": (
                self.route2_exact_infeasibility_certified
            ),
            "toy_positivstellensatz_empty_set_certified": (
                self.toy_positivstellensatz_empty_set_certified
            ),
            "full_octic_half_degree_two_basis_size": (
                self.full_octic_half_degree_two_basis_size
            ),
            "full_octic_half_degree_two_gram_entries": (
                self.full_octic_half_degree_two_gram_entries
            ),
            "scheme_basic_closed_encoding_supplied": (
                self.scheme_basic_closed_encoding_supplied
            ),
            "symmetry_reduction_complete_for_all_realizations": (
                self.symmetry_reduction_complete_for_all_realizations
            ),
            "octic_scheme_obstructed": self.octic_scheme_obstructed,
            "part_a_22_oval_realized": self.part_a_22_oval_realized,
            "hilbert16_part_a_solved": False,
            "full_hilbert16_solved": False,
            "scope": (
                "Exact polygon-layout and finite SOS-capability audit. A fixed "
                "layout or symmetry ansatz is not a general obstruction to an "
                "isotopy scheme."
            ),
        }


def audit_part_a_obstruction_route(
    *,
    run_route2_lp: bool = False,
) -> PartAObstructionAudit:
    """Test H7 on the wide/deep open ``(19,3)`` octic scheme."""
    target = wide_deep_octic_region_tree()
    annuli = layout_annuli_for_tree(target)
    validate_annulus_layout(annuli, target)
    signs = signs_by_nesting_depth(annuli)
    system = assemble_barrier_system(annuli, signs, symmetry="full")
    lp = (
        search_route2_lp(
            annuli,
            signs,
            symmetry="full",
            certify_infeasible=True,
            barrier_system=system,
        )
        if run_route2_lp
        else None
    )
    exact_infeasible = bool(lp is not None and lp.farkas is not None)
    dimensions = {
        name: len(octic_basis_orbits(name))
        for name in ("d4", "klein", "central", "full")
    }
    encoding = False
    complete_reduction = False
    obstructed = exact_infeasible and encoding and complete_reduction
    return PartAObstructionAudit(
        target_annulus_count=len(annuli),
        polygonal_layout_valid=True,
        route2_constraint_count=system.n_constraints,
        coefficient_dimensions=dimensions,
        route2_lp_ran=lp is not None,
        route2_lp_candidate_found=(
            lp.candidate_found if lp is not None else None
        ),
        route2_exact_infeasibility_certified=exact_infeasible,
        toy_positivstellensatz_empty_set_certified=(
            _toy_empty_set_certified()
        ),
        full_octic_half_degree_two_basis_size=sos_basis_size(45, 2),
        full_octic_half_degree_two_gram_entries=symmetric_gram_entries(45, 2),
        scheme_basic_closed_encoding_supplied=encoding,
        symmetry_reduction_complete_for_all_realizations=complete_reduction,
        octic_scheme_obstructed=obstructed,
        part_a_22_oval_realized=False,
    )
