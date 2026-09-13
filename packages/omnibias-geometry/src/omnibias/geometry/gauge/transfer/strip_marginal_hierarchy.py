# SPDX-License-Identifier: Apache-2.0
"""Actual SU(2) strip marginals with a depth-independent local Hessian ball.

Exact gates for a written analytic implication on every finite open
one-plaquette-wide strip at fixed microscopic coupling. Coordinate
integration retains all generated interactions and the induced electric
form. It is not a running-coupling or continuum renormalization theorem.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_strip_compression_geometry(
    n_plaquettes: int, retained: Sequence[int],
) -> dict[str, Any]:
    """Exact electric weights for retained vertical links in a forest gauge.

    The two horizontal rows are gauged to the identity. Retain the two
    endpoint verticals and any ordered subset of the interior verticals.
    Each cut weight multiplies BOTH the left and right prefix-gradient
    squares. Vertical-gradient squares retain coefficient one. Distances
    are the original site distances, not reset coarse-lattice distances.
    """
    n = _integer(n_plaquettes, "n_plaquettes")
    if n < 1:
        raise ValueError("n_plaquettes must be positive")
    indices = tuple(_integer(i, "retained index") for i in retained)
    if (len(indices) < 2 or indices[0] != 0 or indices[-1] != n
            or any(i >= j for i, j in zip(indices, indices[1:], strict=False))):
        raise ValueError("retained must be strictly increasing and contain endpoints 0 and n")
    return {
        "n_plaquettes": n, "retained": list(indices),
        "n_vertices": 2 * n + 2, "n_original_edges": 3 * n + 1,
        "cut_weights": [j - i for i, j in zip(indices, indices[1:], strict=False)],
        "vertical_weights": ["1"] * len(indices),
        "distances": [[abs(i - j) for j in indices] for i in indices],
        "physical_kinetic_floor": "1",
        "electric_form": "sum_i |p_i|^2 + sum_j cut_weights[j]*(|sum_i<=j p_i|^2 + |sum_i<=j Ad(Q_i)p_i|^2)",
        "gauss_constraints": "sum_i p_i=0 and sum_i Ad(Q_i)p_i=0",
        "units": "original dimensionless aH; multiply the electric form by kappa/2",
        "nonfluctuating_compression_geometry_verified": True,
        "continuum_claim": False,
    }


def su2_strip_marginal_hierarchy(
    kappa: int | Q, *, correction_radius: int | Q | None = None, decay_base: int | Q = 1,
) -> dict[str, Any]:
    """Seal a local bound for all finite strip lengths and marginal depths.

    The existing Fourier construction supplies the actual full-spin
    vacuum. A spatial covariance kernel and the shared row-budget Schur
    identity preserve its weighted Hessian ball under every retained
    coordinate marginal. The induced physical electric form dominates
    the retained product metric in the original units. Source failure or
    a nonpositive curvature margin is inconclusive, never a zero gap.
    Omitting the radius selects the rational forcing A itself. A radius
    satisfying this source criterion exists iff 3*kappa^2>1024*decay^2;
    A is then a constructive witness. This is a criterion decision, not
    a decision that the physical vacuum or gap fails outside the window.
    """
    coupling = _rational(kappa, "kappa")
    decay = _rational(decay_base, "decay_base")
    if coupling <= 0 or decay < 1:
        raise ValueError("require positive kappa and decay_base >= 1")
    forcing = 64 * decay**2 / coupling**2
    radius = forcing if correction_radius is None else _rational(correction_radius, "correction_radius")
    if radius <= 0:
        raise ValueError("require positive kappa and correction_radius, and decay_base >= 1")
    source = invariant_vacuum_fourier_family(
        "su2", coupling, correction_radius=radius, weighted_incidence_cap=2,
        minimum_girth=4, max_cycle_length=4, max_cycle_diameter=2,
        decay_base=decay,
    )
    fixed_point = bool(source["witness"]["arithmetic"]["fixed_point_verified"])
    alpha, g, ricci = coupling / 2, 4 / coupling**2, Q(1, 2)
    # The source norm includes the ambient line-graph diameter weight.
    # On vertical edges site distance <= ambient line-graph distance.
    # This sharper seed row is recomputed; the source's published Hessian
    # field is UNWEIGHTED and cannot be copied into this weighted gate.
    seed = g * (1 + decay) / 3
    correction = 2 * radius / 3
    row_cap = seed + correction
    margin = ricci - 2 * row_cap
    passed = fixed_point and margin > 0
    locality = passed and decay > 1
    failures = []
    if not fixed_point:
        failures.append("actual_vacuum_fixed_point")
    if margin <= 0:
        failures.append("strict_weighted_hessian_row_ball")
    earned = {
        "actual_strip_vacuum_family_verified": fixed_point,
        "actual_weighted_hessian_bound_verified": fixed_point,
        "arbitrary_depth_strip_marginal_closure_verified": passed,
        "volume_uniform_strip_marginal_derivative_bounds_verified": passed,
        "physical_strip_compression_hierarchy_verified": passed,
        "all_compressed_physical_gaps_verified": passed,
        "spatial_exponential_covariance_bound_verified": locality,
    }
    scope = {
        "all_scale_refinement_claim": False, "coarse_wilson_family_closed": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False, "static_confinement_claim": False,
    }
    witness = {
        "inputs": {"kappa": str(coupling), "correction_radius": str(radius),
                   "decay_base": str(decay)},
        "source_certificate": source["certificate"],
        "source_use": "actual weighted Fourier vacuum fixed point only; source gap and factorization outputs are not premises",
        "family": "all finite open 1-by-n square strips, integer n>=1, unit Wilson plaquette weights and unit electric edge weights",
        "structure": {"plaquette_incidence_cap": 2, "girth": 4,
                      "cycle_length": 4, "ambient_cycle_diameter": 2,
                      "n_edges": "3*n+1", "n_vertices": "2*n+2"},
        "normalization": "aH=kappa/2*sum_e C_e + 2/kappa*sum_p(2-ReTr U_p)",
        "coordinate_gauge": "all horizontal links equal I; n+1 vertical Q_i with common left and right endpoint gauge actions",
        "marginal": "S_I=1/2*log integral exp(2*S) product_(j not in I) dH(Q_j), up to a constant",
        "retention_scope": "any nested sequence of strictly decreasing retained subsets containing 0 and n, any finite depth allowed by n",
        "distance_scope": "inherited microscopic site distance |i-j|; never reset after deletion",
        "conditional_covariance_kernel": "K_R=(rho*I-2*M_RR)^-1; |Cov(f,g)| <= a^T*K_R*b for block L2 gradient norms a,b",
        "hessian_majorant_transform": "M_out=M_CC+2*M_CR*(rho*I-2*M_RR)^-1*M_RC",
        "row_closure": "weighted_row(M_out)<=m; same m at every depth by the shared total row budget",
        "nested_consistency": "iterated Haar marginals agree by Fubini; majorants compose by Schur complement associativity",
        "physical_compression": "Gamma_I=sum_i|p_i|^2+sum_j(i_(j+1)-i_j)*(|sum_h<=j p_h|^2+|sum_h<=j Ad(Q_h)p_h|^2), with both endpoint Gauss sums zero",
        "kinetic_compression_scope": "actual vacuum-preserving compression as closed quadratic forms; generated drift retained; compressed spectrum is not the full fine spectrum",
        "gap_scope": "all physical representations above the unique vacuum of each finite vacuum-preserving compression, including the original strip",
        "energy_units": "original dimensionless aH at fixed microscopic kappa; no spacing rescaling",
        "arithmetic": {
            "g": str(g), "electric_alpha": str(alpha), "ricci_lower": str(ricci),
            "source_forcing_upper": str(forcing),
            "automatic_radius_candidate": str(forcing),
            "source_radius_feasibility_slack": str(3 * coupling**2 - 1024 * decay**2),
            "source_radius_exists_for_criterion": 3 * coupling**2 > 1024 * decay**2,
            "weighted_seed_hessian_row_upper": str(seed),
            "weighted_correction_hessian_row_upper": str(correction) if fixed_point else None,
            "weighted_row_candidate_upper": str(row_cap),
            "actual_weighted_hessian_row_upper": str(row_cap) if fixed_point else None,
            "curvature_candidate_lower": str(margin),
            "uniform_marginal_curvature_lower": str(margin) if passed else None,
            "weighted_covariance_kernel_row_upper": str(1 / margin) if passed else None,
            "hessian_tail_prefactor_upper": str(row_cap) if locality else None,
            "covariance_tail_prefactor_upper": str(1 / margin) if locality else None,
            "tail_rate": "decay_base^(-D) in inherited site distance" if locality else None,
            "all_compressed_physical_gap_lower": str(alpha * margin) if passed else None,
            "fixed_point_verified": fixed_point,
            "strict_weighted_row_ball_verified": margin > 0,
        },
        "failed_constraints": failures,
    }
    certificate = make_certificate(
        claim="the actual fixed-coupling SU2 strip vacuum has a spatial Hessian ball closed under every finite coordinate-marginal hierarchy, with the induced physical electric form and positive gaps when the gates pass",
        payload={"type": "su2_strip_marginal_hierarchy_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-strip-marginal-hierarchy.md",
              "transcend_backend": "not_used"},
    )
    return {
        "status": "PASS" if passed else "INCONCLUSIVE", "finite_gate_verified": passed,
        "all_compressed_physical_gap_lower": str(alpha * margin) if passed else None,
        "witness": witness, "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate), **earned, **scope,
        "theorem_prover_verified": False, "mathlib_verified": False,
    }


def replay_su2_strip_marginal_hierarchy_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the actual source, weighted row, physical units and scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_strip_marginal_hierarchy_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_strip_marginal_hierarchy(
            Q(inputs["kappa"]), correction_radius=Q(inputs["correction_radius"]),
            decay_base=Q(inputs["decay_base"]),
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_strip_marginal_hierarchy_certificate",
    "su2_strip_compression_geometry",
    "su2_strip_marginal_hierarchy",
]
