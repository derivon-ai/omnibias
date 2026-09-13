# SPDX-License-Identifier: Apache-2.0
"""Static fundamental-source bounds from the actual cubic Wilson vacuum."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.wilson_polar_source import (
    replay_su2_wilson_polar_vacuum_certificate,
)
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    replay_su2_wilson_linear_vacuum_certificate,
)


def su2_wilson_static_family(source_certificate: dict[str, Any]) -> dict[str, Any]:
    """Convert original-edge conditional curvature into linear static energy.

    The input must be a canonical cubic linear-source or polar-source certificate.
    In particular, the strip's vertical-chart curvature and an inverse-only
    reference certificate cannot supply the original-edge cut premise.
    No caller-supplied curvature or physical mass is accepted.
    """
    if not (replay_su2_wilson_linear_vacuum_certificate(source_certificate)
            or replay_su2_wilson_polar_vacuum_certificate(source_certificate)):
        raise ValueError("a canonical Wilson linear-source or polar-source certificate is required")
    source = source_certificate["payload"]["witness"]
    if source["inputs"]["family"] != "cubic":
        raise ValueError("the source must supply original-edge cubic curvature")
    coupling = Q(source["inputs"]["kappa"])
    arithmetic = source["arithmetic"]
    fixed_point = bool(arithmetic["fixed_point_verified"])
    curvature = Q(arithmetic["curvature_lower"]) if arithmetic["curvature_lower"] is not None else None
    passed = fixed_point and curvature is not None and curvature > 0
    lower = coupling * curvature / 2 if passed and curvature is not None else None
    upper = 3 * coupling / 8 if passed else None
    witness = {
        "source_certificate": source_certificate,
        "family": source["family"],
        "normalization": source["normalization"],
        "energy_units": "dimensionless original aH",
        "kappa": str(coupling),
        "quantifier": "every stated finite open cubic box and every pair of distinct source vertices",
        "source_sector": "one fundamental and one antifundamental static source; singlet Gauss law at every other vertex",
        "dynamical_matter": False,
        "source_bare_rest_energy": "0",
        "vacuum_subtraction": "the true neutral ground energy E0, via its positive vacuum Dirichlet transform",
        "distance": "ordinary undirected graph distance d(s,t), in microscopic edges",
        "conditional_curvature_lower": str(curvature) if curvature is not None else None,
        "cut_argument": "a center gauge transformation on a source-separating vertex set flips all cut links and makes every charged component conditionally odd; original-edge conditional Poincare gives rho per cut",
        "cut_count": "the d(s,t) distance-ball edge cuts are nonempty and pairwise edge disjoint",
        "upper_trial": "fundamental transport along a shortest path, normalized pointwise Hilbert-Schmidt norm1 and total Casimir3*d(s,t)/4",
        "linear_static_energy_coefficients": [str(lower), str(upper)] if passed else None,
        "conclusion": "lower*d(s,t)<=E_st-E0<=upper*d(s,t)",
        "all_spins_included": True,
        "failed_constraints": [] if passed else ["actual_cubic_vacuum_with_positive_original_edge_curvature"],
    }
    earned = {"finite_graph_family_confinement": passed,
              "actual_vacuum_cut_poincare_verified": passed}
    scope = {
        "infinite_volume_static_potential_claim": False,
        "asymptotic_string_tension_claim": False,
        "isotropic_euclidean_wilson_identification_verified": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="linear all-spin static fundamental-source energy bounds in the replayed fixed-coupling cubic family",
        payload={"type": "su2_wilson_static_family_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-wilson-static-source.md",
              "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if passed else "INCONCLUSIVE", "finite_gate_verified": passed,
            "witness": witness, "certificate": certificate,
            "digest_verified": verify_certificate_digest(certificate),
            **earned, **scope, "theorem_prover_verified": False, "mathlib_verified": False}


def replay_su2_wilson_static_family_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the static bounds, including replay of their complete source."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_static_family_v1":
            return False
        result = su2_wilson_static_family(payload["witness"]["source_certificate"])
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_wilson_static_family_certificate",
    "su2_wilson_static_family",
]
