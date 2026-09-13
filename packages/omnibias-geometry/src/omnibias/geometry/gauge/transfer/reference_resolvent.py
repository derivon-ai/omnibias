# SPDX-License-Identifier: Apache-2.0
"""Exact premises for the reference diffusion's inverse, not the YM vacuum."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_reference_linearized_inverse(
    kappa: int | Q, *, family: str = "cubic",
    decay_base: int | Q = 1, exponent_steps: int = 4,
) -> dict[str, Any]:
    """Bound the inverse of C-2*Gamma(S_star,.) on centered L2(reference).

    S_star=(g/3)*sum chi_p and reference density is proportional to
    exp(2*S_star), not the unknown quantum vacuum density. The strip
    product route holds on gauge-invariant functions; cubic curvature
    holds on the full original-link product and hence its physical sector.
    No inverse in the original Fourier nuclear norm is asserted.
    """
    if family not in ("strip", "cubic"):
        raise ValueError("family must be 'strip' or 'cubic'")
    coupling = _rational(kappa, "kappa")
    decay = _rational(decay_base, "decay_base")
    steps = _integer(exponent_steps, "exponent_steps")
    if coupling <= 0 or decay < 1 or steps < 1:
        raise ValueError("require kappa>0, decay_base>=1, exponent_steps>=1")
    cap = 2 if family == "strip" else 4
    g = 4 / coupling**2
    curvature = Q(1, 2) - Q(4, 3) * cap * g
    weighted_row = cap * g * (1 + decay)**2 / 6
    weighted_margin = Q(1, 2) - 2 * weighted_row
    # Strip's alternative tree fixes all top and vertical links. The
    # remaining bottom links give a product reference, each with chi range4.
    omega = 8 * g / 3
    product_floor = Q(3, 4) * (1 - omega / steps)**steps if omega < steps else None
    floors = [curvature] if curvature > 0 else []
    if family == "strip" and product_floor is not None:
        floors.append(product_floor)
    floor = max(floors) if floors else None
    passed = floor is not None
    spatial = weighted_margin > 0 and decay > 1
    legacy_linear = Q(32, 3) * cap * g * decay**2
    exact_residual_linear = legacy_linear  # The previous residual improvement left this unchanged.
    witness = {
        "inputs": {"kappa": str(coupling), "family": family, "decay_base": str(decay),
                   "exponent_steps": steps},
        "family": "all finite open 1-by-n square strips, n>=1" if family == "strip" else
                  "all finite open rectangular three-dimensional cubic boxes with at least one cell per direction",
        "reference": "S_star=(g/3)*sum elementary chi_(1/2); nu=exp(2*S_star)*Haar/Z",
        "operator": "A=C-2*Gamma(S_star,.), C=sum original-edge Casimirs",
        "operator_units": "unscaled diffusion A; not aH and not an actual quantum Hamiltonian gap",
        "domain": "nu-centered gauge-invariant L2(nu); Friedrichs generator on the finite compact link product",
        "input_projection": "Pi_nu h=h-integral(h dnu), not the Haar projection",
        "haar_inverse_identity": "(Pi_H A)^(-1)=Pi_H*A_nu^(-1)*Pi_nu on Haar-centered smooth inputs",
        "preconditioned_identity": "(I-2*C0^(-1)*Pi_H*Gamma(S_star,.))^(-1)=Pi_H*A_nu^(-1)*Pi_nu*C",
        "derivative_loss": "the preconditioned identity has C on the input; L2(nu) inversion does not bound the Fourier-norm inverse",
        "strip_product_route": "top and vertical links form a spanning tree; bottom-link plaquette variables carry product exp((2g/3)*chi) density; original electric form dominates their unit product gradient form",
        "curvature_route": "original-edge Hessian row<=2*q*g/3, Ricci=1/2",
        "spatial_route": "weighted original-edge Hessian row<=q*g*(1+b)^2/6; covariance kernel weighted row<=1/(1/2-2*m_b) when positive",
        "arithmetic": {
            "g": str(g), "plaquettes_per_edge": cap,
            "curvature_lower": str(curvature),
            "weighted_hessian_row_upper": str(weighted_row),
            "weighted_covariance_margin_lower": str(weighted_margin),
            "weighted_covariance_kernel_row_upper": str(1 / weighted_margin)
            if weighted_margin > 0 else None,
            "strip_one_link_log_density_oscillation_upper": str(omega),
            "strip_exponential_domain_verified": omega < steps,
            "strip_product_poincare_lower": str(product_floor)
            if family == "strip" and product_floor is not None else None,
            "reference_poincare_lower": str(floor) if floor is not None else None,
            "reference_inverse_l2_upper": str(1 / floor) if floor is not None else None,
            "reference_dirichlet_over_forcing_l2_squared_upper": str(1 / floor)
            if floor is not None else None,
            "previous_fourier_linear_upper": str(exact_residual_linear),
        },
        "failed_constraints": [] if passed else ["no_positive_quantitative_reference_inverse_route"],
    }
    earned = {
        "reference_inverse_verified": passed,
        "volume_uniform_reference_inverse_verified": passed,
        "spatial_reference_covariance_verified": spatial,
        "beyond_previous_fourier_linear_gate_verified": passed and exact_residual_linear >= 1,
    }
    scope = {
        "fourier_nuclear_inverse_verified": False, "actual_vacuum_verified": False,
        "target_hamiltonian_gap_verified": False, "nonlinear_correction_verified": False,
        "infinite_volume_claim": False, "uniform_in_a_claim": False,
        "continuum_claim": False, "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="quantitative inverse for the auxiliary SU2 reference diffusion on centered L2(reference), under the written finite-family hypotheses",
        payload={"type": "su2_reference_linearized_inverse_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={"analytic_implication": "docs/api/gauge-reference-resolvent.md",
              "transcend_backend": "not_used"},
    )
    return {"status": "PASS" if passed else "INCONCLUSIVE",
            "finite_gate_verified": passed, "witness": witness, "certificate": certificate,
            "digest_verified": verify_certificate_digest(certificate), **earned, **scope,
            "theorem_prover_verified": False, "mathlib_verified": False}


def replay_su2_reference_linearized_inverse_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild all rational premises, norm statements and refusal flags."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_reference_linearized_inverse_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_reference_linearized_inverse(
            Q(inputs["kappa"]), family=inputs["family"], decay_base=Q(inputs["decay_base"]),
            exponent_steps=inputs["exponent_steps"],
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_reference_linearized_inverse_certificate",
    "su2_reference_linearized_inverse",
]
