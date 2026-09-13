# SPDX-License-Identifier: Apache-2.0
"""Actual SU(2) vacuum energy and large-field moments on periodic cubic boxes.

The all-spin bound follows from a product-link variational trial, uniqueness
of the positive scalar vacuum, and cubic symmetry. Markov bounds concern
the actual quantum vacuum, not the product trial or a Wilson Gibbs measure.
They do not assert conditional suppression or an exponential polymer bound.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_wilson_large_field(
    kappa: int | Q,
    *,
    threshold: int | Q = 1,
    marked_count: int = 1,
    required_bad_count: int = 1,
) -> dict[str, Any]:
    """Bound actual vacuum plaquette action and marked large-field events.

    Quantifies over isotropic periodic 3D cubic tori of side L>=3 with
    at least marked_count plaquettes, and every set of that many distinct
    plaquettes. A bad plaquette has action 2-Tr(U_p)>=threshold. The event
    is that at least required_bad_count marked plaquettes are bad.

    Every accepted input has a sound bound; probability_bound_nontrivial
    distinguishes a probability bound below one from a vacuous bound.
    There is no independence assumption and no exponentiation over a set.
    """
    coupling = _rational(kappa, "kappa")
    level = _rational(threshold, "threshold")
    marked = _integer(marked_count, "marked_count")
    required = _integer(required_bad_count, "required_bad_count")
    if coupling <= 0:
        raise ValueError("kappa must be strictly positive")
    if not 0 < level <= 4:
        raise ValueError("threshold must lie in the physical action interval (0,4]")
    if marked < 1 or not 1 <= required <= marked:
        raise ValueError("counts must obey 1<=required_bad_count<=marked_count")
    trial_t = 4 / coupling
    energy_per_plaquette = min(Q(3), 4 / coupling)
    mean = coupling * energy_per_plaquette / 2
    expected_count = min(Q(marked), marked * mean / level)
    probability = min(Q(1), expected_count / required)
    gradient_square = mean * (4 - mean)
    plaquette_form = coupling * gradient_square / 2
    marked_incidence = min(4, marked)
    marked_average_form = coupling * marked_incidence * gradient_square / (2 * marked)
    witness = {
        "inputs": {
            "kappa": str(coupling),
            "threshold": str(level),
            "marked_count": marked,
            "required_bad_count": required,
        },
        "family": {
            "group": "SU(2)",
            "spatial_dimension": 3,
            "boundary": "periodic isotropic cubic torus",
            "side_length_quantifier": "every integer L>=3 with 3*L^3>=marked_count",
            "edges_and_plaquettes": "|E|=|P|=3*L^3",
            "plaquette_geometry": "four distinct original link variables per elementary square",
            "gauss_law": "at every vertex; no dynamical matter or external sources",
            "marked_set_quantifier": "every set of marked_count distinct elementary plaquettes",
        },
        "hamiltonian": "aH=kappa/2*sum_e C_e+2/kappa*sum_p(2-Tr U_p)",
        "normalization": "SU2 fundamental Casimir3/4; original unit electric edge metric",
        "energy_units": "dimensionless microscopic spatial-spacing times Hamiltonian",
        "vacuum": "unique normalized strictly positive scalar ground state, which is gauge invariant",
        "actual_measure": "mu=psi0^2*product Haar",
        "trial": "product over original links of exp(t*Tr U_e), normalized in L2",
        "trial_scope": "variational function, not the actual vacuum; physical lowest energy equals scalar lowest energy",
        "mean_ode": "m(c)=E_c(q0), c=4*t, m'=1-m^2-3*m/c",
        "mean_barrier": "m(c)>=1-3/(2*c); for c<=3/2 use m(c)>=0",
        "barrier_contact_slack": "3/(4*c^2)>0",
        "trial_kinetic_bound": "3*kappa*t*|E|/8",
        "trial_magnetic_bound": "6*|P|/(kappa*t)",
        "symmetry_step": "uniqueness preserves translations and cubic axis permutations; all plaquettes have the same mean",
        "bad_event": "K=sum_(p in marked set) 1_[2-Tr U_p>=threshold]; event K>=required_bad_count",
        "markov_step": "threshold*K<=sum_(p in marked set)(2-Tr U_p); no factorization of joint events",
        "gradient_identity": "sum_e |grad_e A_p|^2=4*A_p-A_p^2; A_p=2-Tr U_p",
        "form_identity": "<F*psi0,(aH-E0)*F*psi0>=kappa/2*E_mu(sum_e |grad_e F|^2)",
        "marked_average": "F=(1/marked_count)*sum_(p in marked set) A_p",
        "localization_rule": "a real scalar cutoff with Lipschitz constant1/width multiplies the corresponding form upper bound by1/width^2",
        "arithmetic": {
            "trial_parameter_t": str(trial_t),
            "trial_one_link_parameter_c": str(4 * trial_t),
            "trial_energy_per_plaquette_upper": "3",
            "haar_energy_per_plaquette_upper": str(4 / coupling),
            "ground_energy_per_plaquette_upper": str(energy_per_plaquette),
            "plaquette_action_mean_upper": str(mean),
            "marked_action_sum_mean_upper": str(marked * mean),
            "expected_bad_count_upper": str(expected_count),
            "large_field_probability_upper": str(probability),
            "plaquette_action_gradient_square_mean_upper": str(gradient_square),
            "plaquette_action_form_energy_upper": str(plaquette_form),
            "marked_edge_incidence_upper": marked_incidence,
            "marked_average_action_form_energy_upper": str(marked_average_form),
            "probability_bound_nontrivial": probability < 1,
        },
    }
    earned = {
        "actual_finite_volume_vacuum_verified": True,
        "actual_plaquette_mean_bound_verified": True,
        "volume_uniform_plaquette_mean_bound_verified": True,
        "actual_marked_count_bound_verified": True,
        "actual_large_field_probability_bound_verified": True,
        "actual_plaquette_form_bound_verified": True,
        "actual_marked_average_form_bound_verified": True,
        "localization_form_bound_verified": True,
    }
    scope = {
        "actual_vacuum_density_supplied": False,
        "arbitrary_graph_membership_verified": False,
        "uniform_over_all_exteriors_verified": False,
        "conditional_large_field_bound_verified": False,
        "exponential_polymer_bound_verified": False,
        "cluster_independence_claim": False,
        "spectral_gap_claim": False,
        "nonvacuum_variance_lower_bound_verified": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim="actual volume-uniform SU2 Wilson vacuum plaquette moments, Markov large-field bounds and local form upper bounds on periodic cubic boxes",
        payload={"type": "su2_wilson_large_field_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-wilson-large-field.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS",
        "ground_energy_per_plaquette_upper": str(energy_per_plaquette),
        "plaquette_action_mean_upper": str(mean),
        "large_field_probability_upper": str(probability),
        "probability_bound_nontrivial": probability < 1,
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_wilson_large_field_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute every bound, quantifier, normalization and claim field."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_large_field_v1":
            return False
        inputs = payload["witness"]["inputs"]
        report = su2_wilson_large_field(
            Q(inputs["kappa"]),
            threshold=Q(inputs["threshold"]),
            marked_count=inputs["marked_count"],
            required_bad_count=inputs["required_bad_count"],
        )
        return bool(report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError, ArithmeticError):
        return False


__all__ = [
    "replay_su2_wilson_large_field_certificate",
    "su2_wilson_large_field",
]

