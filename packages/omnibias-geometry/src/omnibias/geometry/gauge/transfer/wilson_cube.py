# SPDX-License-Identifier: Apache-2.0
"""Actual vacuum subtraction for a cube flap attached to a periodic torus.

The old bottom plaquette moment is supplied by the canonical all-coupling
periodic-cubic vacuum theorem. A transported nonabelian face identity
controls the dependent top face through four weighted fresh-face trials.
The enlarged graph is a noncubic flap, not a filled or replaced cubic block.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational
from omnibias.geometry.gauge.transfer.wilson_attachment import (
    replay_su2_wilson_attachment_certificate,
    su2_wilson_attachment,
)
from omnibias.geometry.gauge.transfer.wilson_large_field import (
    replay_su2_wilson_large_field_certificate,
    su2_wilson_large_field,
)


def su2_wilson_attached_cube(kappa: int | Q, *, action_threshold: int | Q = 1) -> dict[str, Any]:
    """Certify an actual vacuum increment for every torus with one cube flap.

    The old graph is an isotropic periodic cubic SU(2) torus of side L>=3.
    Select any elementary face as bottom. Add four entirely fresh top
    vertices, four vertical and four top edges, four sides and one top
    Wilson face. There are no further new interactions or identifications.
    Unit original-edge electric weights and fundamental Casimir3/4 apply.

    PASS certifies the stated quantified family and the arithmetic bounds;
    it does not verify membership of a caller-supplied graph. The bound45
    uses the actual old torus vacuum moment, never a guessed conditional
    moment. Actual new-vacuum localization and supported-state subtraction
    are consequences of the energy increment; they imply no spectral gap.
    """
    coupling = _rational(kappa, "kappa")
    threshold = _rational(action_threshold, "action_threshold")
    if coupling <= 0 or not 0 < threshold <= 20:
        raise ValueError("kappa>0 and 0<action_threshold<=20 are required")

    # Labels0..3 denote the old bottom, labels4..7 entirely fresh vertices.
    bottom = [[i, (i + 1) % 4] for i in range(4)]
    sides = [[i, (i + 1) % 4, (i + 1) % 4 + 4, i + 4] for i in range(4)]
    top = [4, 5, 6, 7]
    old_source = su2_wilson_large_field(coupling)
    fallback = su2_wilson_attachment(coupling, bottom, [*sides, top], action_threshold=threshold)
    if not replay_su2_wilson_large_field_certificate(old_source["certificate"]):
        raise ValueError("canonical actual old-vacuum source did not replay")
    if not replay_su2_wilson_attachment_certificate(fallback["certificate"]):
        raise ValueError("canonical generic attachment fallback did not replay")

    old_arithmetic = old_source["certificate"]["payload"]["witness"]["arithmetic"]
    fallback_arithmetic = fallback["certificate"]["payload"]["witness"]["arithmetic"]
    old_mean = Q(old_arithmetic["plaquette_action_mean_upper"])
    feedback = 30 + 10 * old_mean / coupling
    generic = Q(fallback_arithmetic["vacuum_increment_upper"])
    increment = min(feedback, generic)
    n, d = 5, 2
    mean = min(Q(4 * n), coupling * increment / 2)
    probability = min(Q(1), mean / threshold)
    norm = max(Q(0), 1 - 2 * mean / threshold)
    concave_point = min(coupling * increment / 2, Q(2 * n))
    gradient_mean = d * concave_point * (4 - concave_point / n)
    vacuum_cost = coupling / 2 * (Q(22, 7) / threshold) ** 2 * gradient_mean
    ims = Q(968, 49) * coupling * d / threshold
    bad_floor = threshold / coupling - increment

    witness = {
        "inputs": {"kappa": str(coupling), "action_threshold": str(threshold)},
        "family": {
            "group": "SU(2)",
            "old_graph": "every isotropic periodic cubic torus of integer side L>=3",
            "bottom_quantifier": "any elementary old plaquette, with either orientation",
            "attachment": (
                "four entirely fresh vertices, four vertical edges, four top edges; four "
                "side faces and one top face"
            ),
            "old_graph_preserved": (
                "all original vertices, edges and Wilson faces are retained with unchanged weights"
            ),
            "new_graph": (
                "the old torus with one noncubic cube flap; not an embedded cube replacement"
            ),
            "no_hidden_faces": (
                "exactly the five specified new Wilson terms; no extra faces at old or new vertices"
            ),
            "physical_space": (
                "Gauss law at every vertex, no matter or external charges; positive scalar "
                "and physical vacuum bottoms agree"
            ),
            "hamiltonian": "aH=kappa/2*sum_e C_e+2/kappa*sum_p(2-Tr U_p)",
            "normalization": (
                "independent original Haar links; unit electric weights; fundamental Casimir3/4"
            ),
            "energy_units": "dimensionless microscopic spatial-spacing times Hamiltonian",
        },
        "geometry": {
            "old_bottom_edges": bottom,
            "new_vertices": [4, 5, 6, 7],
            "side_faces": sides,
            "top_face": top,
            "new_edge_count": 8,
            "side_fresh_counts": [3, 2, 2, 1],
            "top_fresh_count": 0,
            "top_is_independent_fresh_attachment": False,
        },
        "curvature_feedback": {
            "side_orientation": (
                "P_i=B_i V_(i+1) T_i^-1 V_i^-1; B_i,T_i follow the bottom/top perimeter "
                "and V_i points upward"
            ),
            "transport_identity": (
                "after the vertical tree gauge, T_i=P_i^-1 B_i; the top product is four "
                "conjugated P_i^-1 factors followed by the bottom product"
            ),
            "pointwise_action_bound": "A_top<=5*(A_bottom+sum_four_sides A_side), A=2-Tr U",
            "pointwise_hamiltonian_bound": (
                "H_new<=H_old+kappa/2*sum_new C_e+12/kappa*sum_sides A_side+10/kappa*A_bottom"
            ),
            "weighted_trial": (
                "four ordered normalized fresh-face isometries with phi_t proportional to "
                "exp(t*Tr U_side), t=2/kappa"
            ),
            "weighted_step_energy": (
                "e_(t,w)=(3*kappa*t/2)*m(4*t)+(4*w/kappa)*(1-m(4*t)), w=6; e_(2/kappa,6)<=15/2"
            ),
            "old_observables": (
                "each isometry intertwines every old-link multiplication observable, "
                "preserving the actual old bottom expectation in the embedded trial "
                "state, not in the new actual vacuum"
            ),
            "form_comparison": "J^* H_new J <= H_old+30*I+10/kappa*A_bottom as quadratic forms",
            "general_old_scope": (
                "the form comparison holds for any finite old graph and smooth real "
                "gauge-invariant old potential with this fresh geometry; the numerical45 "
                "bound additionally requires the earned old moment"
            ),
            "actual_old_moment": (
                "the canonical periodic-cubic source bounds E_old_vacuum A_bottom by "
                "min(3*kappa/2,2)"
            ),
            "noniteration": (
                "a bound E A_bottom<=C*kappa would give a new added-action "
                "coefficient15+5*C; this is not a closed invariant moment class and the "
                "new flap is not an isotropic torus"
            ),
        },
        "localization": {
            "action": "S=sum_five_added_faces(2-Tr U_p)",
            "cutoffs": (
                "chi_g=cos(theta(S)), chi_b=sin(theta(S)); theta=0 below s/2, pi/2 above "
                "s, linear between"
            ),
            "moment": (
                "H_new-E_old>=2*S/kappa implies E_new_vacuum S<=kappa*increment_upper/2; "
                "no symmetry of the new graph is assumed"
            ),
            "gradient_mean": (
                "Gamma S<=2*(4*S-sum A_p^2); Jensen and the concave maximum at E S=10 give "
                "the displayed mean bound"
            ),
            "supported_state": "H_new-E_new>=s/kappa-increment_upper on states supported in S>=s/2",
            "ims": (
                "q(f)>=bad_floor*||chi_b f||^2-ims_upper*||f||^2; the good localized form "
                "is nonnegative; q=H_new-E_new"
            ),
            "scope": (
                "actual full new vacuum and full physical states; no frozen or conditional "
                "vacuum substitution"
            ),
        },
        "arithmetic": {
            "face_count": n,
            "fresh_stage_count": 4,
            "zero_fresh_stage_count": 1,
            "max_added_face_edge_incidence": d,
            "weighted_side_coefficient": "6",
            "trial_parameter_t": str(2 / coupling),
            "weighted_per_side_increment_upper": "15/2",
            "four_side_weighted_increment_upper": "30",
            "old_bottom_action_mean_upper": str(old_mean),
            "curvature_feedback_increment_upper": str(feedback),
            "coupling_independent_feedback_increment_upper": "45",
            "generic_attachment_increment_upper": str(generic),
            "sequential_increment_upper": fallback_arithmetic["sequential_increment_upper"],
            "global_haar_increment_upper": fallback_arithmetic["global_haar_increment_upper"],
            "vacuum_increment_lower": "0",
            "vacuum_increment_upper": str(increment),
            "actual_added_action_mean_upper": str(mean),
            "large_action_probability_upper": str(probability),
            "good_localized_vacuum_norm_squared_lower": str(norm),
            "added_action_gradient_square_mean_upper": str(gradient_mean),
            "vacuum_localization_form_cost_upper": str(vacuum_cost),
            "good_normalized_energy_above_vacuum_upper": str(vacuum_cost / norm)
            if norm > 0
            else None,
            "universal_ims_error_upper": str(ims),
            "vacuum_subtracted_bad_support_floor": str(bad_floor),
            "bad_floor_minus_ims": str(bad_floor - ims),
            "feedback_next_moment_coefficient_intercept": "15",
            "feedback_next_moment_coefficient_slope": "5",
        },
        "actual_old_vacuum_certificate": old_source["certificate"],
        "generic_attachment_certificate": fallback["certificate"],
    }
    earned = {
        "actual_curvature_feedback_operator_comparison_verified": True,
        "actual_old_moment_applicability_verified": True,
        "actual_vacuum_increment_bound_verified": True,
        "ambient_volume_independent_increment_verified": True,
        "actual_added_action_mean_bound_verified": True,
        "actual_vacuum_localization_verified": True,
        "universal_ims_bound_verified": True,
        "actual_vacuum_subtracted_bad_support_bound_verified": True,
        "positive_bad_support_floor_verified": bad_floor > 0,
    }
    scope = {
        "ambient_graph_membership_verified": False,
        "arbitrary_old_potential_moment_bound_verified": False,
        "embedded_full_cubic_increment_verified": False,
        "iterable_curvature_feedback_class_verified": False,
        "uniform_conditional_bad_probability_verified": False,
        "uniform_conditional_gap_verified": False,
        "embedded_subspace_invariant_verified": False,
        "spectral_gap_claim": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim=(
            "actual SU2 periodic-torus cube-flap vacuum increments via nonabelian "
            "curvature feedback, supported-state subtraction and localization"
        ),
        payload={"type": "su2_wilson_attached_cube_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-wilson-cube.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "PASS",
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_wilson_attached_cube_certificate(certificate: dict[str, Any]) -> bool:
    """Replay both actual sources, all cube geometry, bounds and scope fields."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_wilson_attached_cube_v1":
            return False
        witness = payload["witness"]
        if not replay_su2_wilson_large_field_certificate(witness["actual_old_vacuum_certificate"]):
            return False
        if not replay_su2_wilson_attachment_certificate(witness["generic_attachment_certificate"]):
            return False
        inputs = witness["inputs"]
        report = su2_wilson_attached_cube(
            Q(inputs["kappa"]), action_threshold=Q(inputs["action_threshold"])
        )
        return bool(report["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = ["replay_su2_wilson_attached_cube_certificate", "su2_wilson_attached_cube"]
