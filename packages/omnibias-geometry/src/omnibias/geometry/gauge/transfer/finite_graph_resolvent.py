# SPDX-License-Identifier: Apache-2.0
"""Conditional all-spin tails for an original-link SU(2) reference linearization."""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_finite_graph_linearized_tail(
    kappa: int | Q,
    *,
    plaquette_count: int,
    plaquettes_per_edge: int,
    energy_root: int | Q,
    decay_base: int | Q = 1,
) -> dict[str, Any]:
    """Replay a tail inequality under explicit finite-graph Fourier premises.

    Let K=2*C0^-1*Pi_H*Gamma(S_star,.) and S_star=(4/(3*kappa**2))*sum chi_p.
    The returned rational number bounds ||K*Q_R|| in the original invariant
    Fourier nuclear norm if R=energy_root**2 and all listed graph, source,
    coefficient-normalization and omitted-energy premises hold.

    Counts alone do not verify those premises. No actual finite inverse,
    actual vacuum, graph geometry, or continuum implication is earned here.
    """
    coupling = _rational(kappa, "kappa")
    count = _integer(plaquette_count, "plaquette_count")
    cap = _integer(plaquettes_per_edge, "plaquettes_per_edge")
    root = _rational(energy_root, "energy_root")
    decay = _rational(decay_base, "decay_base")
    if coupling <= 0 or count < 1 or cap < 1 or root <= 0 or decay < 1:
        raise ValueError(
            "require kappa>0, plaquette_count>=1, plaquettes_per_edge>=1, "
            "energy_root>0 and decay_base>=1"
        )
    g = 4 / coupling**2
    cutoff = root**2
    anchored = 16 * g * decay**2 * count / root
    shifted = 12 * g * decay**2 * cap / cutoff
    tail = anchored + shifted
    witness = {
        "inputs": {
            "kappa": str(coupling),
            "plaquette_count": count,
            "plaquettes_per_edge": cap,
            "energy_root": str(root),
            "decay_base": str(decay),
        },
        "operator": "K=2*C0^-1*Pi_H*Gamma(S_star,.); S_star=(g/3)*sum chi_(1/2)",
        "norm": (
            "N_b(u)=max_i sum_{spin labels!=0} "
            "j_i*E_label*b^diam(support)*||original_link_coefficient||_nuclear"
        ),
        "projection": "Q_R retains input electric energies E_label>=R; R=energy_root^2",
        "approximation": (
            "K*P_R with P_R=I-Q_R retains every fusion output of retained inputs; "
            "it is not a square Galerkin truncation"
        ),
        "units": "unscaled reference linearization, g=4/kappa^2; no Hamiltonian gap",
        "external_premises": [
            "finite open square strip or rectangular cubic graph in canonical positive-axis orientation",
            "gauge invariance at every vertex and Haar mean zero",
            "each of plaquette_count elementary squares has four distinct original edges",
            "each original edge belongs to at most plaquettes_per_edge listed squares",
            "each fundamental square has original-link Fourier coefficient nuclear norm eight",
            "unit original-edge electric Casimirs with fundamental value three quarters",
            "diameter uses the ambient original edge line-graph metric and square diameter at most two",
            "omitted input Fourier labels have electric energy at least energy_root squared",
            "full omitted coefficient series belongs to the original anchored Fourier Banach space",
        ],
        "arithmetic": {
            "g": str(g),
            "energy_cutoff": str(cutoff),
            "square_nuclear_norm_premise": "8",
            "anchored_derivative_tail_upper": str(anchored),
            "shifted_anchor_tail_upper": str(shifted),
            "operator_tail_upper": str(tail),
            "operator_tail_formula": "g*b^2*(16*P/t+12*q/t^2)",
            "tail_strictly_below_one": tail < 1,
        },
        "conditional_conclusion": (
            "If every listed external premise holds, ||K*Q_R||_(N_b to N_b) "
            "<=operator_tail_upper. For each fixed graph this tends to zero as "
            "energy_root tends to infinity."
        ),
    }
    earned = {
        "conditional_tail_arithmetic_verified": True,
        "numeric_tail_below_one": tail < 1,
    }
    scope = {
        "actual_graph_verified": False,
        "actual_reference_verified": False,
        "finite_inverse_verified": False,
        "fourier_nuclear_inverse_verified": False,
        "actual_vacuum_verified": False,
        "target_hamiltonian_gap_verified": False,
        "volume_uniform_inverse_verified": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim=(
            "exact rational high-electric-energy input-tail bound conditional on "
            "the listed original-link SU2 Fourier premises"
        ),
        payload={"type": "su2_finite_graph_linearized_tail_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-finite-graph-resolvent.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": "CONDITIONAL_BOUND",
        "finite_gate_verified": True,
        "operator_tail_upper": str(tail),
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_finite_graph_linearized_tail_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute arithmetic, all explicit premises, and the complete claim scope."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_finite_graph_linearized_tail_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_finite_graph_linearized_tail(
            Q(inputs["kappa"]),
            plaquette_count=inputs["plaquette_count"],
            plaquettes_per_edge=inputs["plaquettes_per_edge"],
            energy_root=Q(inputs["energy_root"]),
            decay_base=Q(inputs["decay_base"]),
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_finite_graph_linearized_tail_certificate",
    "su2_finite_graph_linearized_tail",
]
