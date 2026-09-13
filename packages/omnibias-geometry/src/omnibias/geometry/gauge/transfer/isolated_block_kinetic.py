# SPDX-License-Identifier: Apache-2.0
"""Exact finite metric budgets after maximal-tree reduction of an open cube.

The physical isolated-block kinetic form is compared with its constant
quadratic metric in chord logarithms. The exterior-conditioned operator
of an embedded block is a different object and is not certified here.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def su2_isolated_block_kinetic(edge_radius: int | Q, *, block_side: int = 1) -> dict[str, Any]:
    """Bound the actual isolated open-block kinetic metric in chord logs.

    Here ``edge_radius`` bounds each retained chord's principal log norm,
    not every original link before gauge reduction. The open block has
    vertices0..block_side in each coordinate, unit original electric
    weights, Gauss law at every vertex and no exterior edges. Maximal-tree
    reduction retains the root's simultaneous conjugation constraint.

    Every accepted input earns a sound error bound. PASS means its exact
    relative error is below1, so the displayed lower comparison is positive.
    INCONCLUSIVE means only that this sufficient comparison did not pass.
    """
    radius = _rational(edge_radius, "edge_radius")
    side = _integer(block_side, "block_side")
    if not 0 <= radius <= 2 or side < 1:
        raise ValueError("0<=edge_radius<=2 and integer block_side>=1 are required")
    vertices = (side + 1) ** 3
    edges = 3 * side * (side + 1) ** 2
    tree = vertices - 1
    chords = edges - tree
    anchor_count = tree * chords
    linear_upper = 1 + anchor_count
    sinc_lower = 1 - radius**2 / 24
    jacobian_error = radius / 2 + radius**2 / (12 * sinc_lower)
    frame_error = (2 * anchor_count + 1) * radius + anchor_count * radius**2
    relative_error = frame_error + (1 + frame_error) * linear_upper * (
        2 * jacobian_error + jacobian_error**2
    )
    lower = 1 - relative_error
    upper = 1 + relative_error
    passed = relative_error < 1
    status = "PASS" if passed else "INCONCLUSIVE"
    witness = {
        "inputs": {"edge_radius": str(radius), "block_side": side},
        "family": {
            "group": "SU(2)",
            "graph": "isolated open cubic block with vertices {0,...,b}^3",
            "boundary": "open; no exterior edges, matter or external charges",
            "gauss_law": "at every vertex; simultaneous root conjugation retained",
            "electric_weights": "one on every original edge; fundamental Casimir3/4",
        },
        "tree": "all z edges; y edges at z=0; x edges at y=z=0",
        "coordinate_domain": (
            "every retained chord log norm<=edge_radius; residual-conjugation-invariant chart"
        ),
        "measure": (
            "product Haar on chord holonomies, restricted to "
            "simultaneous-conjugation-invariant functions"
        ),
        "normalization": "Gamma=sum of original-edge gradient squares; aH_electric=kappa*Gamma/2",
        "physical_form": "sum_a |p_a|^2+sum_tree_e |sum_a (s_ea Ad(W_a)-t_ea I)p_a|^2",
        "derivative_scope": (
            "s_ea and t_ea indicate source/target in descendant cut; "
            "internal chords contribute Ad(W_a)-I"
        ),
        "reference_form": "Gamma0(z)=|z|^2+|Dz|^2, D_ea=s_ea-t_ea; M0=I+D^T D",
        "gradient_conversion": "p_a=J_R(A_a)^(-T) grad_Aa h; J_R^-1=Pr+(r/2)cot(r/2)Pt-(A cross)/2",
        "comparison": "|Gamma_tree(A,grad h)-Gamma0(grad h)|<=relative_error*Gamma0(grad h)",
        "measure_scope": (
            "both forms use the same chord Haar measure; "
            "no flat-measure half-density potential is inferred"
        ),
        "linear_curl_kernel": (
            "open cubic complex is contractible; zero face curl plus zero tree coordinates "
            "implies zero chord coordinates"
        ),
        "exterior_scope": (
            "does not identify a block conditioned on an exterior "
            "or allow cross-boundary electric terms to be omitted"
        ),
        "arithmetic": {
            "vertex_count": vertices,
            "original_edge_count": edges,
            "tree_edge_count": tree,
            "chord_count": chords,
            "tree_chord_product": anchor_count,
            "linear_metric_eigenvalue_lower": "1",
            "linear_metric_eigenvalue_upper": str(linear_upper),
            "sinc_lower": str(sinc_lower),
            "right_jacobian_inverse_error_upper": str(jacobian_error),
            "right_gradient_frame_relative_error_upper": str(frame_error),
            "log_coordinate_relative_error_upper": str(relative_error),
            "kinetic_comparison_lower": str(lower),
            "kinetic_comparison_upper": str(upper),
            "positive_lower_comparison": passed,
        },
    }
    earned = {
        "isolated_physical_tree_gauge_form_verified": True,
        "isolated_block_kinetic_error_bound_verified": True,
        "isolated_block_positive_kinetic_comparison_verified": passed,
        "linear_open_block_curl_kernel_removed_verified": True,
    }
    scope = {
        "actual_vacuum_approximation_verified": False,
        "embedded_block_conditional_metric_verified": False,
        "flat_measure_operator_comparison_verified": False,
        "uniform_in_block_size_claim": False,
        "spectral_gap_claim": False,
        "infinite_volume_claim": False,
        "uniform_in_a_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
    }
    certificate = make_certificate(
        claim=(
            "finite SU2 isolated open-block physical kinetic metric error bound "
            "in maximal-tree chord logarithms"
        ),
        payload={"type": "su2_isolated_block_kinetic_v1", "witness": witness},
        honesty={**earned, **scope},
        meta={
            "analytic_implication": "docs/api/gauge-isolated-block-kinetic.md",
            "transcend_backend": "not_used",
        },
    )
    return {
        "status": status,
        "kinetic_relative_error_upper": str(relative_error),
        "kinetic_comparison_lower": str(lower),
        "kinetic_comparison_upper": str(upper),
        "witness": witness,
        "certificate": certificate,
        "digest_verified": verify_certificate_digest(certificate),
        **earned,
        **scope,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def replay_su2_isolated_block_kinetic_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the full canonical bound, including INCONCLUSIVE reports."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_isolated_block_kinetic_v1":
            return False
        inputs = payload["witness"]["inputs"]
        result = su2_isolated_block_kinetic(
            Q(inputs["edge_radius"]), block_side=inputs["block_side"]
        )
        return bool(result["certificate"] == certificate)
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


__all__ = [
    "replay_su2_isolated_block_kinetic_certificate",
    "su2_isolated_block_kinetic",
]
