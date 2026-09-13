# SPDX-License-Identifier: Apache-2.0
"""Actual theta physical-block bounds and conditional overlap arithmetic.

See docs/api/gauge-physical-blocks.md for the internal-Gauss Poisson proof.
Only the theta source consumer earns actual-vacuum consequences. A numerical
joint residual or a generic overlap budget never supplies its own premises.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from math import isqrt
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.adjacent_gap_comparison import (
    su2_adjacent_vacuum_gap_comparison,
)
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational

_SCOPE = {
    "uniform_in_volume_claim": False,
    "arbitrary_exterior_graph_claim": False,
    "all_scale_refinement_claim": False,
    "continuum_claim": False,
    "yang_mills_claim": False,
    "yang_mills_mass_gap_claim": False,
    "analytic_proof_formally_verified": False,
    "theorem_prover_verified": False,
    "mathlib_verified": False,
}


def _count(value: int, name: str, upper: int) -> int:
    result = _integer(value, name)
    if not 1 <= result <= upper:
        raise ValueError(f"{name} must lie in [1,{upper}]")
    return result


def _read_q(inputs: dict[str, Any], name: str) -> Q:
    text = inputs[name]
    if type(text) is not str:
        raise TypeError("serialized rationals must be strings")
    value = Q(text)
    if str(value) != text:
        raise ValueError("serialized rationals must use canonical spelling")
    return value


def _seal(kind: str, payload: dict[str, Any], claim: str) -> dict[str, Any]:
    payload = {"type": kind, **payload, **_SCOPE}
    return {
        **deepcopy(payload),
        "certificate": make_certificate(
            claim=claim,
            payload=deepcopy(payload),
            honesty={
                key: value
                for key, value in payload.items()
                if type(value) is bool
                and key not in {"theorem_prover_verified", "mathlib_verified"}
            },
            meta={
                "transcend_backend": "not_used",
                "analytic_implication": "docs/api/gauge-physical-blocks.md",
            },
        ),
    }


def _projection_bounds(delta: Q, sweeps: int) -> dict[str, Any]:
    return {
        "maximal_correlation_upper": str(delta),
        "alternating_sweeps": sweeps,
        "alternating_projection_norm_upper": str(delta ** (2 * sweeps - 1)),
        "two_block_heat_bath_gap_lower": str(1 - delta),
        "heat_bath_normalization": "L_hb=(I-E[.|A])+(I-E[.|B]); rate one for each block",
        "heat_bath_gap_is_quantum_hamiltonian_gap": False,
        "projection_scope": "two conditional projections for the same joint measure; common constants removed",
    }


def conditional_overlap_budget(
    epsilon: int | Q,
    marginal_x_lower: int | Q,
    marginal_y_lower: int | Q,
    *,
    sweeps: int = 1,
    sqrt_bits: int = 64,
) -> dict[str, Any]:
    """Check ||p-pX*pY||_L2 <= epsilon and marginal-floor consequences.

    The base measures must be probability measures. L-infinity error also
    bounds this L2 error. These are UNVERIFIED input hypotheses. No total
    variation bound may be substituted for epsilon without a separate proof.
    A dyadic upper square root supplies explicit projection/heat-bath bounds.
    """
    error = _rational(epsilon, "epsilon")
    mx, my = (
        _rational(marginal_x_lower, "marginal_x_lower"),
        _rational(marginal_y_lower, "marginal_y_lower"),
    )
    n, bits = _count(sweeps, "sweeps", 256), _count(sqrt_bits, "sqrt_bits", 512)
    if error < 0 or not 0 < mx <= 1 or not 0 < my <= 1:
        raise ValueError("require epsilon>=0 and marginal probability-density floors in (0,1]")
    squared = error**2 / (mx * my)
    scale = 1 << bits
    integer = isqrt(squared.numerator * scale**2 // squared.denominator)
    if Q(integer, scale) ** 2 < squared:
        integer += 1
    delta = min(Q(1), Q(integer, scale))
    passed = delta < 1
    return _seal(
        "conditional_overlap_budget_v1",
        {
            "inputs": {
                "epsilon": str(error),
                "marginal_x_lower": str(mx),
                "marginal_y_lower": str(my),
                "sweeps": n,
                "sqrt_bits": bits,
            },
            "status": "PASS" if passed else "INCONCLUSIVE",
            "finite_gate_verified": passed,
            "arithmetic": {
                "correlation_squared_majorant": str(squared),
                **_projection_bounds(delta, n),
            },
            "premises": [
                "p is the actual normalized joint density relative to two probability base measures",
                "pX,pY are its own actual marginals, with pX>=mX and pY>=mY",
                "||p-pX*pY||_L2(base product)<=epsilon; L-infinity control suffices",
                "the two projections are conditional expectations for this same measure",
            ],
            "input_analytic_premises_verified": False,
            "actual_vacuum_verified": False,
            "actual_joint_overlap_verified": False,
            "physical_gap_verified": False,
        },
        "conditional finite overlap and two-projection budgets; analytic input hypotheses are not verified",
    )


def replay_conditional_overlap_certificate(certificate: dict[str, Any]) -> bool:
    """Canonical finite replay does not establish the supplied density bounds."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "conditional_overlap_budget_v1":
            return False
        i = payload["inputs"]
        expected = conditional_overlap_budget(
            _read_q(i, "epsilon"),
            _read_q(i, "marginal_x_lower"),
            _read_q(i, "marginal_y_lower"),
            sweeps=i["sweeps"],
            sqrt_bits=i["sqrt_bits"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def conditional_overlap_transfer_budget(
    local_gap: int | Q,
    maximal_correlation_upper: int | Q,
    *,
    decompositions: int = 1,
) -> dict[str, Any]:
    """Finite two-block-to-larger-block implication with supplied geometry.

    Each of s decompositions must have the same conditional local gap and
    angle bounds. The averaged block energy must be <=(1+1/s) times the
    larger energy, for example by pairwise disjoint overlap regions. Neither
    that geometry nor any actual-measure premise is established by this API.
    """
    gamma = _rational(local_gap, "local_gap")
    delta = _rational(maximal_correlation_upper, "maximal_correlation_upper")
    count = _count(decompositions, "decompositions", 10**6)
    if gamma <= 0 or not 0 <= delta <= 1:
        raise ValueError("require local_gap>0 and maximal_correlation_upper in [0,1]")
    multiplicity = 1 + Q(1, count)
    floor = gamma * (1 - delta) / multiplicity
    return _seal(
        "conditional_overlap_transfer_budget_v1",
        {
            "inputs": {
                "local_gap": str(gamma),
                "maximal_correlation_upper": str(delta),
                "decompositions": count,
            },
            "status": "PASS" if floor > 0 else "INCONCLUSIVE",
            "finite_gate_verified": floor > 0,
            "arithmetic": {
                "averaged_energy_multiplicity_upper": str(multiplicity),
                "larger_block_gap_lower": str(floor),
            },
            "premises": [
                "every decomposition covers the larger block and uses its SAME actual conditional measure for every exterior",
                "each component block has conditional Poincare gap at least local_gap on the stated physical domain",
                "each pair of conditional projections has maximal correlation at most delta after common constants are removed",
                "the average of the two block energies over s decompositions is at most (1+1/s) times the larger-block energy",
                "internal Gauss domains, all boundary flux sectors and form domains are compatible with the projections",
            ],
            "input_analytic_premises_verified": False,
            "actual_vacuum_verified": False,
            "physical_gap_verified": False,
            "iteration_at_all_scales_verified": False,
        },
        "conditional finite overlap transfer budget; actual conditional measures and covering inequalities remain premises",
    )


def replay_conditional_overlap_transfer_certificate(certificate: dict[str, Any]) -> bool:
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        p = certificate["payload"]
        if p["type"] != "conditional_overlap_transfer_budget_v1":
            return False
        i = p["inputs"]
        expected = conditional_overlap_transfer_budget(
            _read_q(i, "local_gap"),
            _read_q(i, "maximal_correlation_upper"),
            decompositions=i["decompositions"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def su2_theta_physical_block_gap(
    source_certificate: dict[str, Any] | None = None,
    *,
    comparison_floor: int | Q = Q(1, 5),
    exponent_steps: int = 4,
    sweeps: int = 1,
) -> dict[str, Any]:
    """Actual seven-edge theta physical-block and joint-overlap certificate.

    None constructs the canonical kappa=6, r=6/25 all-spin source. Supplied
    sources must be supported, canonical actual adjacent-vacuum certificates.
    The block proof removes only internal Gauss transformations and retains
    every boundary-flux sector. Failed canonical sources remain inconclusive.
    """
    floor = _rational(comparison_floor, "comparison_floor")
    steps, n = _count(exponent_steps, "exponent_steps", 256), _count(sweeps, "sweeps", 256)
    if floor <= 0:
        raise ValueError("comparison_floor must be positive")
    if source_certificate is None:
        from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import su2_adjacent_cone_vacuum

        source_certificate = su2_adjacent_cone_vacuum(6, correction_radius=Q(6, 25))["certificate"]
    source = deepcopy(source_certificate)
    # This consumer canonically replays the actual source and derives the
    # original coefficient-to-oscillation theorem. No caller gap is used.
    comparison = su2_adjacent_vacuum_gap_comparison(source, exponent_steps=steps)
    a = comparison["witness"]["arithmetic"]
    actual = comparison["actual_vacuum_verified"] is True
    coupling, g = Q(a["kappa"]), Q(a["g"])
    radius = (
        Q(a["selected_correction_radius"]) if a["selected_correction_radius"] is not None else None
    )
    gamma = Q(3, 4) * Q(a["neutral_exp_negative_lower"]) if actual else None
    cross = 5 * g / 12 + 2 * radius / 3 if actual and radius is not None else None
    diagonal_a = gamma - floor if gamma is not None else None
    diagonal_b = 2 * gamma - floor if gamma is not None else None
    determinant = (
        diagonal_a * diagonal_b - 4 * cross**2
        if diagonal_a is not None and diagonal_b is not None and cross is not None
        else None
    )
    passed = bool(
        actual
        and diagonal_a is not None
        and diagonal_b is not None
        and determinant is not None
        and min(diagonal_a, diagonal_b, determinant) >= 0
    )
    haar_minorant = Q(a["full_scalar_exp_negative_lower"]) if actual else None
    overlap = {
        "actual_joint_overlap_verified": actual,
        "marginals_are_actual_product_haar": actual,
        "joint_density_lower": str(haar_minorant) if haar_minorant is not None else None,
        "proof": "both block marginals are Haar; p>=m gives p=m+(1-m)r with r a doubly stochastic density",
        **(_projection_bounds(1 - haar_minorant, n) if haar_minorant is not None else {}),
    }
    return _seal(
        "su2_theta_physical_blocks_v1",
        {
            "inputs": {"comparison_floor": str(floor), "exponent_steps": steps, "sweeps": n},
            "status": "PASS" if passed else "INCONCLUSIVE",
            "finite_gate_verified": passed,
            "actual_vacuum_verified": actual,
            "physical_gap_verified": passed,
            "actual_joint_overlap_verified": actual,
            "physical_gap_lower": str(coupling * floor / 2) if passed else None,
            "source_certificate": source,
            "coefficient_comparison_certificate": comparison["certificate"],
            "geometry": {
                "vertices": 6,
                "edges": [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]],
                "block_A_edges": [1, 2, 6],
                "block_B_edges": [3, 4, 5, 7],
                "internal_A_vertices": [1],
                "internal_B_vertices": [3, 5],
                "boundary_vertices": [0, 2, 4],
                "path_coordinates": "a=e1^-1,b=e2,s=e6,P=e5*e3,Q=e7*e4^-1; x=s^-1*a*P,y=s^-1*b*Q",
                "refinement": "six-edge outer cycle gains shared edge e6 and one new physical loop",
                "conditional_A_form": "Cx+Cy+C_diagonal_left >= Cx+Cy on internal-Gauss-invariant functions",
                "conditional_B_form": "2*(Cx+Cy) on internal-Gauss-invariant functions",
                "boundary_flux_sectors": "all; no neutral boundary condition is imposed",
            },
            "arithmetic": {
                "kappa": str(coupling),
                "g": str(g),
                "source_radius": str(radius) if radius is not None else None,
                "conditional_product_gap_lower": str(gamma) if gamma is not None else None,
                "conditional_A_gap_lower": str(gamma) if gamma is not None else None,
                "conditional_B_gap_lower": str(2 * gamma) if gamma is not None else None,
                "seed_cross_hessian_coefficient_upper": "5/12",
                "seed_cross_incidence_gram_eigenvalues": ["0", "2", "6"],
                "correction_cross_hessian_coefficient_upper": "2/3",
                "mixed_log_vacuum_hessian_upper": str(cross) if cross is not None else None,
                "shifted_diagonal_A": str(diagonal_a) if diagonal_a is not None else None,
                "shifted_diagonal_B": str(diagonal_b) if diagonal_b is not None else None,
                "shifted_determinant": str(determinant) if determinant is not None else None,
                "comparison_matrix_floor": str(floor) if passed else None,
            },
            "actual_overlap": overlap,
            "normalization": "dimensionless aH=kappa*C_original/2+2*(4-chi_p-chi_q)/kappa; C_fund=3/4",
            "source_gap_used_as_premise": False,
            "restricted_poisson_argument_verified_in_written_analysis": passed,
            "conditional_internal_gauss_gaps_verified": actual,
            "all_boundary_flux_sectors_retained": True,
            "new_weak_coupling_window_claim": False,
            "new_family_gap_improvement_claim": False,
            "iteration_scope": "two conditional projections on this fixed actual theta joint measure only",
        },
        "actual theta internal-Gauss block comparison and source-bound two-block overlap; fixed graph and all spins",
    )


def replay_su2_theta_physical_block_certificate(certificate: dict[str, Any]) -> bool:
    """Recompute the actual source, nested coefficient proof, geometry and flags."""
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        p = certificate["payload"]
        if p["type"] != "su2_theta_physical_blocks_v1":
            return False
        i = p["inputs"]
        expected = su2_theta_physical_block_gap(
            p["source_certificate"],
            comparison_floor=_read_q(i, "comparison_floor"),
            exponent_steps=i["exponent_steps"],
            sweeps=i["sweeps"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError, IndexError):
        return False


def physical_block_controls(
    *,
    side: int = 3,
    mass_squared: int | Q = 1,
    rare_probability: int | Q = Q(1, 1000),
    regulator: int | Q = Q(1, 64),
    chain_length: int = 16,
    overlap_width: int = 4,
) -> dict[str, Any]:
    """Exact controls for gap, overlap, gauge modes and Gaussian comparison.

    The cycle precision is m²I+L_cycle for every integer side>=3. Its
    smallest eigenvalue is m² by the squared-difference identity and the
    constant vector. The m=0 control is an unnormalized Gaussian precision,
    not an asserted probability density on the full Euclidean space.
    """
    size = _integer(side, "side")
    length, width = _integer(chain_length, "chain_length"), _integer(overlap_width, "overlap_width")
    mass, p, eps = (
        _rational(mass_squared, "mass_squared"),
        _rational(rare_probability, "rare_probability"),
        _rational(regulator, "regulator"),
    )
    if not 3 <= size <= 10**6 or mass < 0 or not 0 < p <= Q(1, 2) or not 0 < eps <= Q(1, 2):
        raise ValueError(
            "require side in [3,10^6], mass_squared>=0, and probability/regulator in (0,1/2]"
        )
    if not 4 <= length <= 10**6 or not 0 < width <= min(512, length - 1) or length % 2 or width % 2:
        raise ValueError(
            "require even 4<=chain_length<=10^6 and even 0<overlap_width<=min(512,chain_length-1)"
        )
    marginal = p + p**3
    covariance = p - marginal**2
    tv = 2 * covariance
    # sqrt(6+eps²)>=2 gives this deliberately rational row upper bound.
    row = (3 * eps - 2) / 2
    return _seal(
        "physical_block_controls_v1",
        {
            "inputs": {
                "side": size,
                "mass_squared": str(mass),
                "rare_probability": str(p),
                "regulator": str(eps),
                "chain_length": length,
                "overlap_width": width,
            },
            "status": "PASS",
            "finite_gate_verified": True,
            "gaussian_cycle": {
                "precision": "m² I+L_cycle",
                "smallest_eigenvalue": str(mass),
                "comparison_margin": str(mass),
                "constant_vector_quadratic_form": str(size * mass),
                "identity": "x^T Q x=m² sum_i x_i²+sum_i(x_i-x_(i+1))²",
                "all_size_formula": "every integer N>=3; parameters do not depend on N",
                "probability_defined_on_full_space": mass > 0,
                "massless_constant_witness_eigenvalue": "0",
                "normalized_vanishing_mass_control": "m_N²=1/N² gives actual Gaussian gaps tending to zero",
            },
            "rare_bernoulli": {
                "joint_probabilities": [[str(1 - p - 2 * p**3), str(p**3)], [str(p**3), str(p)]],
                "total_variation_to_own_product": str(tv),
                "maximal_correlation": str(1 - p**3 / (marginal * (1 - marginal))),
                "strictly_positive_joint": True,
                "limit_as_probability_tends_to_zero": "TV tends to 0 while maximal correlation tends to 1",
                "density_residual_Linf_on_uniform_base": str(4 * covariance),
                "marginal_density_lower_on_uniform_base": str(2 * min(marginal, 1 - marginal)),
                "residual_over_marginal_floor": str(2 * covariance / min(marginal, 1 - marginal)),
                "small_total_variation_implies_overlap_contraction": False,
            },
            "pinned_gaussian_overlap": {
                "chain": "sites 0..L, endpoints fixed to zero; wings [1,(L-w)/2] and [(L+w)/2,L-1]",
                "massless_precision": "Dirichlet nearest-neighbor Laplacian; a normalized finite Gaussian probability",
                "massless_maximal_correlation": str(Q(length - width, length + width)),
                "fixed_overlap_uniform_contraction": False,
                "proportional_overlap_example": "w=L/2 gives delta=1/3 when L is a multiple of 4",
                "massive_precision": "Dirichlet nearest-neighbor Laplacian plus I",
                "massive_maximal_correlation_upper": str(Q(2, 5) ** width),
                "massive_all_size_bound": "every admissible L,w; inverse correlation length root (3-sqrt(5))/2 < 2/5",
                "endpoint_markov_reduction_verified_in_written_analysis": True,
            },
            "original_link_gaussian": {
                "reference": "Omega_epsilon=sqrt(curl*curl+epsilon² I); positive regulator retained",
                "comparison": "diag(Omega)-abs(offdiag(Omega)); multiply by 2/kappa for Poincare units",
                "unscaled_row_margin_upper": str(row),
                "positive_definite_comparison_possible": False,
                "positive_dual_weights_can_repair": False,
                "proof": "axis-constant torons give same-axis row sum epsilon; a cross-axis Fourier entry has modulus >=(sqrt(6+epsilon²)-epsilon)/2",
                "actual_nonlinear_vacuum_no_go_proved": False,
            },
            "internal_star_cap": {
                "hypothesis": "block contains a full vertex star of a gauge-invariant joint measure; all exterior links frozen",
                "test_function": "Tr(U_e)/2 on an incident link, which is gauge variant",
                "variance": "1/4",
                "dirichlet_energy": "3/16",
                "full_scalar_conditional_gap_upper": "3/4",
                "physical_internal_gauss_gap_capped_by_this_test": False,
                "actual_model_hypotheses_verified": False,
            },
            "actual_vacuum_verified": False,
            "physical_gap_verified": False,
        },
        "exact Gaussian and rare-event controls for physical-block methods; no actual Yang-Mills conclusion",
    )


def replay_physical_block_control_certificate(certificate: dict[str, Any]) -> bool:
    if not isinstance(certificate, dict):
        return False
    try:
        if not verify_certificate_digest(certificate):
            return False
        p = certificate["payload"]
        if p["type"] != "physical_block_controls_v1":
            return False
        i = p["inputs"]
        expected = physical_block_controls(
            side=i["side"],
            mass_squared=_read_q(i, "mass_squared"),
            rare_probability=_read_q(i, "rare_probability"),
            regulator=_read_q(i, "regulator"),
            chain_length=i["chain_length"],
            overlap_width=i["overlap_width"],
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "conditional_overlap_budget",
    "conditional_overlap_transfer_budget",
    "physical_block_controls",
    "replay_conditional_overlap_certificate",
    "replay_conditional_overlap_transfer_certificate",
    "replay_physical_block_control_certificate",
    "replay_su2_theta_physical_block_certificate",
    "su2_theta_physical_block_gap",
]
