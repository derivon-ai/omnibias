# SPDX-License-Identifier: Apache-2.0
"""Actual finite-strip own-marginal kernel tails, with exact symbolic bounds.

The original-link proof is in docs/api/gauge-strip-kernel-tail.md. Constants
depend on the number of plaquettes. Neither a volume-uniform tail nor an
explicit weak-coupling correlation threshold follows from this certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational
from omnibias.geometry.gauge.transfer.theta_kernel_tail import _ceil_log_two, _dyadic_exponent
from omnibias.geometry.gauge.transfer.theta_weak_blocks import _plaquette_trial_source


def _partition(n: int, retained: Sequence[int] | None) -> tuple[int, ...]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("n_plaquettes must be an integer at least two")
    if retained is None:
        return tuple(range(1, n // 2 + 1))
    if not isinstance(retained, (list, tuple)):
        raise ValueError("retained_cycles must be a list or tuple of cycle IDs")
    if any(isinstance(i, bool) or not isinstance(i, int) or not 1 <= i <= n for i in retained):
        raise ValueError("cycle IDs must be integers from 1 through n_plaquettes")
    if len(set(retained)) != len(retained) or not 0 < len(retained) < n:
        raise ValueError("retained_cycles must be a nonempty proper subset without duplicates")
    return tuple(sorted(retained))


def _strip_geometry(n: int) -> dict[str, Any]:
    # Bottom, top, then vertical original links; signed words use one-based IDs.
    edges = [[i, i + 1] for i in range(n)]
    edges += [[n + 1 + i, n + 2 + i] for i in range(n)]
    edges += [[i, n + 1 + i] for i in range(n + 1)]
    return {
        "n_vertices": 2 * n + 2,
        "n_original_links": 3 * n + 1,
        "oriented_edges": edges,
        "plaquette_words": [
            [i + 1, 2 * n + i + 2, -(n + i + 1), -(2 * n + i + 1)] for i in range(n)
        ],
        "tree_link_ids": list(range(n + 1, 3 * n + 2)),
        "chord_link_ids": list(range(1, n + 1)),
        "root_vertex": n + 1,
        "coordinates": "top-and-rung tree gauge: each bottom chord is its elementary plaquette",
        "boundary_condition": "isolated open 1-by-n strip; all original vertices obey Gauss",
    }


def su2_strip_normalized_kernel_tail(
    kappa: int | Q,
    n_plaquettes: int,
    radius: int | Q,
    *,
    retained_cycles: Sequence[int] | None = None,
    target: int | Q | None = None,
) -> dict[str, Any]:
    """Bound an actual finite-strip squared Hilbert--Schmidt tail.

    For n>=2, 0<kappa<=1/64, and any proper cycle bipartition I,J,
    the actual kernel rho/sqrt(rho_I*rho_J) has squared tail on F0>=R*kappa
    at most exp(1600000*n*n+13120*n-17*R/330). Here
    F0=sum_i 8*(1-cos(theta_i/2)), theta_i in [0,pi].
    Cycle IDs are one-based. The optional exact rational target has a separate
    outward-dyadic gate. Huge positive powers and floating exponentials are
    never materialized. Out-of-range couplings return INCONCLUSIVE.
    """
    retained = _partition(n_plaquettes, retained_cycles)
    n = n_plaquettes
    coupling, cutoff = _rational(kappa, "kappa"), _rational(radius, "radius")
    requested = None if target is None else _rational(target, "target")
    if coupling <= 0 or cutoff <= 0 or (requested is not None and requested <= 0):
        raise ValueError("kappa, radius and any target must be strictly positive")
    source, source_valid, single_ground = _plaquette_trial_source(coupling)
    ground = n * single_ground if source_valid and single_ground is not None else None
    exponent = Q(1_600_000 * n * n + 13_120 * n) - Q(17, 330) * cutoff
    dyadic = _dyadic_exponent(exponent)
    empty = cutoff * coupling >= 8 * n
    gates = {
        "canonical_single_plaquette_trial_source": source_valid,
        "coupling_in_proved_interval": coupling <= Q(1, 64),
        "actual_product_trial_energy_at_most_3n": ground is not None and ground <= 3 * n,
        "bochner_scaled_potential_separation": 2**18 * n * n >= 2 * 32774 * n,
        "bochner_contradiction": Q(64, 3) > Q(29, 200),
        "bochner_first_remainder": 222**2 > 49152 and Q(1776, 2**18) < Q(1, 100),
        "bochner_second_remainder": Q(3, 256) < Q(1, 64),
        "bochner_third_remainder": Q(3, 2**18) < Q(1, 100),
        "core_path_log_ratio": Q(131072 * 22, 21) < 200000,
        "supersolution_outside_core": Q(2, 25) * 2048 > Q(783, 5),
        "subsolution_outside_core": Q(9, 2) - Q(2048, 4) < 0,
        "smoothed_action_ratio": 1 / (1 + Q(2, 64)) == Q(32, 33),
        "upper_barrier_core_allowance": Q(4, 5) * 4096 < 3277,
        "normalized_kernel_decay": 4 * Q(4, 5) * Q(32, 33) - 3 == Q(17, 165),
        "marginal_haar_integral_lower": Q(175, 32076) > Q(1, 200),
        "tail_haar_integral_upper": Q(121, 98) * Q(330, 17) ** 2 < 466,
        "prefactor_absorbed_in_exponential": 200 * 466 == 93200 and Q(8, 3) ** 12 > 93200,
    }
    passed = all(gates.values())
    threshold = None if requested is None else _ceil_log_two(1 / requested)
    target_passed = bool(passed and threshold is not None and (empty or dyadic >= threshold))
    symbolic = {
        "prefactor": "1",
        "exponent": str(exponent),
        "meaning": "exp(exponent); squared Hilbert-Schmidt tail",
    }
    arithmetic = {
        "kappa": str(coupling),
        "proved_kappa_upper": "1/64",
        "n_plaquettes": n,
        "radius": str(cutoff),
        "unscaled_tail_threshold": str(cutoff * coupling),
        "maximum_F0": str(8 * n),
        "single_plaquette_ground_energy_upper": str(single_ground)
        if single_ground is not None
        else None,
        "ground_energy_upper": str(ground) if ground is not None else None,
        "original_dimension": 9 * n + 3,
        "central_phase_gradient_lower": "2",
        "central_phase_gradient_upper": "6",
        "cutoff_support_scale": str(8192 * n),
        "core_action_scale": str(2048 * n),
        "bochner_scaled_gradient_cutoff": str(2**18 * n * n),
        "core_gradient_coefficient": str(Q(2048 * n, 3)),
        "core_log_ratio_to_identity_upper": str(200000 * n * n),
        "barrier_regularization": "1/64",
        "upper_barrier_action_coefficient": "4/5",
        "lower_barrier_action_coefficient": "3/2",
        "smoothed_action_ratio_lower": "32/33",
        "lower_barrier_log_prefactor": str(-200000 * n * n),
        "upper_barrier_log_prefactor": str(200000 * n * n + 3277 * n),
        "normalized_kernel_decay_coefficient": "17/165",
        "unabsorbed_tail_prefactor_base": "93200",
        "unabsorbed_tail_prefactor_power": n,
        "squared_hs_tail_prefactor": "1",
        "squared_hs_tail_exponent": str(exponent),
        "dyadic_upper_negative_integer_exponent": dyadic,
        "target_required_dyadic_exponent": threshold,
        "gates": gates,
    }
    payload = {
        "type": "su2_strip_normalized_kernel_tail_v1",
        "status": "PASS" if passed else "INCONCLUSIVE",
        "inputs": {
            "kappa": str(coupling),
            "n_plaquettes": n,
            "radius": str(cutoff),
            "retained_cycles": list(retained),
            "target": str(requested) if requested is not None else None,
        },
        "plaquette_trial_source_certificate": source,
        "geometry": _strip_geometry(n),
        "retained_cycles": list(retained),
        "eliminated_cycles": [i for i in range(1, n + 1) if i not in retained],
        "normalization": "H=kappa*sum_original_C/2+2*sum_plaquettes(2-Tr(U_p))/kappa; C_fund=3/4",
        "vacuum": "actual unique positive original-link ground state, in rooted tree coordinates SU2^n",
        "kernel": "rho/sqrt(rho_I*rho_J), with the actual joint density and its own block marginals",
        "tail_region": "sum_i 8*(1-cos(theta_i/2)) >= radius*kappa; theta_i in[0,pi]",
        "tail_measure": "product normalized Haar; integral of the squared kernel",
        "squared_hs_tail_upper": symbolic if passed else None,
        "effective_squared_hs_tail_upper": ({"exact_rational": "0"} if empty else symbolic)
        if passed
        else None,
        "factored_dyadic_squared_hs_tail_upper": {
            "prefactor": "1",
            "base": 2,
            "negative_integer_exponent": dyadic,
        }
        if passed
        else None,
        "tail_region_has_zero_haar_measure": empty,
        "tail_target_status": "NOT_REQUESTED"
        if requested is None
        else "PASS"
        if target_passed
        else "INCONCLUSIVE",
        "tail_target_verified": target_passed,
        "arithmetic": arithmetic,
        "actual_positive_strip_vacuum_identified_in_written_analysis": source_valid,
        "actual_product_trial_energy_verified_in_written_analysis": source_valid,
        "actual_normalized_kernel_tail_verified_in_written_analysis": passed,
        "fixed_n_uniform_rescaled_hs_tail_verified_in_written_analysis": passed,
        "radial_marginal_kernel_tail_verified_in_written_analysis": passed,
        "all_reduced_modes_included": passed,
        "source_gap_used_as_premise": False,
        "full_reduced_metric_bounded_by_path_matrix_claim": False,
        "gaussian_kernel_comparison_verified": False,
        "maximal_correlation_upper_verified": False,
        "explicit_correlation_coupling_threshold_verified": False,
        "ambient_exterior_uniformity_verified": False,
        "uniform_in_volume_claim": False,
        "all_scale_refinement_claim": False,
        "continuum_claim": False,
        "yang_mills_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }
    certificate = make_certificate(
        claim="actual fixed-strip own-marginal joint-kernel squared-Hilbert-Schmidt tail",
        payload=payload,
        meta={
            "analytic_implication": "docs/api/gauge-strip-kernel-tail.md",
            "proof_register": "written original-link product-trial, Bochner, barriers and marginal denominators",
            "transcend_backend": "symbolic exponential and outward factored dyadic bound",
        },
    )
    return {**payload, "certificate": certificate}


def replay_su2_strip_kernel_tail_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild the actual source, original geometry, partition, arithmetic and scope."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        if payload["type"] != "su2_strip_normalized_kernel_tail_v1":
            return False
        inputs = payload["inputs"]
        expected = su2_strip_normalized_kernel_tail(
            Q(inputs["kappa"]),
            inputs["n_plaquettes"],
            Q(inputs["radius"]),
            retained_cycles=inputs["retained_cycles"],
            target=None if inputs["target"] is None else Q(inputs["target"]),
        )
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "replay_su2_strip_kernel_tail_certificate",
    "su2_strip_normalized_kernel_tail",
]
