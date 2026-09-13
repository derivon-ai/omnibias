# SPDX-License-Identifier: Apache-2.0
"""Actual SU(2) corner vacuum: original-link metric and uniform IMS budgets.

The compact analytic implication is proved in docs/api/gauge-corner-vacuum.md.
Certificates replay its exact rational constants, not a Lean proof of analysis.
This is one nine-link graph, without an exterior or a continuum limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _rational as rational


def corner_geometry() -> dict[str, Any]:
    """Three corner-adjacent squares, coherently based at vertex000."""
    return {
        "vertices": 7,
        "edges": [[0, 1], [0, 2], [0, 4], [1, 3], [2, 3], [2, 6], [4, 6], [4, 5], [1, 5]],
        "faces": [[1, 4, -5, -2], [2, 6, -7, -3], [3, 8, -9, -1]],
        "tree_edges": [1, 2, 3, 5, 7, 9],
        "chord_edges": [4, 6, 8],
        "boundary": [-9, 4, -5, 6, -7, 8],
        "rooted_space": "SU2^3, followed by residual simultaneous conjugation",
        "reduced_kinetic_form": "2 sum_i |L_i f|^2 + sum_i |L_i f-R_(i-1) f|^2",
        "flat_curl_gram": [[4, -1, -1], [-1, 4, -1], [-1, -1, 4]],
    }


def _seal(
    kind: str, inputs: dict[str, Any], arithmetic: dict[str, Any], *, passed: bool = True
) -> dict[str, Any]:
    payload = {
        "type": kind,
        "inputs": inputs,
        "arithmetic": arithmetic,
        "status": "PASS" if passed else "INCONCLUSIVE",
    }
    return {
        **payload,
        "certificate": make_certificate(
            claim="finite rational corner metric or compact IMS implication budget",
            payload=payload,
            meta={
                "transcend_backend": "not_used",
                "analytic_implication": "docs/api/gauge-corner-vacuum.md",
            },
        ),
        "theorem_prover_verified": False,
        "mathlib_verified": False,
        "uniform_in_volume_claim": False,
        "continuum_claim": False,
        "physical_gap_claim": False,
        "yang_mills_mass_gap_claim": False,
    }


def corner_reduced_metric(holonomies: Sequence[Sequence[int | Q]]) -> dict[str, Any]:
    """Exact9x9 symbol in the vector-part hemisphere coordinates.

    Negative scalar parts are legal for this pointwise algebra; local IMS
    comparisons separately require the positive hemisphere.
    """
    from omnibias.geometry.gauge.stochastic.lattice import _unit, quaternion_product

    if len(holonomies) != 3:
        raise ValueError("three exact unit holonomies required")
    values = [_unit(u) for u in holonomies]
    left, right = [], []
    for u in values:
        lrows, rrows = [], []
        for axis in range(3):
            generator = [Q(0), *[Q(1, 2) if j == axis else Q(0) for j in range(3)]]
            lrows.append(list(quaternion_product(generator, u)[1:]))
            rrows.append(list(quaternion_product(u, generator)[1:]))
        left.append(lrows)
        right.append(rrows)
    rows: list[list[Q]] = []
    for i in range(3):
        for _ in range(2):
            for axis in range(3):
                rows.append([left[i][axis][j % 3] if j // 3 == i else Q(0) for j in range(9)])
    for i in range(3):
        prev = (i - 1) % 3
        for axis in range(3):
            rows.append(
                [
                    left[i][axis][j % 3]
                    if j // 3 == i
                    else -right[prev][axis][j % 3]
                    if j // 3 == prev
                    else Q(0)
                    for j in range(9)
                ]
            )
    metric = [[sum((row[i] * row[j] for row in rows), Q(0)) for j in range(9)] for i in range(9)]
    product = quaternion_product(quaternion_product(values[0], values[1]), values[2])
    s = sum((2 - 2 * u[0] for u in values), Q(0))
    ad = 2 - 2 * product[0]
    return _seal(
        "su2_corner_reduced_metric_v1",
        {"holonomies": [[str(v) for v in u] for u in values]},
        {
            "metric": [[str(v) for v in row] for row in metric],
            "original_derivative_rows": [[str(v) for v in row] for row in rows],
            "old_action_sum": str(s),
            "boundary_action": str(ad),
            "tilted_potential": str((5 * s - ad) / 2),
            "positive_potential_slack": str((3 * s - ad) / 2),
            "source": "one exact corner configuration; not a vacuum or ensemble",
        },
    )


def su2_corner_vacuum_bound(tau_upper: int | Q = Q(1, 256)) -> dict[str, Any]:
    """Bound the actual two ground energies uniformly for0<kappa<=tau_upper^5.

    Global localization radius r=tau². All error functions are monotone
    in the asserted domain; finite inputs certify the full smaller interval.
    No excited gap, finite spin cutoff, or approximate vacuum is an input.
    """
    tau = rational(tau_upper, "tau_upper")
    if not 0 < tau <= Q(1, 4):
        raise ValueError("tau_upper must lie in(0,1/4]")
    r = tau * tau
    base_upper = Q(7417, 840)
    domain = r <= Q(1, 128) and 12 * tau < 1
    upper = (
        ((1 + 4 * r * r) * (1 + 64 * r) * (base_upper + 13 * tau / (1 - 12 * tau)))
        if domain
        else None
    )
    lower_local = 9 * (1 - 64 * r) / (1 + 4 * r * r)
    lower_bad = 1 / (4 * tau)
    lower = min(lower_local, lower_bad) - 10 * tau
    error_upper = upper - base_upper if upper is not None else None
    error_lower = 9 - lower
    margin = lower - upper if upper is not None else None
    gates = {
        "chart_domain": domain,
        "bad_branch_above9": lower_bad >= 9,
        "upper_error_at_most_one_sixteenth": error_upper is not None and error_upper <= Q(1, 16),
        "lower_error_at_most_one_sixteenth": error_lower <= Q(1, 16),
        "positive_ground_energy_separation": margin is not None and margin > 0,
    }
    passed = all(gates.values())
    return _seal(
        "su2_corner_vacuum_bound_v1",
        {"tau_upper": str(tau)},
        {
            "geometry": corner_geometry(),
            "kappa_upper": str(tau**5),
            "localization_radius_upper": str(r),
            "upper_harmonic_energy": str(base_upper),
            "tilted_harmonic_energy": "9",
            "reference_energy_margin": "143/840",
            "density_ratio_upper": str(1 + 4 * r * r),
            "metric_relative_error": str(64 * r),
            "gaussian_cut_norm_lower": str(1 - 12 * tau),
            "ims_error_upper": str(10 * tau),
            "tilted_local_floor": str(lower_local),
            "tilted_bad_region_floor": str(lower_bad),
            "actual_ground_energy_upper": str(upper) if passed else None,
            "actual_tilted_ground_energy_lower": str(lower) if passed else None,
            "proposed_ground_energy_upper": str(upper) if upper is not None else None,
            "proposed_tilted_ground_energy_lower": str(lower),
            "upper_remainder": str(error_upper) if error_upper is not None else None,
            "lower_remainder": str(error_lower),
            "energy_separation_lower": str(margin) if passed else None,
            "observable_difference_over_kappa_upper": str(-2 * margin)
            if passed and margin is not None
            else None,
            "observable": "actual vacuum expectation of A(D)-sum_old A",
            "tilted_operator": "Htilde=H+(sum_old A-A(D))/(2*kappa)",
            "averaged_seam_constant": "8667/512",
            "averaged_seam_theta": "363/392",
            "gates": gates,
            "actual_corner_vacuum_decorrelation_verified": passed,
            "all_spin_analytic_comparison_verified": passed,
            "uniform_on_declared_kappa_interval_verified": passed,
            "analytic_proof_formally_verified": False,
            "exterior_coupled_vacuum_verified": False,
            "proof_register": "written compact analytic proof plus exact rational budget; not Lean",
        },
        passed=passed,
    )


def search_corner_vacuum_interval(max_dyadic_exponent: int = 32) -> dict[str, Any]:
    if type(max_dyadic_exponent) is not int or not 2 <= max_dyadic_exponent <= 32:
        raise ValueError("max_dyadic_exponent must be an integer in[2,32]")
    attempts = []
    for exponent in range(2, max_dyadic_exponent + 1):
        row = su2_corner_vacuum_bound(Q(1, 2**exponent))
        attempts.append(row)
        if row["status"] == "PASS":
            return {
                "status": "PASS",
                "attempts": attempts,
                "accepted": row,
                "selection": "largest passing tau among the declared descending dyadic search",
            }
    return {
        "status": "INCONCLUSIVE",
        "attempts": attempts,
        "accepted": None,
        "selection": "finite search exhausted; no nonexistence claim",
    }


def replay_corner_vacuum_certificate(certificate: dict[str, Any]) -> bool:
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        inputs = payload["inputs"]
        if payload["type"] == "su2_corner_vacuum_bound_v1":
            expected = su2_corner_vacuum_bound(Q(inputs["tau_upper"]))
        elif payload["type"] == "su2_corner_reduced_metric_v1":
            expected = corner_reduced_metric([[Q(v) for v in row] for row in inputs["holonomies"]])
        else:
            return False
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "corner_geometry",
    "corner_reduced_metric",
    "replay_corner_vacuum_certificate",
    "search_corner_vacuum_interval",
    "su2_corner_vacuum_bound",
]
