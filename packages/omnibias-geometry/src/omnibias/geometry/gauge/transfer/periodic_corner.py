# SPDX-License-Identifier: Apache-2.0
"""Periodic SU(2) harmonic comparisons and an actual global upper trial.

The volume-uniform analytic implications are proved in
``docs/api/gauge-periodic-corner.md``. Exact arithmetic and canonical replay
do not constitute a formal proof of the Fourier or log-concavity theorems.
The nonlinear tilted lower estimate is deliberately a separate premise.
"""

from __future__ import annotations

from fractions import Fraction as Q
from typing import Any

from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.static_sources import _integer, _rational


def _seal(
    kind: str,
    inputs: dict[str, Any],
    arithmetic: dict[str, Any],
    *,
    passed: bool = True,
    actual_upper: bool = False,
    minimum_side: int = 3,
) -> dict[str, Any]:
    payload = {
        "type": kind,
        "inputs": inputs,
        "arithmetic": arithmetic,
        "status": "PASS" if passed else "INCONCLUSIVE",
        "model": "aH=kappa/2 sum_e C_e + 2/kappa sum_p (2-Tr U_p)",
        "group": "SU(2)",
        "energy_density_units": "dimensionless aH per spatial lattice site",
        "family": f"periodic cubic spatial lattice N^3, every integer N>={minimum_side}",
        "proof_register": "written analytic implication and exact rational replay",
        "actual_periodic_upper_density_verified": actual_upper,
        "actual_nonlinear_upper_verified_in_written_analysis": actual_upper,
        "harmonic_bound_verified_in_written_analysis": kind == "su2_periodic_corner_harmonic_v1",
        "actual_periodic_tilted_lower_density_verified": False,
        "actual_periodic_vacuum_decorrelation_verified": False,
        "uniform_nonlinear_semiclassical_comparison_verified": False,
        "continuum_claim": False,
        "physical_gap_claim": False,
        "yang_mills_mass_gap_claim": False,
        "analytic_proof_formally_verified": False,
    }
    return {
        **payload,
        "certificate": make_certificate(
            claim="periodic harmonic theorem, actual upper density, or conditional comparison budget",
            payload=payload,
            meta={
                "transcend_backend": "not_used",
                "analytic_implication": "docs/api/gauge-periodic-corner.md",
            },
        ),
        "theorem_prover_verified": False,
        "mathlib_verified": False,
    }


def su2_periodic_corner_harmonic() -> dict[str, Any]:
    """Universal identity-background harmonic bounds, with all gauge zeros removed.

    The common gauge/toron kernel is projected out only in this Gaussian
    reference. No statement about the actual nonlinear vacuum follows alone.
    """
    tilt = Q(1, 4)
    slope = Q(9, 28)
    curvature = Q(1, 2)
    secant = slope - curvature * tilt / 2
    return _seal(
        "su2_periodic_corner_harmonic_v1",
        {},
        {
            "tilt_endpoint": str(tilt),
            "tilt": "H_t=H+t sum_x(S_x-A_D,x)/(2*kappa)",
            "face_defect": "D=(I_3-ones(3,3))/4",
            "curl_symbol_rows": [["-z2", "z1", "0"], ["0", "-z3", "z2"], ["z3", "0", "-z1"]],
            "z_definition": "z_i=exp(i*k_i)-1, k_i=2*pi*n_i/N",
            "lambda": "sum_i |z_i|^2",
            "second_frequency_squared": "(1-t/2)*lambda+t*|z1+z2+z3|^2/4",
            "gauge_zero_modes_per_color": "N^3-1",
            "constant_toron_modes_per_color": 3,
            "nonzero_modes_per_color": "2*N^3-2",
            "mean_q_i": "2",
            "mean_q_i_q_j_distinct": "4",
            "mean_lambda": "6",
            "lambda_upper": "12",
            "initial_density_derivative_lower": str(slope),
            "density_second_derivative_lower": str(-curvature),
            "second_derivative_bound_squared": "675/5488",
            "second_derivative_squared_slack": str(Q(1, 4) - Q(675, 5488)),
            "harmonic_density_secant_lower": str(secant),
            "harmonic_density_gain_lower": str(tilt * secant),
            "local_tilt_energy_gain_lower": "413/7942",
            "gaussian_action_ratio_upper": "2/3",
            "gaussian_action_difference_over_kappa_lower": "9/14",
            "local_logdet_reference": "(1-2/(s+6)+1/(s+12))*(1+1/(s+6)+1/(4*(s+12)))^2",
            "local_logdet_ratio_on_0_4_lower": "1083/1024",
            "volume_uniform_harmonic_comparison_verified": True,
            "uniform_over_flat_backgrounds_verified": False,
            "reference_only": True,
        },
    )


def su2_periodic_upper_density(tau_upper: int | Q = Q(1, 8192)) -> dict[str, Any]:
    """Actual upper density for every N>=3 and 0<kappa<=tau_upper^3<=1.

    ``tau_upper`` is an exact upper bound for kappa^(1/3); no floating
    root is taken. The trial retains every gauge and toron coordinate.
    """
    tau = _rational(tau_upper, "tau_upper")
    if not 0 < tau <= 1:
        raise ValueError("tau_upper must lie in (0,1]")
    upper_error = 65 * tau
    return _seal(
        "su2_periodic_upper_density_v1",
        {"tau_upper": str(tau)},
        {
            "kappa_upper": str(tau**3),
            "upper_density_error": str(upper_error),
            "density_error_upper": str(upper_error),
            "pointwise_error_formula": "65*kappa^(1/3)",
            "density_comparison": "e_actual(kappa,0)<=e_harmonic_identity(0)+65*kappa^(1/3)",
            "regularization": "eta=kappa^(2/3), Omega=sqrt(K+eta*I)",
            "coordinate_cutoff_radius": "r=kappa^(1/4)",
            "trial": "J^(-1/2)*exp(-A.Omega.A/(2*kappa))*product_i cos(pi*A_i/(2*r))_+",
            "cutoff_domain": "each scalar coordinate |A_i|<r; zero extension outside",
            "scalar_coordinates_per_site": 9,
            "links_per_site": 3,
            "plaquettes_per_site": 3,
            "variance_bound": "sigma^2=kappa^(2/3)/2",
            "link_fourth_moment_coefficient": "21",
            "magnetic_error_coefficient": "42",
            "regularization_error_coefficient": "9/2",
            "cutoff_energy_coefficient_pi_squared": "9/8",
            "metric_multiplier": "(1-kappa^(1/2)/8)^(-2)",
            "metric_multiplier_upper": "64/49",
            "metric_excess_coefficient": "16/49",
            "upper_density_error_cuberoot_coefficient": "2346/49",
            "upper_density_error_squareroot_coefficient": "40728/2401",
            "combined_error_coefficient": "155682/2401",
            "rounded_coefficient": "65",
            "rounding_slack": str(65 - Q(155682, 2401)),
            "target_one_sixty_fourth_met": upper_error <= Q(1, 64),
            "volume_uniform_upper_density_verified": True,
            "original_link_kinetic_metric_included": True,
            "all_gauge_and_toron_coordinates_retained": True,
            "global_cutoff_union_bound_used": False,
            "gauge_projection_rayleigh_monotonicity_assumed": False,
            "cubic_expectation_exactly_zero": True,
        },
        actual_upper=True,
    )


def periodic_corner_energy_budget(
    upper_error: int | Q = Q(1, 64), lower_error: int | Q = Q(1, 64)
) -> dict[str, Any]:
    """Conditional secant implication; input errors are NOT verified premises.

    Required analytic hypotheses are the actual untilted upper and actual
    tilted lower energy-density comparisons at the same coupling and volume.
    PASS means their stated error budget implies the target, not that the
    nonlinear lower estimate has been constructed.
    """
    upper = _rational(upper_error, "upper_error")
    lower = _rational(lower_error, "lower_error")
    if min(upper, lower) < 0:
        raise ValueError("error bounds must be nonnegative")
    secant = Q(29, 112) - 4 * (upper + lower)
    action_difference = 2 * secant
    passed = action_difference >= Q(15, 56)
    return _seal(
        "su2_periodic_corner_energy_budget_v1",
        {"upper_error": str(upper), "lower_error": str(lower)},
        {
            "tilt_endpoint": "1/4",
            "harmonic_density_gain_lower": "29/448",
            "sum_density_errors": str(upper + lower),
            "conditional_actual_density_gain_lower": str(Q(29, 448) - upper - lower),
            "conditional_density_secant_lower": str(secant),
            "conditional_action_difference_over_kappa_lower": str(action_difference),
            "target_action_difference_over_kappa": "15/56",
            "target_slack": str(action_difference - Q(15, 56)),
            "one_quarter_target_slack": str(action_difference - Q(1, 4)),
            "premises": [
                "e_actual(kappa,0)<=e_harmonic_identity(0)+upper_error",
                "e_actual(kappa,1/4)>=e_harmonic_identity(1/4)-lower_error",
                "both comparisons apply to the same actual periodic model and volume",
            ],
            "input_error_hypotheses_verified": False,
            "conditional_energy_budget_verified": passed,
            "spectral_gap_or_differentiability_premise_used": False,
        },
        passed=passed,
    )


def periodic_corner_box_surface(box_side: int = 8192) -> dict[str, Any]:
    """Exact 54/ell harmonic surface loss for original-link open boxes.

    Box vertex side lengths lie in [ell,2*ell-1]. Crossing corner cells
    retain positive internal-face replacements; this is not deletion of
    every crossing potential. No nonlinear box lower bound is inferred.
    """
    ell = _integer(box_side, "box_side")
    if ell < 2:
        raise ValueError("box_side must be at least two")
    return _seal(
        "su2_periodic_corner_box_surface_v1",
        {"box_side": ell},
        {
            "minimum_periodic_side": max(3, ell),
            "box_vertex_side_range": [ell, 2 * ell - 1],
            "tilt_interval": ["0", "1/4"],
            "internal_links": "nonwrapping consecutive original links inside each box",
            "crossing_cell_replacement": "(2-t)*sum of wholly internal face actions/kappa",
            "quadratic_face_replacement": "(1-t/2)*P_internal_faces",
            "quadratic_spectrum_upper": "51/4",
            "rounded_frequency_upper": "4",
            "rank_per_crossing_cell_upper": 3,
            "su2_harmonic_energy_per_crossing_cell_upper": "18",
            "crossing_cell_density_upper": str(Q(3, ell)),
            "surface_density_error_upper": str(Q(54, ell)),
            "harmonic_comparison": "e_periodic_harmonic-sum_box_E_harmonic/N^3<=54/ell",
            "zero_modes_included_in_rank_interlacing": True,
            "volume_uniform_surface_comparison_verified": True,
            "nonlinear_box_lower_premise_verified": False,
        },
        minimum_side=max(3, ell),
    )


def periodic_corner_box_lower_budget(
    box_side: int = 8192, local_error: int | Q = Q(1, 128)
) -> dict[str, Any]:
    """Conditional lower budget; local_error is a hypothesis, not a source.

    An actual lower source must prove its estimate for every retained
    rectangular box at the same coupling, including the crossing-face
    replacement. This helper only adds the proved harmonic surface loss.
    """
    surface = periodic_corner_box_surface(box_side)
    error = _rational(local_error, "local_error")
    if error < 0:
        raise ValueError("local_error must be nonnegative")
    surface_error = Q(surface["arithmetic"]["surface_density_error_upper"])
    total = surface_error + error
    passed = total <= Q(1, 64)
    return _seal(
        "su2_periodic_corner_box_lower_budget_v1",
        {"box_side": box_side, "local_error": str(error)},
        {
            "minimum_periodic_side": max(3, box_side),
            "surface_density_error_upper": str(surface_error),
            "assumed_nonlinear_box_density_error_upper": str(error),
            "conditional_total_density_error_upper": str(total),
            "target_density_error": "1/64",
            "target_slack": str(Q(1, 64) - total),
            "premise": "E_actual_box/box_vertices>=E_harmonic_box/box_vertices-local_error for every retained box",
            "nonlinear_box_lower_premise_verified": False,
            "conditional_box_lower_budget_verified": passed,
        },
        passed=passed,
        minimum_side=max(3, box_side),
    )


def _read_rational(inputs: dict[str, Any], key: str) -> Q:
    value = inputs[key]
    if type(value) is not str:
        raise TypeError("serialized rational inputs must be strings")
    result = Q(value)
    if str(result) != value:
        raise ValueError("rational inputs must use canonical fraction spelling")
    return result


def replay_periodic_corner_certificate(certificate: dict[str, Any]) -> bool:
    """Rebuild a complete PASS or INCONCLUSIVE certificate and all scope fields."""
    try:
        if not isinstance(certificate, dict) or not verify_certificate_digest(certificate):
            return False
        payload = certificate["payload"]
        inputs = payload["inputs"]
        kind = payload["type"]
        if kind == "su2_periodic_corner_harmonic_v1":
            expected = su2_periodic_corner_harmonic()
        elif kind == "su2_periodic_upper_density_v1":
            expected = su2_periodic_upper_density(_read_rational(inputs, "tau_upper"))
        elif kind == "su2_periodic_corner_energy_budget_v1":
            expected = periodic_corner_energy_budget(
                _read_rational(inputs, "upper_error"), _read_rational(inputs, "lower_error")
            )
        elif kind == "su2_periodic_corner_box_surface_v1":
            expected = periodic_corner_box_surface(inputs["box_side"])
        elif kind == "su2_periodic_corner_box_lower_budget_v1":
            expected = periodic_corner_box_lower_budget(
                inputs["box_side"], _read_rational(inputs, "local_error")
            )
        else:
            return False
        return bool(certificate == expected["certificate"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


__all__ = [
    "periodic_corner_box_lower_budget",
    "periodic_corner_box_surface",
    "periodic_corner_energy_budget",
    "replay_periodic_corner_certificate",
    "su2_periodic_corner_harmonic",
    "su2_periodic_upper_density",
]
