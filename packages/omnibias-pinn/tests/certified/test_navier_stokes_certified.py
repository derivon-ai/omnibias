# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Navier-Stokes certified-evidence substrate and CAP bundle tests."""

from __future__ import annotations

import json
import math
from dataclasses import asdict

import numpy as np
import pytest
from omnibias.pinn.certified import (
    CCF_SELFSIMILAR_SCHEMA_VERSION,
    CLM_BLOWUP_SCHEMA_VERSION,
    CLM_MULTIZERO_SCHEMA_VERSION,
    GCLM_BLOWUP_SCHEMA_VERSION,
    GCLM_GRADIENT_AMP_SCHEMA_VERSION,
    CandidateGateConfig,
    HonestyLabels,
    TailBound,
    active_projector_error_certificate,
    active_subspace_absorption_frontier_report,
    active_subspace_completeness_theorem_attempt,
    active_subspace_invariance_report,
    active_subspace_tail_contraction_attempt,
    active_tail_contraction_lift_certificate,
    analytic_tail_error_certificate,
    assemble_axisymmetric_active_subspace_operator,
    assemble_axisymmetric_linearized_operator,
    axisymmetric_axis_smoothness_certificate,
    axisymmetric_basis_count,
    axisymmetric_basis_metadata,
    axisymmetric_basis_regular_interval_checks,
    axisymmetric_basis_tensor,
    axisymmetric_coefficient_loss,
    axisymmetric_coefficients_to_fields,
    axisymmetric_compactified_metadata,
    axisymmetric_function_space_metadata,
    axisymmetric_meridional_replay_grid,
    axisymmetric_nontriviality_gate,
    axisymmetric_physical_axes,
    axisymmetric_residual_hessian_operator_norm,
    axisymmetric_swirl_ansatz_metadata,
    axisymmetric_swirl_residual_samples,
    blowup_contract,
    blowup_proof_obligation_bundles,
    blowup_route_lemma_package,
    build_active_tail_lift_error_budget,
    build_analytic_closure_report,
    build_axisymmetric_active_subspace_closure_report,
    build_axisymmetric_blowup_closure_report,
    build_axisymmetric_interval_report,
    build_axisymmetric_swirl_candidate_artifact,
    build_blowup_closure_report,
    build_candidate_artifact,
    build_certificate_manifest,
    build_formal_proof_package,
    build_ns_cap_bundle,
    build_ns_proof_program_report,
    build_ns_solve_or_falsify_report,
    build_ns_theorem_ladder_report,
    build_refined_axisymmetric_swirl_candidate_artifact,
    build_regularity_closure_report,
    build_regularity_inequality_report,
    build_theorem_grade_closure_attempt,
    candidate_artifact_schema_errors,
    candidate_upgrade_gates,
    certified_candidate_refinement_report,
    certified_ccf_linearized_operator_bound,
    certified_ccf_linearized_operator_bound_schema_errors,
    certified_ccf_selfsimilar_blowup_attempt,
    certified_ccf_selfsimilar_blowup_attempt_schema_errors,
    certified_clm_blowup,
    certified_clm_blowup_schema_errors,
    certified_clm_multizero_first_blowup,
    certified_clm_multizero_first_blowup_schema_errors,
    certified_gclm_gradient_amplification,
    certified_gclm_gradient_amplification_schema_errors,
    certified_gclm_selfsimilar_blowup,
    certified_gclm_selfsimilar_blowup_schema_errors,
    certified_tail_bounds_from_artifact,
    classical_assumptions_readiness_gate,
    coefficient_interval_boxes,
    compactification_map_interval,
    compactified_coefficient_set,
    compactified_r3_metadata,
    compactified_sandbox_replay_grid,
    componentwise_radii_polynomial_certificate,
    conditioning_preserving_ansatz_report,
    continuum_banach_invertibility_attempt,
    continuum_neumann_inequality_certificate,
    continuum_residual_certificates,
    continuum_residual_upper_bound,
    default_ccf_collocation_nodes,
    energy_diagnostics,
    exact_navier_stokes_equation_contracts,
    exact_profile_norm_divergence_attempt,
    external_review_gate,
    external_verification_record,
    finite_active_tail_contraction_diagnostic,
    finite_energy_interval_bounds,
    finite_energy_tail_certificate,
    global_regularity_contract,
    ingest_theorem_verifier_bundle,
    initial_axisymmetric_coefficients,
    interval_add,
    interval_cap_backend_contract,
    interval_div,
    interval_from_bounds,
    interval_jacobian_error_certificate,
    interval_mul,
    interval_sqrt,
    interval_square,
    interval_sub,
    interval_trapezoid_bound,
    lean_formalization_package,
    leray_project_periodic,
    manufactured_abc_flow,
    nonlinear_tail_remainder_certificate,
    norm_divergence_certificate,
    ns_cap_schema_errors,
    operator_theoretic_invertibility_certificate,
    pressure_poisson_residual_periodic,
    primitive_residual_periodic,
    proof_contract_bundle,
    proof_obligation_bundle,
    radii_polynomial_certificate,
    radii_polynomial_closure,
    refine_axisymmetric_coefficients,
    refine_ccf_selfsimilar_profile,
    regularity_all_data_proof_attempt,
    regularity_counterexample_sweep,
    regularity_proof_obligation_bundles,
    regularity_route_lemma_package,
    residual_interval_envelopes,
    scalar_interval,
    scalar_interval_contains,
    spectral_divergence,
    theorem_claim_gate,
    theorem_grade_function_space_contract,
    theorem_grade_function_space_definitions,
    theorem_grade_radii_polynomial_attempt,
    theorem_interval_backend_readiness_report,
    theorem_verifier_record,
    verify_external_proof_package,
    vorticity_residual_periodic,
    weighted_analytic_tail_norm_contract,
    write_candidate_artifact,
    write_ns_cap_bundle,
)


def _grid_2d(n: int) -> tuple[np.ndarray, np.ndarray]:
    x = 2.0 * np.pi * np.arange(n) / n
    return np.meshgrid(x, x, indexing="ij")


def _taylor_green(n: int, viscosity: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = _grid_2d(n)
    velocity = np.stack([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])
    pressure = 0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y))
    velocity_t = -2.0 * viscosity * velocity
    return velocity, pressure, velocity_t


def test_contract_and_honesty_labels_do_not_overclaim() -> None:
    regularity = global_regularity_contract()
    blowup = blowup_contract()
    assert regularity.target == "global_regularity"
    assert blowup.target == "finite_time_blowup"

    bundle = proof_contract_bundle(regularity, honesty=HonestyLabels(notes="test"))
    assert bundle["honesty"]["unproven_claim"] is False
    assert bundle["honesty"]["exact_solution_claim"] is False
    assert any(o["name"] == "a_priori_estimate" for o in bundle["proof_obligations"])


def test_leray_projection_makes_periodic_field_divergence_free() -> None:
    n = 32
    x, y = _grid_2d(n)
    raw = np.stack([np.sin(x) + 0.2 * np.cos(y), np.cos(x - y)])
    before = np.max(np.abs(spectral_divergence(raw)))
    projected = leray_project_periodic(raw)
    after = np.max(np.abs(spectral_divergence(projected)))
    assert before > 0.1
    assert after < 1e-10


def test_taylor_green_exact_flow_residual_and_pressure_poisson() -> None:
    viscosity = 0.1
    velocity, pressure, velocity_t = _taylor_green(64, viscosity)
    residual, continuity = primitive_residual_periodic(
        velocity, pressure, velocity_t=velocity_t, viscosity=viscosity
    )
    pressure_res = pressure_poisson_residual_periodic(velocity, pressure)
    assert float(np.max(np.abs(residual))) < 1e-10
    assert float(np.max(np.abs(continuity))) < 1e-10
    assert float(np.max(np.abs(pressure_res))) < 1e-10


def test_abc_3d_manufactured_flow_is_exact_periodic_solution() -> None:
    mms = manufactured_abc_flow(24, viscosity=0.07, density=1.0)
    residual, continuity = primitive_residual_periodic(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        density=mms["density"],
        lengths=mms["lengths"],
    )
    pressure_res = pressure_poisson_residual_periodic(
        mms["velocity"], mms["pressure"], density=mms["density"], lengths=mms["lengths"]
    )
    assert float(np.max(np.abs(residual))) < 1e-10
    assert float(np.max(np.abs(continuity))) < 1e-10
    assert float(np.max(np.abs(pressure_res))) < 1e-10


def test_abc_3d_vorticity_residual_is_exact_in_inviscid_form() -> None:
    mms = manufactured_abc_flow(24, viscosity=0.0, density=1.0)
    residual = vorticity_residual_periodic(
        mms["velocity"],
        velocity_t=mms["velocity_t"],
        viscosity=0.0,
        lengths=mms["lengths"],
    )
    assert float(np.max(np.abs(residual))) < 1e-10


def test_energy_diagnostics_are_finite_and_include_proof_quantities() -> None:
    velocity, pressure, _ = _taylor_green(32, 0.1)
    diag = energy_diagnostics(velocity, pressure=pressure)
    for key in (
        "kinetic_energy",
        "enstrophy",
        "palinstrophy",
        "max_abs_divergence",
        "bkm_vorticity_proxy",
        "pressure_poisson_max_abs",
    ):
        assert key in diag
        assert np.isfinite(diag[key])
    assert diag["kinetic_energy"] > 0.0
    assert diag["max_abs_divergence"] < 1e-10


def test_ns_cap_bundle_schema_roundtrip_and_summary(tmp_path) -> None:
    viscosity = 0.1
    velocity, pressure, velocity_t = _taylor_green(32, viscosity)
    bundle = build_ns_cap_bundle(
        velocity, pressure, velocity_t=velocity_t, viscosity=viscosity
    )
    assert ns_cap_schema_errors(bundle) == []
    assert bundle["honesty"]["unproven_claim"] is False
    assert bundle["honesty"]["periodic_model_only"] is True
    assert bundle["residual_diagnostics"]["max_abs_momentum_residual"] < 1e-10

    reloaded = json.loads(json.dumps(bundle))
    assert reloaded["schema_version"] == bundle["schema_version"]
    path = write_ns_cap_bundle(bundle, tmp_path)
    assert path.exists()
    assert (tmp_path / "navier_stokes_cap_summary.md").exists()


def test_candidate_bridge_containers_schema_roundtrip_and_summary(tmp_path) -> None:
    grid = compactified_sandbox_replay_grid(n_radial=4, n_theta=4, n_phi=8)
    coeffs = compactified_coefficient_set(
        "u",
        np.arange(12, dtype=float).reshape(3, 4),
        tail_l1_bound=1e-12,
        finite_energy_estimate=2.5,
    )
    artifact = build_candidate_artifact(
        candidate_type="regularity_growth_law",
        replay_grid=grid,
        replay_inputs={
            "time": [0.0, 0.5, 1.0],
            "traces": {"enstrophy": [1.0, 1.5, 2.25]},
        },
        result={"fit_rmse": 0.0, "global_regularity_claim": False},
        coefficients=(coeffs,),
        proof_obligations=("tail_bounds", "independent_replay"),
        notes="schema test",
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["honesty"]["unproven_claim"] is False
    assert artifact["replay_grid"]["domain_type"] == "compactified_r3"
    assert artifact["coefficients"][0]["tail_bound"]["tail_l1_bound"] == 1e-12

    reloaded = json.loads(json.dumps(artifact))
    assert candidate_artifact_schema_errors(reloaded) == []
    assert reloaded["coefficients"][0]["shape"] == [3, 4]
    path = write_candidate_artifact(artifact, tmp_path)
    assert path.exists()
    assert (tmp_path / "navier_stokes_candidate_summary.md").exists()


def test_candidate_bridge_rejects_missing_compactified_coefficients() -> None:
    grid = compactified_sandbox_replay_grid(n_radial=4, n_theta=4, n_phi=8)
    artifact = build_candidate_artifact(
        candidate_type="self_similar_blowup_rate",
        replay_grid=asdict(grid),
        replay_inputs={"time": [0.0], "norm_values": [1.0]},
        result={"rate_fit": {"alpha": 0.5}},
    )
    errors = candidate_artifact_schema_errors(artifact)
    assert any("coefficient payloads" in error for error in errors)


def test_axisymmetric_grid_metadata_and_residual_diagnostics() -> None:
    grid = axisymmetric_meridional_replay_grid(n_radial=6, n_axial=7)
    radial, axial = axisymmetric_physical_axes(grid)
    assert grid.coordinate_names == ("rho", "zeta")
    assert radial.shape == (6,)
    assert axial.shape == (7,)
    assert np.all(radial > 0.0)

    meta = axisymmetric_compactified_metadata()
    ansatz = axisymmetric_swirl_ansatz_metadata()
    assert "theta_independent" in meta.symmetry_assumptions
    assert ansatz.representation == "streamfunction_swirl_pressure"

    rr = radial[:, None]
    zz = axial[None, :]
    streamfunction = rr * rr * np.exp(-(rr * rr + zz * zz))
    swirl = rr * np.exp(-(rr * rr + zz * zz))
    pressure = np.zeros_like(streamfunction)
    residual = axisymmetric_swirl_residual_samples(
        streamfunction,
        swirl,
        pressure,
        radial_axis=radial,
        axial_axis=axial,
        viscosity=0.01,
    )
    diag = residual["residual_diagnostics"]
    assert np.isfinite(diag["max_abs_momentum_residual"])
    assert np.isfinite(diag["max_abs_continuity"])
    assert diag["unproven_claim"] is False


def test_axisymmetric_candidate_artifact_schema_roundtrip() -> None:
    artifact = build_axisymmetric_swirl_candidate_artifact(
        seed=7,
        n_radial=6,
        n_axial=7,
        viscosity=0.01,
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["candidate_type"] == "axisymmetric_swirl_sandbox"
    assert artifact["honesty"]["unproven_claim"] is False
    assert artifact["replay_inputs"]["ansatz_metadata"]["representation"] == (
        "streamfunction_swirl_pressure"
    )
    assert len(artifact["coefficients"]) == 3

    reloaded = json.loads(json.dumps(artifact))
    assert candidate_artifact_schema_errors(reloaded) == []
    assert reloaded["result"]["finite_energy_estimate"] > 0.0


def test_axisymmetric_basis_has_expected_axis_behavior() -> None:
    grid = axisymmetric_meridional_replay_grid(n_radial=6, n_axial=7)
    radial, axial = axisymmetric_physical_axes(grid)
    metadata = axisymmetric_basis_metadata(radial_degree=1, axial_degree=1)
    assert axisymmetric_basis_count(metadata) == 4
    psi_basis = axisymmetric_basis_tensor(
        radial, axial, component="streamfunction", metadata=metadata
    )
    swirl_basis = axisymmetric_basis_tensor(
        radial, axial, component="swirl", metadata=metadata
    )
    pressure_basis = axisymmetric_basis_tensor(
        radial, axial, component="pressure", metadata=metadata
    )
    assert psi_basis.shape == (4, 6, 7)
    assert np.all(np.abs(psi_basis[:, 0, :]) < np.abs(pressure_basis[:, 0, :]) + 1e-12)
    assert np.all(np.abs(swirl_basis[:, 0, :]) < np.abs(pressure_basis[:, 0, :]) + 1e-12)


def test_axisymmetric_coefficient_loss_and_refiner_are_deterministic() -> None:
    metadata = axisymmetric_basis_metadata(radial_degree=1, axial_degree=1)
    grid = axisymmetric_meridional_replay_grid(n_radial=6, n_axial=7)
    radial, axial = axisymmetric_physical_axes(grid)
    coeffs = initial_axisymmetric_coefficients(seed=5, metadata=metadata)
    fields = axisymmetric_coefficients_to_fields(
        coeffs,
        radial_axis=radial,
        axial_axis=axial,
        metadata=metadata,
    )
    assert fields["streamfunction"].shape == (6, 7)
    initial = axisymmetric_coefficient_loss(
        coeffs,
        grid=grid,
        metadata=metadata,
        viscosity=0.01,
    )
    refined_a = refine_axisymmetric_coefficients(
        seed=5,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=3,
        step_size=0.01,
        viscosity=0.01,
    )
    refined_b = refine_axisymmetric_coefficients(
        seed=5,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=3,
        step_size=0.01,
        viscosity=0.01,
    )
    assert refined_a["train"]["loss"] <= initial["loss"]
    assert refined_a["loss_history"] == refined_b["loss_history"]
    assert np.allclose(refined_a["coefficients"], refined_b["coefficients"])
    assert "residual_diagnostics" in refined_a["holdout"]


def test_refined_axisymmetric_artifact_schema_roundtrip() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=5,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=3,
        step_size=0.01,
        viscosity=0.01,
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["candidate_type"] == "axisymmetric_swirl_refined"
    assert artifact["honesty"]["unproven_claim"] is False
    assert artifact["result"]["residual_descended"] is True
    assert artifact["result"]["final_loss"] <= artifact["result"]["initial_loss"]
    assert artifact["result"]["holdout"]["loss"] >= 0.0

    reloaded = json.loads(json.dumps(artifact))
    assert candidate_artifact_schema_errors(reloaded) == []
    assert reloaded["replay_inputs"]["basis_metadata"]["basis_name"] == (
        "compact_polynomial_envelope"
    )


def test_scalar_interval_and_axisymmetric_interval_report_roundtrip() -> None:
    interval = scalar_interval(1.0, absolute_padding=0.1, relative_padding=0.01)
    assert scalar_interval_contains(interval, 1.0)
    assert interval.lower < 1.0 < interval.upper

    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=19,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    coeff_boxes = coefficient_interval_boxes(artifact)
    assert "streamfunction_coefficients" in coeff_boxes
    assert coeff_boxes["streamfunction_coefficients"]["tail_certified"] is False

    residuals = residual_interval_envelopes(artifact)
    energy = finite_energy_interval_bounds(artifact)
    axis_checks = axisymmetric_basis_regular_interval_checks(artifact)
    assert axis_checks["all_basis_checks_pass"] is True
    for section in ("train", "holdout"):
        diag = artifact["result"][section]["residual_diagnostics"]
        assert scalar_interval_contains(
            residuals[section]["max_abs_momentum_residual"]["interval"],
            diag["max_abs_momentum_residual"],
        )
        assert scalar_interval_contains(
            energy[section]["interval"],
            artifact["result"][section]["finite_energy_estimate"],
        )

    report = build_axisymmetric_interval_report(artifact)
    assert report["candidate_type"] == "axisymmetric_interval_report"
    assert report["honesty"]["unproven_claim"] is False
    assert report["honesty"]["interval_verified"] is True
    assert report["upgrade_gate"]["tail_bounds_certified"] is True
    assert report["upgrade_gate"]["continuum_bound_certified"] is True
    assert report["upgrade_gate"]["finite_energy_certified"] is True
    reloaded = json.loads(json.dumps(report))
    assert reloaded["upgrade_gate"]["unproven_claim"] is False
    assert reloaded["axis_regular_checks"]["all_basis_checks_pass"] is True
    assert reloaded["axis_regular_checks"]["certified_smooth_axis"] is True
    assert "continuum_residual_certificates" in reloaded
    assert "finite_energy_tail_certificate" in reloaded


def test_interval_arithmetic_and_compactification_smoke_checks() -> None:
    left = interval_from_bounds(1.0, 2.0)
    right = interval_from_bounds(3.0, 4.0)
    assert scalar_interval_contains(interval_add(left, right), 5.0)
    assert scalar_interval_contains(interval_sub(right, left), 2.0)
    assert scalar_interval_contains(interval_mul(left, right), 8.0)
    assert scalar_interval_contains(interval_div(right, left), 2.0)
    assert scalar_interval_contains(interval_square(interval_from_bounds(-2.0, 3.0)), 9.0)
    assert scalar_interval_contains(interval_sqrt(interval_from_bounds(4.0, 9.0)), 3.0)

    comp = compactification_map_interval(interval_from_bounds(0.25, 0.5))
    assert scalar_interval_contains(comp["radius"], 1.0)
    assert scalar_interval_contains(comp["dr_drho"], 4.0)

    quad = interval_trapezoid_bound(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.5, 1.0]))
    assert scalar_interval_contains(quad, 0.5)


def test_tail_continuum_energy_and_axis_certificates_are_machine_readable() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=29,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    tails = certified_tail_bounds_from_artifact(artifact)
    assert tails
    assert all(tail["certified"] for tail in tails.values())
    continuum = continuum_residual_certificates(artifact, tail_bounds=tails)
    assert continuum["continuum_bound_certified"] is True
    energy = finite_energy_tail_certificate(artifact, tail_bounds=tails)
    assert energy["certified"] is True
    axis = axisymmetric_axis_smoothness_certificate(artifact)
    assert axis["certified_smooth_axis"] is True


def test_schema_rejects_unverified_unproven_claim() -> None:
    velocity, pressure, velocity_t = _taylor_green(16, 0.1)
    bundle = build_ns_cap_bundle(
        velocity,
        pressure,
        velocity_t=velocity_t,
        viscosity=0.1,
        honesty=HonestyLabels(unproven_claim=True),
    )
    errors = ns_cap_schema_errors(bundle)
    assert any("unproven_claim requires" in e for e in errors)


def test_compactified_schema_requires_concrete_tail_and_energy_fields() -> None:
    mms = manufactured_abc_flow(12, viscosity=0.05)
    bundle = build_ns_cap_bundle(
        mms["velocity"], mms["pressure"],
        velocity_t=mms["velocity_t"], forcing=mms["forcing"],
        viscosity=mms["viscosity"], domain_type="compactified_r3",
        compactification=compactified_r3_metadata(),
        tail_bounds=[
            TailBound("u", "chebyshev_radial_fourier_angular", 32, 1e-12),
            TailBound("p", "chebyshev_radial_fourier_angular", 32, 1e-12),
        ],
        finite_energy_checks={
            "kinetic_energy_upper_bound": 10.0,
            "tail_energy_upper_bound": 1e-10,
        },
    )
    assert ns_cap_schema_errors(bundle) == []
    assert bundle["domain"]["type"] == "compactified_r3"
    assert bundle["domain"]["compactification"]["map_name"] == "rational_radial"
    assert len(bundle["domain"]["tail_bounds"]) == 2


def test_candidate_upgrade_gates_are_conservative() -> None:
    mms = manufactured_abc_flow(12, viscosity=0.05)
    bundle = build_ns_cap_bundle(
        mms["velocity"], mms["pressure"],
        velocity_t=mms["velocity_t"], forcing=mms["forcing"],
        viscosity=mms["viscosity"],
    )
    no_report = candidate_upgrade_gates(bundle)
    assert no_report["stage"] == "numerical_artifact"
    report = candidate_upgrade_gates(
        bundle,
        independent_report={"residual_samples_match": True},
        config=CandidateGateConfig(require_tail_bounds_for_interval=True),
    )
    assert report["stage"] == "cap_candidate"
    assert report["unproven_claim"] is False

    uncertified = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        domain_type="compactified_r3",
        compactification=compactified_r3_metadata(),
        tail_bounds=[
            TailBound("u", "chebyshev_radial_fourier_angular", 32, 1e-12),
        ],
        finite_energy_checks={"kinetic_energy_upper_bound": 10.0},
        honesty=HonestyLabels(finite_energy_verified=True),
    )
    uncertified_gate = candidate_upgrade_gates(
        uncertified,
        independent_report={"residual_samples_match": True},
    )
    assert uncertified_gate["stage"] == "cap_candidate"
    assert uncertified_gate["tail_bounds_present"] is True
    assert uncertified_gate["tail_bounds_certified"] is False

    interval_ready = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        domain_type="compactified_r3",
        compactification=compactified_r3_metadata(),
        tail_bounds=[
            TailBound("u", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
            TailBound("p", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
        ],
        finite_energy_checks={"kinetic_energy_upper_bound": 10.0},
        honesty=HonestyLabels(finite_energy_verified=True),
    )
    interval_gate = candidate_upgrade_gates(
        interval_ready,
        independent_report={"residual_samples_match": True},
    )
    assert interval_gate["stage"] == "interval_obligation_ready"

    proof_ready = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        domain_type="compactified_r3",
        compactification=compactified_r3_metadata(),
        tail_bounds=[
            TailBound("u", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
            TailBound("p", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
        ],
        finite_energy_checks={"kinetic_energy_upper_bound": 10.0},
        honesty=HonestyLabels(finite_energy_verified=True, interval_verified=True),
    )
    proof_gate = candidate_upgrade_gates(
        proof_ready,
        independent_report={"residual_samples_match": True},
    )
    assert proof_gate["stage"] == "proof_assistant_obligation_ready"

    external = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        domain_type="compactified_r3",
        compactification=compactified_r3_metadata(),
        tail_bounds=[
            TailBound("u", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
            TailBound("p", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
        ],
        finite_energy_checks={"kinetic_energy_upper_bound": 10.0},
        honesty=HonestyLabels(
            finite_energy_verified=True,
            interval_verified=True,
            theorem_prover_verified=True,
        ),
    )
    external_gate = candidate_upgrade_gates(
        external,
        independent_report={"residual_samples_match": True},
    )
    # Self-declared theorem_prover_verified is never trusted: schema refuses the
    # forgery, and promotion consults check_certificate (absent here → unverified).
    assert external_gate["stage"] == "invalid"
    assert any("theorem_prover_verified" in e for e in external_gate["schema_errors"])
    assert external_gate["unproven_claim"] is False

    # Without the forged honesty flag the stage stops at the proof-assistant gate.
    kernel_ready = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        domain_type="compactified_r3",
        compactification=compactified_r3_metadata(),
        tail_bounds=[
            TailBound("u", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
            TailBound("p", "chebyshev_radial_fourier_angular", 32, 1e-12, certified=True),
        ],
        finite_energy_checks={"kinetic_energy_upper_bound": 10.0},
        honesty=HonestyLabels(finite_energy_verified=True, interval_verified=True),
    )
    kernel_gate = candidate_upgrade_gates(
        kernel_ready,
        independent_report={"residual_samples_match": True},
    )
    assert kernel_gate["stage"] == "proof_assistant_obligation_ready"


def test_certified_refinement_closure_formal_package_and_manifest() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=31,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    refinement = certified_candidate_refinement_report(artifact, interval_report)
    assert refinement["survived_certified_objectives"] is True
    assert refinement["honesty"]["unproven_claim"] is False

    blowup = build_blowup_closure_report(
        interval_report,
        approximate_inverse_norm=0.5,
        nonlinear_lipschitz_bound=0.1,
        residual_bound=0.1,
        norm_growth_exponent=0.25,
    )
    assert blowup["formalizable"] is True
    regularity = build_regularity_closure_report(
        inequality_name="synthetic_enstrophy_control",
        coefficients={"enstrophy": 1.0},
        counterexample_count=0,
    )
    assert regularity["formalizable"] is False
    closure = build_analytic_closure_report(
        interval_report,
        regularity_report=regularity,
        blowup_report=blowup,
    )
    assert closure["selected_route"] == "finite_time_blowup"

    formal = build_formal_proof_package(closure, target="written_cap")
    assert formal["theorem_prover_verified"] is False
    assert formal["unproven_claim"] is False
    assert formal["obligations"]
    assert formal["obligation_ids"] == [item["id"] for item in formal["obligations"]]
    assert formal["theorem_name"] in formal["proof_assistant_stub"]["source"]

    manifest = build_certificate_manifest(
        candidate_artifact=artifact,
        interval_report=interval_report,
        closure_report=closure,
        formal_package=formal,
    )
    assert "interval_report" in manifest["artifacts_present"]
    assert manifest["claim_boundary"]["unproven_claim"] is False


def test_axisymmetric_closure_certificates_are_machine_readable() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=41,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    function_space = axisymmetric_function_space_metadata(interval_report)
    assert function_space.theorem_grade is False
    assert function_space.coefficient_basis == "compact_polynomial_envelope"

    linearized = assemble_axisymmetric_linearized_operator(artifact)
    assert linearized["matrix_shape"][1] == len(artifact["replay_inputs"]["coefficients"])
    assert linearized["finite_dimensional_certified"] is True
    assert linearized["operator_theoretic_certified"] is False
    assert "continuum_operator_invertibility" in linearized["open_obligations"]
    active = assemble_axisymmetric_active_subspace_operator(artifact, active_indices=(0, 1, 2, 3))
    assert active["method"] == "central_finite_difference_active_subspace_residual_jacobian"
    assert active["active_coefficient_indices"] == [0, 1, 2, 3]
    assert active["active_coefficient_count"] == 4
    assert active["ambient_coefficient_count"] == len(artifact["replay_inputs"]["coefficients"])
    assert active["finite_dimensional_certified"] is True
    assert "active_subspace_completeness" in active["open_obligations"]
    operator = operator_theoretic_invertibility_certificate(interval_report, linearized)
    assert operator["projection"]["name"] == "axisymmetric_compact_polynomial_truncation"
    assert operator["operator_theoretic_certified"] is False
    assert "external_banach_space_invertibility_proof" in operator["open_obligations"]

    radii = radii_polynomial_certificate(interval_report, linearized)
    assert radii["certified"] is True
    assert radii["closure_interval"]["upper"] >= radii["closure_interval"]["lower"]
    assert "unproven_claim" in radii
    componentwise = componentwise_radii_polynomial_certificate(interval_report, linearized, operator)
    assert componentwise["components"]
    assert componentwise["proof_prep_certified"] is True

    norm = norm_divergence_certificate(interval_report, growth_exponent=0.25)
    assert norm["certified"] is False
    assert "link_norm_trace_to_field_profile" in norm["open_obligations"]
    linked_norm = norm_divergence_certificate(
        interval_report,
        growth_exponent=0.25,
        linked_to_field_profile=True,
    )
    assert linked_norm["certified"] is True

    blowup = build_axisymmetric_blowup_closure_report(
        interval_report,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    assert blowup["candidate_type"] == "blowup_analytic_closure_report"
    assert blowup["closure_consistency_verified"] is True
    assert blowup["formalizable"] is False
    assert "operator_theoretic_invertibility" in blowup["open_obligations"]
    assert blowup["blocker_resolution"]["operator_theoretic_invertibility"]["proof_prep_certified"] is True
    active_closure = build_axisymmetric_active_subspace_closure_report(
        interval_report,
        active_indices=(0, 1, 2, 3),
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    assert active_closure["candidate_type"] == "active_subspace_blowup_closure_report"
    assert active_closure["active_subspace"]["active_coefficient_indices"] == [0, 1, 2, 3]
    assert "active_subspace_completeness" in active_closure["open_obligations"]
    assert active_closure["unproven_claim"] is False
    invariance = active_subspace_invariance_report(artifact, active_indices=(0, 1, 2, 3))
    assert invariance["candidate_type"] == "active_subspace_invariance_report"
    assert invariance["active_indices"] == [0, 1, 2, 3]
    assert "external_sparse_ansatz_completeness_proof" in invariance["open_obligations"]
    assert invariance["unproven_claim"] is False
    frontier = active_subspace_absorption_frontier_report(
        interval_report,
        active_indices=(0, 1, 2, 3),
        max_combination_order=1,
    )
    assert frontier["candidate_type"] == "active_subspace_absorption_frontier_report"
    assert frontier["base_active_indices"] == [0, 1, 2, 3]
    assert frontier["passed_count"] >= 1
    assert frontier["unproven_claim"] is False
    finite_q = finite_active_tail_contraction_diagnostic(
        artifact,
        active_indices=(0, 1, 2, 3),
        tail_modes=frontier["required_tail_control_modes"],
    )
    assert finite_q["candidate_type"] == "finite_active_tail_contraction_diagnostic"
    assert finite_q["finite_contraction_ratio_upper"] >= 0.0
    assert finite_q["finite_tail_contraction_surrogate_passed"] is (
        finite_q["finite_contraction_ratio_upper"] < 1.0
    )
    assert "lift_finite_tail_contraction_to_weighted_analytic_tail_space" in finite_q["open_obligations"]
    tail_contract = weighted_analytic_tail_norm_contract(frontier)
    assert tail_contract["candidate_type"] == "weighted_analytic_tail_norm_contract"
    assert tail_contract["required_tail_modes"] == frontier["required_tail_control_modes"]
    assert tail_contract["weighted_tail_contract_certified"] is False
    lift = active_tail_contraction_lift_certificate(finite_q, tail_contract)
    assert lift["candidate_type"] == "active_tail_contraction_lift_certificate"
    assert lift["analytic_lift_certified"] is False
    assert lift["q_finite_upper"] == finite_q["finite_contraction_ratio_upper"]
    assert lift["margin_after_finite_q"] == 1.0 - finite_q["finite_contraction_ratio_upper"]
    assert "interval_jacobian_error_upper" in lift["missing_error_budget_terms"]

    # --- four lift error-term certificates + assembled budget ----------------
    projector_err = active_projector_error_certificate(finite_q)
    assert projector_err["candidate_type"] == "active_projector_error_certificate"
    assert projector_err["finite_dimensional_certified"] is True
    assert projector_err["theorem_grade_certified"] is False
    assert projector_err["projector_error_upper"] >= 0.0
    assert projector_err["sigma_min_active"] > 0.0
    assert "external_active_projector_perturbation_proof" in projector_err["open_obligations"]
    assert projector_err["unproven_claim"] is False

    interval_err = interval_jacobian_error_certificate(finite_q)
    assert interval_err["candidate_type"] == "interval_jacobian_error_certificate"
    assert interval_err["finite_dimensional_certified"] is True
    assert interval_err["theorem_grade_certified"] is False
    assert interval_err["interval_jacobian_error_upper"] >= 0.0
    assert (
        "replace_float64_jacobian_with_directed_interval_arithmetic"
        in interval_err["open_obligations"]
    )

    nonlinear_err = nonlinear_tail_remainder_certificate(finite_q)
    assert nonlinear_err["candidate_type"] == "nonlinear_tail_remainder_certificate"
    assert nonlinear_err["sampled_estimate_only"] is True
    assert nonlinear_err["theorem_grade_certified"] is False
    assert nonlinear_err["proof_status"] == "blocked_sampled_nonlinear_remainder"
    assert (
        "certify_second_derivative_bound_over_solution_ball"
        in nonlinear_err["open_obligations"]
    )
    # A supplied ball radius scales the (still blocked) sampled remainder linearly.
    nonlinear_small = nonlinear_tail_remainder_certificate(finite_q, solution_ball_radius=1e-3)
    assert nonlinear_small["solution_ball_radius_used"] == 1e-3
    assert nonlinear_small["theorem_grade_certified"] is False

    # Rigorous operator-norm Hessian bound (the residual is exactly quadratic, so
    # the Hessian is a constant tensor; the diagonal proxy is *not* an upper bound).
    op_norm = axisymmetric_residual_hessian_operator_norm(artifact)
    assert op_norm["candidate_type"] == "axisymmetric_residual_hessian_operator_norm"
    assert op_norm["exact_quadratic_max_rel_error"] < 1e-8
    assert op_norm["jacobian_lipschitz_operator_norm_upper"] >= op_norm["diagonal_hessian_proxy"]
    assert op_norm["remainder_operator_norm_upper"] >= 0.0
    nonlinear_rig = nonlinear_tail_remainder_certificate(
        finite_q, certify_hessian_operator_norm=True
    )
    assert nonlinear_rig["hessian_bound_rigorous"] is True
    assert (
        nonlinear_rig["hessian_bound_method"]
        == "certified_jacobian_lipschitz_operator_norm_upper"
    )
    assert nonlinear_rig["diagonal_hessian_proxy"] == nonlinear_err["hessian_operator_norm_proxy"]
    assert (
        abs(
            nonlinear_rig["hessian_operator_norm_proxy"]
            - op_norm["jacobian_lipschitz_operator_norm_upper"]
        )
        <= 1e-9
    )
    # The sampled-second-derivative obligation is discharged by the rigorous bound.
    assert (
        "certify_second_derivative_bound_over_solution_ball"
        not in nonlinear_rig["open_obligations"]
    )
    # The honest (rigorous) bound is at least as large as the optimistic sampled one.
    assert (
        nonlinear_rig["nonlinear_remainder_error_upper"]
        >= nonlinear_err["nonlinear_remainder_error_upper"]
    )

    analytic_err = analytic_tail_error_certificate(tail_contract)
    assert analytic_err["candidate_type"] == "analytic_tail_error_certificate"
    assert analytic_err["decay_rate_assumed_not_proven"] is True
    assert analytic_err["theorem_grade_certified"] is False
    assert (
        "prove_enriched_coefficient_geometric_decay_rate"
        in analytic_err["open_obligations"]
    )
    # A weight*decay product >= 1 makes the geometric tail diverge (blocked).
    analytic_divergent = analytic_tail_error_certificate(
        tail_contract, coefficient_decay_rate=2.0
    )
    assert analytic_divergent["geometric_series_convergent"] is False
    assert analytic_divergent["analytic_tail_error_upper"] is None

    budget = build_active_tail_lift_error_budget(
        finite_q,
        tail_contract,
        projector_certificate=projector_err,
        interval_jacobian_certificate=interval_err,
        nonlinear_remainder_certificate=nonlinear_err,
        analytic_tail_certificate=analytic_err,
    )
    assert budget["candidate_type"] == "active_tail_lift_error_budget"
    assert budget["all_terms_present"] is True
    assert budget["all_terms_theorem_grade"] is False
    assert set(budget["proof_prep_certified_terms"]) == {
        "interval_jacobian_error_upper",
        "projector_error_upper",
    }
    assert set(budget["blocked_terms"]) == {
        "analytic_tail_error_upper",
        "nonlinear_remainder_error_upper",
    }
    # The budget must compute a (conditional) q_total or name the blocking terms.
    assert budget["q_total_upper"] is not None
    assert budget["blocking_terms"]
    assert budget["analytic_lift_certified"] is False
    assert budget["lift_certificate"]["candidate_type"] == "active_tail_contraction_lift_certificate"
    assert budget["unproven_claim"] is False
    tail_contraction = active_subspace_tail_contraction_attempt(
        frontier,
        tail_contract,
        finite_diagnostic=finite_q,
        analytic_lift=lift,
    )
    assert tail_contraction["candidate_type"] == "active_subspace_tail_contraction_attempt"
    assert tail_contraction["finite_tail_contraction_surrogate_passed"] is finite_q[
        "finite_tail_contraction_surrogate_passed"
    ]
    assert tail_contraction["analytic_lift_certified"] is False
    assert "external_active_subspace_tail_contraction_proof" in tail_contraction["open_obligations"]
    completeness = active_subspace_completeness_theorem_attempt(
        frontier,
        tail_contraction,
        invariance_report=invariance,
    )
    assert completeness["candidate_type"] == "active_subspace_completeness_theorem_attempt"
    assert completeness["active_subspace_complete"] is False
    assert "active_subspace_tail_contraction" in completeness["open_obligations"]
    assumptions_gate = classical_assumptions_readiness_gate(
        interval_report,
        exact_bridge={"exact_profile_verified": False},
        norm_divergence={"norm_divergence_certified": False},
    )
    assert assumptions_gate["candidate_type"] == "classical_assumptions_readiness_gate"
    assert assumptions_gate["classical_assumptions_ready"] is False
    assert blowup["blocker_resolution"]["radii_polynomial_closure"]["components"]
    assert blowup["blocker_resolution"]["norm_divergence"]["linked_to_field_profile"] is True
    assert blowup["unproven_claim"] is False
    assert continuum_residual_upper_bound(interval_report) > 0.0


def test_regularity_inequality_report_and_formal_manifest_blockers() -> None:
    time = np.linspace(0.0, 1.0, 80)
    enstrophy = np.exp(time)
    traces = {
        "energy": 0.5 * enstrophy,
        "enstrophy": enstrophy,
        "palinstrophy": 2.0 * enstrophy,
        "bkm_vorticity_proxy": np.sqrt(enstrophy),
    }
    regularity_artifact = {
        "candidate_type": "regularity_growth_law",
        "replay_inputs": {
            "time": time.tolist(),
            "traces": {name: value.tolist() for name, value in traces.items()},
            "target": "enstrophy",
        },
        "result": {
            "coefficients": {"enstrophy": 1.0},
            "feature_names": ["energy", "enstrophy", "palinstrophy", "bkm_vorticity_proxy"],
        },
    }
    report = build_regularity_inequality_report(regularity_artifact, residual_tolerance=0.1)
    assert report["candidate_type"] == "regularity_inequality_report"
    assert report["obligations"]["a_priori_estimate_candidate"] is True
    assert report["counterexample_sweep"]["passed"] is True
    assert report["formalizable"] is False
    assert "all_smooth_finite_energy_data_proof" in report["open_obligations"]
    sweep = regularity_counterexample_sweep(
        {"enstrophy": 1.0},
        traces=traces,
        target="enstrophy",
        tolerance=0.1,
    )
    assert sweep["passed"] is True

    closure = build_analytic_closure_report(
        {"upgrade_gate": {}, "honesty": {}},
        regularity_report=report,
        blowup_report={
            "route": "finite_time_blowup",
            "formalizable": False,
            "open_obligations": ["operator_theoretic_invertibility"],
            "unproven_claim": False,
        },
    )
    formal = build_formal_proof_package(closure)
    assert "regularity_inequality_obligations" in formal["blocker_obligation_sections"]
    manifest = build_certificate_manifest(closure_report=closure, closure_certificates={"regularity": report})
    assert "all_smooth_finite_energy_data_proof" in manifest["open_obligations"]


def test_theorem_grade_attempts_remain_blocked_without_external_proof() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=43,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval_report,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    theorem_space = theorem_grade_function_space_contract(interval_report)
    assert theorem_space["theorem_grade"] is True
    assert theorem_space["unproven_claim"] is False

    linearized = blowup["closure_certificates"]["linearized_operator"]
    operator_attempt = continuum_banach_invertibility_attempt(interval_report, linearized)
    assert operator_attempt["operator_theoretic_certified"] is False
    assert "external_banach_space_invertibility_proof" in operator_attempt["open_obligations"]

    radii_attempt = theorem_grade_radii_polynomial_attempt(
        interval_report,
        operator_attempt,
        blowup["closure_certificates"]["radii_polynomial"]["componentwise"],
    )
    assert radii_attempt["radii_polynomial_closure"] is False
    assert "external_radii_polynomial_proof" in radii_attempt["open_obligations"]

    norm_attempt = exact_profile_norm_divergence_attempt(
        interval_report,
        blowup["closure_certificates"]["norm_divergence"],
    )
    assert norm_attempt["norm_divergence"] is False
    assert "blocked_profile_not_exact_solution" in norm_attempt["open_obligations"]

    regularity = build_regularity_closure_report(
        inequality_name="blocked_universal_bound",
        coefficients={},
    )
    regularity_attempt = regularity_all_data_proof_attempt(regularity)
    assert regularity_attempt["all_smooth_finite_energy_data_proof"] is False
    assert "all_smooth_finite_energy_data_proof" in regularity_attempt["open_obligations"]

    theorem_attempt = build_theorem_grade_closure_attempt(
        interval_report,
        blowup_report=blowup,
        regularity_report=regularity,
    )
    assert theorem_attempt["candidate_type"] == "theorem_grade_closure_attempt"
    assert theorem_attempt["proof_status"] in {
        "blocked_with_precise_obligations",
        "falsified_with_counterexample",
    }
    assert theorem_attempt["open_obligations"]
    formal = build_formal_proof_package(theorem_attempt)
    missing = verify_external_proof_package(formal, None)
    assert missing["verified"] is False
    gate = theorem_claim_gate(theorem_attempt, formal)
    assert gate["unproven_claim"] is False
    fake_external = external_verification_record(
        verifier="unit-test",
        theorem_name=formal["theorem_name"],
        discharged_obligations=tuple(formal["obligation_ids"]),
        artifact_sha256="abc123",
    )
    checked = verify_external_proof_package(formal, fake_external)
    assert checked["verified"] is True
    gated = theorem_claim_gate(theorem_attempt, formal, external_verification=fake_external)
    assert gated["unproven_claim"] is False


def _interval_report_with_residual(upper: float) -> dict[str, object]:
    """Minimal interval report whose continuum residual upper bound is ``upper``."""
    return {
        "continuum_residual_certificates": {
            "sections": {
                "axis": {
                    "row": {"continuum_sup_norm_interval": {"upper": upper}},
                },
            },
        },
        "upgrade_gate": {
            "finite_energy_certified": True,
            "axis_regular_certified": True,
        },
        "replay_inputs": {"refined_artifact": {}},
    }


def test_exact_profile_uses_certified_tolerance_not_exact_float_equality() -> None:
    norm_cert = {"growth_exponent": 1.0, "norm_name": "trace"}

    # Bit-exact zero residual: exact profile under the default eps = 0.0.
    zero = exact_profile_norm_divergence_attempt(
        _interval_report_with_residual(0.0), norm_cert
    )
    assert zero["field_profile_linkage"]["exact_profile_verified"] is True
    assert zero["field_profile_linkage"]["residual_tolerance"] == 0.0

    # A tiny positive certified upper bound is NOT exact under eps = 0.0.
    tiny = exact_profile_norm_divergence_attempt(
        _interval_report_with_residual(1e-12), norm_cert
    )
    assert tiny["field_profile_linkage"]["exact_profile_verified"] is False
    assert "blocked_profile_not_exact_solution" in tiny["open_obligations"]

    # ... but it IS accepted as exact-within-tolerance for a certified eps.
    within = exact_profile_norm_divergence_attempt(
        _interval_report_with_residual(1e-12),
        norm_cert,
        residual_tolerance=1e-10,
    )
    assert within["field_profile_linkage"]["exact_profile_verified"] is True
    assert within["field_profile_linkage"]["residual_tolerance"] == 1e-10

    # The predicate is a closed <= eps (boundary included), and a bound just
    # above eps is rejected.
    boundary = exact_profile_norm_divergence_attempt(
        _interval_report_with_residual(1e-10), norm_cert, residual_tolerance=1e-10
    )
    assert boundary["field_profile_linkage"]["exact_profile_verified"] is True
    above = exact_profile_norm_divergence_attempt(
        _interval_report_with_residual(2e-10), norm_cert, residual_tolerance=1e-10
    )
    assert above["field_profile_linkage"]["exact_profile_verified"] is False


def test_exact_profile_rejects_negative_residual_tolerance() -> None:
    with pytest.raises(ValueError, match="non-negative certified epsilon"):
        exact_profile_norm_divergence_attempt(
            _interval_report_with_residual(0.0),
            {"growth_exponent": 1.0},
            residual_tolerance=-1e-12,
        )


def test_full_ns_proof_program_report_tracks_open_lemmas() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=47,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval_report,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    regularity = build_regularity_closure_report(
        inequality_name="proof_program_placeholder",
        coefficients={},
    )
    theorem_attempt = build_theorem_grade_closure_attempt(
        interval_report,
        blowup_report=blowup,
        regularity_report=regularity,
    )
    exact = exact_navier_stokes_equation_contracts()
    assert "primitive" in exact["contracts"]
    spaces = theorem_grade_function_space_definitions()
    assert "finite_time_blowup" in spaces["definitions"]
    backend = interval_cap_backend_contract()
    assert backend["theorem_grade_ready"] is False

    blowup_lemmas = blowup_route_lemma_package(theorem_attempt)
    regularity_lemmas = regularity_route_lemma_package(theorem_attempt["route_attempts"]["regularity_all_data"])
    assert blowup_lemmas["open_lemmas"]
    assert regularity_lemmas["open_lemmas"]

    report = build_ns_proof_program_report(theorem_attempt=theorem_attempt)
    assert report["candidate_type"] == "navier_stokes_proof_program_report"
    assert report["proof_status"] in {"blocked_with_named_missing_lemma", "candidate_falsified"}
    assert report["open_obligations"]
    assert report["open_lemmas"]
    assert report["unproven_claim"] is False

    lean = lean_formalization_package(report)
    assert lean["artifact_sha256"]
    assert lean["proof_assistant_verified"] is False
    review = external_review_gate(report, lean)
    assert review["unproven_claim"] is False
    assert "external_review_missing_or_stale" in review["open_obligations"]
    assert report["proof_obligation_bundles"]
    assert report["verifier_ingestion"]["finite_time_blowup"]["accepted_obligations"] == []


def test_solve_or_falsify_report_materializes_all_roadmap_gates() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=51,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval_report,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    theorem_attempt = build_theorem_grade_closure_attempt(interval_report, blowup_report=blowup)
    linearized = blowup["closure_certificates"]["linearized_operator"]
    componentwise = blowup["closure_certificates"]["radii_polynomial"]["componentwise"]

    nontriviality = axisymmetric_nontriviality_gate(interval_report, blowup_report=blowup)
    assert nontriviality["candidate_type"] == "axisymmetric_nontriviality_gate"
    assert "certify_nontrivial_blowup_scaling_law" in nontriviality["open_obligations"]

    ansatz = conditioning_preserving_ansatz_report(linearized)
    assert "residual_sensitivity_orthogonalized_basis" in ansatz["recommended_ansatz_families"]

    backend = theorem_interval_backend_readiness_report(interval_report)
    assert backend["theorem_grade_ready"] is False
    continuum = continuum_neumann_inequality_certificate(
        interval_report,
        linearized,
        interval_backend_report=backend,
    )
    assert continuum["continuum_neumann_certified"] is False
    assert "certified_directed_rounding_interval_backend" in continuum["open_obligations"]
    assert componentwise["unproven_claim"] is False

    report = build_ns_solve_or_falsify_report(
        interval_report,
        blowup_report=blowup,
        theorem_attempt=theorem_attempt,
        candidate_family_status=[{"family": "unit", "status": "blocked_with_named_missing_lemma"}],
    )
    assert report["candidate_type"] == "navier_stokes_solve_or_falsify_report"
    assert report["unproven_claim"] is False
    assert report["phases"]["baseline"]["baseline_sha256"]
    assert report["phases"]["proof_program"]["candidate_type"] == "navier_stokes_proof_program_report"
    assert report["phases"]["formal_verification"]["final_claim_gate"]["unproven_claim"] is False
    assert report["open_obligations"]
    frontier = active_subspace_absorption_frontier_report(
        interval_report,
        active_indices=(0, 1, 2),
        max_combination_order=1,
    )
    finite_q = finite_active_tail_contraction_diagnostic(
        artifact,
        active_indices=(0, 1, 2),
        tail_modes=frontier["required_tail_control_modes"],
    )
    lift = active_tail_contraction_lift_certificate(
        finite_q,
        weighted_analytic_tail_norm_contract(frontier),
    )
    ladder = build_ns_theorem_ladder_report(
        interval_report,
        frontier,
        blowup_report=blowup,
        theorem_attempt=theorem_attempt,
        finite_tail_diagnostic=finite_q,
        analytic_tail_lift=lift,
    )
    assert ladder["candidate_type"] == "navier_stokes_theorem_ladder_report"
    assert ladder["phases"]["finite_theorem"]["candidate_type"] == "active_subspace_finite_theorem_report"
    assert ladder["phases"]["tail_norm_contract"]["candidate_type"] == "weighted_analytic_tail_norm_contract"
    assert ladder["phases"]["active_subspace_completeness"]["active_subspace_complete"] is False
    assert ladder["route_summary"]["finite_tail_contraction_surrogate_passed"] is finite_q[
        "finite_tail_contraction_surrogate_passed"
    ]
    assert ladder["route_summary"]["analytic_tail_lift_certified"] is False
    assert ladder["route_summary"]["classical_assumptions_ready"] is False
    assert ladder["unproven_claim"] is False


def test_theorem_gate_closure_requires_matching_obligation_evidence() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=49,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval_report,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    regularity = build_regularity_closure_report(
        inequality_name="proof_program_placeholder",
        coefficients={},
    )
    theorem_attempt = build_theorem_grade_closure_attempt(
        interval_report,
        blowup_report=blowup,
        regularity_report=regularity,
    )
    blowup_obligations = blowup_proof_obligation_bundles(theorem_attempt)
    regularity_obligations = regularity_proof_obligation_bundles(
        theorem_attempt["route_attempts"]["regularity_all_data"]
    )
    assert len(blowup_obligations) == 6
    assert len(regularity_obligations) == 4

    first_obligation = blowup_obligations[0]["obligation_id"]
    verifier = theorem_verifier_record(
        blowup_obligations,
        discharged_obligations=[first_obligation],
        reviewed_at_utc="2026-06-23T00:00:00Z",
    )
    ingestion = ingest_theorem_verifier_bundle(blowup_obligations, verifier)
    assert ingestion["accepted_obligations"] == [first_obligation]
    assert len(ingestion["rejected_obligations"]) == len(blowup_obligations) - 1

    partially_closed = blowup_route_lemma_package(theorem_attempt, verifier_bundle=verifier)
    assert partially_closed["lemmas"][first_obligation.split(".")[-1]]["status"] == "closed"
    assert partially_closed["proof_status"] == "blocked_with_named_missing_lemma"

    stale = dict(verifier)
    stale["reviewed_at_utc"] = ""
    stale_ingestion = ingest_theorem_verifier_bundle(blowup_obligations, stale)
    assert stale_ingestion["accepted_obligations"] == []
    assert any("freshness" in reason for reason in stale_ingestion["rejected_obligations"].values())

    mismatched = json.loads(json.dumps(verifier))
    mismatched["proof_records"][0]["theorem_name"] = "WrongTheorem"
    mismatch_ingestion = ingest_theorem_verifier_bundle(blowup_obligations, mismatched)
    assert mismatch_ingestion["accepted_obligations"] == []
    assert any("theorem_name" in reason for reason in mismatch_ingestion["rejected_obligations"].values())

    report = build_ns_proof_program_report(
        theorem_attempt=theorem_attempt,
        verifier_bundles={"blowup": verifier},
    )
    assert first_obligation in report["verifier_ingestion"]["finite_time_blowup"]["accepted_obligations"]
    assert report["unproven_claim"] is False
    formal = build_formal_proof_package(report)
    gate = theorem_claim_gate(theorem_attempt, formal, external_verification=verifier)
    assert gate["unproven_claim"] is False
    assert "proof_obligation_verifier_evidence" in gate["open_obligations"]


def test_proof_obligation_bundle_hash_changes_with_statement() -> None:
    bundle = proof_obligation_bundle(
        route="finite_time_blowup",
        lemma_id="unit_test_lemma",
        theorem_statement="Unit theorem statement.",
        assumptions=("assumption",),
        dependencies=("dependency",),
        source_artifact={"artifact": "source"},
    )
    changed = proof_obligation_bundle(
        route="finite_time_blowup",
        lemma_id="unit_test_lemma",
        theorem_statement="Changed theorem statement.",
        assumptions=("assumption",),
        dependencies=("dependency",),
        source_artifact={"artifact": "source"},
    )
    assert bundle["obligation_id"] == changed["obligation_id"]
    assert bundle["obligation_sha256"] != changed["obligation_sha256"]


# --------------------------------------------------------------------------- #
# Constantin-Lax-Majda (CLM) 1D finite-time blow-up certificate
# --------------------------------------------------------------------------- #


def test_certified_clm_blowup_matches_closed_form_at_origin() -> None:
    """Interval enclosures at x=0 contain the exact rational reference values."""
    coeffs = [-1.0, 0.5]
    scales = [1.0, 2.0]
    cert = certified_clm_blowup(coeffs=coeffs, scales=scales)

    # omega0 is odd => omega0(0) = 0 exactly; H omega0(0) = -sum c/a; omega0'(0) = sum c/a^2.
    ref_hw0 = -sum(c / a for c, a in zip(coeffs, scales, strict=True))
    ref_w0p = sum(c / a**2 for c, a in zip(coeffs, scales, strict=True))

    w0 = cert["omega0_at_zero"]
    hw0 = cert["hilbert_omega0_at_zero"]
    w0p = cert["omega0_prime_at_zero"]
    assert w0["lower"] <= 0.0 <= w0["upper"]
    assert hw0["lower"] <= ref_hw0 <= hw0["upper"]
    assert w0p["lower"] <= ref_w0p <= w0p["upper"]
    # tight, certified, no quadrature
    assert hw0["upper"] - hw0["lower"] < 1e-12
    assert hw0["certified"] is True


def test_certified_clm_blowup_certifies_finite_time_singularity() -> None:
    """omega0 = -q_1 satisfies the CLM criterion: H omega0(0) = 1 > 0, T = 2."""
    cert = certified_clm_blowup(coeffs=[-1.0], scales=[1.0])
    assert cert["singularity_certified"] is True
    assert cert["omega0_vanishes_at_zero"] is True
    assert cert["hilbert_positive_at_zero"] is True
    assert cert["gradient_nontrivial"] is True
    bt = cert["blowup_time"]
    assert bt is not None
    assert bt["lower"] <= 2.0 <= bt["upper"]
    assert bt["lower"] > 0.0
    assert certified_clm_blowup_schema_errors(cert) == []
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["honesty"]["one_dimensional_model"] is True
    assert cert["three_d_claim"] is False


def test_certified_clm_blowup_honestly_uncertified_when_hilbert_nonpositive() -> None:
    """omega0 = +q_1 has H omega0(0) = -1 < 0: no certified singularity, no time."""
    cert = certified_clm_blowup(coeffs=[1.0], scales=[1.0])
    assert cert["singularity_certified"] is False
    assert cert["hilbert_positive_at_zero"] is False
    assert cert["blowup_time"] is None
    assert certified_clm_blowup_schema_errors(cert) == []


def test_certified_clm_blowup_nodes_enclose_independent_reference() -> None:
    """Reported node midpoints reproduce the independent closed-form omega0 / H omega0."""
    coeffs = [-1.0, 0.3]
    scales = [1.0, 2.5]
    cert = certified_clm_blowup(coeffs=coeffs, scales=scales)
    xs = np.asarray(cert["nodes"])
    w0 = np.asarray(cert["omega0_on_nodes"])
    hw0 = np.asarray(cert["hilbert_omega0_on_nodes"])
    # omega0 = sum c_i x/(x^2+a_i^2);  H omega0 = -sum c_i a_i/(x^2+a_i^2)
    ref_w0 = sum(c * xs / (xs**2 + a**2) for c, a in zip(coeffs, scales, strict=True))
    ref_hw0 = -sum(c * a / (xs**2 + a**2) for c, a in zip(coeffs, scales, strict=True))
    assert np.allclose(w0, ref_w0, atol=1e-12, rtol=0.0)
    assert np.allclose(hw0, ref_hw0, atol=1e-12, rtol=0.0)


def test_clm_exact_solution_satisfies_pde_and_hilbert_identity() -> None:
    """Independent check that the CLM closed form solves omega_t = omega H(omega).

    Validates the literature formula used by the certificate by (a) confirming the
    explicit time-dependent Hilbert transform against a high-precision *principal
    value* quadrature, and (b) confirming the PDE residual via a central time
    difference -- so the certified construction is verified, not asserted.
    """
    mp = pytest.importorskip("mpmath")

    # Initial data omega0 = -q_1  (=> H omega0 = p_1, blow-up time T = 2).
    def omega0(s: object) -> object:
        return -s / (s * s + 1)

    def h0(s: object) -> object:
        return 1 / (s * s + 1)

    def denom(s: object, t: float) -> object:
        return (2 - t * h0(s)) ** 2 + (t * omega0(s)) ** 2

    def omega(s: object, t: float) -> object:
        return 4 * omega0(s) / denom(s, t)

    def hilbert_omega_closed(s: object, t: float) -> object:
        num = 2 * h0(s) * (2 - t * h0(s)) - 2 * t * omega0(s) ** 2
        return num / denom(s, t)

    def pv_hilbert(t: float, x0: float) -> float:
        """(1/pi) p.v. int omega(s,t)/(x0 - s) ds via the desingularized integrand."""
        with mp.workdps(40):
            fx0 = omega(mp.mpf(x0), t)

            def integrand(s: object) -> object:
                return (omega(s, t) - fx0) / (mp.mpf(x0) - s)

            return float(mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi)

    for t in (0.5, 1.0):  # both safely below T = 2
        for x in (0.3, -0.7, 1.5):
            # (a) closed-form Hilbert transform matches PV quadrature
            closed = float(hilbert_omega_closed(mp.mpf(x), t))
            assert abs(closed - pv_hilbert(t, x)) < 1e-9

            # (b) PDE residual omega_t - omega * H(omega) ~ 0 (central difference in t)
            d = 1e-6
            wt = float((omega(mp.mpf(x), t + d) - omega(mp.mpf(x), t - d)) / (2 * d))
            rhs = float(omega(mp.mpf(x), t)) * closed
            assert abs(wt - rhs) < 1e-5

    # The certified blow-up time for this datum is exactly T = 2.
    cert = certified_clm_blowup(coeffs=[-1.0], scales=[1.0])
    assert cert["blowup_time"]["lower"] <= 2.0 <= cert["blowup_time"]["upper"]


def test_certified_clm_blowup_schema_validation() -> None:
    cert = certified_clm_blowup(coeffs=[-1.0], scales=[1.0])
    assert cert["schema_version"] == CLM_BLOWUP_SCHEMA_VERSION
    assert certified_clm_blowup_schema_errors(cert) == []

    bad_claim = json.loads(json.dumps(cert))
    bad_claim["honesty"]["unproven_claim"] = True
    assert any("unproven_claim" in e for e in certified_clm_blowup_schema_errors(bad_claim))

    bad_three_d = json.loads(json.dumps(cert))
    bad_three_d["three_d_claim"] = True
    assert any("three_d_claim" in e for e in certified_clm_blowup_schema_errors(bad_three_d))

    bad_version = json.loads(json.dumps(cert))
    bad_version["schema_version"] = "wrong"
    assert any("schema_version" in e for e in certified_clm_blowup_schema_errors(bad_version))

    missing = json.loads(json.dumps(cert))
    del missing["blowup_time"]
    assert any("blowup_time" in e for e in certified_clm_blowup_schema_errors(missing))


def test_certified_clm_blowup_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        certified_clm_blowup(coeffs=[], scales=[])
    with pytest.raises(ValueError):
        certified_clm_blowup(coeffs=[1.0], scales=[1.0, 2.0])
    with pytest.raises(ValueError):
        certified_clm_blowup(coeffs=[1.0], scales=[-1.0])
    with pytest.raises(ValueError):
        certified_clm_blowup(coeffs=[1.0], scales=[0.0])


def test_certified_clm_blowup_json_and_provenance_are_deterministic() -> None:
    cert_a = certified_clm_blowup(coeffs=[-1.0, 0.25], scales=[1.0, 3.0])
    cert_b = certified_clm_blowup(coeffs=[-1.0, 0.25], scales=[1.0, 3.0])
    # JSON-serializable
    assert json.loads(json.dumps(cert_a)) == cert_a
    # deterministic content hash (no wall-clock timestamp)
    assert cert_a["provenance"]["sha256"] == cert_b["provenance"]["sha256"]
    assert len(cert_a["provenance"]["sha256"]) == 64
    assert cert_a["provenance"]["harness"].endswith("certified_clm_blowup")


# --------------------------------------------------------------------------- #
# CLM multi-zero EARLIEST blow-up certificate
# --------------------------------------------------------------------------- #


def test_clm_multizero_reduces_to_single_point_at_origin() -> None:
    """One basis term has only the origin zero; T* matches the single-point cert."""
    cert = certified_clm_multizero_first_blowup(coeffs=[-1.0], scales=[1.0])
    assert cert["schema_version"] == CLM_MULTIZERO_SCHEMA_VERSION
    assert certified_clm_multizero_first_blowup_schema_errors(cert) == []
    assert cert["n_distinct_positive_roots"] == 0
    assert cert["n_zeros_enclosed"] == 1  # origin only
    assert cert["completeness_certified"] is True
    assert cert["earliest_first_blowup_certified"] is True
    bt = cert["first_blowup_time"]
    assert bt["lower"] <= 2.0 <= bt["upper"]
    # earliest zero is the origin
    z = cert["earliest_zero_location"]
    assert z["lower"] <= 0.0 <= z["upper"]


def test_clm_multizero_enumerates_all_real_zeros() -> None:
    """A profile with k positive roots of P enumerates 1 + 2k real zeros."""
    coeffs = [-1.0, 2.0, -0.5]
    scales = [0.5, 1.5, 3.0]
    cert = certified_clm_multizero_first_blowup(coeffs=coeffs, scales=scales)
    k = cert["n_distinct_positive_roots"]
    assert cert["n_zeros_enclosed"] == 1 + 2 * k
    assert cert["completeness_certified"] is True
    assert certified_clm_multizero_first_blowup_schema_errors(cert) == []

    # Each reported x-zero genuinely encloses a root of omega0 = sum c_i x/(x^2+a^2).
    for z in cert["zero_locations"]:
        x = z["midpoint"]
        omega0 = sum(c * x / (x**2 + a**2) for c, a in zip(coeffs, scales, strict=True))
        assert abs(omega0) < 1e-9


def test_clm_multizero_two_sided_time_brackets_numpy_reference() -> None:
    """The certified earliest time encloses an independent numpy max-Hilbert value."""
    coeffs = [-1.0, 0.6, -0.3]
    scales = [0.7, 1.3, 2.6]
    cert = certified_clm_multizero_first_blowup(coeffs=coeffs, scales=scales)
    assert cert["completeness_certified"] is True

    # Independent reference: all zeros via the numerator polynomial roots, then
    # H omega0 = -sum c_i a_i/(x^2+a_i^2) evaluated at each zero.
    a2 = [a * a for a in scales]
    n = len(coeffs)
    # P(u) = sum_i c_i prod_{j!=i}(u + a_j^2), ascending coeffs.
    poly = np.array([0.0])
    for i in range(n):
        term = np.array([1.0])
        for j in range(n):
            if j != i:
                term = np.polynomial.polynomial.polymul(term, [a2[j], 1.0])
        poly = np.polynomial.polynomial.polyadd(poly, coeffs[i] * term)
    roots = np.polynomial.polynomial.polyroots(poly)
    us = [0.0] + [
        float(r.real) for r in roots if abs(r.imag) < 1e-9 and r.real > 0.0
    ]
    hvals = [-sum(c * a / (u + a * a) for c, a in zip(coeffs, scales, strict=True)) for u in us]
    h_max = max(hvals)
    assert h_max > 0.0
    t_ref = 2.0 / h_max

    bt = cert["first_blowup_time"]
    assert bt["lower"] <= t_ref <= bt["upper"]
    # The two-sided enclosure is tight (no quadrature anywhere).
    assert bt["upper"] - bt["lower"] < 1e-6


def test_clm_multizero_earliest_can_be_a_nonorigin_zero() -> None:
    """When a non-origin zero has the largest H omega0, it drives the earliest time."""
    # Searched datum: the interior zero (not the origin) attains max H omega0.
    coeffs = [-0.59, 0.08]
    scales = [1.45, 0.41]
    cert = certified_clm_multizero_first_blowup(coeffs=coeffs, scales=scales)
    assert cert["singularity_certified"] is True
    assert cert["completeness_certified"] is True
    # The earliest zero is genuinely away from the origin ...
    assert abs(cert["earliest_zero_location"]["midpoint"]) > 1e-2
    # ... and its Hilbert value strictly exceeds the origin's.
    h_origin = cert["hilbert_omega0_at_zeros"][0]["midpoint"]
    h_earliest = cert["earliest_zero_hilbert"]["midpoint"]
    assert h_earliest > h_origin
    # x = 0 is still recorded among the enumerated zeros.
    assert any(abs(z["midpoint"]) < 1e-12 for z in cert["zero_locations"])


def test_clm_multizero_honestly_uncertified_when_no_positive_hilbert() -> None:
    """If H omega0 <= 0 at every zero, no singularity is certified and time is None."""
    cert = certified_clm_multizero_first_blowup(coeffs=[1.0], scales=[1.0])
    assert cert["singularity_certified"] is False
    assert cert["first_blowup_time"] is None
    assert cert["first_blowup_time_upper_bound"] is None
    assert certified_clm_multizero_first_blowup_schema_errors(cert) == []


def test_clm_multizero_schema_and_honesty_guards() -> None:
    cert = certified_clm_multizero_first_blowup(coeffs=[-1.0, 0.4], scales=[1.0, 2.5])
    assert certified_clm_multizero_first_blowup_schema_errors(cert) == []
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["three_d_claim"] is False
    assert cert["continuum_navier_stokes_claim"] is False

    bad_claim = json.loads(json.dumps(cert))
    bad_claim["honesty"]["unproven_claim"] = True
    assert any(
        "unproven_claim" in e
        for e in certified_clm_multizero_first_blowup_schema_errors(bad_claim)
    )

    bad_complete = json.loads(json.dumps(cert))
    bad_complete["completeness_certified"] = False
    bad_complete["earliest_first_blowup_certified"] = True
    assert any(
        "completeness_certified" in e
        for e in certified_clm_multizero_first_blowup_schema_errors(bad_complete)
    )


def test_clm_multizero_json_and_provenance_are_deterministic() -> None:
    cert_a = certified_clm_multizero_first_blowup(coeffs=[-1.0, 0.4], scales=[1.0, 2.5])
    cert_b = certified_clm_multizero_first_blowup(coeffs=[-1.0, 0.4], scales=[1.0, 2.5])
    assert json.loads(json.dumps(cert_a)) == cert_a
    assert cert_a["provenance"]["sha256"] == cert_b["provenance"]["sha256"]
    assert cert_a["provenance"]["harness"].endswith("certified_clm_multizero_first_blowup")


def test_clm_multizero_independent_numpy_replay_twin_agrees() -> None:
    """The numpy-only symbolic twin (no omnibias.pinn.certified import) reproduces the certificate."""
    symbolic = pytest.importorskip("omnibias.symbolic")
    verify = symbolic.verify_clm_multizero_first_blowup
    data = (
        ([-1.0], [1.0]),
        ([-0.59, 0.08], [1.45, 0.41]),  # non-origin earliest zero
        ([-1.0, 2.0, -0.5], [0.5, 1.5, 3.0]),
        ([1.0], [1.0]),  # no certified singularity
    )
    for coeffs, scales in data:
        cert = certified_clm_multizero_first_blowup(coeffs=coeffs, scales=scales)
        report = verify(cert)
        assert report["replay_match"] is True, (coeffs, scales, report)
        assert report["zeros_are_genuine"] is True
        assert report["n_distinct_positive_roots_match"] is True
        assert report["hilbert_max_in_certificate_interval"] is True
        assert report["first_blowup_time_in_certificate_interval"] is True
        assert report["unproven_claim"] is False


def test_clm_multizero_earliest_time_matches_mpmath_principal_value() -> None:
    """Anti-faking: the certified earliest time follows from a PV-quadrature Hilbert.

    Uses a datum whose *non-origin* zero attains the maximum line Hilbert value, so
    the new multi-zero path (not just the origin) is what is being certified.  The
    nonlocal operator at the earliest zero is recomputed by an independent
    high-precision principal-value quadrature and the resulting blow-up time is
    checked against the certificate's two-sided interval.
    """
    mp = pytest.importorskip("mpmath")
    coeffs = [-0.59, 0.08]
    scales = [1.45, 0.41]
    cert = certified_clm_multizero_first_blowup(coeffs=coeffs, scales=scales)
    assert cert["singularity_certified"] is True
    assert cert["completeness_certified"] is True
    x_star = cert["earliest_zero_location"]["midpoint"]
    assert abs(x_star) > 1e-2  # genuinely a non-origin zero

    def omega0(s: object) -> object:
        return sum(c * s / (s * s + a * a) for c, a in zip(coeffs, scales, strict=True))

    # (a) x_star is a genuine zero of omega0.
    assert abs(float(omega0(mp.mpf(x_star)))) < 1e-9

    # (b) H omega0(x_star) via desingularized principal-value quadrature.
    def pv_hilbert(x0: float) -> float:
        with mp.workdps(40):
            f0 = omega0(mp.mpf(x0))

            def integrand(s: object) -> object:
                return (omega0(s) - f0) / (mp.mpf(x0) - s)

            return float(mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi)

    h_pv = pv_hilbert(x_star)
    h_iv = cert["earliest_zero_hilbert"]
    assert h_iv["lower"] <= h_pv <= h_iv["upper"]
    assert h_pv > 0.0

    # (c) the earliest blow-up time T* = 2 / H omega0(x_star) matches the interval.
    t_star = 2.0 / h_pv
    bt = cert["first_blowup_time"]
    assert bt["lower"] <= t_star <= bt["upper"]

    # (d) the CLM denominator collapses at (x_star, T*): genuine finite-time blow-up.
    denom = (2.0 - t_star * h_pv) ** 2 + (t_star * float(omega0(mp.mpf(x_star)))) ** 2
    assert abs(denom) < 1e-12


# --------------------------------------------------------------------------- #
# Cordoba-Cordoba-Fontelos self-similar radii-polynomial attempt
# --------------------------------------------------------------------------- #
_CCF_SCALES = [0.6, 1.3, 2.1]


def test_ccf_default_nodes_count_and_positivity() -> None:
    for n in (1, 2, 3, 5):
        nodes = default_ccf_collocation_nodes(n)
        assert len(nodes) == n
        assert all(y > 0.0 for y in nodes)
        assert list(nodes) == sorted(nodes)
    with pytest.raises(ValueError):
        default_ccf_collocation_nodes(0)


def test_ccf_generic_candidate_is_blocked_with_quantified_gap() -> None:
    """An un-refined candidate fails closure; the gap is reported, not faked."""
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=[1.0, 0.4, -0.2], scales=_CCF_SCALES, lam=0.5
    )
    assert cert["schema_version"] == CCF_SELFSIMILAR_SCHEMA_VERSION
    assert certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert) == []
    assert cert["closure_certified"] is False
    assert cert["selfsimilar_profile_certified"] is False
    assert cert["lambda_enclosure"] is None
    report = cert["closure_report"]
    # The discriminant is genuinely negative and the failure is explained.
    assert report["discriminant_lower"] < 0.0
    assert "discriminant" in report["failed_inequality"]
    assert report["residual_normal_form_Y0"] > 0.0


def test_ccf_refined_candidate_closes_and_encloses_lambda() -> None:
    """Refine -> certify: the radii polynomial closes with a two-sided lambda."""
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    assert refined["residual_max_abs"] < 1e-10
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    assert certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert) == []
    assert cert["operator_invertible_certified"] is True
    assert cert["closure_certified"] is True
    assert cert["selfsimilar_profile_certified"] is True
    report = cert["closure_report"]
    assert report["discriminant_lower"] >= 0.0
    assert report["linear_defect_Z1"] < 1.0
    # lambda is enclosed two-sided around the refined value.
    enc = cert["lambda_enclosure"]
    assert enc["lower"] <= refined["lam"] <= enc["upper"]
    assert cert["profile_enclosure_radius"] is not None
    assert enc["upper"] - enc["lower"] < 1e-6


def test_ccf_flux_form_also_closes() -> None:
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6, form="flux"
    )
    assert refined["residual_max_abs"] < 1e-10
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"], form="flux",
    )
    assert cert["form"] == "flux"
    assert cert["closure_certified"] is True
    assert certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert) == []


def test_ccf_far_field_and_sampled_residual_quantify_remaining_gap() -> None:
    """Even when collocation closes, the whole-line residual is honestly nonzero."""
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    report = cert["closure_report"]
    # A certified far-field tail bound exists and is finite.
    assert report["far_field_residual_bound"] > 0.0
    assert math.isfinite(report["far_field_residual_bound"])
    # The sampled between-node residual is NOT zero -> collocation-only, honest.
    assert report["residual_sampled_sup"] > 1e-3
    # The reused line tail machinery bounds H on the core.
    assert report["hilbert_far_field_tail_on_core"] is not None
    assert cert["honesty"]["whole_line_certified"] is False
    assert cert["honesty"]["collocation_only"] is True


def test_ccf_certified_residual_sup_bounds_between_nodes() -> None:
    """The Taylor-model whole-line residual sup rigorously bounds *every* y."""
    from omnibias.pinn.certified.navier_stokes import _ccf_residual_interval

    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    report = cert["closure_report"]
    # New certified fields are present and well-formed.
    assert report["between_node_residual_certified"] is True
    assert report["residual_taylor_model_order"] == 6
    assert report["residual_taylor_model_leaf_cells"] >= 64
    certified = report["residual_certified_sup"]
    assert math.isfinite(certified) and certified > 0.0
    # The rigorous bound dominates the merely sampled estimate ...
    assert certified + 1e-9 >= report["residual_sampled_sup"]
    # ... yet stays tight (a Taylor model, not a wrapping-blown interval).
    assert certified <= 10.0 * report["residual_sampled_sup"]

    # Anti-faking: sample E on a grid 10x finer than the harness used; EVERY
    # point the original sampling skipped must still respect the certified sup.
    cs = cert["coeffs"]
    as_ = cert["scales"]
    lam = cert["lambda_candidate"]
    s = cert["velocity_sign"]
    yt = report["far_field_trunc"]
    n = 2000
    fine = max(_ccf_residual_interval(cs, as_, lam, yt * t / n, "transport", s).mag for t in range(n + 1))
    assert fine <= certified + 1e-9


def test_ccf_certified_residual_sup_obligation_is_discharged() -> None:
    """The between-node residual sup is no longer an open obligation."""
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    obligations = cert["linearized_operator"]["open_obligations"]
    assert "between_node_and_whole_line_residual_sup" not in obligations
    assert "shrink_certified_residual_sup_below_function_space_closure_threshold" in obligations
    # The continuum Frechet-derivative bound is now discharged (continuum_operator);
    # what remains is the Neumann invertibility threshold rho < 1.
    assert "continuum_frechet_derivative_bound" not in obligations
    assert "continuum_linearized_neumann_rho_below_one" in obligations
    # The result is still honestly collocation-only (no whole-line existence).
    assert cert["honesty"]["whole_line_certified"] is False
    assert certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert) == []


def test_ccf_certified_residual_sup_flux_form() -> None:
    """The Taylor-model residual path also covers the flux form ((H Theta)')."""
    from omnibias.pinn.certified.navier_stokes import _ccf_residual_interval

    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6, form="flux"
    )
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"], form="flux",
    )
    report = cert["closure_report"]
    certified = report["residual_certified_sup"]
    assert math.isfinite(certified) and certified > 0.0
    cs = cert["coeffs"]
    as_ = cert["scales"]
    lam = cert["lambda_candidate"]
    s = cert["velocity_sign"]
    yt = report["far_field_trunc"]
    n = 1500
    fine = max(_ccf_residual_interval(cs, as_, lam, yt * t / n, "flux", s).mag for t in range(n + 1))
    assert fine <= certified + 1e-9


def test_ccf_linearized_operator_bound_exact_scaling_resolvent() -> None:
    """kappa = 1/|1/2 + 3/2 lambda| is the exact dilation-generator resolvent norm."""
    for lam in (0.6, -1.2831742073, 1.7):
        op = certified_ccf_linearized_operator_bound(
            coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=lam,
        )
        assert op["supported"] is True
        assert op["scaling_invertible"] is True
        a = 0.5 + 1.5 * lam
        assert op["scaling_shift"] == pytest.approx(a)
        # certified kappa is a tight UPPER bound on the exact value.
        assert op["scaling_inverse_norm_bound"] >= 1.0 / abs(a) - 1e-12
        assert op["scaling_inverse_norm_bound"] <= 1.0 / abs(a) + 1e-9
        assert certified_ccf_linearized_operator_bound_schema_errors(op) == []


def test_ccf_linearized_operator_bound_blocks_o1_profile() -> None:
    """The O(1) refined profile's linearization is far from invertible (rho >> 1)."""
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    op = certified_ccf_linearized_operator_bound(
        coeffs=refined["coeffs"], scales=refined["scales"], lam=refined["lam"],
    )
    assert op["neumann_rho"] > 1.0
    assert op["rho_closes"] is False
    assert op["continuum_invertible_certified"] is False
    assert op["inverse_norm_bound"] is None
    assert "continuum_linearized_neumann_rho_below_one" in op["open_obligations"]
    # The forward operator-norm bound is finite (a genuine continuum Frechet bound).
    assert math.isfinite(op["forward_operator_bound"]) and op["forward_operator_bound"] > 0.0


def test_ccf_linearized_operator_bound_closes_small_amplitude() -> None:
    """A small-amplitude profile makes rho < 1: continuum invertibility is certified."""
    op = certified_ccf_linearized_operator_bound(
        coeffs=[0.03, -0.02, 0.01], scales=_CCF_SCALES, lam=0.6,
    )
    assert op["rho_closes"] is True
    assert op["neumann_rho"] < 1.0
    assert op["continuum_invertible_certified"] is True
    # ||DE^{-1}|| <= kappa/(1-rho), as a rigorous upper bound.
    kappa = op["scaling_inverse_norm_bound"]
    rho = op["neumann_rho"]
    assert op["inverse_norm_bound"] >= kappa / (1.0 - rho) - 1e-9
    assert op["open_obligations"] == []
    assert certified_ccf_linearized_operator_bound_schema_errors(op) == []


def test_ccf_linearized_operator_sups_dominate_dense_samples() -> None:
    """Anti-faking: certified coefficient sups dominate a dense direct sampling."""
    cs = [1.0, -0.5, 0.3]
    op = certified_ccf_linearized_operator_bound(coeffs=cs, scales=_CCF_SCALES, lam=0.6)
    as_ = _CCF_SCALES
    yt = op["far_field_trunc"]
    n = 5000
    hthy_s = thp_s = mult_s = 0.0
    for t in range(n + 1):
        y = yt * t / n
        d = [(y * y + a * a) for a in as_]
        hthy = sum(c / di for c, di in zip(cs, d, strict=True))
        thp = sum(c * (-2.0 * a * y) / (di * di) for c, a, di in zip(cs, as_, d, strict=True))
        mult = (1.0 + 0.6) + 1.0 * hthy
        hthy_s = max(hthy_s, abs(hthy))
        thp_s = max(thp_s, abs(thp))
        mult_s = max(mult_s, abs(mult))
    assert hthy_s <= op["htheta_over_y_sup"] + 1e-9
    assert thp_s <= op["theta_prime_sup"] + 1e-9
    assert mult_s <= op["multiplier_sup"] + 1e-9


def test_ccf_linearized_operator_flux_unsupported() -> None:
    """The flux form's extra Theta (H h)' term needs H^1; honestly reported."""
    op = certified_ccf_linearized_operator_bound(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6, form="flux",
    )
    assert op["supported"] is False
    assert op["continuum_invertible_certified"] is False
    assert "flux_form_continuum_linearization_requires_h1_space" in op["open_obligations"]
    assert certified_ccf_linearized_operator_bound_schema_errors(op) == []


def test_ccf_continuum_operator_embedded_and_replays() -> None:
    """The main certificate embeds the continuum operator block and it replays."""
    from omnibias.symbolic.ccf import verify_ccf_linearized_operator_bound

    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    block = cert["continuum_operator"]
    assert block["supported"] is True
    assert certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert) == []
    replay = verify_ccf_linearized_operator_bound(block)
    assert replay["replay_match"] is True
    assert replay["kappa_match"] and replay["rho_match"] and replay["verdict_match"]
    assert replay["sup_dominates_samples"] is True


def test_ccf_continuum_operator_replay_catches_forged_sup() -> None:
    """Replaying a certificate with an understated sup must fail (anti-faking)."""
    from omnibias.symbolic.ccf import verify_ccf_linearized_operator_bound

    op = certified_ccf_linearized_operator_bound(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6,
    )
    forged = json.loads(json.dumps(op))
    forged["htheta_over_y_sup"] = 1e-6  # impossibly small
    replay = verify_ccf_linearized_operator_bound(forged)
    assert replay["sup_dominates_samples"] is False
    assert replay["replay_match"] is False


def test_ccf_radii_polynomial_closure_pure_function() -> None:
    """The closure helper: closes on small residual, fails on large, exact algebra."""
    # Y0 small, Z1 ~ 0, modest Z2 -> closes with a sensible existence radius.
    ok = radii_polynomial_closure(1e-6, 1e-12, 50.0)
    assert ok["passed"] is True
    assert ok["discriminant_lower"] >= 0.0
    assert 0.0 < ok["r_minus"] < ok["r_plus"]
    # Large residual -> discriminant negative, blocked with explanation.
    bad = radii_polynomial_closure(1.0, 1e-12, 50.0)
    assert bad["passed"] is False
    assert bad["discriminant_lower"] < 0.0
    assert "discriminant" in bad["failed_inequality"]
    # Defect >= 1 -> the linearization is uncontrolled.
    nondef = radii_polynomial_closure(1e-6, 1.5, 50.0)
    assert nondef["passed"] is False
    assert "Z1" in nondef["failed_inequality"]


def test_ccf_schema_and_honesty_guards() -> None:
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    cert = certified_ccf_selfsimilar_blowup_attempt(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    assert certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert) == []
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["three_d_claim"] is False
    assert cert["continuum_navier_stokes_claim"] is False

    forged_claim = json.loads(json.dumps(cert))
    forged_claim["honesty"]["unproven_claim"] = True
    assert any(
        "unproven_claim" in e
        for e in certified_ccf_selfsimilar_blowup_attempt_schema_errors(forged_claim)
    )

    forged_line = json.loads(json.dumps(cert))
    forged_line["honesty"]["whole_line_certified"] = True
    assert any(
        "whole_line_certified" in e
        for e in certified_ccf_selfsimilar_blowup_attempt_schema_errors(forged_line)
    )


def test_ccf_json_and_provenance_are_deterministic() -> None:
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    kw = dict(
        coeffs=refined["coeffs"], scales=refined["scales"],
        lam=refined["lam"], nodes=refined["nodes"],
    )
    cert_a = certified_ccf_selfsimilar_blowup_attempt(**kw)
    cert_b = certified_ccf_selfsimilar_blowup_attempt(**kw)
    assert json.loads(json.dumps(cert_a)) == cert_a
    assert cert_a["provenance"]["sha256"] == cert_b["provenance"]["sha256"]
    assert cert_a["provenance"]["harness"].endswith(
        "certified_ccf_selfsimilar_blowup_attempt"
    )


def test_ccf_independent_numpy_replay_twin_agrees() -> None:
    """The numpy-only symbolic twin (no omnibias.pinn.certified import) reproduces the closure."""
    symbolic = pytest.importorskip("omnibias.symbolic")
    verify = symbolic.verify_ccf_selfsimilar_blowup_attempt
    # both a closing (refined) and a non-closing (generic) certificate.
    refined = refine_ccf_selfsimilar_profile(
        coeffs=[1.0, -0.5, 0.3], scales=_CCF_SCALES, lam=0.6
    )
    cases = [
        certified_ccf_selfsimilar_blowup_attempt(
            coeffs=refined["coeffs"], scales=refined["scales"],
            lam=refined["lam"], nodes=refined["nodes"],
        ),
        certified_ccf_selfsimilar_blowup_attempt(
            coeffs=[1.0, 0.4, -0.2], scales=_CCF_SCALES, lam=0.5
        ),
    ]
    for cert in cases:
        report = verify(cert)
        assert report["replay_match"] is True, (cert["closure_certified"], report)
        assert report["verdict_match"] is True
        assert report["unproven_claim"] is False


def test_ccf_hilbert_matches_mpmath_principal_value() -> None:
    """Anti-faking: the exact even-profile Hilbert equals a PV quadrature on the line.

    The certificate's whole closure rests on ``H[p_a] = q_a`` being the *exact*
    line Hilbert transform.  Here that closed form is checked, at the collocation
    nodes, against an independent high-precision desingularized principal-value
    quadrature of ``H[Theta](x0) = (1/pi) p.v. int Theta(t)/(x0 - t) dt``.
    """
    mp = pytest.importorskip("mpmath")
    coeffs = [1.0, -0.5, 0.3]
    scales = _CCF_SCALES

    def theta(s: object) -> object:
        return sum(c * a / (s * s + a * a) for c, a in zip(coeffs, scales, strict=True))

    def h_theta_closed(x: float) -> float:
        return sum(c * x / (x * x + a * a) for c, a in zip(coeffs, scales, strict=True))

    def pv_hilbert(x0: float) -> float:
        with mp.workdps(40):
            f0 = theta(mp.mpf(x0))

            def integrand(s: object) -> object:
                return (theta(s) - f0) / (mp.mpf(x0) - s)

            return float(mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi)

    for x0 in default_ccf_collocation_nodes(len(coeffs)):
        assert abs(h_theta_closed(x0) - pv_hilbert(x0)) < 1e-9


def test_certified_gclm_selfsimilar_blowup_exact_profile_identity() -> None:
    """The a=1/2 profile solves the self-similar ODE as an exact rational identity."""
    cert = certified_gclm_selfsimilar_blowup(a=0.5)
    assert cert["profile_equation_exactly_satisfied"] is True
    # cleared-denominator residual polynomial coefficients are *exactly* zero
    assert cert["profile_residual_polynomial"] == [0.0, 0.0, 0.0]
    # interval re-check on the grid: residual encloses 0 and is tiny
    assert cert["residual_contains_zero_on_nodes"] is True
    assert cert["max_profile_residual_abs"] < 1e-12
    assert cert["blowup_certified"] is True
    assert certified_gclm_selfsimilar_blowup_schema_errors(cert) == []
    assert cert["honesty"]["unproven_claim"] is False
    assert cert["honesty"]["one_dimensional_model"] is True
    assert cert["honesty"]["advective_model"] is True
    assert cert["three_d_claim"] is False
    assert cert["continuum_navier_stokes_claim"] is False


def test_certified_gclm_selfsimilar_constants_and_exponents() -> None:
    """Eigenvalues, focusing rate and amplitude exponent match the OSW a=1/2 theory."""
    cert = certified_gclm_selfsimilar_blowup(a=0.5)
    assert cert["advection_parameter_a"] == 0.5
    assert cert["c_l"] == pytest.approx(1.0 / 6.0)
    assert cert["c_omega"] == pytest.approx(-0.5)
    assert cert["poisson_scale_squared"] == pytest.approx(3.0 / 8.0)
    # gamma = -c_l / c_omega = 1/3 (focusing, spatially shrinking at x=0)
    assert cert["gamma"] == pytest.approx(1.0 / 3.0)
    assert cert["gamma"] == pytest.approx(-cert["c_l"] / cert["c_omega"])
    assert cert["gamma_exact"] == "1/3"
    assert cert["focusing_at_origin"] is True
    # amplitude exponent lambda = -1 (forced by the quadratic vortex-stretching term)
    assert cert["amplitude_exponent_lambda"] == -1
    # ||Omega_bar||_inf = sqrt(3)/2 (exact extremum at X* = sqrt(1/8))
    assert cert["profile_sup_norm"] == pytest.approx(math.sqrt(3.0) / 2.0)


def test_certified_gclm_selfsimilar_hilbert_matches_mpmath() -> None:
    """Independently verify H[Omega_bar] = (1/2) q_b' and the profile ODE via mpmath.

    The exact whole-line Hilbert transform used by the certificate comes from the
    verified line rotation identity ``H[p_a] = q_a`` (differentiated).  Here we
    confirm it against a high-precision *principal value* quadrature and check the
    self-similar ODE residual numerically -- so the construction is verified, not
    asserted.
    """
    mp = pytest.importorskip("mpmath")
    c = 3.0 / 8.0
    b = math.sqrt(c)
    c_l, a_param, c_omega = 1.0 / 6.0, 0.5, -0.5

    def omega(s: object) -> object:
        return -b * s / (s * s + c) ** 2  # Omega_bar = (1/2) p_b'

    def omega_x(s: object) -> object:
        return b * (3 * s * s - c) / (s * s + c) ** 3

    def hilbert_closed(s: object) -> object:
        return 0.5 * (c - s * s) / (s * s + c) ** 2  # (1/2) q_b'

    def u_vel(s: object) -> object:
        return 0.5 * s / (s * s + c)

    def pv_hilbert(x0: float) -> float:
        with mp.workdps(40):
            fx0 = omega(mp.mpf(x0))

            def integrand(s: object) -> object:
                return (omega(s) - fx0) / (mp.mpf(x0) - s)

            return float(mp.quad(integrand, [mp.ninf, x0, mp.inf]) / mp.pi)

    cert = certified_gclm_selfsimilar_blowup(a=0.5)
    nodes = cert["nodes"]
    hcl_nodes = cert["hilbert_profile_on_nodes"]
    for x in (0.0, 0.3, -0.7, 1.5):
        closed = float(hilbert_closed(mp.mpf(x)))
        # (a) closed-form Hilbert transform matches PV quadrature
        assert abs(closed - pv_hilbert(x)) < 1e-9
        # (b) self-similar ODE residual ~ 0 with these eigenvalues
        ux = closed  # U_X = H[Omega_bar]
        res = float(
            (c_l * x + a_param * float(u_vel(mp.mpf(x)))) * float(omega_x(mp.mpf(x)))
            - (c_omega + ux) * float(omega(mp.mpf(x)))
        )
        assert abs(res) < 1e-9

    # (c) the certificate's reported (line.py) Hilbert nodes match the PV reference
    for x, hc in zip(nodes, hcl_nodes, strict=True):
        assert abs(hc - float(hilbert_closed(mp.mpf(x)))) < 1e-12


def test_certified_gclm_nodes_reproduce_closed_form_profile() -> None:
    """Reported node midpoints reproduce the closed-form profile and its transform."""
    cert = certified_gclm_selfsimilar_blowup(a=0.5)
    xs = np.asarray(cert["nodes"])
    om = np.asarray(cert["omega_profile_on_nodes"])
    hom = np.asarray(cert["hilbert_profile_on_nodes"])
    c = 3.0 / 8.0
    b = math.sqrt(c)
    ref_om = -b * xs / (xs**2 + c) ** 2
    ref_hom = 0.5 * (c - xs**2) / (xs**2 + c) ** 2
    assert np.allclose(om, ref_om, atol=1e-12, rtol=0.0)
    assert np.allclose(hom, ref_hom, atol=1e-12, rtol=0.0)


def test_certified_gclm_selfsimilar_blowup_rejects_unsupported_a() -> None:
    """Only a=1/2 has an elementary Poisson-basis profile; others raise."""
    for bad in (0.0, 1.0, 0.7, -0.5):
        with pytest.raises(NotImplementedError):
            certified_gclm_selfsimilar_blowup(a=bad)


def test_certified_gclm_selfsimilar_blowup_custom_nodes() -> None:
    """Custom evaluation nodes are honoured and stay on the certified-zero residual."""
    nodes = [-2.0, -0.5, 0.0, 0.5, 2.0]
    cert = certified_gclm_selfsimilar_blowup(a=0.5, nodes=nodes)
    assert cert["nodes"] == nodes
    assert len(cert["omega_profile_on_nodes"]) == len(nodes)
    assert cert["residual_contains_zero_on_nodes"] is True
    assert cert["max_profile_residual_abs"] < 1e-12


def test_certified_gclm_selfsimilar_blowup_schema_validation() -> None:
    cert = certified_gclm_selfsimilar_blowup(a=0.5)
    assert cert["schema_version"] == GCLM_BLOWUP_SCHEMA_VERSION
    assert certified_gclm_selfsimilar_blowup_schema_errors(cert) == []

    bad_claim = json.loads(json.dumps(cert))
    bad_claim["honesty"]["unproven_claim"] = True
    assert any(
        "unproven_claim" in e
        for e in certified_gclm_selfsimilar_blowup_schema_errors(bad_claim)
    )

    bad_three_d = json.loads(json.dumps(cert))
    bad_three_d["three_d_claim"] = True
    assert any(
        "three_d_claim" in e
        for e in certified_gclm_selfsimilar_blowup_schema_errors(bad_three_d)
    )

    bad_version = json.loads(json.dumps(cert))
    bad_version["schema_version"] = "wrong"
    assert any(
        "schema_version" in e
        for e in certified_gclm_selfsimilar_blowup_schema_errors(bad_version)
    )

    # a blow-up claim with a non-zero residual polynomial must be rejected
    bad_residual = json.loads(json.dumps(cert))
    bad_residual["profile_residual_polynomial"] = [1.0, 0.0, 0.0]
    assert any(
        "profile_residual_polynomial" in e
        for e in certified_gclm_selfsimilar_blowup_schema_errors(bad_residual)
    )

    # a blow-up claim that is not exactly satisfied must be rejected
    bad_exact = json.loads(json.dumps(cert))
    bad_exact["profile_equation_exactly_satisfied"] = False
    assert any(
        "profile_equation_exactly_satisfied" in e
        for e in certified_gclm_selfsimilar_blowup_schema_errors(bad_exact)
    )


def test_certified_gclm_selfsimilar_blowup_json_and_provenance_are_deterministic() -> None:
    cert_a = certified_gclm_selfsimilar_blowup(a=0.5)
    cert_b = certified_gclm_selfsimilar_blowup(a=0.5)
    assert json.loads(json.dumps(cert_a)) == cert_a
    assert cert_a["provenance"]["sha256"] == cert_b["provenance"]["sha256"]
    assert len(cert_a["provenance"]["sha256"]) == 64
    assert cert_a["provenance"]["harness"].endswith("certified_gclm_selfsimilar_blowup")


def test_certified_gclm_gradient_amplification_rate_matches_spatial_derivative() -> None:
    """The certified t=0 rate equals d/dx(RHS)(0) by independent mpmath quadrature.

    The stagnation-point rate ``d/dt omega_x(0,t) = (1-a) H omega(0,t) omega_x(0,t)``
    is, at ``t=0``, the spatial ``x``-derivative of the gCLM right-hand side
    ``RHS(x) = -a u(x) omega0'(x) + H omega0(x) omega0(x)`` at the stagnation point.
    We confirm it independently: ``H omega0(0)`` against a principal-value quadrature,
    and the rate against a central difference of ``RHS`` -- so the derivation (the
    ``(1-a)`` factor and the advection/stretching cancellation) is verified.
    """
    mp = pytest.importorskip("mpmath")
    coeffs = [-1.0, 0.4]
    scales = [1.0, 2.0]

    def omega0(s: object) -> object:
        return sum(c * s / (s * s + a * a) for c, a in zip(coeffs, scales, strict=True))

    def omega0_p(s: object) -> object:
        return sum(
            c * (a * a - s * s) / (s * s + a * a) ** 2
            for c, a in zip(coeffs, scales, strict=True)
        )

    def hilbert_closed(x: object) -> object:  # H q_a = -p_a = -a/(x^2+a^2)
        return sum(
            c * (-a) / (x * x + a * a) for c, a in zip(coeffs, scales, strict=True)
        )

    def velocity(x: object) -> object:  # u = int_0^x H omega0 = -sum c_i atan(x/a_i)
        return sum(
            c * (-mp.atan(x / a)) for c, a in zip(coeffs, scales, strict=True)
        )

    def pv_hilbert(x0: float) -> float:
        with mp.workdps(40):
            f0 = omega0(mp.mpf(x0))
            return float(
                mp.quad(lambda s: (omega0(s) - f0) / (mp.mpf(x0) - s), [mp.ninf, x0, mp.inf])
                / mp.pi
            )

    # H omega0(0) = -sum c/a, validated against PV quadrature
    hw0_0 = -sum(c / a for c, a in zip(coeffs, scales, strict=True))
    assert abs(hw0_0 - pv_hilbert(0.0)) < 1e-9
    w0p_0 = sum(c / a**2 for c, a in zip(coeffs, scales, strict=True))

    for a_param in (0.0, 0.5, 1.0, 1.5):

        def rhs(x: object, a_param: float = a_param) -> float:
            return float(
                -a_param * float(velocity(x)) * float(omega0_p(x))
                + float(hilbert_closed(x)) * float(omega0(x))
            )

        h = 1e-5
        approx = (rhs(mp.mpf(h)) - rhs(mp.mpf(-h))) / (2 * h)
        exact = (1.0 - a_param) * hw0_0 * w0p_0
        assert abs(approx - exact) < 1e-6

        cert = certified_gclm_gradient_amplification(
            a=a_param, coeffs=coeffs, scales=scales
        )
        gdt = cert["gradient_time_derivative_at_zero"]
        assert gdt["lower"] <= exact <= gdt["upper"]
        rate = cert["amplification_rate"]
        assert rate["lower"] <= (1.0 - a_param) * hw0_0 <= rate["upper"]
        assert certified_gclm_gradient_amplification_schema_errors(cert) == []


def test_certified_gclm_gradient_amplification_de_gregorio_neutral() -> None:
    """At a=1 (De Gregorio) the stagnation rate vanishes for *any* odd datum."""
    for coeffs, scales in (
        ([-1.0], [1.0]),
        ([-1.0, 2.0, 0.3], [0.5, 1.5, 3.0]),
    ):
        cert = certified_gclm_gradient_amplification(a=1.0, coeffs=coeffs, scales=scales)
        assert cert["regime"] == "neutral"
        assert cert["neutral"] is True
        assert cert["instantaneous_amplification_certified"] is False
        assert cert["instantaneous_damping_certified"] is False
        rate = cert["amplification_rate"]
        assert rate["lower"] <= 0.0 <= rate["upper"]
        assert cert["closes_to_finite_time_blowup"] is False
        assert cert["critical_advection_parameter"] == 1.0
        assert certified_gclm_gradient_amplification_schema_errors(cert) == []


def test_certified_gclm_gradient_amplification_reduces_to_clm_at_a0() -> None:
    """At a=0 the rate is H omega0(0) and the cross-reference time matches CLM."""
    coeffs = [-1.0, 0.4]
    scales = [1.0, 2.0]
    cert = certified_gclm_gradient_amplification(a=0.0, coeffs=coeffs, scales=scales)
    assert cert["closes_to_finite_time_blowup"] is True
    assert cert["instantaneous_rate_only"] is False
    assert cert["honesty"]["global_blowup_proof"] is True
    # rate at a=0 equals H omega0(0) = -sum c/a = 0.8
    hw0_0 = -sum(c / a for c, a in zip(coeffs, scales, strict=True))
    rate = cert["amplification_rate"]
    assert rate["lower"] <= hw0_0 <= rate["upper"]
    # cross-reference blow-up time matches the standalone CLM certificate (T = 2/Hw0)
    clm = certified_clm_blowup(coeffs=coeffs, scales=scales)
    xref = cert["blowup_cross_reference"]
    assert xref is not None
    assert xref["blowup_time"]["lower"] <= 2.0 / hw0_0 <= xref["blowup_time"]["upper"]
    assert (
        abs(xref["blowup_time"]["midpoint"] - clm["blowup_time"]["midpoint"]) < 1e-9
    )


def test_certified_gclm_gradient_amplification_regimes() -> None:
    """a<1 amplifies, a>1 damps when H omega0(0) > 0; a=1 is the threshold."""
    coeffs = [-1.0]  # H omega0(0) = +1 > 0
    scales = [1.0]
    amp = certified_gclm_gradient_amplification(a=0.25, coeffs=coeffs, scales=scales)
    assert amp["regime"] == "instantaneous_amplification"
    assert amp["instantaneous_amplification_certified"] is True
    assert amp["amplification_rate"]["lower"] > 0.0

    damp = certified_gclm_gradient_amplification(a=2.0, coeffs=coeffs, scales=scales)
    assert damp["regime"] == "instantaneous_damping"
    assert damp["instantaneous_damping_certified"] is True
    assert damp["amplification_rate"]["upper"] < 0.0

    neutral = certified_gclm_gradient_amplification(a=1.0, coeffs=coeffs, scales=scales)
    assert neutral["regime"] == "neutral"


def test_certified_gclm_gradient_amplification_schema_validation() -> None:
    cert = certified_gclm_gradient_amplification(a=0.5, coeffs=[-1.0], scales=[1.0])
    assert cert["schema_version"] == GCLM_GRADIENT_AMP_SCHEMA_VERSION
    assert certified_gclm_gradient_amplification_schema_errors(cert) == []

    bad_claim = json.loads(json.dumps(cert))
    bad_claim["honesty"]["unproven_claim"] = True
    assert any(
        "unproven_claim" in e
        for e in certified_gclm_gradient_amplification_schema_errors(bad_claim)
    )

    bad_three_d = json.loads(json.dumps(cert))
    bad_three_d["three_d_claim"] = True
    assert any(
        "three_d_claim" in e
        for e in certified_gclm_gradient_amplification_schema_errors(bad_three_d)
    )

    bad_version = json.loads(json.dumps(cert))
    bad_version["schema_version"] = "wrong"
    assert any(
        "schema_version" in e
        for e in certified_gclm_gradient_amplification_schema_errors(bad_version)
    )

    # a global blow-up claim is only honest at a == 0
    bad_closure = json.loads(json.dumps(cert))
    bad_closure["closes_to_finite_time_blowup"] = True  # but a = 0.5
    assert any(
        "closes_to_finite_time_blowup" in e
        for e in certified_gclm_gradient_amplification_schema_errors(bad_closure)
    )

    # an amplification claim with a non-positive certified rate must be rejected
    bad_amp = json.loads(json.dumps(cert))
    bad_amp["instantaneous_amplification_certified"] = True
    bad_amp["amplification_rate"]["lower"] = -1.0
    assert any(
        "amplification_rate.lower" in e
        for e in certified_gclm_gradient_amplification_schema_errors(bad_amp)
    )


def test_certified_gclm_gradient_amplification_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        certified_gclm_gradient_amplification(a=0.5, coeffs=[], scales=[])
    with pytest.raises(ValueError):
        certified_gclm_gradient_amplification(a=0.5, coeffs=[1.0], scales=[1.0, 2.0])
    with pytest.raises(ValueError):
        certified_gclm_gradient_amplification(a=0.5, coeffs=[1.0], scales=[-1.0])
    with pytest.raises(ValueError):
        certified_gclm_gradient_amplification(a=0.5, coeffs=[1.0], scales=[0.0])


def test_certified_gclm_gradient_amplification_json_and_provenance_are_deterministic() -> None:
    cert_a = certified_gclm_gradient_amplification(a=0.5, coeffs=[-1.0, 0.25], scales=[1.0, 3.0])
    cert_b = certified_gclm_gradient_amplification(a=0.5, coeffs=[-1.0, 0.25], scales=[1.0, 3.0])
    assert json.loads(json.dumps(cert_a)) == cert_a
    assert cert_a["provenance"]["sha256"] == cert_b["provenance"]["sha256"]
    assert len(cert_a["provenance"]["sha256"]) == 64
    assert cert_a["provenance"]["harness"].endswith("certified_gclm_gradient_amplification")
