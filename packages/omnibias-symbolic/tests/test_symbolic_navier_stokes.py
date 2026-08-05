# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent numpy validation for Navier-Stokes certified-evidence bundles."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
from omnibias.pinn.certified import (
    active_projector_error_certificate,
    active_subspace_absorption_frontier_report,
    active_subspace_completeness_theorem_attempt,
    active_subspace_invariance_report,
    active_subspace_tail_contraction_attempt,
    active_tail_contraction_lift_certificate,
    analytic_tail_error_certificate,
    build_active_tail_lift_error_budget,
    build_axisymmetric_active_subspace_closure_report,
    build_axisymmetric_blowup_closure_report,
    build_axisymmetric_interval_report,
    build_axisymmetric_swirl_candidate_artifact,
    build_ns_cap_bundle,
    build_ns_proof_program_report,
    build_ns_solve_or_falsify_report,
    build_ns_theorem_ladder_report,
    build_refined_axisymmetric_swirl_candidate_artifact,
    build_regularity_inequality_report,
    build_theorem_grade_closure_attempt,
    candidate_artifact_schema_errors,
    compactified_coefficient_set,
    compactified_sandbox_replay_grid,
    deterministic_periodic_replay_grid,
    finite_active_tail_contraction_diagnostic,
    interval_jacobian_error_certificate,
    manufactured_abc_flow,
    nonlinear_tail_remainder_certificate,
    theorem_verifier_record,
    weighted_analytic_tail_norm_contract,
)
from omnibias.symbolic.navier_stokes import (
    assess_blowup_candidate,
    assess_navier_stokes_candidate,
    build_axisymmetric_candidate_bridge_artifacts,
    build_blowup_candidate_artifact,
    build_regularity_candidate_artifact,
    fit_regularity_growth_bound,
    fit_self_similar_blowup_rate,
    pressure_poisson_residual_periodic,
    primitive_residual_periodic,
    regularity_feature_vector,
    replay_candidate_artifact,
    run_regularity_search,
    verify_active_projector_error_certificate,
    verify_active_subspace_absorption_frontier_report,
    verify_active_subspace_closure_report,
    verify_active_subspace_completeness_theorem_attempt,
    verify_active_subspace_invariance_report,
    verify_active_subspace_tail_contraction_attempt,
    verify_active_tail_contraction_lift_certificate,
    verify_active_tail_lift_error_budget,
    verify_analytic_tail_error_certificate,
    verify_axisymmetric_candidate_bridge_artifacts,
    verify_axisymmetric_interval_report,
    verify_axisymmetric_swirl_candidate_artifact,
    verify_blowup_closure_report,
    verify_finite_active_tail_contraction_diagnostic,
    verify_interval_jacobian_error_certificate,
    verify_nonlinear_tail_remainder_certificate,
    verify_ns_cap_bundle,
    verify_ns_proof_program_report,
    verify_ns_solve_or_falsify_report,
    verify_ns_theorem_ladder_report,
    verify_proof_obligation_bundle,
    verify_refined_axisymmetric_swirl_candidate_artifact,
    verify_regularity_inequality_report,
    verify_theorem_grade_closure_attempt,
    verify_theorem_verifier_bundle,
    verify_weighted_analytic_tail_norm_contract,
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


def test_independent_primitive_residual_on_taylor_green() -> None:
    viscosity = 0.1
    velocity, pressure, velocity_t = _taylor_green(64, viscosity)
    residual, continuity = primitive_residual_periodic(
        velocity, pressure, velocity_t=velocity_t, viscosity=viscosity
    )
    pressure_res = pressure_poisson_residual_periodic(velocity, pressure)
    assert float(np.max(np.abs(residual))) < 1e-10
    assert float(np.max(np.abs(continuity))) < 1e-10
    assert float(np.max(np.abs(pressure_res))) < 1e-10


def test_verify_ns_cap_bundle_recomputes_pinn_bundle_independently() -> None:
    viscosity = 0.1
    velocity, pressure, velocity_t = _taylor_green(32, viscosity)
    bundle = build_ns_cap_bundle(
        velocity, pressure, velocity_t=velocity_t, viscosity=viscosity
    )
    report = verify_ns_cap_bundle(bundle)
    assert report["residual_samples_match"], report
    assert report["agreement_momentum_max_abs_diff"] < 1e-10
    assert report["unproven_claim"] is False


def test_verify_abc_3d_cap_bundle_independently() -> None:
    mms = manufactured_abc_flow(18, viscosity=0.04)
    bundle = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        density=mms["density"],
        lengths=mms["lengths"],
    )
    report = verify_ns_cap_bundle(bundle)
    assert report["residual_samples_match"], report
    assert report["momentum_max_abs"] < 1e-10
    assert report["continuity_max_abs"] < 1e-10


def test_assess_candidate_keeps_honesty_flags_false() -> None:
    velocity, pressure, velocity_t = _taylor_green(32, 0.1)
    bundle = build_ns_cap_bundle(velocity, pressure, velocity_t=velocity_t, viscosity=0.1)
    out = assess_navier_stokes_candidate(bundle)
    assert out["honesty"]["unproven_claim"] is False
    assert out["honesty"]["exact_solution_claim"] is False
    assert out["verification"]["residual_samples_match"]
    assert out["regularity_features"]["max_abs_divergence"] < 1e-10


def test_regularity_feature_vector_contains_expected_quantities() -> None:
    velocity, pressure, _ = _taylor_green(32, 0.1)
    features = regularity_feature_vector(velocity, pressure=pressure)
    assert features["kinetic_energy"] > 0.0
    assert features["enstrophy"] > 0.0
    assert features["palinstrophy"] > 0.0
    assert features["pressure_poisson_max_abs"] < 1e-10
    assert features["unproven_claim"] is False


def test_fit_regularity_growth_bound_recovers_simple_growth_law() -> None:
    time = np.linspace(0.0, 1.0, 200)
    quantity = np.exp(time)
    out = fit_regularity_growth_bound(
        time,
        quantity,
        {"Q": quantity, "one": np.ones_like(quantity)},
        alpha=1e-12,
        threshold=1e-8,
    )
    assert abs(out["coefficients"]["Q"] - 1.0) < 1e-3
    assert out["fit_rmse"] < 1e-3
    assert out["global_regularity_claim"] is False


def test_run_regularity_search_wraps_feature_library() -> None:
    time = np.linspace(0.0, 1.0, 200)
    enstrophy = np.exp(time)
    traces = {
        "energy": 0.5 * enstrophy,
        "enstrophy": enstrophy,
        "palinstrophy": 2.0 * enstrophy,
        "bkm_vorticity_proxy": np.sqrt(enstrophy),
    }
    out = run_regularity_search(
        time, traces, target="enstrophy", include_quadratic=False,
        alpha=1e-12, threshold=1e-8,
    )
    assert out["candidate_type"] == "regularity_growth_law"
    assert out["target"] == "enstrophy"
    assert out["global_regularity_claim"] is False
    assert out["fit_rmse"] < 1e-3


def test_regularity_candidate_artifact_roundtrips_and_replays_independently() -> None:
    time = np.linspace(0.0, 1.0, 200)
    enstrophy = np.exp(time)
    traces = {
        "energy": 0.5 * enstrophy,
        "enstrophy": enstrophy,
        "palinstrophy": 2.0 * enstrophy,
        "bkm_vorticity_proxy": np.sqrt(enstrophy),
    }
    grid = asdict(deterministic_periodic_replay_grid(dimension=3, n=4))
    artifact = build_regularity_candidate_artifact(
        time,
        traces,
        target="enstrophy",
        include_quadratic=False,
        alpha=1e-12,
        threshold=1e-8,
        replay_grid=grid,
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["honesty"]["unproven_claim"] is False

    reloaded = json.loads(json.dumps(artifact))
    report = replay_candidate_artifact(reloaded)
    assert report["replay_match"], report
    assert report["unproven_claim"] is False


def test_fit_self_similar_blowup_rate_recovers_power_law() -> None:
    time = np.linspace(0.0, 0.9, 100)
    norm = 3.0 * (1.0 - time) ** -1.25
    out = fit_self_similar_blowup_rate(time, norm, blowup_time=1.0)
    assert abs(float(out["alpha"]) - 1.25) < 1e-12
    assert float(out["log_fit_rmse"]) < 1e-12
    assert out["finite_time_blowup_claim"] is False
    assert out["unproven_claim"] is False


def test_assess_blowup_candidate_is_strict_nonclaim() -> None:
    time = np.linspace(0.0, 0.9, 80)
    norm = 2.0 * (1.0 - time) ** -0.75
    out = assess_blowup_candidate(
        time,
        norm,
        blowup_time=1.0,
        ansatz_metadata={"class": "axisymmetric_swirl_sandbox"},
        residual_metrics={"max_abs_residual": 1e-6},
    )
    assert abs(out["rate_fit"]["alpha"] - 0.75) < 1e-12
    assert out["honesty"]["finite_time_blowup_claim"] is False
    assert out["honesty"]["unproven_claim"] is False
    assert "finite_energy_initial_data" in out["proof_obligations"]


def test_blowup_candidate_artifact_roundtrips_and_replays_independently() -> None:
    time = np.linspace(0.0, 0.9, 80)
    norm = 2.0 * (1.0 - time) ** -0.75
    grid = asdict(compactified_sandbox_replay_grid(n_radial=4, n_theta=4, n_phi=8))
    coeffs = asdict(
        compactified_coefficient_set(
            "u",
            np.ones((3, 4), dtype=float),
            tail_l1_bound=1e-10,
            finite_energy_estimate=3.0,
        )
    )
    artifact = build_blowup_candidate_artifact(
        time,
        norm,
        blowup_time=1.0,
        ansatz_metadata={"class": "axisymmetric_swirl_sandbox"},
        residual_metrics={"max_abs_residual": 1e-6},
        replay_grid=grid,
        coefficients=[coeffs],
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["honesty"]["finite_time_blowup_claim"] is False

    reloaded = json.loads(json.dumps(artifact))
    report = replay_candidate_artifact(reloaded)
    assert report["replay_match"], report
    assert report["alpha_abs_diff"] < 1e-12


def test_axisymmetric_candidate_artifact_roundtrips_and_replays_independently() -> None:
    artifact = build_axisymmetric_swirl_candidate_artifact(
        seed=11,
        n_radial=6,
        n_axial=7,
        viscosity=0.01,
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["candidate_type"] == "axisymmetric_swirl_sandbox"

    reloaded = json.loads(json.dumps(artifact))
    direct = verify_axisymmetric_swirl_candidate_artifact(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert direct["replay_match"], direct
    assert via_dispatch["replay_match"], via_dispatch
    assert direct["unproven_claim"] is False
    assert via_dispatch["stage"] == "candidate_replay_ready"


def test_axisymmetric_candidate_bridge_emits_replayable_companion_artifacts() -> None:
    axisym = build_axisymmetric_swirl_candidate_artifact(
        seed=13,
        n_radial=6,
        n_axial=7,
        viscosity=0.01,
    )
    bridge = build_axisymmetric_candidate_bridge_artifacts(axisym)
    assert bridge["honesty"]["unproven_claim"] is False
    assert bridge["regularity"]["candidate_type"] == "regularity_growth_law"
    assert bridge["blowup"]["candidate_type"] == "self_similar_blowup_rate"

    regularity_report = replay_candidate_artifact(json.loads(json.dumps(bridge["regularity"])))
    blowup_report = replay_candidate_artifact(json.loads(json.dumps(bridge["blowup"])))
    bridge_report = verify_axisymmetric_candidate_bridge_artifacts(json.loads(json.dumps(bridge)))
    bridge_dispatch = replay_candidate_artifact(json.loads(json.dumps(bridge)))
    assert regularity_report["replay_match"], regularity_report
    assert blowup_report["replay_match"], blowup_report
    assert bridge_report["replay_match"], bridge_report
    assert bridge_dispatch["replay_match"], bridge_dispatch


def test_refined_axisymmetric_candidate_roundtrips_and_replays_independently() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=17,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=3,
        step_size=0.01,
        viscosity=0.01,
    )
    assert candidate_artifact_schema_errors(artifact) == []
    assert artifact["result"]["final_loss"] <= artifact["result"]["initial_loss"]

    reloaded = json.loads(json.dumps(artifact))
    direct = verify_refined_axisymmetric_swirl_candidate_artifact(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert direct["replay_match"], direct
    assert via_dispatch["replay_match"], via_dispatch
    assert direct["unproven_claim"] is False
    assert via_dispatch["candidate_type"] == "axisymmetric_swirl_refined"

    bridge = build_axisymmetric_candidate_bridge_artifacts(reloaded)
    assert replay_candidate_artifact(json.loads(json.dumps(bridge["regularity"])))["replay_match"]
    assert replay_candidate_artifact(json.loads(json.dumps(bridge["blowup"])))["replay_match"]


def test_axisymmetric_interval_report_verifies_midpoint_containment() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=23,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    report = build_axisymmetric_interval_report(artifact)
    reloaded = json.loads(json.dumps(report))
    verified = verify_axisymmetric_interval_report(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert verified["interval_report_match"], verified
    assert via_dispatch["interval_report_match"], via_dispatch
    assert verified["max_interval_violation"] == 0.0
    assert verified["tail_certified"] is True
    assert verified["axis_certified"] is True
    assert verified["continuum_certified"] is True
    assert verified["unproven_claim"] is False
    assert verified["stage"] == "interval_obligation_ready"


def test_blowup_closure_report_replays_and_detects_tampering() -> None:
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
    interval = build_axisymmetric_interval_report(artifact)
    closure = build_axisymmetric_blowup_closure_report(interval, norm_growth_exponent=0.25)
    reloaded = json.loads(json.dumps(closure))
    verified = verify_blowup_closure_report(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert verified["closure_report_match"], verified
    assert via_dispatch["closure_report_match"], via_dispatch
    assert verified["interval_replay"]["interval_report_match"] is True
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(closure))
    tampered["radii_polynomial"]["closure_interval"]["upper"] += 1.0
    failed = verify_blowup_closure_report(tampered)
    assert failed["closure_report_match"] is False
    assert failed["max_closure_violation"] > 0.5


def test_regularity_inequality_report_replays_and_detects_tampering() -> None:
    time = np.linspace(0.0, 1.0, 80)
    enstrophy = np.exp(time)
    artifact = build_regularity_candidate_artifact(
        time,
        {
            "energy": 0.5 * enstrophy,
            "enstrophy": enstrophy,
            "palinstrophy": 2.0 * enstrophy,
            "bkm_vorticity_proxy": np.sqrt(enstrophy),
        },
        target="enstrophy",
        include_quadratic=False,
        alpha=1e-12,
        threshold=1e-8,
    )
    closure = build_regularity_inequality_report(artifact, residual_tolerance=0.1)
    reloaded = json.loads(json.dumps(closure))
    verified = verify_regularity_inequality_report(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert verified["regularity_report_match"], verified
    assert via_dispatch["regularity_report_match"], via_dispatch
    assert verified["counterexample_sweep"]["passed"] == closure["counterexample_sweep"]["passed"]
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(closure))
    tampered["trace_residual"]["max_abs_residual"] += 1.0
    failed = verify_regularity_inequality_report(tampered)
    assert failed["regularity_report_match"] is False
    assert failed["max_regularity_violation"] > 0.5


def test_theorem_grade_closure_attempt_replays_and_detects_tampering() -> None:
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
    interval = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    bridge = build_axisymmetric_candidate_bridge_artifacts(artifact)
    regularity = build_regularity_inequality_report(bridge["regularity"], residual_tolerance=0.1)
    theorem = build_theorem_grade_closure_attempt(
        interval,
        blowup_report=blowup,
        regularity_report=regularity,
    )
    reloaded = json.loads(json.dumps(theorem))
    verified = verify_theorem_grade_closure_attempt(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert verified["theorem_grade_report_match"], verified
    assert via_dispatch["theorem_grade_report_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(theorem))
    tampered["route_attempts"]["operator_invertibility"]["operator_theoretic_certified"] = True
    failed = verify_theorem_grade_closure_attempt(tampered)
    assert failed["theorem_grade_report_match"] is False
    assert failed["max_theorem_grade_violation"] == float("inf")


def test_ns_proof_program_report_replays_and_detects_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=37,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    bridge = build_axisymmetric_candidate_bridge_artifacts(artifact)
    regularity = build_regularity_inequality_report(bridge["regularity"], residual_tolerance=0.1)
    theorem = build_theorem_grade_closure_attempt(
        interval,
        blowup_report=blowup,
        regularity_report=regularity,
    )
    report = build_ns_proof_program_report(
        theorem_attempt=theorem,
    )
    reloaded = json.loads(json.dumps(report))
    verified = verify_ns_proof_program_report(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert verified["proof_program_report_match"], verified
    assert via_dispatch["proof_program_report_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(report))
    tampered["open_lemmas"].append("tampered_extra_lemma")
    failed = verify_ns_proof_program_report(tampered)
    assert failed["proof_program_report_match"] is False
    assert failed["max_proof_program_violation"] == float("inf")


def test_active_subspace_closure_report_replays_and_detects_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=40,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    report = build_axisymmetric_active_subspace_closure_report(
        interval,
        active_indices=(0, 1, 2),
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    verified = verify_active_subspace_closure_report(json.loads(json.dumps(report)))
    via_dispatch = replay_candidate_artifact(json.loads(json.dumps(report)))
    assert verified["active_subspace_closure_match"], verified
    assert via_dispatch["active_subspace_closure_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(report))
    tampered["closure_certificates"]["linearized_operator"]["smallest_singular_value"] += 1.0
    failed = verify_active_subspace_closure_report(tampered)
    assert failed["active_subspace_closure_match"] is False
    assert failed["max_active_subspace_violation"] > 0.5


def test_active_subspace_invariance_report_replays_and_detects_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=42,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    report = active_subspace_invariance_report(artifact, active_indices=(0, 1, 2))
    verified = verify_active_subspace_invariance_report(json.loads(json.dumps(report)))
    via_dispatch = replay_candidate_artifact(json.loads(json.dumps(report)))
    assert verified["active_subspace_invariance_match"], verified
    assert via_dispatch["active_subspace_invariance_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(report))
    tampered["post_newton_leakage_ratio"] += 1.0
    failed = verify_active_subspace_invariance_report(tampered)
    assert failed["active_subspace_invariance_match"] is False
    assert failed["max_active_subspace_invariance_violation"] > 0.5


def test_active_subspace_absorption_frontier_replays_and_detects_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=44,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    report = active_subspace_absorption_frontier_report(
        interval,
        active_indices=(0, 1, 2),
        max_combination_order=1,
    )
    verified = verify_active_subspace_absorption_frontier_report(json.loads(json.dumps(report)))
    via_dispatch = replay_candidate_artifact(json.loads(json.dumps(report)))
    assert verified["active_subspace_absorption_frontier_match"], verified
    assert via_dispatch["active_subspace_absorption_frontier_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(report))
    tampered["required_tail_control_modes"] = [999]
    failed = verify_active_subspace_absorption_frontier_report(tampered)
    assert failed["active_subspace_absorption_frontier_match"] is False
    assert failed["max_active_subspace_absorption_violation"] == float("inf")


def test_active_tail_theorem_artifacts_replay_and_detect_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=45,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    frontier = active_subspace_absorption_frontier_report(
        interval,
        active_indices=(0, 1, 2),
        max_combination_order=1,
    )
    finite_q = finite_active_tail_contraction_diagnostic(
        artifact,
        active_indices=(0, 1, 2),
        tail_modes=frontier["required_tail_control_modes"],
    )
    contract = weighted_analytic_tail_norm_contract(frontier)
    lift = active_tail_contraction_lift_certificate(finite_q, contract)
    contraction = active_subspace_tail_contraction_attempt(
        frontier,
        contract,
        finite_diagnostic=finite_q,
        analytic_lift=lift,
    )
    completeness = active_subspace_completeness_theorem_attempt(frontier, contraction)

    finite_replay = verify_finite_active_tail_contraction_diagnostic(json.loads(json.dumps(finite_q)))
    lift_replay = verify_active_tail_contraction_lift_certificate(json.loads(json.dumps(lift)))
    contract_replay = verify_weighted_analytic_tail_norm_contract(json.loads(json.dumps(contract)))
    contraction_replay = verify_active_subspace_tail_contraction_attempt(json.loads(json.dumps(contraction)))
    completeness_replay = verify_active_subspace_completeness_theorem_attempt(json.loads(json.dumps(completeness)))
    via_dispatch = replay_candidate_artifact(json.loads(json.dumps(finite_q)))
    assert finite_replay["finite_active_tail_contraction_match"], finite_replay
    assert via_dispatch["finite_active_tail_contraction_match"], via_dispatch
    assert finite_q["finite_tail_contraction_surrogate_passed"] is (
        finite_q["finite_contraction_ratio_upper"] < 1.0
    )
    assert lift_replay["active_tail_contraction_lift_match"], lift_replay
    assert lift["analytic_lift_certified"] is False
    assert contract_replay["weighted_analytic_tail_norm_match"], contract_replay
    assert contraction_replay["active_subspace_tail_contraction_match"], contraction_replay
    assert completeness_replay["active_subspace_completeness_match"], completeness_replay
    assert completeness["active_subspace_complete"] is False

    tampered = json.loads(json.dumps(contract))
    tampered["required_tail_modes"] = [999]
    failed = verify_weighted_analytic_tail_norm_contract(tampered)
    assert failed["weighted_analytic_tail_norm_match"] is False
    assert failed["max_weighted_analytic_tail_norm_violation"] == float("inf")

    tampered_q = json.loads(json.dumps(finite_q))
    tampered_q["finite_contraction_ratio_upper"] += 1.0
    failed_q = verify_finite_active_tail_contraction_diagnostic(tampered_q)
    assert failed_q["finite_active_tail_contraction_match"] is False
    assert failed_q["max_finite_active_tail_contraction_violation"] > 0.5

    tampered_lift = json.loads(json.dumps(lift))
    tampered_lift["missing_error_budget_terms"] = []
    failed_lift = verify_active_tail_contraction_lift_certificate(tampered_lift)
    assert failed_lift["active_tail_contraction_lift_match"] is False
    assert failed_lift["max_active_tail_contraction_lift_violation"] == float("inf")


def test_active_tail_lift_error_terms_replay_and_detect_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=45,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    frontier = active_subspace_absorption_frontier_report(
        interval,
        active_indices=(0, 1, 2),
        max_combination_order=1,
    )
    finite_q = finite_active_tail_contraction_diagnostic(
        artifact,
        active_indices=(0, 1, 2),
        tail_modes=frontier["required_tail_control_modes"],
    )
    contract = weighted_analytic_tail_norm_contract(frontier)
    projector = active_projector_error_certificate(finite_q)
    interval_jac = interval_jacobian_error_certificate(finite_q)
    nonlinear = nonlinear_tail_remainder_certificate(finite_q)
    analytic = analytic_tail_error_certificate(contract)
    budget = build_active_tail_lift_error_budget(
        finite_q,
        contract,
        projector_certificate=projector,
        interval_jacobian_certificate=interval_jac,
        nonlinear_remainder_certificate=nonlinear,
        analytic_tail_certificate=analytic,
    )

    projector_replay = verify_active_projector_error_certificate(json.loads(json.dumps(projector)))
    interval_replay = verify_interval_jacobian_error_certificate(json.loads(json.dumps(interval_jac)))
    nonlinear_replay = verify_nonlinear_tail_remainder_certificate(json.loads(json.dumps(nonlinear)))
    analytic_replay = verify_analytic_tail_error_certificate(json.loads(json.dumps(analytic)))
    budget_replay = verify_active_tail_lift_error_budget(json.loads(json.dumps(budget)))
    via_dispatch = replay_candidate_artifact(json.loads(json.dumps(budget)))

    assert projector_replay["active_projector_error_match"], projector_replay
    assert interval_replay["interval_jacobian_error_match"], interval_replay
    assert nonlinear_replay["nonlinear_tail_remainder_match"], nonlinear_replay
    assert analytic_replay["analytic_tail_error_match"], analytic_replay
    assert budget_replay["active_tail_lift_error_budget_match"], budget_replay
    assert budget_replay["sub_certificate_replays_match"] is True
    assert via_dispatch["active_tail_lift_error_budget_match"], via_dispatch
    assert budget["unproven_claim"] is False
    assert budget["analytic_lift_certified"] is False

    # Rigorous operator-norm Hessian path: the independent replay must rebuild the
    # constant residual Hessian and match the certified Jacobian-Lipschitz bound.
    nonlinear_rig = nonlinear_tail_remainder_certificate(
        finite_q, certify_hessian_operator_norm=True
    )
    assert nonlinear_rig["hessian_bound_rigorous"] is True
    assert (
        "certify_second_derivative_bound_over_solution_ball"
        not in nonlinear_rig["open_obligations"]
    )
    nonlinear_rig_replay = verify_nonlinear_tail_remainder_certificate(
        json.loads(json.dumps(nonlinear_rig))
    )
    assert nonlinear_rig_replay["nonlinear_tail_remainder_match"], nonlinear_rig_replay
    # Tamper the rigorous bound: the recompute-from-artifact replay must catch it.
    tampered_rig = json.loads(json.dumps(nonlinear_rig))
    tampered_rig["hessian_operator_norm_proxy"] = float(tampered_rig["hessian_operator_norm_proxy"]) * 0.5
    failed_rig = verify_nonlinear_tail_remainder_certificate(tampered_rig)
    assert failed_rig["nonlinear_tail_remainder_match"] is False

    # Tamper the projector bound: the recompute-from-artifact replay must catch it.
    tampered_proj = json.loads(json.dumps(projector))
    tampered_proj["projector_error_upper"] = 0.5
    failed_proj = verify_active_projector_error_certificate(tampered_proj)
    assert failed_proj["active_projector_error_match"] is False
    assert failed_proj["max_active_projector_error_violation"] > 0.0

    # Tamper the analytic decay ratio.
    tampered_analytic = json.loads(json.dumps(analytic))
    tampered_analytic["weighted_ratio_rho_gamma"] += 1.0
    failed_analytic = verify_analytic_tail_error_certificate(tampered_analytic)
    assert failed_analytic["analytic_tail_error_match"] is False

    # Tamper a sub-certificate inside the assembled budget: the budget replay
    # re-verifies each sub-certificate from its own inputs and must flag it.
    tampered_budget = json.loads(json.dumps(budget))
    tampered_budget["replay_inputs"]["projector_certificate"]["projector_error_upper"] = 0.9
    failed_budget = verify_active_tail_lift_error_budget(tampered_budget)
    assert failed_budget["sub_certificate_replays_match"] is False
    assert failed_budget["active_tail_lift_error_budget_match"] is False


def test_ns_theorem_ladder_report_replays_and_detects_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=46,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    frontier = active_subspace_absorption_frontier_report(
        interval,
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
    blowup = build_axisymmetric_blowup_closure_report(
        interval,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    theorem = build_theorem_grade_closure_attempt(interval, blowup_report=blowup)
    report = build_ns_theorem_ladder_report(
        interval,
        frontier,
        blowup_report=blowup,
        theorem_attempt=theorem,
        finite_tail_diagnostic=finite_q,
        analytic_tail_lift=lift,
    )
    verified = verify_ns_theorem_ladder_report(json.loads(json.dumps(report)))
    via_dispatch = replay_candidate_artifact(json.loads(json.dumps(report)))
    assert verified["theorem_ladder_report_match"], verified
    assert via_dispatch["theorem_ladder_report_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(report))
    tampered["phases"]["tail_norm_contract"]["required_tail_modes"] = [999]
    failed = verify_ns_theorem_ladder_report(tampered)
    assert failed["theorem_ladder_report_match"] is False
    assert failed["phase_hashes_match"] is False


def test_solve_or_falsify_report_replays_and_detects_tampering() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=39,
        n_radial=5,
        n_axial=6,
        radial_degree=0,
        axial_degree=1,
        max_iterations=1,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    theorem = build_theorem_grade_closure_attempt(interval, blowup_report=blowup)
    report = build_ns_solve_or_falsify_report(
        interval,
        blowup_report=blowup,
        theorem_attempt=theorem,
        candidate_family_status=[{"family": "unit", "status": "blocked_with_named_missing_lemma"}],
    )
    reloaded = json.loads(json.dumps(report))
    verified = verify_ns_solve_or_falsify_report(reloaded)
    via_dispatch = replay_candidate_artifact(reloaded)
    assert verified["solve_or_falsify_report_match"], verified
    assert via_dispatch["solve_or_falsify_report_match"], via_dispatch
    assert verified["unproven_claim"] is False

    tampered = json.loads(json.dumps(report))
    tampered["phases"]["baseline"]["metrics"]["train_loss"] += 1.0
    failed = verify_ns_solve_or_falsify_report(tampered)
    assert failed["solve_or_falsify_report_match"] is False
    assert failed["phase_hashes_match"] is False


def test_proof_obligation_and_verifier_bundles_replay_symbolically() -> None:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=38,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    interval = build_axisymmetric_interval_report(artifact)
    blowup = build_axisymmetric_blowup_closure_report(
        interval,
        norm_growth_exponent=0.25,
        linked_norm_profile=True,
    )
    bridge = build_axisymmetric_candidate_bridge_artifacts(artifact)
    regularity = build_regularity_inequality_report(bridge["regularity"], residual_tolerance=0.1)
    theorem = build_theorem_grade_closure_attempt(
        interval,
        blowup_report=blowup,
        regularity_report=regularity,
    )
    report = build_ns_proof_program_report(
        theorem_attempt=theorem,
    )
    obligations = [
        dict(bundle)
        for bundle in report["lemma_packages"]["finite_time_blowup"]["proof_obligation_bundles"]
    ]
    replay = verify_proof_obligation_bundle(obligations[0])
    assert replay["obligation_bundle_match"], replay

    tampered = json.loads(json.dumps(obligations[0]))
    tampered["theorem_statement"] = "Tampered theorem statement."
    failed_obligation = verify_proof_obligation_bundle(tampered)
    assert failed_obligation["obligation_bundle_match"] is False

    verifier = theorem_verifier_record(
        obligations,
        discharged_obligations=[obligations[0]["obligation_id"]],
        reviewed_at_utc="2026-06-23T00:00:00Z",
    )
    verified = verify_theorem_verifier_bundle(obligations, verifier)
    assert verified["accepted_obligations"] == [obligations[0]["obligation_id"]]
    assert len(verified["rejected_obligations"]) == len(obligations) - 1

    mismatch = json.loads(json.dumps(verifier))
    mismatch["proof_records"][0]["source_artifact_sha256"] = "0" * 64
    failed_verifier = verify_theorem_verifier_bundle(obligations, mismatch)
    assert failed_verifier["accepted_obligations"] == []
    assert any("source_hash" in reason for reason in failed_verifier["rejected_obligations"].values())


def test_candidate_replay_detects_tampered_result() -> None:
    time = np.linspace(0.0, 1.0, 80)
    enstrophy = np.exp(time)
    artifact = build_regularity_candidate_artifact(
        time,
        {
            "energy": 0.5 * enstrophy,
            "enstrophy": enstrophy,
            "palinstrophy": 2.0 * enstrophy,
            "bkm_vorticity_proxy": np.sqrt(enstrophy),
        },
        target="enstrophy",
        include_quadratic=False,
    )
    tampered = json.loads(json.dumps(artifact))
    tampered["result"]["fit_rmse"] = float(tampered["result"]["fit_rmse"]) + 1.0
    report = replay_candidate_artifact(tampered)
    assert report["replay_match"] is False
    assert report["fit_rmse_abs_diff"] > 0.5
