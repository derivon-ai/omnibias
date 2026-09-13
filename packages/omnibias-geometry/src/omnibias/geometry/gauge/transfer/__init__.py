# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite lattice spectra, charged sectors and static-source certificates.

Builds a **fixed**, finite-dimensional transfer matrix with rigorously enclosed
entries (:mod:`.matrices`), certifies a lower bound on its lattice-unit mass gap
``m a = -ln(|lambda_1| / lambda_0)`` with the rigorous engines of
:mod:`omnibias.core.verified.eig` (:mod:`.gap`), and seals the result into a
tamper-evident certificate whose rational obligation the Mathlib-free Lean kernel
can discharge (:mod:`.certificates`).

The ingredients were already here and simply never connected: the gap engines name
these very matrices in their docstrings, ``quadratic_casimir`` returns an *exact*
``Fraction``, ``besseli_iv`` encloses the Wilson character coefficients, and the
Lean kernel's ``spectral_gap_pos`` lemma already discharges a
``subdominant_ratio_upper`` obligation.

Scope
-----
Transfer and truncated-Hamiltonian certificates concern **fixed finite
matrices**. The :mod:`.static_sources` graph bounds instead include all link
spins through their written parity and groundstate-transform proofs, and
:mod:`.charged_sectors` counts exact finite representation dimensions.
These are distinct scopes. ``continuum_claim`` is always ``False``,
:func:`heat_kernel_gap_scaling_report` is labelled evidence rather than proof, and
nothing here is a claim about the Yang-Mills mass gap.

The :mod:`.commutator_matrix` and :mod:`.compact_commutator` bounds include
the full Hilbert spaces of specified finite configuration models, with
their singlet and center-sector restrictions explicit. The
:mod:`.physical_blocks` actual-vacuum pilot and generic conditional overlap
budgets do not establish weak-coupling volume or continuum uniformity.
"""

from __future__ import annotations

from omnibias.geometry.gauge.transfer.adjacent_cone_inverse import (
    replay_su2_adjacent_cone_inverse_certificate,
    su2_adjacent_cone_inverse,
)
from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import (
    replay_su2_adjacent_cone_vacuum_certificate,
    su2_adjacent_cone_vacuum,
)
from omnibias.geometry.gauge.transfer.adjacent_gap_comparison import (
    replay_su2_adjacent_vacuum_gap_comparison_certificate,
    su2_adjacent_vacuum_gap_comparison,
)
from omnibias.geometry.gauge.transfer.adjacent_resolvent import (
    replay_su2_adjacent_linearized_inverse_certificate,
    su2_adjacent_linearized_inverse,
)
from omnibias.geometry.gauge.transfer.adjacent_vacuum import (
    replay_su2_adjacent_preconditioned_vacuum_certificate,
    su2_adjacent_preconditioned_vacuum,
)
from omnibias.geometry.gauge.transfer.ambient_conditional import (
    replay_su2_ambient_conditional_certificate,
    su2_ambient_conditional_gap,
)
from omnibias.geometry.gauge.transfer.certificates import (
    FINITE_GAUGE_REPORT_KIND,
    FINITE_GAUGE_REPORT_SCHEMA_VERSION,
    FOUR_PLAQUETTE_GAP_KIND,
    HAMILTONIAN_GAP_KIND,
    HAMILTONIAN_GAP_SCHEMA_VERSION,
    POLYMER_DOMAIN_KIND,
    POLYMER_DOMAIN_SCHEMA_VERSION,
    STRIP_RP_KIND,
    STRIP_RP_SCHEMA_VERSION,
    STRONG_COUPLING_KIND,
    STRONG_COUPLING_SCHEMA_VERSION,
    THREE_PLAQUETTE_GAP_KIND,
    TORUS_RP_KIND,
    TORUS_RP_SCHEMA_VERSION,
    TRANSFER_GAP_KIND,
    TRANSFER_GAP_SCHEMA_VERSION,
    WILSON_CHARACTER_DOMAIN_KIND,
    WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION,
    finite_gauge_report_schema_errors,
    hamiltonian_gap_schema_errors,
    polymer_domain_schema_errors,
    replay_finite_gauge_report,
    replay_hamiltonian_gap,
    replay_polymer_domain,
    replay_strip_rp,
    replay_strong_coupling_gap,
    replay_transfer_matrix_gap,
    replay_wilson_character_domain,
    seal_finite_gauge_report_certificate,
    seal_hamiltonian_gap_certificate,
    seal_polymer_domain_certificate,
    seal_strip_rp_certificate,
    seal_strong_coupling_certificate,
    seal_transfer_gap_certificate,
    seal_wilson_character_domain_certificate,
    strip_rp_schema_errors,
    strong_coupling_schema_errors,
    transfer_gap_schema_errors,
    wilson_character_domain_schema_errors,
)
from omnibias.geometry.gauge.transfer.character_algebra import (
    su2_character_convolution,
    su2_character_even_tail,
    su2_character_laplacian,
    su2_character_log_bound,
    su2_character_norm,
    su2_character_potential_bound,
    su2_character_product,
    su2_character_round,
)
from omnibias.geometry.gauge.transfer.charged_amplitude import (
    replay_su2_hamiltonian_rectangle_certificate,
    su2_hamiltonian_rectangle_enclosure,
)
from omnibias.geometry.gauge.transfer.charged_confinement import (
    replay_su2_static_confinement_certificate,
    su2_static_confinement_bounds,
    su2_static_confinement_family,
)
from omnibias.geometry.gauge.transfer.charged_sectors import (
    charged_spin_network_dimension,
    su2_singlet_multiplicity,
)
from omnibias.geometry.gauge.transfer.commutator_matrix import (
    replay_su2_commutator_matrix_gap_certificate,
    su2_commutator_matrix_gap,
)
from omnibias.geometry.gauge.transfer.compact_commutator import (
    replay_compact_commutator_certificate,
    su2_compact_commutator_gap,
)
from omnibias.geometry.gauge.transfer.corner_vacuum import (
    corner_geometry,
    corner_reduced_metric,
    replay_corner_vacuum_certificate,
    search_corner_vacuum_interval,
    su2_corner_vacuum_bound,
)
from omnibias.geometry.gauge.transfer.finite_graph_resolvent import (
    replay_su2_finite_graph_linearized_tail_certificate,
    su2_finite_graph_linearized_tail,
)
from omnibias.geometry.gauge.transfer.forest_projection import (
    gauge_forest_projection,
    replay_gauge_forest_projection_certificate,
)
from omnibias.geometry.gauge.transfer.gap import (
    BIRKHOFF_METHOD,
    DIAGONAL_SPECTRUM_METHOD,
    LEHMANN_METHOD,
    SYMMETRIC_METHOD,
    EffectiveMassCurve,
    EffectiveMassPoint,
    GapCandidate,
    MultistepGapResult,
    ScalingPoint,
    ScalingReport,
    TransferGapResult,
    certified_effective_mass_curve,
    certified_gap_scaling_table,
    certified_multistep_gap_refinement,
    certified_transfer_matrix_gap,
    heat_kernel_gap_scaling_report,
)
from omnibias.geometry.gauge.transfer.hamiltonian import (
    COUPLING_LOCK,
    FOUR_PLAQUETTE_ELECTRIC,
    GaugeHamiltonian,
    HamiltonianGapResult,
    certified_hamiltonian_gap,
    four_plaquette_basis,
    plaquette_holonomy_trial_space,
    standard_basis_trial_space,
    su2_four_plaquette_hamiltonian,
    su2_three_plaquette_hamiltonian,
    su2_two_plaquette_hamiltonian,
)
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_bounds,
    invariant_vacuum_fourier_family,
    replay_invariant_vacuum_fourier_certificate,
)
from omnibias.geometry.gauge.transfer.isolated_block_kinetic import (
    replay_su2_isolated_block_kinetic_certificate,
    su2_isolated_block_kinetic,
)
from omnibias.geometry.gauge.transfer.joint_vacuum import (
    su2_theta_norm,
    su2_theta_product_characters,
    su2_theta_product_reference_bound,
    su2_theta_vacuum_tail,
)
from omnibias.geometry.gauge.transfer.marginal_majorant import (
    conditional_poincare_schur,
    marginal_majorant_closure,
    replay_conditional_poincare_schur_certificate,
    replay_marginal_majorant_closure_certificate,
)
from omnibias.geometry.gauge.transfer.matrices import (
    TransferMatrix,
    su2_class_angle_transfer,
    su2_heat_kernel_transfer,
    su2_wilson_transfer,
    su3_heat_kernel_transfer,
    su3_wilson_transfer,
    u1_heat_kernel_transfer,
)
from omnibias.geometry.gauge.transfer.montecarlo import (
    MonteCarloGapCheck,
    PathEnsemble,
    certified_gap_versus_monte_carlo,
    sample_transfer_path_ensemble,
)
from omnibias.geometry.gauge.transfer.nonabelian_small_field import (
    replay_su2_nonabelian_small_field_certificate,
    su2_plaquette_trace_jet,
    su2_wilson_block_small_field,
    su2_wilson_small_field_budget,
)
from omnibias.geometry.gauge.transfer.physical_blocks import (
    conditional_overlap_budget,
    conditional_overlap_transfer_budget,
    physical_block_controls,
    replay_conditional_overlap_certificate,
    replay_conditional_overlap_transfer_certificate,
    replay_physical_block_control_certificate,
    replay_su2_theta_physical_block_certificate,
    su2_theta_physical_block_gap,
)
from omnibias.geometry.gauge.transfer.plaquette_path import (
    replay_su2_plaquette_path_conditional_certificate,
    su2_plaquette_path_conditional_gap,
)
from omnibias.geometry.gauge.transfer.plaquette_resolvent import (
    replay_su2_plaquette_linearized_inverse_certificate,
    su2_plaquette_linearized_inverse,
)
from omnibias.geometry.gauge.transfer.quadratic_vacuum import (
    quadratic_vacuum_root,
    replay_quadratic_vacuum_root_certificate,
)
from omnibias.geometry.gauge.transfer.reference_resolvent import (
    replay_su2_reference_linearized_inverse_certificate,
    su2_reference_linearized_inverse,
)
from omnibias.geometry.gauge.transfer.report import (
    REPORT_SU3_N_CELLS,
    FiniteGaugeReport,
    FiniteGaugeSpec,
    HaarIdentityCheck,
    MeasuredG1,
    default_finite_gauge_spec,
    finite_gauge_report,
    finite_gauge_spec_from_mapping,
    finite_gauge_spec_to_mapping,
)
from omnibias.geometry.gauge.transfer.scale_gap_budget import (
    replay_scale_gap_budget_certificate,
    scale_gap_budget,
)
from omnibias.geometry.gauge.transfer.shared_strip_refinement import (
    replay_su2_shared_strip_refinement_certificate,
    su2_shared_strip_refinement,
)
from omnibias.geometry.gauge.transfer.shared_strip_two_step import (
    replay_su2_shared_strip_two_step_certificate,
    su2_shared_strip_two_step,
)
from omnibias.geometry.gauge.transfer.static_sources import (
    replay_su2_static_source_certificate,
    su2_static_source_bounds,
)
from omnibias.geometry.gauge.transfer.strip import (
    STRIP_COUPLING_LOCK,
    certified_strip_cluster_tail,
    certified_strip_reflection_positivity,
    su2_spatial_strip_transfer,
    su2_spatial_torus_transfer,
)
from omnibias.geometry.gauge.transfer.strip_conditional_hierarchy import (
    replay_su2_strip_conditional_hierarchy_certificate,
    su2_strip_conditional_hierarchy,
)
from omnibias.geometry.gauge.transfer.strip_face_obstruction import (
    replay_su2_strip_face_obstruction_certificate,
    su2_strip_face_obstruction,
)
from omnibias.geometry.gauge.transfer.strip_kernel_tail import (
    replay_su2_strip_kernel_tail_certificate,
    su2_strip_normalized_kernel_tail,
)
from omnibias.geometry.gauge.transfer.strip_marginal_hierarchy import (
    replay_su2_strip_marginal_hierarchy_certificate,
    su2_strip_compression_geometry,
    su2_strip_marginal_hierarchy,
)
from omnibias.geometry.gauge.transfer.strong_coupling import (
    BACKTRACK_POLYMER_METHOD,
    BETA_LOCK,
    BETA_LOCK_CRUDE,
    CLUSTER_POLYMER_METHOD,
    CRUDE_POLYMER_METHOD,
    POLYMER_BETA_DOMAIN_METHOD,
    POLYMER_BETA_GRID,
    POLYMER_METHOD,
    WILSON_CHARACTER_BETA_DOMAIN_METHOD,
    WILSON_CHARACTER_BETA_GRID,
    WILSON_CHARACTER_CONTRAST_BETA,
    WILSON_CHARACTER_METHOD,
    PolymerDomainResult,
    StrongCouplingGapResult,
    VolumeUniformStrongCouplingFamily,
    WilsonCharacterDomainResult,
    WilsonCharacterGapResult,
    certified_polymer_beta_domain,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_beta_domain,
    certified_wilson_character_gap,
    polymer_coordination,
    polymer_coordination_backtrack,
    polymer_first_step,
    su2_wilson_activity,
    volume_uniform_strong_coupling_family,
)
from omnibias.geometry.gauge.transfer.su3_vacuum_fourier import (
    replay_su3_vacuum_fourier_certificate,
    su3_vacuum_fourier_bounds,
    su3_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.theta_kernel_tail import (
    replay_su2_theta_kernel_tail_certificate,
    su2_theta_normalized_kernel_tail,
)
from omnibias.geometry.gauge.transfer.theta_kernel_threshold import (
    replay_su2_theta_kernel_contraction_certificate,
    su2_theta_kernel_contraction,
)
from omnibias.geometry.gauge.transfer.theta_quasimode import (
    replay_su2_theta_quasimode_certificate,
    su2_theta_quasimode_error,
)
from omnibias.geometry.gauge.transfer.theta_refinement import (
    replay_su2_theta_refinement_certificate,
    su2_theta_refinement,
)
from omnibias.geometry.gauge.transfer.theta_vacuum_refinement import (
    replay_su2_theta_vacuum_refinement_certificate,
    su2_theta_vacuum_refinement,
)
from omnibias.geometry.gauge.transfer.theta_weak_blocks import (
    replay_su2_theta_weak_block_certificate,
    su2_theta_weak_block_gaps,
)
from omnibias.geometry.gauge.transfer.trial import (
    GRAM_COND_THRESHOLD,
    Loop,
    TrialSpace,
    holonomy_trial_space,
    su2_holonomy_trace,
)
from omnibias.geometry.gauge.transfer.vacuum_constraints import (
    replay_vacuum_constraint_certificate,
    solve_vacuum_constraints,
)
from omnibias.geometry.gauge.transfer.vacuum_fourier import (
    replay_su2_vacuum_fourier_certificate,
    su2_vacuum_fourier_bounds,
    su2_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.vacuum_local import (
    replay_su2_vacuum_local_certificate,
    su2_vacuum_local_bounds,
)
from omnibias.geometry.gauge.transfer.vmf_poincare import (
    replay_su2_vmf_poincare_certificate,
    su2_strip_reference_poincare,
    su2_vmf_poincare_bound,
)
from omnibias.geometry.gauge.transfer.weak_plaquette import (
    replay_su2_weak_plaquette_gap_certificate,
    su2_weak_plaquette_gap,
)
from omnibias.geometry.gauge.transfer.wilson_attachment import (
    replay_su2_wilson_attachment_certificate,
    su2_wilson_attachment,
)
from omnibias.geometry.gauge.transfer.wilson_conditional_block import (
    replay_su2_wilson_conditional_block_certificate,
    su2_wilson_conditional_block,
)
from omnibias.geometry.gauge.transfer.wilson_cube import (
    replay_su2_wilson_attached_cube_certificate,
    su2_wilson_attached_cube,
)
from omnibias.geometry.gauge.transfer.wilson_large_field import (
    replay_su2_wilson_large_field_certificate,
    su2_wilson_large_field,
)
from omnibias.geometry.gauge.transfer.wilson_polar_source import (
    replay_su2_wilson_polar_vacuum_certificate,
    su2_wilson_polar_vacuum,
)
from omnibias.geometry.gauge.transfer.wilson_residual_source import (
    replay_su2_wilson_linear_vacuum_certificate,
    replay_su2_wilson_residual_vacuum_certificate,
    su2_wilson_linear_vacuum,
    su2_wilson_residual_vacuum,
)
from omnibias.geometry.gauge.transfer.wilson_static_source import (
    replay_su2_wilson_static_family_certificate,
    su2_wilson_static_family,
)

__all__ = [
    "BACKTRACK_POLYMER_METHOD",
    "BETA_LOCK",
    "BETA_LOCK_CRUDE",
    "BIRKHOFF_METHOD",
    "CLUSTER_POLYMER_METHOD",
    "COUPLING_LOCK",
    "CRUDE_POLYMER_METHOD",
    "DIAGONAL_SPECTRUM_METHOD",
    "EffectiveMassCurve",
    "EffectiveMassPoint",
    "FINITE_GAUGE_REPORT_KIND",
    "FINITE_GAUGE_REPORT_SCHEMA_VERSION",
    "FOUR_PLAQUETTE_ELECTRIC",
    "FOUR_PLAQUETTE_GAP_KIND",
    "FiniteGaugeReport",
    "FiniteGaugeSpec",
    "GRAM_COND_THRESHOLD",
    "GapCandidate",
    "GaugeHamiltonian",
    "HAMILTONIAN_GAP_KIND",
    "HAMILTONIAN_GAP_SCHEMA_VERSION",
    "HaarIdentityCheck",
    "HamiltonianGapResult",
    "LEHMANN_METHOD",
    "Loop",
    "MeasuredG1",
    "MonteCarloGapCheck",
    "MultistepGapResult",
    "POLYMER_BETA_DOMAIN_METHOD",
    "POLYMER_BETA_GRID",
    "POLYMER_DOMAIN_KIND",
    "POLYMER_DOMAIN_SCHEMA_VERSION",
    "POLYMER_METHOD",
    "PathEnsemble",
    "PolymerDomainResult",
    "REPORT_SU3_N_CELLS",
    "STRIP_COUPLING_LOCK",
    "STRIP_RP_KIND",
    "STRIP_RP_SCHEMA_VERSION",
    "STRONG_COUPLING_KIND",
    "STRONG_COUPLING_SCHEMA_VERSION",
    "SYMMETRIC_METHOD",
    "ScalingPoint",
    "ScalingReport",
    "StrongCouplingGapResult",
    "THREE_PLAQUETTE_GAP_KIND",
    "TORUS_RP_KIND",
    "TORUS_RP_SCHEMA_VERSION",
    "TRANSFER_GAP_KIND",
    "TRANSFER_GAP_SCHEMA_VERSION",
    "TransferGapResult",
    "TransferMatrix",
    "TrialSpace",
    "VolumeUniformStrongCouplingFamily",
    "WILSON_CHARACTER_BETA_DOMAIN_METHOD",
    "WILSON_CHARACTER_BETA_GRID",
    "WILSON_CHARACTER_CONTRAST_BETA",
    "WILSON_CHARACTER_DOMAIN_KIND",
    "WILSON_CHARACTER_DOMAIN_SCHEMA_VERSION",
    "WILSON_CHARACTER_METHOD",
    "WilsonCharacterDomainResult",
    "WilsonCharacterGapResult",
    "certified_effective_mass_curve",
    "certified_gap_scaling_table",
    "certified_gap_versus_monte_carlo",
    "certified_hamiltonian_gap",
    "certified_multistep_gap_refinement",
    "certified_polymer_beta_domain",
    "certified_strip_cluster_tail",
    "certified_strip_reflection_positivity",
    "certified_strong_coupling_glueball_bound",
    "certified_transfer_matrix_gap",
    "certified_wilson_character_beta_domain",
    "certified_wilson_character_gap",
    "charged_spin_network_dimension",
    "conditional_overlap_budget",
    "conditional_overlap_transfer_budget",
    "conditional_poincare_schur",
    "corner_geometry",
    "corner_reduced_metric",
    "default_finite_gauge_spec",
    "finite_gauge_report",
    "finite_gauge_report_schema_errors",
    "finite_gauge_spec_from_mapping",
    "finite_gauge_spec_to_mapping",
    "four_plaquette_basis",
    "gauge_forest_projection",
    "hamiltonian_gap_schema_errors",
    "heat_kernel_gap_scaling_report",
    "holonomy_trial_space",
    "invariant_vacuum_fourier_bounds",
    "invariant_vacuum_fourier_family",
    "marginal_majorant_closure",
    "physical_block_controls",
    "plaquette_holonomy_trial_space",
    "polymer_coordination",
    "polymer_coordination_backtrack",
    "polymer_domain_schema_errors",
    "polymer_first_step",
    "quadratic_vacuum_root",
    "replay_compact_commutator_certificate",
    "replay_conditional_overlap_certificate",
    "replay_conditional_overlap_transfer_certificate",
    "replay_conditional_poincare_schur_certificate",
    "replay_corner_vacuum_certificate",
    "replay_finite_gauge_report",
    "replay_gauge_forest_projection_certificate",
    "replay_hamiltonian_gap",
    "replay_invariant_vacuum_fourier_certificate",
    "replay_marginal_majorant_closure_certificate",
    "replay_physical_block_control_certificate",
    "replay_polymer_domain",
    "replay_quadratic_vacuum_root_certificate",
    "replay_scale_gap_budget_certificate",
    "replay_strip_rp",
    "replay_strong_coupling_gap",
    "replay_su2_adjacent_cone_inverse_certificate",
    "replay_su2_adjacent_cone_vacuum_certificate",
    "replay_su2_adjacent_linearized_inverse_certificate",
    "replay_su2_adjacent_preconditioned_vacuum_certificate",
    "replay_su2_adjacent_vacuum_gap_comparison_certificate",
    "replay_su2_ambient_conditional_certificate",
    "replay_su2_commutator_matrix_gap_certificate",
    "replay_su2_finite_graph_linearized_tail_certificate",
    "replay_su2_hamiltonian_rectangle_certificate",
    "replay_su2_isolated_block_kinetic_certificate",
    "replay_su2_nonabelian_small_field_certificate",
    "replay_su2_plaquette_linearized_inverse_certificate",
    "replay_su2_plaquette_path_conditional_certificate",
    "replay_su2_reference_linearized_inverse_certificate",
    "replay_su2_shared_strip_refinement_certificate",
    "replay_su2_shared_strip_two_step_certificate",
    "replay_su2_static_confinement_certificate",
    "replay_su2_static_source_certificate",
    "replay_su2_strip_conditional_hierarchy_certificate",
    "replay_su2_strip_face_obstruction_certificate",
    "replay_su2_strip_kernel_tail_certificate",
    "replay_su2_strip_marginal_hierarchy_certificate",
    "replay_su2_theta_kernel_contraction_certificate",
    "replay_su2_theta_kernel_tail_certificate",
    "replay_su2_theta_physical_block_certificate",
    "replay_su2_theta_quasimode_certificate",
    "replay_su2_theta_refinement_certificate",
    "replay_su2_theta_vacuum_refinement_certificate",
    "replay_su2_theta_weak_block_certificate",
    "replay_su2_vacuum_fourier_certificate",
    "replay_su2_vacuum_local_certificate",
    "replay_su2_vmf_poincare_certificate",
    "replay_su2_weak_plaquette_gap_certificate",
    "replay_su2_wilson_attached_cube_certificate",
    "replay_su2_wilson_attachment_certificate",
    "replay_su2_wilson_conditional_block_certificate",
    "replay_su2_wilson_large_field_certificate",
    "replay_su2_wilson_linear_vacuum_certificate",
    "replay_su2_wilson_polar_vacuum_certificate",
    "replay_su2_wilson_residual_vacuum_certificate",
    "replay_su2_wilson_static_family_certificate",
    "replay_su3_vacuum_fourier_certificate",
    "replay_transfer_matrix_gap",
    "replay_vacuum_constraint_certificate",
    "replay_wilson_character_domain",
    "sample_transfer_path_ensemble",
    "scale_gap_budget",
    "seal_finite_gauge_report_certificate",
    "seal_hamiltonian_gap_certificate",
    "seal_polymer_domain_certificate",
    "seal_strip_rp_certificate",
    "seal_strong_coupling_certificate",
    "seal_transfer_gap_certificate",
    "seal_wilson_character_domain_certificate",
    "search_corner_vacuum_interval",
    "solve_vacuum_constraints",
    "standard_basis_trial_space",
    "strip_rp_schema_errors",
    "strong_coupling_schema_errors",
    "su2_adjacent_cone_inverse",
    "su2_adjacent_cone_vacuum",
    "su2_adjacent_linearized_inverse",
    "su2_adjacent_preconditioned_vacuum",
    "su2_adjacent_vacuum_gap_comparison",
    "su2_ambient_conditional_gap",
    "su2_character_convolution",
    "su2_character_even_tail",
    "su2_character_laplacian",
    "su2_character_log_bound",
    "su2_character_norm",
    "su2_character_potential_bound",
    "su2_character_product",
    "su2_character_round",
    "su2_class_angle_transfer",
    "su2_commutator_matrix_gap",
    "su2_compact_commutator_gap",
    "su2_corner_vacuum_bound",
    "su2_finite_graph_linearized_tail",
    "su2_four_plaquette_hamiltonian",
    "su2_hamiltonian_rectangle_enclosure",
    "su2_heat_kernel_transfer",
    "su2_holonomy_trace",
    "su2_isolated_block_kinetic",
    "su2_plaquette_linearized_inverse",
    "su2_plaquette_path_conditional_gap",
    "su2_plaquette_trace_jet",
    "su2_reference_linearized_inverse",
    "su2_shared_strip_refinement",
    "su2_shared_strip_two_step",
    "su2_singlet_multiplicity",
    "su2_spatial_strip_transfer",
    "su2_spatial_torus_transfer",
    "su2_static_confinement_bounds",
    "su2_static_confinement_family",
    "su2_static_source_bounds",
    "su2_strip_compression_geometry",
    "su2_strip_conditional_hierarchy",
    "su2_strip_face_obstruction",
    "su2_strip_marginal_hierarchy",
    "su2_strip_normalized_kernel_tail",
    "su2_strip_reference_poincare",
    "su2_theta_kernel_contraction",
    "su2_theta_norm",
    "su2_theta_normalized_kernel_tail",
    "su2_theta_physical_block_gap",
    "su2_theta_product_characters",
    "su2_theta_product_reference_bound",
    "su2_theta_quasimode_error",
    "su2_theta_refinement",
    "su2_theta_vacuum_refinement",
    "su2_theta_vacuum_tail",
    "su2_theta_weak_block_gaps",
    "su2_three_plaquette_hamiltonian",
    "su2_two_plaquette_hamiltonian",
    "su2_vacuum_fourier_bounds",
    "su2_vacuum_fourier_family",
    "su2_vacuum_local_bounds",
    "su2_vmf_poincare_bound",
    "su2_weak_plaquette_gap",
    "su2_wilson_activity",
    "su2_wilson_attached_cube",
    "su2_wilson_attachment",
    "su2_wilson_block_small_field",
    "su2_wilson_conditional_block",
    "su2_wilson_large_field",
    "su2_wilson_linear_vacuum",
    "su2_wilson_polar_vacuum",
    "su2_wilson_residual_vacuum",
    "su2_wilson_small_field_budget",
    "su2_wilson_static_family",
    "su2_wilson_transfer",
    "su3_heat_kernel_transfer",
    "su3_vacuum_fourier_bounds",
    "su3_vacuum_fourier_family",
    "su3_wilson_transfer",
    "transfer_gap_schema_errors",
    "u1_heat_kernel_transfer",
    "volume_uniform_strong_coupling_family",
    "wilson_character_domain_schema_errors",
]
