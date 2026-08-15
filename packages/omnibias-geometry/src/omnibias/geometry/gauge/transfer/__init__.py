# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified spectral gaps of finite lattice transfer matrices.

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
Every certificate is a statement about **one fixed matrix, at one fixed lattice
spacing, in finite dimension**.  ``continuum_claim`` is always ``False``,
:func:`heat_kernel_gap_scaling_report` is labelled evidence rather than proof, and
nothing here is a claim about the Yang-Mills mass gap.
"""

from __future__ import annotations

from omnibias.geometry.gauge.transfer.certificates import (
    STRONG_COUPLING_KIND,
    STRONG_COUPLING_SCHEMA_VERSION,
    TRANSFER_GAP_KIND,
    TRANSFER_GAP_SCHEMA_VERSION,
    replay_strong_coupling_gap,
    replay_transfer_matrix_gap,
    seal_strong_coupling_certificate,
    seal_transfer_gap_certificate,
    strong_coupling_schema_errors,
    transfer_gap_schema_errors,
)
from omnibias.geometry.gauge.transfer.gap import (
    BIRKHOFF_METHOD,
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
    certified_multistep_gap_refinement,
    certified_transfer_matrix_gap,
    heat_kernel_gap_scaling_report,
)
from omnibias.geometry.gauge.transfer.matrices import (
    TransferMatrix,
    su2_class_angle_transfer,
    su2_heat_kernel_transfer,
    su2_wilson_transfer,
    su3_heat_kernel_transfer,
    u1_heat_kernel_transfer,
)
from omnibias.geometry.gauge.transfer.montecarlo import (
    MonteCarloGapCheck,
    PathEnsemble,
    certified_gap_versus_monte_carlo,
    sample_transfer_path_ensemble,
)
from omnibias.geometry.gauge.transfer.strong_coupling import (
    BETA_LOCK,
    POLYMER_METHOD,
    WILSON_CHARACTER_METHOD,
    StrongCouplingGapResult,
    WilsonCharacterGapResult,
    certified_strong_coupling_glueball_bound,
    certified_wilson_character_gap,
    polymer_coordination,
    su2_wilson_activity,
)
from omnibias.geometry.gauge.transfer.trial import (
    GRAM_COND_THRESHOLD,
    Loop,
    TrialSpace,
    holonomy_trial_space,
    su2_holonomy_trace,
)

__all__ = [
    "BETA_LOCK",
    "BIRKHOFF_METHOD",
    "GRAM_COND_THRESHOLD",
    "LEHMANN_METHOD",
    "POLYMER_METHOD",
    "STRONG_COUPLING_KIND",
    "STRONG_COUPLING_SCHEMA_VERSION",
    "SYMMETRIC_METHOD",
    "TRANSFER_GAP_KIND",
    "TRANSFER_GAP_SCHEMA_VERSION",
    "WILSON_CHARACTER_METHOD",
    "EffectiveMassCurve",
    "EffectiveMassPoint",
    "GapCandidate",
    "Loop",
    "MonteCarloGapCheck",
    "MultistepGapResult",
    "PathEnsemble",
    "ScalingPoint",
    "ScalingReport",
    "StrongCouplingGapResult",
    "TransferGapResult",
    "TransferMatrix",
    "TrialSpace",
    "WilsonCharacterGapResult",
    "certified_effective_mass_curve",
    "certified_gap_versus_monte_carlo",
    "certified_multistep_gap_refinement",
    "certified_strong_coupling_glueball_bound",
    "certified_transfer_matrix_gap",
    "certified_wilson_character_gap",
    "heat_kernel_gap_scaling_report",
    "holonomy_trial_space",
    "polymer_coordination",
    "replay_strong_coupling_gap",
    "replay_transfer_matrix_gap",
    "sample_transfer_path_ensemble",
    "seal_strong_coupling_certificate",
    "seal_transfer_gap_certificate",
    "strong_coupling_schema_errors",
    "su2_class_angle_transfer",
    "su2_heat_kernel_transfer",
    "su2_holonomy_trace",
    "su2_wilson_activity",
    "su2_wilson_transfer",
    "su3_heat_kernel_transfer",
    "transfer_gap_schema_errors",
    "u1_heat_kernel_transfer",
]
