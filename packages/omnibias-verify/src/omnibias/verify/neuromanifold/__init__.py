# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified consumers of numerical realization geometry and collision charts."""

from .confluence import (
    ConfluenceCertificate,
    certify_pair_collapse,
    certify_transition,
    integral_error,
    linear_residual_error,
    nonlinear_residual_error,
    replay_confluence_certificate,
)
from .formal import (
    FormalGeometryReplay,
    formalize_confluence,
    formalize_minimum,
    minimum_replay_certificates,
    replay_finite_geometry,
)
from .minima import (
    IntervalObjective,
    MinimumCertificate,
    certify_quotient_minimum,
    certify_slice_minimum,
    dual_sigmoid,
    dual_tanh,
    verify_affine_chart,
)
from .scientific import (
    IdentifiabilityCertificate,
    QuantumGeometryCertificate,
    ReductionCertificate,
    certify_identifiability,
    certify_quantum_geometry,
    certify_reduction,
    replay_quantum_geometry,
)
from .selection import (
    ArchitectureSelection,
    decode_architecture,
    select_architectures_jax,
    select_architectures_torch,
)

__all__ = [
    "ArchitectureSelection",
    "ConfluenceCertificate",
    "FormalGeometryReplay",
    "IdentifiabilityCertificate",
    "IntervalObjective",
    "MinimumCertificate",
    "QuantumGeometryCertificate",
    "ReductionCertificate",
    "certify_identifiability",
    "certify_pair_collapse",
    "certify_quantum_geometry",
    "certify_quotient_minimum",
    "certify_reduction",
    "certify_slice_minimum",
    "certify_transition",
    "decode_architecture",
    "dual_sigmoid",
    "dual_tanh",
    "formalize_confluence",
    "formalize_minimum",
    "integral_error",
    "linear_residual_error",
    "minimum_replay_certificates",
    "nonlinear_residual_error",
    "replay_confluence_certificate",
    "replay_finite_geometry",
    "replay_quantum_geometry",
    "select_architectures_jax",
    "select_architectures_torch",
    "verify_affine_chart",
]
