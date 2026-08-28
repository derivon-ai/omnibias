# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Lie point-symmetry discovery (theory 03-11).

Point symmetries only. The ansatz bounds what can be found.
Jets come from the founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two.
"""

from __future__ import annotations

from omnibias.symbolic.symmetry._core import (
    DISCLAIMER,
    ExactSymmetryReport,
    Generator,
    LinearPoly,
    PDESpec,
    Sample,
    SymmetryBasis,
    SymmetryResult,
    affine_basis,
    designed_samples,
    determining_matrix,
    discover_symmetries,
    eta_t,
    eta_xx,
    exact_symmetry_report,
    heat_known_coeffs,
    honesty_payload,
    matrix_condition,
    negative_control,
    noether_wave_residual,
    pr_for,
    pr_heat,
    pr_heat_fd,
    suite,
)

__all__ = [
    "DISCLAIMER",
    "ExactSymmetryReport",
    "Generator",
    "LinearPoly",
    "PDESpec",
    "Sample",
    "SymmetryBasis",
    "SymmetryResult",
    "affine_basis",
    "designed_samples",
    "determining_matrix",
    "discover_symmetries",
    "eta_t",
    "eta_xx",
    "exact_symmetry_report",
    "heat_known_coeffs",
    "honesty_payload",
    "matrix_condition",
    "negative_control",
    "noether_wave_residual",
    "pr_for",
    "pr_heat",
    "pr_heat_fd",
    "suite",
]
