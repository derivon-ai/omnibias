# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Finite-domain CSP on the discrete seam (theory 03-03).

Simplex sharpness and clause sharpness are two independent
``beta -> inf`` temperature-collapse knobs (feasibility). Neither is
the founding bias collapse (``delta -> 0``). Do not conflate the two.
A relaxation is not a complete solver. Certified gaps are instance-wise,
not P vs NP.
"""

from __future__ import annotations

from omnibias.discrete.csp._core import (
    CSP,
    CSPCertificate,
    CSPResult,
    GlobalConstraint,
    Relation,
    Variable,
    assignment_onehot,
    backtrack_sat,
    brute_force_sat,
    certify_csp,
    clause_gap_bound,
    csp_solve,
    default_clause_schedule,
    default_simplex_schedule,
    honesty_payload,
    not_equal_relation,
    random_binary_csp,
    soft_arc_consistency,
    softmax_rows,
    triangle_colouring,
)

__all__ = [
    "CSP",
    "CSPCertificate",
    "CSPResult",
    "GlobalConstraint",
    "Relation",
    "Variable",
    "assignment_onehot",
    "backtrack_sat",
    "brute_force_sat",
    "certify_csp",
    "clause_gap_bound",
    "csp_solve",
    "default_clause_schedule",
    "default_simplex_schedule",
    "honesty_payload",
    "not_equal_relation",
    "random_binary_csp",
    "soft_arc_consistency",
    "softmax_rows",
    "triangle_colouring",
]
