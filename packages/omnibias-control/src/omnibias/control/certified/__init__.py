# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified enclosures for the jet-adjoint policy-optimization stack (theory 10-03).

Claims here follow `.cursor/rules/omnibias.md` (Frontier program): every claim
in this subpackage must be a sound, finite enclosure of a stated quantity, not
an assertion about the true (unknowable in general) policy-gradient bias.
"""

from __future__ import annotations

from omnibias.control.certified.gradient_bias import (
    GradientBiasReport,
    honesty_payload,
    terminal_adjoint_error_bound,
    truncation_bias_bound,
)

__all__ = [
    "GradientBiasReport",
    "honesty_payload",
    "terminal_adjoint_error_bound",
    "truncation_bias_bound",
]
