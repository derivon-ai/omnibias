# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Uncertainty slabs (verify surface; theory 04-02).

Sound enclosures live here. Conformal intervals do not seal.
The band slab is a finite gap, the opposite of founding bias
collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two.
"""

from __future__ import annotations

from omnibias.core.uncertainty import (
    DISCLAIMER,
    WORKED_RESIDUALS,
    CalibrationReport,
    CombinedStatement,
    GuaranteeKind,
    UncertaintyInterval,
    adaptive_conformal,
    adaptive_vs_fixed,
    calibration_report,
    combine_enclosure_with_conformal,
    conformal_index,
    honesty_payload,
    refuse_conformal_seal,
    resample_coverage,
    shift_diagnostic,
    softplus_width,
    softplus_width_prime,
    split_conformal,
    theoretical_coverage,
    worked_example,
)

__all__ = [
    "CalibrationReport",
    "CombinedStatement",
    "DISCLAIMER",
    "GuaranteeKind",
    "UncertaintyInterval",
    "WORKED_RESIDUALS",
    "adaptive_conformal",
    "adaptive_vs_fixed",
    "calibration_report",
    "combine_enclosure_with_conformal",
    "conformal_index",
    "honesty_payload",
    "refuse_conformal_seal",
    "resample_coverage",
    "shift_diagnostic",
    "softplus_width",
    "softplus_width_prime",
    "split_conformal",
    "theoretical_coverage",
    "worked_example",
]
