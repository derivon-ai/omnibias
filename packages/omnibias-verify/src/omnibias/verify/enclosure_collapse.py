# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Public import path for Enclosure Collapse (theory 01-14).

Re-exports :mod:`omnibias.verify._core.enclosure_collapse`. The founding
bias collapse (``delta -> 0``) yields a derivative. Temperature collapse
(``beta -> inf``, feasibility) yields a 0/1 step. Enclosure Collapse is
the ``width -> 0`` limit of a sound enclosure: a point plus a proof.
Not a derivative and not a 0/1 step. Do not conflate the three.
"""

from __future__ import annotations

from omnibias.verify._core.enclosure_collapse import (
    SqueezeKind,
    SqueezeReport,
    SqueezeStatus,
    report_digest_ok,
    squeeze,
    squeeze_existence,
    squeeze_flow,
    squeeze_identifiability,
    squeeze_peak,
    squeeze_remainder,
    squeeze_residual,
)

__all__ = [
    "SqueezeKind",
    "SqueezeReport",
    "SqueezeStatus",
    "report_digest_ok",
    "squeeze",
    "squeeze_existence",
    "squeeze_flow",
    "squeeze_identifiability",
    "squeeze_peak",
    "squeeze_remainder",
    "squeeze_residual",
]
