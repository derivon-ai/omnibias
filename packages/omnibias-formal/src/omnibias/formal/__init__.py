# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""omnibias-formal: Mathlib-backed formal checking for omnibias certificates.

This optional package drives the Mathlib-backed Lean project
``formal/omnibias-analytic`` to discharge a certificate's rational / real finite
obligations, on a trust tier (:data:`~omnibias.formal.mathlib_check.MATHLIB_CLAIM_KEY`,
``"mathlib_verified"``) that is **distinct** from the Mathlib-free minimal
kernel's ``theorem_prover_verified`` (:mod:`omnibias.core.proof.lean_check`).

It is deliberately kept out of :mod:`omnibias.core` so a normal install never
pulls a Mathlib expectation, and it degrades gracefully when no Lean toolchain is
present.  A green Mathlib build certifies the certificate's *finite* obligation;
it never implies ``unproven_claim`` and the analytic statement track keeps honest
``sorry``s.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PkgNotFound
from importlib.metadata import version as _pkg_version

from omnibias.formal.augment import MathlibVerdict, evaluate_with_mathlib
from omnibias.formal.drive import DriveReport, drive_obligation
from omnibias.formal.mathlib_check import (
    MATHLIB_CLAIM_KEY,
    MathlibCheckResult,
    analytic_root,
    check_certificate,
    classify_obligation,
    generate_obligation,
    mathlib_check_available,
)
from omnibias.formal.tower import (
    LEGAL_TOWER_FAMILIES,
    tower_coeffs,
    tower_coeffs_certificate,
)

try:
    __version__ = _pkg_version("omnibias-formal")
except _PkgNotFound:  # pragma: no cover - bare source checkout
    __version__ = "0.0.0+unknown"

# Founding-idea lineage (see docs/theory.md "Two senses of collapse").
__lineage__ = "exempt: infrastructure"

__all__ = [
    "DriveReport",
    "LEGAL_TOWER_FAMILIES",
    "MATHLIB_CLAIM_KEY",
    "MathlibCheckResult",
    "MathlibVerdict",
    "__lineage__",
    "__version__",
    "analytic_root",
    "check_certificate",
    "classify_obligation",
    "drive_obligation",
    "evaluate_with_mathlib",
    "generate_obligation",
    "mathlib_check_available",
    "tower_coeffs",
    "tower_coeffs_certificate",
]
