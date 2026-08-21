# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Public import path for certified scan localization (theory 03-08).

Re-exports :mod:`omnibias.verify._core.localization`. A scan template
is founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.
"""

from __future__ import annotations

from omnibias.verify._core.localization import (
    Inconclusive,
    LocalizationCertificate,
    ScanResponse,
    branch_and_bound_peak,
    certify_multiple_peaks,
    certify_peak,
    honesty_payload,
    seal,
    sealed_digest_ok,
    tight_deriv,
)

__all__ = [
    "Inconclusive",
    "LocalizationCertificate",
    "ScanResponse",
    "branch_and_bound_peak",
    "certify_multiple_peaks",
    "certify_peak",
    "honesty_payload",
    "seal",
    "sealed_digest_ok",
    "tight_deriv",
]
