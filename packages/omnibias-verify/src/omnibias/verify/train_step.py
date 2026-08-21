# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Public import path for the 08-09 certified-step policy.

Re-exports :mod:`omnibias.verify._core.train_step`. Optimizers propose
``theta'``; this module decides whether the I/O enclosure stays in cap.
Do not import this from ``omnibias.core`` or the T1 torch/jax packages.
"""

from __future__ import annotations

from omnibias.verify._core.train_step import (
    HONESTY,
    CertifiedStepConfig,
    CertifiedStepForbidden,
    CertifiedStepResult,
    CollapseName,
    PropertyName,
    StepReason,
    apply_flat_theta,
    certified_accept,
    honesty_payload,
    select_certified_theta,
)

__all__ = [
    "CertifiedStepConfig",
    "CertifiedStepForbidden",
    "CertifiedStepResult",
    "CollapseName",
    "HONESTY",
    "PropertyName",
    "StepReason",
    "apply_flat_theta",
    "certified_accept",
    "honesty_payload",
    "select_certified_theta",
]
