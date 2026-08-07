# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Readout-independence gate for the frozen-feature linear solver.

The collocation plan reuses cached :class:`~omnibias.fields._core.state.FieldState`
objects while sweeping the readout. That is sound only for fields that declare
the marker named by
:data:`~omnibias.fields._core.field_base.READOUT_INDEPENDENT_ATTR`.
"""

from __future__ import annotations

from typing import Any

from omnibias.fields._core.field_base import READOUT_INDEPENDENT_ATTR


class ReadoutDependentError(TypeError):
    """Raised when a frozen-feature solve is asked of a readout-dependent field."""


def requires_readout_independent(field: Any) -> None:
    """Refuse a field whose per-state caches depend on the readout parameters."""
    if getattr(field, READOUT_INDEPENDENT_ATTR, False):
        return
    name = type(field).__name__
    raise ReadoutDependentError(
        f"{name} does not declare readout-independent FieldState caches "
        f"({READOUT_INDEPENDENT_ATTR}=True). The frozen-feature linear solver "
        "reuses cached states while sweeping the readout, which is only sound "
        "when every cached quantity is independent of the readout parameters. "
        "Fields that are nonlinear in the readout (e.g. IntegralConservationField, "
        "NormConservationField) cannot participate; use solve_optimize instead, "
        "or unwrap the nonlinear cage."
    )


__all__ = [
    "ReadoutDependentError",
    "requires_readout_independent",
]
