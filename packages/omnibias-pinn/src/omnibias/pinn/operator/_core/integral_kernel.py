# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Integral-kernel operator schemas (theory 09-14).

Re-exports the shared 1-D OMBU ``integral`` cell. Not BEM-Net.
founding bias collapse (``delta -> 0``) of the window is not the
default. Temperature collapse (``beta -> inf``, feasibility) does
not appear. do not conflate the two.
"""

from __future__ import annotations

from omnibias.core.integral_kernel import (
    DEFAULT_CONFIG,
    DISCLAIMER,
    IntegralKernelConfig,
    antiderivative_skill,
    honesty_payload,
    integral_cell,
    integral_kernel_apply,
    kernel,
    volterra_apply,
    worked_example,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "IntegralKernelConfig",
    "antiderivative_skill",
    "honesty_payload",
    "integral_cell",
    "integral_kernel_apply",
    "kernel",
    "volterra_apply",
    "worked_example",
]
