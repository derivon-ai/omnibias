# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parallel-interface transmission PINN (theory 02-05, gated).

Import :class:`Interface` from this package, **not** from
:mod:`omnibias.pinn._core.interface` (that is the XPINN penalty glue).
The :class:`TransmissionInterface` alias exists so docs and skills cannot
grab the wrong type by accident.

``alpha -> inf`` is interface sharpening, neither collapse. Parallel
interfaces only.
"""

from __future__ import annotations

from omnibias.pinn.interface._core import (
    Interface,
    TransmissionInterface,
    order_for_condition,
    smoothing_error_bound,
)

__all__ = [
    "Interface",
    "MultiInterfaceField",
    "TransmissionInterface",
    "order_for_condition",
    "smoothing_error_bound",
]


def __getattr__(name: str) -> object:
    if name == "MultiInterfaceField":
        from omnibias.pinn.interface.torch import MultiInterfaceField

        return MultiInterfaceField
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
