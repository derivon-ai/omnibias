# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Integral-kernel DeepONet cell (torch; theory 09-14).

Volumetric apply. Not BEM-Net. founding bias collapse
(``delta -> 0``) of the window is not the default. Temperature
collapse (``beta -> inf``, feasibility) does not appear. do not
conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.pinn.operator._core import integral_kernel as core
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
IntegralKernelConfig = core.IntegralKernelConfig
honesty_payload = core.honesty_payload


def integral_cell(z: Tensor, b_lo: float, b_hi: float) -> Tensor:
    """``S(z + b_hi) - S(z + b_lo)``. Matches ``OperatorBlock(op='integral')``."""
    vals = [core.integral_cell(float(v), b_lo, b_hi) for v in z.reshape(-1).tolist()]
    return z.new_tensor(vals).reshape(z.shape)


def integral_kernel_apply(
    source: Tensor,
    coords: Tensor,
    params: Sequence[float] | None = None,
    *,
    config: IntegralKernelConfig | None = None,
    nodes: Tensor | None = None,
) -> Tensor:
    """Volumetric apply. Not BEM-Net."""
    src = [float(v) for v in source.reshape(-1).tolist()]
    xs = [float(v) for v in coords.reshape(-1).tolist()]
    ys = [float(v) for v in nodes.reshape(-1).tolist()] if nodes is not None else None
    out = core.integral_kernel_apply(src, xs, params, config=config, nodes=ys)
    return coords.new_tensor(out)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "IntegralKernelConfig",
    "honesty_payload",
    "integral_cell",
    "integral_kernel_apply",
    "worked_example",
]
