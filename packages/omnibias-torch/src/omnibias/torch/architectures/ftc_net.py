# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""FTC-Net + dual-FTC loss (torch; theory 09-03 / 09-17).

The cell is the ``integral`` role. The derivative head is FTC of the
same window. The founding bias collapse (``delta -> 0``) is the
collapse head ``I / delta``. Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import ftc as core_ftc

import torch
from torch import Tensor

DISCLAIMER = core_ftc.DISCLAIMER
DualFTCConfig = core_ftc.DualFTCConfig
FTCNetConfig = core_ftc.FTCNetConfig
honesty_payload = core_ftc.honesty_payload
worked_example = core_ftc.worked_example

__doc__ += f"\n\n{DISCLAIMER}\n"


def ftc_block(
    x: Tensor,
    w: Tensor,
    b_lo: Tensor,
    b_hi: Tensor,
    *,
    activation: str = "sigmoid",
) -> tuple[Tensor, Tensor, Tensor]:
    """Bit-identical to :func:`omnibias.core.ftc.ftc_block` on scalars."""
    _ = activation
    xs = x.detach().reshape(-1).tolist()
    ws = w.detach().reshape(-1).tolist()
    los = b_lo.detach().reshape(-1).tolist()
    his = b_hi.detach().reshape(-1).tolist()
    n = max(len(xs), len(ws), len(los), len(his))

    def _at(vals: list[float], i: int) -> float:
        return vals[0] if len(vals) == 1 else vals[i]

    integrals: list[float] = []
    derivs: list[float] = []
    collapses: list[float] = []
    for i in range(n):
        integral, deriv, collapse = core_ftc.ftc_block(
            _at(xs, i), _at(ws, i), _at(los, i), _at(his, i), activation=activation
        )
        integrals.append(integral)
        derivs.append(deriv)
        collapses.append(collapse)
    dtype = x.dtype
    if x.ndim == 0:
        return (
            torch.tensor(integrals[0], dtype=dtype, device=x.device),
            torch.tensor(derivs[0], dtype=dtype, device=x.device),
            torch.tensor(collapses[0], dtype=dtype, device=x.device),
        )
    return (
        torch.tensor(integrals, dtype=dtype, device=x.device).reshape(x.shape),
        torch.tensor(derivs, dtype=dtype, device=x.device).reshape(x.shape),
        torch.tensor(collapses, dtype=dtype, device=x.device).reshape(x.shape),
    )


class FTCNet:
    """Config holder; evaluation is the closed-form pack sum in ``omnibias.core.ftc``."""

    def __init__(self, config: FTCNetConfig | None = None) -> None:
        self.config = FTCNetConfig() if config is None else config


def dual_ftc_loss(
    I_vals: Sequence[float] | Tensor,
    dI_vals: Sequence[float] | Tensor,
    f_vals: Sequence[float] | Tensor,
    F_vals: Sequence[float] | Tensor,
    *,
    config: DualFTCConfig | None = None,
    I_a: float | None = None,
) -> core_ftc.DualFTCResult:
    def _as_list(v: Sequence[float] | Tensor) -> list[float]:
        if isinstance(v, Tensor):
            return [float(x) for x in v.detach().reshape(-1).tolist()]
        return [float(x) for x in v]

    return core_ftc.dual_ftc_loss(
        _as_list(I_vals),
        _as_list(dI_vals),
        _as_list(f_vals),
        _as_list(F_vals),
        config=core_ftc.DEFAULT_DUAL if config is None else config,
        I_a=I_a,
    )


__all__ = [
    "DISCLAIMER",
    "DualFTCConfig",
    "FTCNet",
    "FTCNetConfig",
    "dual_ftc_loss",
    "ftc_block",
    "honesty_payload",
    "worked_example",
]
