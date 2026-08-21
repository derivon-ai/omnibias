# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pack-MoE (torch; theory 09-07).

The router is slab mass (window knob), not softmax. Expert collapse
heads may use founding bias collapse (``delta -> 0``). Temperature
collapse (``beta -> inf``, feasibility) is recorded when ``beta != 1``.
Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import pack_moe as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
ExpertWindow = core.ExpertWindow
PackMoEConfig = core.PackMoEConfig
honesty_payload = core.honesty_payload


def pack_moe_forward(
    x: Tensor,
    experts: Sequence[float] | Tensor,
    windows: Sequence[ExpertWindow],
    *,
    config: PackMoEConfig | None = None,
) -> Tensor:
    xs = [float(v) for v in x.reshape(-1).tolist()]
    fs = (
        [float(v) for v in experts.reshape(-1).tolist()]
        if isinstance(experts, Tensor)
        else [float(v) for v in experts]
    )
    ys = [core.pack_moe_forward(xi, fs, windows, config=config) for xi in xs]
    return x.new_tensor(ys).reshape(x.shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "ExpertWindow",
    "PackMoEConfig",
    "honesty_payload",
    "pack_moe_forward",
    "worked_example",
]
