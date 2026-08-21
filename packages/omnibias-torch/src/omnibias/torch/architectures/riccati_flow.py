# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Riccati flow net (torch; theory 09-10).

Depth is Riccati time, not a DEQ and not a CNF. This forward is not
founding bias collapse (no ``delta -> 0`` pack). Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.
"""

from __future__ import annotations

from omnibias.core import riccati_flow as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
RiccatiFlowConfig = core.RiccatiFlowConfig
honesty_payload = core.honesty_payload


def riccati_flow(s0: Tensor, *, config: RiccatiFlowConfig | None = None) -> Tensor:
    xs = [float(v) for v in s0.reshape(-1).tolist()]
    ys = [core.riccati_flow(x, config=config) for x in xs]
    return s0.new_tensor(ys).reshape(s0.shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "RiccatiFlowConfig",
    "honesty_payload",
    "riccati_flow",
    "worked_example",
]
