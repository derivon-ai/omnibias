# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Collapse-Net (torch; theory 09-11).

Train uses a founding stencil; inference is founding bias collapse
(``delta -> 0``) to the named derivative. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.
"""

from __future__ import annotations

from omnibias.core import collapse_net as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
CollapseNetConfig = core.CollapseNetConfig
honesty_payload = core.honesty_payload


def collapse_net_forward(
    x: Tensor,
    params: object | None = None,
    *,
    config: CollapseNetConfig | None = None,
) -> Tensor:
    xs = [float(v) for v in x.reshape(-1).tolist()]
    ys = [core.collapse_net_forward(v, params, config=config) for v in xs]
    return x.new_tensor(ys).reshape(x.shape)


def collapse_remainder(x: Tensor, *, config: CollapseNetConfig | None = None) -> Tensor:
    xs = [float(v) for v in x.reshape(-1).tolist()]
    ys = [core.collapse_remainder(v, config=config) for v in xs]
    return x.new_tensor(ys).reshape(x.shape)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "CollapseNetConfig",
    "DISCLAIMER",
    "collapse_net_forward",
    "collapse_remainder",
    "honesty_payload",
    "worked_example",
]
