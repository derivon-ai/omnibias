# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Characteristic-Net (torch; theory 09-08).

Time integral is the window knob. ``v`` jets are founding bias
collapse (``delta -> 0``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Callable

from omnibias.pinn import characteristic as core
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
CharacteristicConfig = core.CharacteristicConfig
honesty_payload = core.honesty_payload


def characteristic_eval(
    x: Tensor,
    t: Tensor | float,
    v_fn: Callable[[float], float],
    u0_fn: Callable[[float], float],
    *,
    config: CharacteristicConfig | None = None,
) -> tuple[Tensor, Tensor]:
    xs = [float(v) for v in x.reshape(-1).tolist()]
    tt = float(t.reshape(-1).tolist()[0]) if isinstance(t, Tensor) else float(t)
    us: list[float] = []
    flags: list[float] = []
    for xi in xs:
        u, crossed = core.characteristic_eval(xi, tt, v_fn, u0_fn, config=config)
        us.append(u)
        flags.append(1.0 if crossed else 0.0)
    return (
        x.new_tensor(us).reshape(x.shape),
        x.new_tensor(flags).reshape(x.shape),
    )


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "CharacteristicConfig",
    "characteristic_eval",
    "honesty_payload",
    "worked_example",
]
