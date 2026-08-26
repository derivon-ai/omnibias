# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Plant PID layer (torch; theory 09-29).

Exact I / D from founding bias collapse (``delta -> 0``) plus the
``integral`` role. Temperature collapse (``beta -> inf``, feasibility)
does not appear. do not conflate the two. Not 08-10. Not cruise SOTA.
"""

from __future__ import annotations

from omnibias.core import pid_layer as core

from torch import Tensor

DISCLAIMER = core.DISCLAIMER
PlantPIDConfig = core.PlantPIDConfig
PlantPIDReport = core.PlantPIDReport
honesty_payload = core.honesty_payload


def plant_pid(
    t: Tensor | float,
    *,
    config: PlantPIDConfig | None = None,
) -> PlantPIDReport:
    """Scalar plant PID. ``t`` may be a 0-dim / 1-element tensor."""
    time = float(t.reshape(-1)[0].detach()) if isinstance(t, Tensor) else float(t)
    return core.plant_pid(time, config=config)


def worked_example() -> dict[str, float | bool]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "PlantPIDConfig",
    "PlantPIDReport",
    "honesty_payload",
    "plant_pid",
    "worked_example",
]
