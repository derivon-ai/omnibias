# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parameter-space mixed jets (torch; theory 09-27).

founding bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two. Not a ParamPINN package.
"""

from __future__ import annotations

from omnibias.core import parameter_jets as core
from torch import Tensor

DISCLAIMER = core.DISCLAIMER
ParameterJetSpec = core.ParameterJetSpec
honesty_payload = core.honesty_payload


def mixed_jet(
    field: object | None,
    coords: Tensor | tuple[float, float],
    parameters: Tensor | float,
    *,
    spec: ParameterJetSpec | None = None,
    wave: float = 1.0,
) -> float:
    if isinstance(coords, Tensor):
        flat = coords.reshape(-1)
        pair = (float(flat[0].detach()), float(flat[1].detach()))
    else:
        pair = (float(coords[0]), float(coords[1]))
    mu = (
        float(parameters.reshape(-1)[0].detach())
        if isinstance(parameters, Tensor)
        else float(parameters)
    )
    return core.mixed_jet(field, pair, mu, spec=spec, wave=wave)


def worked_example() -> dict[str, float]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "ParameterJetSpec",
    "honesty_payload",
    "mixed_jet",
    "worked_example",
]
