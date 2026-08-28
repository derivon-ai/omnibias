# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Kantorovich homotopy continuation (jax; theory 09-20).

Each knot is an 08-04 accept. Jets / ``sigma''`` use founding bias
collapse (``delta -> 0``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. do not conflate the two.
"""

from __future__ import annotations

from omnibias.core import homotopy as core

from jax import Array

DISCLAIMER = core.DISCLAIMER
HomotopyConfig = core.HomotopyConfig
HomotopyReport = core.HomotopyReport
honesty_payload = core.honesty_payload


def homotopy_train(
    h_fn: object | None = None,
    theta0: Array | float = 1.0,
    *,
    config: HomotopyConfig | None = None,
    family: str = "quadratic",
) -> HomotopyReport:
    """Returns ``{theta, tau_final, halted, balls}``. Halted if 08-04 rejects."""
    if isinstance(theta0, Array):
        start = float(theta0.reshape(-1)[0])
    else:
        start = float(theta0)
    return core.homotopy_train(h_fn, start, config=config, family=family)


def homotopy_step(
    theta: Array | float,
    tau: float,
    *,
    config: HomotopyConfig | None = None,
    family: str = "quadratic",
) -> tuple[float, float, object]:
    if isinstance(theta, Array):
        start = float(theta.reshape(-1)[0])
    else:
        start = float(theta)
    return core.homotopy_step(start, tau, config=config, family=family)


def worked_example() -> dict[str, float | bool]:
    return core.worked_example()


__all__ = [
    "DISCLAIMER",
    "HomotopyConfig",
    "HomotopyReport",
    "homotopy_step",
    "homotopy_train",
    "honesty_payload",
    "worked_example",
]
