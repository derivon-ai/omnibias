# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""FTC-Net + dual-FTC loss (jax; theory 09-03 / 09-17).

The cell is the ``integral`` role. The derivative head is FTC of the
same window. The founding bias collapse (``delta -> 0``) is the
collapse head ``I / delta``. Temperature collapse (``beta -> inf``,
feasibility) does not appear. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.core import ftc as core_ftc

import jax.numpy as jnp
from jax import Array

DISCLAIMER = core_ftc.DISCLAIMER
DualFTCConfig = core_ftc.DualFTCConfig
FTCNetConfig = core_ftc.FTCNetConfig
honesty_payload = core_ftc.honesty_payload
worked_example = core_ftc.worked_example

__doc__ += f"\n\n{DISCLAIMER}\n"


def ftc_block(
    x: Array,
    w: Array,
    b_lo: Array,
    b_hi: Array,
    *,
    activation: str = "sigmoid",
) -> tuple[Array, Array, Array]:
    """Bit-identical to :func:`omnibias.core.ftc.ftc_block` on scalars."""
    xs = jnp.asarray(x).reshape(-1).tolist()
    ws = jnp.asarray(w).reshape(-1).tolist()
    los = jnp.asarray(b_lo).reshape(-1).tolist()
    his = jnp.asarray(b_hi).reshape(-1).tolist()
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
    dtype = jnp.asarray(x).dtype
    shape = jnp.asarray(x).shape
    if shape == ():
        return (
            jnp.asarray(integrals[0], dtype=dtype),
            jnp.asarray(derivs[0], dtype=dtype),
            jnp.asarray(collapses[0], dtype=dtype),
        )
    return (
        jnp.asarray(integrals, dtype=dtype).reshape(shape),
        jnp.asarray(derivs, dtype=dtype).reshape(shape),
        jnp.asarray(collapses, dtype=dtype).reshape(shape),
    )


class FTCNet:
    """Config holder; evaluation is the closed-form pack sum in ``omnibias.core.ftc``."""

    def __init__(self, config: FTCNetConfig | None = None) -> None:
        self.config = FTCNetConfig() if config is None else config


def dual_ftc_loss(
    I_vals: Sequence[float] | Array,
    dI_vals: Sequence[float] | Array,
    f_vals: Sequence[float] | Array,
    F_vals: Sequence[float] | Array,
    *,
    config: DualFTCConfig | None = None,
    I_a: float | None = None,
) -> core_ftc.DualFTCResult:
    def _as_list(v: Sequence[float] | Array) -> list[float]:
        if hasattr(v, "reshape"):
            return [float(x) for x in jnp.asarray(v).reshape(-1).tolist()]
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
