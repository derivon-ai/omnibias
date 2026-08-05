# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Helmholtz equation residual (jax twin)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import psi_value


@dataclass
class HelmholtzOutput:
    residual: Array
    diag: dict[str, float]


@dataclass
class Helmholtz:
    """JAX twin of :class:`omnibias.qpinn.torch.equations.Helmholtz`."""

    k: float | Callable[[FieldState], Array] = 1.0
    psi: str = "psi"
    source: Callable[[FieldState], Array] | None = None

    def __call__(self, state: FieldState) -> HelmholtzOutput:
        re_name = f"{self.psi}_re"
        im_name = f"{self.psi}_im"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )
        psi_re, psi_im = psi_value(state, self.psi)
        lap_re = state.ops.laplacian(state, re_name)
        lap_im = state.ops.laplacian(state, im_name)
        if callable(self.k):
            k_val = self.k(state)
        else:
            k_val = jnp.asarray(self.k, dtype=psi_re.dtype)
        k_sq = k_val * k_val
        res_re = lap_re + k_sq * psi_re
        res_im = lap_im + k_sq * psi_im
        residual = jnp.stack([res_re, res_im], axis=-1)
        if self.source is not None:
            residual = residual + self.source(state)
        return HelmholtzOutput(
            residual=residual,
            diag={"mean_sq_residual": jnp.mean(residual * residual)},
        )


def helmholtz(
    state: FieldState,
    *,
    k: float | Callable[[FieldState], Array] = 1.0,
    psi: str = "psi",
    source: Callable[[FieldState], Array] | None = None,
) -> HelmholtzOutput:
    """Stateless one-shot wrapper around :class:`Helmholtz`."""
    return Helmholtz(k=k, psi=psi, source=source)(state)


__all__ = ["Helmholtz", "HelmholtzOutput", "helmholtz"]
