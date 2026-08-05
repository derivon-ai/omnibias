# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Solution objects returned by the jax drivers (twins of the torch ones)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from omnibias.pinn.solver._core.system import System


def _as_coords(field_obj: Any, coords: Any) -> Any:
    return jnp.asarray(coords, dtype=field_obj.W.dtype)


@dataclass
class FieldSolution:
    """A fitted field ansatz plus solve diagnostics (collocation drivers)."""

    field: Any
    system: System
    residual_norm: float
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def component_names(self) -> tuple[str, ...]:
        return self.system.component_names()

    def evaluate(self, coords: Any, name: str | None = None) -> Any:
        state = self.field(_as_coords(self.field, coords))
        if name is not None:
            return state.ops.value(state, name)
        return {n: state.ops.value(state, n) for n in self.component_names()}

    def __call__(self, coords: Any, name: str | None = None) -> Any:
        return self.evaluate(coords, name)


@dataclass
class GridSolution:
    """A method-of-lines solution: snapshots on a periodic grid over time."""

    times: Any
    x: Any
    values: dict[str, Any]
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def final(self, name: str) -> Any:
        return self.values[name][-1]

    def at(self, name: str, t_index: int) -> Any:
        return self.values[name][t_index]


__all__ = ["FieldSolution", "GridSolution"]
