# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Solution objects returned by the torch drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from omnibias.pinn.solver._core.system import System
from torch import Tensor


def _as_coords(field_obj: Any, coords: Any) -> Tensor:
    dtype = field_obj.W.weight.dtype
    device = field_obj.W.weight.device
    if isinstance(coords, Tensor):
        return coords.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(coords), dtype=dtype, device=device)


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
        """Evaluate the solution components at ``coords`` (numpy or tensor)."""
        t = _as_coords(self.field, coords)
        state = self.field(t)
        if name is not None:
            return state.ops.value(state, name)
        return {n: state.ops.value(state, n) for n in self.component_names()}

    def __call__(self, coords: Any, name: str | None = None) -> Any:
        return self.evaluate(coords, name)


@dataclass
class GridSolution:
    """A method-of-lines solution: snapshots on a periodic grid over time."""

    times: Any            # (T,) array of times
    x: Any                # (N,) spatial grid
    values: dict[str, Any]  # component name -> (T, N) snapshots
    method: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def final(self, name: str) -> Any:
        return self.values[name][-1]

    def at(self, name: str, t_index: int) -> Any:
        return self.values[name][t_index]


__all__ = ["FieldSolution", "GridSolution"]
