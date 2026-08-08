# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-head branch encoder layout shared by torch / JAX DeepONet twins.

Each conditioning head is independently layer-normalised before a small MLP
maps it to a fixed ``encoder_dim``. A fusion network concatenates the active
head encodings and emits the per-sample branch coefficients. When only function
sensors are present the path reduces to function-encoder -> fusion (the v1
function-only compatibility surface).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from omnibias.pinn.operator._core.conditioning import ConditioningSpec


@dataclass(frozen=True)
class BranchHeadLayout:
    """Active head widths and fusion input size for a :class:`ConditioningSpec`."""

    spec: ConditioningSpec
    encoder_dim: int

    @property
    def n_heads(self) -> int:
        c = self.spec
        return (
            1
            + int(c.has_parameters)
            + int(c.has_boundary)
            + int(c.has_geometry)
        )

    @property
    def fusion_dim(self) -> int:
        return self.n_heads * int(self.encoder_dim)

    @property
    def head_dims(self) -> tuple[tuple[str, int], ...]:
        """``(name, width)`` pairs in fusion order."""
        c = self.spec
        out: list[tuple[str, int]] = [("function", int(c.n_function_sensors))]
        if c.has_parameters:
            out.append(("parameters", int(c.n_parameters)))
        if c.has_boundary:
            out.append(("boundary", int(c.n_boundary_sensors)))
        if c.has_geometry:
            out.append(("geometry", int(c.n_geometry_probes)))
        return tuple(out)


def validate_head_batch(
    name: str,
    tensor: Any | None,
    *,
    width: int,
    batch: int,
) -> Any | None:
    """Raise on missing / mis-shaped optional conditioning heads."""
    if width == 0:
        if tensor is not None:
            raise ValueError(f"{name} provided but conditioning width is 0")
        return None
    if tensor is None:
        raise ValueError(f"{name} required: conditioning width is {width}")
    return tensor


__all__ = ["BranchHeadLayout", "validate_head_batch"]
