# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free operator-learning schemas.

An :class:`OperatorSpec` names the query coordinate system, the output
components, the sensor count ``m`` and the trunk basis width ``p``. It is the
immutable metadata every DeepONet / FNO construction carries; backends specialise
the tensor type but share this description.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec


@dataclass(frozen=True)
class OperatorSpec:
    """Immutable description of a neural operator.

    Parameters
    ----------
    coordinate_spec
        Query-coordinate axes the operator outputs a field over (e.g.
        ``("x", "t")`` for a space-time slab). This is the trunk's input.
    components
        Output field components (e.g. ``("u",)``).
    n_sensors
        Number of sensors ``m`` at which the input function is observed. The
        branch receives an ``(F, m)`` (or ``(F, m, C_in)``) array.
    trunk_width
        Trunk basis width ``p``. The operator is
        ``G(u)(y) = b_0 + sum_{k=1}^{p} b_k(u) t_k(y)``.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    n_sensors: int
    trunk_width: int

    def __post_init__(self) -> None:
        if self.n_sensors < 1:
            raise ValueError(f"n_sensors must be >= 1, got {self.n_sensors}")
        if self.trunk_width < 1:
            raise ValueError(f"trunk_width must be >= 1, got {self.trunk_width}")
        if self.components.n_components < 1:
            raise ValueError("components must contain at least one name")

    @property
    def n_components(self) -> int:
        return int(self.components.n_components)

    @property
    def ndim(self) -> int:
        return int(self.coordinate_spec.ndim)


__all__ = ["OperatorSpec"]
