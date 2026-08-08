# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-free operator-learning schemas.

An :class:`OperatorSpec` names the query coordinate system, the output
components, the sensor count ``m`` and the trunk basis width ``p``. It is the
immutable metadata every DeepONet / FNO construction carries; backends specialise
the tensor type but share this description.

Optional :class:`~omnibias.pinn.operator._core.conditioning.ConditioningSpec`
extends the branch input beyond function sensors to PDE parameters, boundary
encodings, and geometry probes. When omitted, behaviour matches the
function-sensors-only v1 surface.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.pinn.operator._core.conditioning import ConditioningSpec


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
        Number of sensors ``m`` at which the input function is observed. Kept
        for backward compatibility; equals
        ``conditioning.n_function_sensors``.
    trunk_width
        Trunk basis width ``p``. The operator is
        ``G(u)(y) = b_0 + sum_{k=1}^{p} b_k(u) t_k(y)``.
    conditioning
        Optional multi-head conditioning. Defaults to function-sensors-only
        with ``n_function_sensors = n_sensors``.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    n_sensors: int
    trunk_width: int
    conditioning: ConditioningSpec | None = None

    def __post_init__(self) -> None:
        if self.n_sensors < 1:
            raise ValueError(f"n_sensors must be >= 1, got {self.n_sensors}")
        if self.trunk_width < 1:
            raise ValueError(f"trunk_width must be >= 1, got {self.trunk_width}")
        if self.components.n_components < 1:
            raise ValueError("components must contain at least one name")
        if self.conditioning is None:
            object.__setattr__(
                self,
                "conditioning",
                ConditioningSpec.function_only(self.n_sensors),
            )
        else:
            if self.conditioning.n_function_sensors != self.n_sensors:
                raise ValueError(
                    f"conditioning.n_function_sensors="
                    f"{self.conditioning.n_function_sensors} must equal "
                    f"n_sensors={self.n_sensors}"
                )

    @property
    def n_components(self) -> int:
        return int(self.components.n_components)

    @property
    def ndim(self) -> int:
        return int(self.coordinate_spec.ndim)

    @property
    def branch_input_dim(self) -> int:
        """Total concatenated branch-input dimension."""
        assert self.conditioning is not None
        return self.conditioning.total_dim


__all__ = ["OperatorSpec"]
