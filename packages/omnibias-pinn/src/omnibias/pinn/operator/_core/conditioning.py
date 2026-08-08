# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Multi-head conditioning descriptors for neural operators (pure Python).

A DeepONet branch that sees only the input function cannot generalise across
PDE parameters, boundary conditions, or geometry. :class:`ConditioningSpec`
names the four heads that concatenate into the branch input; the default
recovers today's function-sensors-only behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConditioningSpec:
    """Sizes of the branch-conditioning heads.

    Parameters
    ----------
    n_function_sensors
        Sensor count ``m`` for the input function (required, >= 1).
    n_parameters
        Length of the PDE-parameter vector (diffusivity, viscosity, ...).
        ``0`` disables the head.
    n_boundary_sensors
        Encoding length for boundary / initial data. ``0`` disables.
    n_geometry_probes
        Number of SDF probe evaluations that encode the domain shape.
        ``0`` disables.
    """

    n_function_sensors: int
    n_parameters: int = 0
    n_boundary_sensors: int = 0
    n_geometry_probes: int = 0

    def __post_init__(self) -> None:
        if self.n_function_sensors < 1:
            raise ValueError(
                f"n_function_sensors must be >= 1, got {self.n_function_sensors}"
            )
        for name in ("n_parameters", "n_boundary_sensors", "n_geometry_probes"):
            val = getattr(self, name)
            if int(val) < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

    @property
    def total_dim(self) -> int:
        """Concatenated branch-input dimension."""
        return (
            int(self.n_function_sensors)
            + int(self.n_parameters)
            + int(self.n_boundary_sensors)
            + int(self.n_geometry_probes)
        )

    @property
    def has_parameters(self) -> bool:
        return self.n_parameters > 0

    @property
    def has_boundary(self) -> bool:
        return self.n_boundary_sensors > 0

    @property
    def has_geometry(self) -> bool:
        return self.n_geometry_probes > 0

    @classmethod
    def function_only(cls, n_sensors: int) -> ConditioningSpec:
        """Today's default: branch sees only the input-function sensors."""
        return cls(n_function_sensors=int(n_sensors))


__all__ = ["ConditioningSpec"]
