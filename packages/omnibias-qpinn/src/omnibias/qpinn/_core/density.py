# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Split-real encoding of a finite-dimensional density matrix.

A ``d x d`` complex Hermitian matrix ``rho`` is stored as ``2 d**2``
real :class:`ComponentSpec` channels

``{name}_re_{i}_{j}`` / ``{name}_im_{i}_{j}``

grouped under ``name``. Real-channel encoding lets existing
``omnibias.pinn`` fields be used as-is. The GKSL residual consumes this
encoding; the hard cage in ``omnibias.qpinn.*.cage.density`` interprets
the same names as a generator ``G`` and exposes ``rho = G G^dag / Tr``.

The GKSL form is a caller input. Not a Born–Markov derivation.

Do not conflate this with founding bias collapse (``delta -> 0``) or
temperature collapse (``beta -> inf``, feasibility sense).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from omnibias.pinn._core.components import ComponentSpec

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.pinn._core.state import FieldState

T = TypeVar("T")


def rho_entry_names(name: str, i: int, j: int) -> tuple[str, str]:
    """Return ``(re_name, im_name)`` for entry ``rho_{ij}``."""
    return f"{name}_re_{i}_{j}", f"{name}_im_{i}_{j}"


def parse_rho_entry_name(component: str, *, group: str) -> tuple[str, int, int]:
    """Parse ``{group}_{re|im}_{i}_{j}`` into ``(part, i, j)``."""
    prefix = f"{group}_"
    if not component.startswith(prefix):
        raise ValueError(f"{component!r} is not a {group!r} density-matrix channel")
    rest = component[len(prefix) :]
    parts = rest.split("_")
    if len(parts) != 3 or parts[0] not in {"re", "im"}:
        raise ValueError(f"cannot parse density-matrix channel {component!r}")
    return parts[0], int(parts[1]), int(parts[2])


def make_rho_components(name: str = "rho", *, dim: int = 2) -> ComponentSpec:
    """Build a :class:`ComponentSpec` carrying a ``dim x dim`` density matrix."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"name must be a non-empty string, got {name!r}")
    if not isinstance(dim, int) or dim < 1:
        raise ValueError(f"dim must be a positive int, got {dim!r}")
    names: list[str] = []
    groups: dict[str, tuple[str, ...]] = {}
    for i in range(dim):
        for j in range(dim):
            re_name, im_name = rho_entry_names(name, i, j)
            names.extend([re_name, im_name])
            groups[f"{name}_{i}_{j}"] = (re_name, im_name)
    groups[name] = tuple(names)
    return ComponentSpec(names=tuple(names), groups=groups)


@dataclass(frozen=True)
class LindbladSpec:
    """Backend-agnostic GKSL data consumed by the qpinn residual."""

    hamiltonian: tuple[tuple[complex, ...], ...]
    jumps: tuple[tuple[tuple[complex, ...], ...], ...]
    rates: tuple[float, ...]
    dim: int
    group: str = "rho"


def is_rho_group(state: FieldState[Any], group: str, *, dim: int) -> bool:
    """Return whether ``group`` carries ``2 dim**2`` real density channels."""
    components = state.components
    if not components.is_group(group):
        return False
    return len(components.group_members(group)) == 2 * dim * dim


__all__ = [
    "LindbladSpec",
    "is_rho_group",
    "make_rho_components",
    "parse_rho_entry_name",
    "rho_entry_names",
]
