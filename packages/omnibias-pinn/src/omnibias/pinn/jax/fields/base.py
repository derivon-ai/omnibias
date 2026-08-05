# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX :class:`FieldBase`.

Concrete fields subclass this and add their per-architecture parameters
plus a ``_pre_activations(coords)`` method. The base implements the
generic boilerplate of producing a :class:`FieldState`.

The base class does *not* register itself as a pytree node -- doing so
would force every subclass to declare its leaves via
:meth:`tree_flatten`/`tree_unflatten`. Concrete subclasses make that
choice (the v0.1 fields all register as pytrees so they survive a
``jax.jit`` of their forward pass).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.sigma_cache import SigmaCache
from omnibias.pinn._core.state import FieldState

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType


def _import_jax_ops() -> ModuleType:
    from omnibias.pinn.jax import _ops_dispatch
    return _ops_dispatch


class FieldBase:
    """Abstract typed PINN field for the JAX backend."""

    coordinate_spec: CoordinateSpec
    components: ComponentSpec

    def _pre_activations(self, coords: Array) -> Array | None:
        raise NotImplementedError(
            f"{type(self).__name__} must override _pre_activations"
        )

    def _make_sigma_cache(self, coords: Array) -> SigmaCache[Array]:
        z = self._pre_activations(coords)
        return SigmaCache(z=z if z is not None else coords)

    def evaluate(self, coords: Array) -> FieldState[Array]:
        coords = jnp.asarray(coords)
        if coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        cache = self._make_sigma_cache(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=cache,
        )

    def __call__(self, coords: Array) -> FieldState[Array]:
        return self.evaluate(coords)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"axes={self.coordinate_spec.axes}, "
            f"components={self.components.names})"
        )


__all__ = ["FieldBase"]
