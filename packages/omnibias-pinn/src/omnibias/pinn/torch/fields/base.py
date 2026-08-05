# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch :class:`FieldBase` mixin.

Concrete fields inherit from both :class:`torch.nn.Module` and this
class. The mixin holds the immutable :class:`CoordinateSpec` /
:class:`ComponentSpec` metadata and implements the boilerplate of
producing a :class:`FieldState` from a runtime ``coords`` tensor.

The closed-form math is *not* in this base class -- each subclass
provides ``_pre_activations(coords) -> Tensor`` returning the ``z``
that the sigma cache will derive from. The base class then
constructs the :class:`SigmaCache` and ties everything to the dispatch
ops module via :class:`FieldState`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.sigma_cache import SigmaCache
from omnibias.pinn._core.state import FieldState
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType


def _import_torch_ops() -> ModuleType:
    """Import the torch ops dispatch lazily to avoid circular imports."""
    from omnibias.pinn.torch import _ops_dispatch
    return _ops_dispatch


class FieldBase(torch.nn.Module):
    """Abstract typed PINN field for the torch backend.

    Subclasses must:

    - Set :attr:`coordinate_spec` and :attr:`components` in
      ``__init__`` (typically by passing them as constructor arguments).
    - Implement :meth:`_pre_activations(coords)` returning the
      pre-activation tensor ``z = W coords + beta`` that the sigma
      cache will be built around. Returning ``None`` is allowed for
      fields that don't have a single pre-activation tower (e.g.
      spectral fields). Such subclasses must override
      :meth:`evaluate` themselves.
    - Implement any backend-specific operator that is not generic
      (e.g. ``_value`` for the field's own forward pass).
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec

    def __init__(
        self,
        *,
        coordinate_spec: CoordinateSpec,
        components: ComponentSpec,
    ) -> None:
        super().__init__()
        self.coordinate_spec = coordinate_spec
        self.components = components

    def _pre_activations(self, coords: Tensor) -> Tensor | None:
        """Return ``z`` (one tensor per layer or ``None`` if not applicable)."""
        raise NotImplementedError(
            f"{type(self).__name__} must override _pre_activations"
        )

    def _make_sigma_cache(self, coords: Tensor) -> SigmaCache[Tensor]:
        """Build a fresh :class:`SigmaCache` for one evaluation."""
        z = self._pre_activations(coords)
        return SigmaCache(z=z if z is not None else coords)

    def evaluate(self, coords: Tensor) -> FieldState[Tensor]:
        """Evaluate the field at ``coords``; return a :class:`FieldState`.

        ``coords`` has shape ``(B, D)`` where ``D ==
        self.coordinate_spec.ndim``.
        """
        if coords.dim() != 2:
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
            ops=_import_torch_ops(),
            sigma_cache=cache,
        )

    def __call__(self, coords: Tensor) -> FieldState[Tensor]:  # type: ignore[override]
        return self.evaluate(coords)

    # Subclasses should override this if the spec naming should look
    # nicer in `repr(state)`.
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"axes={self.coordinate_spec.axes}, "
            f"components={self.components.names})"
        )

    def to(self, *args: Any, **kwargs: Any) -> FieldBase:  # type: ignore[override]
        """Pass-through ``nn.Module.to`` returning ``self`` typed as FieldBase."""
        return super().to(*args, **kwargs)


__all__ = ["FieldBase"]
