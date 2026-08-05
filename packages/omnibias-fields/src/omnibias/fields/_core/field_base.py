# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Abstract :class:`FieldBase` interface (backend-agnostic part only).

Each backend implements its own concrete subclass that inherits from
``FieldBase`` *and* the backend's neural-network base class
(``torch.nn.Module`` / ``equinox.Module``). The reason ``FieldBase`` is
not an ABC by itself is that the concrete subclasses already inherit
from a heavyweight base class (e.g. ``nn.Module``), and Python multiple
inheritance prefers thin protocols over ABC injection.

This module defines the structural contract a typed PINN field obeys:

- :meth:`evaluate` (alias :meth:`__call__`) takes a runtime ``coords``
  tensor and returns a :class:`FieldState`.
- :attr:`coordinate_spec` and :attr:`components` expose the immutable
  metadata for the field.

The class is small on purpose -- the per-backend file does the heavy
lifting (parameter management, the actual closed-form evaluation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.components import ComponentSpec
    from omnibias.fields._core.coords import CoordinateSpec
    from omnibias.fields._core.state import FieldState


#: Name of the class attribute a concrete field sets to select the backend
#: ops dispatch path (e.g. ``"one_layer"``, ``"spectral"``, ``"chebyshev"``,
#: ``"cage"``). The backend ops read this marker via ``getattr`` so the
#: foundational ``omnibias-fields`` package never has to import concrete field
#: classes from a downstream package (which would create a dependency cycle).
DISPATCH_ATTR = "_omnibias_dispatch"


@runtime_checkable
class FieldBase(Protocol):
    """Structural protocol every typed field must satisfy.

    A concrete field additionally sets the class attribute named by
    :data:`DISPATCH_ATTR` to one of the dispatch tags understood by the
    backend ops (``"one_layer"`` selects the closed-form sigma-tower
    reduction; other tags select the state-method path).
    """

    @property
    def coordinate_spec(self) -> CoordinateSpec: ...

    @property
    def components(self) -> ComponentSpec: ...

    def evaluate(self, coords: Any) -> FieldState: ...

    def __call__(self, coords: Any) -> FieldState: ...


__all__ = ["DISPATCH_ATTR", "FieldBase"]
