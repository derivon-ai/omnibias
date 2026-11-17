# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Opt-in planner routing for deep fields (``_omnibias_dispatch == "jet_mlp"``).

A deep field whose network is a dense chain can say so by defining ``planned_chain(state, names)``, which returns a
:class:`PlannedChain` (the ``(W, b, activation)`` layers of the readout of ``names``, with ``W`` of shape
``(out, in)``).  The Laplacian, bi- and poly-Laplacian and divergence operators of the field substrate then run through
the planner of ``omnibias.<backend>.experimental`` (``projected`` contractions, forward Laplacian, ... chosen by the
per-device cost model) instead of the field's own jet methods.  A field without the hook, or a hook that returns
``None``, keeps the field's own methods.  Gradients reach the layer tensors the hook returns.

The planner does not cache anything on the :class:`~omnibias.fields._core.state.FieldState`: its plans depend on shapes
only and live in a process cache, so the readout-independence contract of ``state.extra`` is untouched.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.state import FieldState

__all__ = [
    "PLANNED_CHAIN_ATTR",
    "PlannedChain",
    "planned_routing",
    "planned_routing_enabled",
    "request_chain",
]

PLANNED_CHAIN_ATTR = "planned_chain"
_ENABLED: ContextVar[bool] = ContextVar("omnibias_fields_planned_routing", default=True)


@dataclass(frozen=True)
class PlannedChain:
    """The dense chain behind the readout of some components of a deep field.

    Attributes
    ----------
    layers:
        ``[(W, b, activation), ...]`` with ``W`` of shape ``(out, in)``, ``b`` of shape ``(out,)`` or ``None`` and
        ``activation`` a registered name (or ``None`` for the linear readout).  The last layer must produce exactly the
        requested components, in the order of ``names``.
    coords:
        The ``(B, D)`` points the chain acts on; ``None`` means ``state.coords``.
    coordinates:
        The input axes the operator acts on (for example the spatial axes of a field that also takes time); ``None``
        means all of them.
    diffusion:
        Optional per-axis scale of the Laplacian metric (shape ``(len(coordinates),)``).
    """

    layers: Sequence[tuple[Any, Any, str | None]]
    coords: Any | None = None
    coordinates: Sequence[int] | None = None
    diffusion: Any | None = None


def planned_routing_enabled() -> bool:
    """Whether fields that define ``planned_chain`` are routed through the planner (default ``True``)."""
    return _ENABLED.get()


@contextmanager
def planned_routing(enabled: bool = True) -> Iterator[None]:
    """Enable or disable planner routing inside the ``with`` block (per thread and per async task)."""
    token = _ENABLED.set(bool(enabled))
    try:
        yield
    finally:
        _ENABLED.reset(token)


def request_chain(state: FieldState, names: tuple[str, ...]) -> PlannedChain | None:
    """The :class:`PlannedChain` of ``names`` if routing is enabled and the field offers one, else ``None``."""
    if not _ENABLED.get():
        return None
    hook = getattr(state.field, PLANNED_CHAIN_ATTR, None)
    if hook is None:
        return None
    chain = hook(state, tuple(names))
    if chain is None:
        return None
    if not isinstance(chain, PlannedChain):
        raise TypeError(
            f"{PLANNED_CHAIN_ATTR} must return a PlannedChain or None, got {type(chain).__name__}"
        )
    return chain
