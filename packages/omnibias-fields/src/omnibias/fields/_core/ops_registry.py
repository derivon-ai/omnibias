# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Lightweight registry for user-extensible ops.

The omnibias-pinn ops surface is intentionally fixed for v0.1 (the dozen
operators listed in section 4.2 of the plan). The registry exists to give
third-party code a clean extension point: a user can write

    >>> from omnibias.fields import ops_registry
    >>> @ops_registry.register("symmetric_laplacian")
    ... def _sym_lap(state, name):
    ...     ...

and afterwards ``state.u.symmetric_laplacian`` will work via the
:class:`ComponentView` ``__getattr__`` fallback. The registry stores
plain callables; the backend ops modules consult it when the view's
attribute lookup misses a built-in property.

The registry is a process-global singleton. We intentionally do *not* try
to namespace by backend: extensions are expected to be backend-pure
Python functions calling into ``state.ops.*`` themselves, which already
dispatches to the right backend.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: The single global registry. Keys are op names (lowercase, no whitespace).
_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: register a new op under ``name``.

    Examples
    --------
    >>> @register("symmetric_laplacian")
    ... def sym_lap(state, name):
    ...     return state.ops.laplacian(state, name)
    """
    if not isinstance(name, str) or not name:
        raise ValueError("op name must be a non-empty string")
    if " " in name or name != name.lower():
        raise ValueError(
            f"op names must be lowercase with no whitespace; got {name!r}"
        )

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"op {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def lookup(name: str) -> Callable[..., Any] | None:
    """Return the registered op or ``None`` if absent."""
    return _REGISTRY.get(name)


def unregister(name: str) -> None:
    """Remove a registered op (mostly used in tests)."""
    _REGISTRY.pop(name, None)


def list_registered() -> tuple[str, ...]:
    """All currently-registered op names."""
    return tuple(sorted(_REGISTRY))


def clear() -> None:
    """Drop every registration (test cleanup)."""
    _REGISTRY.clear()


__all__ = ["clear", "list_registered", "lookup", "register", "unregister"]
