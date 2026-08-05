# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Activation registry (Keras backend): :class:`ActivationSpec` and lookup.

The :class:`ActivationSpec` dataclass is defined generically in
:mod:`omnibias.core.spec`; this module re-exports it and maintains a
process-wide registry so :func:`get_activation` can resolve string names
like ``"tanh"`` to their full spec. The kernels stored in each spec are
written against ``keras.ops`` so they run on any Keras backend.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.spec import ActivationSpec

_REGISTRY: dict[str, ActivationSpec[Any]] = {}


def register_activation(spec: ActivationSpec[Any]) -> ActivationSpec[Any]:
    """Register an :class:`ActivationSpec` under its name and aliases."""
    if not spec.name:
        raise ValueError("ActivationSpec.name must be a non-empty string.")
    name = spec.name.lower()
    if name in _REGISTRY and _REGISTRY[name] is not spec:
        raise ValueError(
            f"Activation {name!r} is already registered with a different spec."
        )
    _REGISTRY[name] = spec
    for alias in spec.aliases:
        akey = alias.lower()
        if akey in _REGISTRY and _REGISTRY[akey] is not spec:
            raise ValueError(
                f"Alias {alias!r} for activation {name!r} clashes with an existing entry."
            )
        _REGISTRY[akey] = spec
    return spec


def get_activation(name: str | ActivationSpec[Any]) -> ActivationSpec[Any]:
    """Resolve a name (or pass-through spec) to an :class:`ActivationSpec`."""
    if isinstance(name, ActivationSpec):
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"Expected str or ActivationSpec, got {type(name).__name__}: {name!r}"
        )
    key = name.lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(set(_REGISTRY.keys())))
        raise KeyError(f"Unknown activation {name!r}. Known: {known}.")
    return _REGISTRY[key]


def list_activations() -> list[str]:
    """Return the canonical (non-alias) activation names, sorted."""
    return sorted({spec.name for spec in _REGISTRY.values()})


def is_registered(name: str) -> bool:
    """True if ``name`` (or one of its aliases) is in the registry."""
    return isinstance(name, str) and name.lower() in _REGISTRY


__all__ = [
    "ActivationSpec",
    "get_activation",
    "is_registered",
    "list_activations",
    "register_activation",
]
