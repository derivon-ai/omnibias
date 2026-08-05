# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Activation registry (PyTorch backend): :class:`ActivationSpec` and global lookup.

The :class:`ActivationSpec` dataclass itself is defined generically in
:mod:`omnibias.core.spec`; this module specialises it to ``torch.Tensor``
and maintains a process-wide registry so :func:`get_activation` can resolve
string names like ``"sigmoid"`` to their full spec at construction time.

The activation modules (:mod:`smooth`, :mod:`proximal`, :mod:`classical`,
:mod:`trigonometric`, :mod:`nqs`) are imported by
:mod:`omnibias.torch.activations.__init__` for side-effect registration.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

# Re-export the backend-agnostic spec so existing call sites keep working.
from omnibias.core.spec import ActivationSpec

from torch import Tensor

#: Public alias so callers do not need to import ``torch.Tensor`` directly
#: when only the type signature matters.
TensorFn = Callable[[Tensor], Tensor]
NthDerivativeFn = Callable[[Tensor, int], Tensor]
IntegralFn = Callable[[Tensor], Tensor]

#: Type alias documenting that the torch backend pins ``TensorT`` to
#: :class:`torch.Tensor`. Equivalent to ``ActivationSpec[torch.Tensor]``.
TorchActivationSpec: TypeAlias = ActivationSpec[Tensor]


_REGISTRY: dict[str, ActivationSpec[Tensor]] = {}


def register_activation(spec: ActivationSpec[Tensor]) -> ActivationSpec[Tensor]:
    """Register an :class:`ActivationSpec` under its name and aliases.

    Returns the spec unchanged so this can be used as a decorator-style
    expression.
    """
    if not spec.name:
        raise ValueError("ActivationSpec.name must be a non-empty string.")
    name = spec.name.lower()
    if name in _REGISTRY and _REGISTRY[name] is not spec:
        raise ValueError(f"Activation {name!r} is already registered with a different spec.")
    _REGISTRY[name] = spec
    for alias in spec.aliases:
        akey = alias.lower()
        if akey in _REGISTRY and _REGISTRY[akey] is not spec:
            raise ValueError(
                f"Alias {alias!r} for activation {name!r} clashes with an existing entry."
            )
        _REGISTRY[akey] = spec
    return spec


def get_activation(name: str | ActivationSpec[Tensor]) -> ActivationSpec[Tensor]:
    """Resolve a name (or pass-through spec) to an :class:`ActivationSpec`."""
    if isinstance(name, ActivationSpec):
        return name
    if not isinstance(name, str):
        raise TypeError(f"Expected str or ActivationSpec, got {type(name).__name__}: {name!r}")
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
    "IntegralFn",
    "NthDerivativeFn",
    "TensorFn",
    "TorchActivationSpec",
    "get_activation",
    "is_registered",
    "list_activations",
    "register_activation",
]
