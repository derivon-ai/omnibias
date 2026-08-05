# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equation registry: name -> :class:`EquationSpec`.

The backend ``equations/`` subpackages register their canonical specs
here at import time so that

    >>> from omnibias.pinn._core.registry import get_equation_spec
    >>> spec = get_equation_spec("navier_stokes")

returns a backend-agnostic descriptor (no torch / jax imports). Each
backend's actual equation *class* (``omnibias.pinn.torch.equations.NavierStokes``
etc.) is registered under the same name in a *separate* per-backend
factory registry, so the cross-backend parity tests can pull the matching
pair via name lookup.

Two registries are exposed:

- :func:`register_spec`/:func:`get_equation_spec` -- the structural specs.
- :func:`register_factory`/:func:`get_equation_factory` -- per-backend
  callable factories, keyed by ``(name, backend)`` where ``backend`` is
  ``"torch"`` or ``"jax"``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from omnibias.pinn._core.pde import EquationSpec

_SPECS: dict[str, EquationSpec] = {}
_FACTORIES: dict[tuple[str, str], Callable[..., Any]] = {}


def register_spec(spec: EquationSpec) -> EquationSpec:
    """Register an equation spec under ``spec.name``."""
    if not isinstance(spec, EquationSpec):
        raise TypeError(
            f"register_spec expects EquationSpec, got {type(spec).__name__}"
        )
    if spec.name in _SPECS and _SPECS[spec.name] != spec:
        raise ValueError(
            f"equation {spec.name!r} already registered with different spec"
        )
    _SPECS[spec.name] = spec
    return spec


def get_equation_spec(name: str) -> EquationSpec:
    if name not in _SPECS:
        raise KeyError(
            f"unknown equation {name!r}; registered: {list_equation_specs()!r}"
        )
    return _SPECS[name]


def list_equation_specs() -> tuple[str, ...]:
    return tuple(sorted(_SPECS))


def has_equation_spec(name: str) -> bool:
    return name in _SPECS


def register_factory(
    name: str,
    backend: str,
    factory: Callable[..., Any],
) -> Callable[..., Any]:
    """Register a backend-specific equation factory."""
    if not callable(factory):
        raise TypeError("factory must be callable")
    if backend not in ("torch", "jax"):
        raise ValueError(
            f"backend must be 'torch' or 'jax', got {backend!r}"
        )
    key = (name, backend)
    if key in _FACTORIES:
        raise ValueError(
            f"equation factory ({name!r}, {backend!r}) already registered"
        )
    _FACTORIES[key] = factory
    return factory


def get_equation_factory(name: str, backend: str) -> Callable[..., Any]:
    key = (name, backend)
    if key not in _FACTORIES:
        raise KeyError(
            f"no equation factory registered for ({name!r}, {backend!r}). "
            f"Registered: {list_equation_factories()!r}"
        )
    return _FACTORIES[key]


def list_equation_factories() -> tuple[tuple[str, str], ...]:
    return tuple(sorted(_FACTORIES))


def clear() -> None:
    """Drop every registration (test cleanup)."""
    _SPECS.clear()
    _FACTORIES.clear()


__all__ = [
    "clear",
    "get_equation_factory",
    "get_equation_spec",
    "has_equation_spec",
    "list_equation_factories",
    "list_equation_specs",
    "register_factory",
    "register_spec",
]
