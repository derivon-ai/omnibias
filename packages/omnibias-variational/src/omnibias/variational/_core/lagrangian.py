# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic Lagrangian schemas.

A :class:`Lagrangian` describes a point-mechanics Lagrangian ``L(q, qdot, t)``
by a *callable* plus the metadata that binds it to a typed omnibias field: the
generalized-coordinate component names ``dof`` and the name of the time axis.
A :class:`LagrangianDensity` is the field-theory analogue ``L(phi, d phi, x)``.

The callables receive backend tensors (torch / jax) and must be written with
last-axis reductions so they work for any leading batch shape (a single sample
``(n_dof,)`` and a batch ``(B, n_dof)`` alike) -- the backend ops rely on this
to take the Lagrangian's partials with ``vmap`` + ``jacrev``.

This module is pure Python (no torch / jax): it only stores the callable and
metadata, exactly like :class:`omnibias.core.spec.ActivationSpec` and
:class:`omnibias.geometry.MetricSpec`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

#: ``L(q, qdot, t)`` (first order) or, for ``order = n``,
#: ``L(q, q^(1), ..., q^(n), t)``: every derivative argument has shape
#: ``(..., n_dof)`` and ``t`` has shape ``(..., 1)`` -> Lagrangian of shape
#: ``(...,)``. The callable therefore takes ``order + 1`` derivative arguments
#: followed by ``t``.
LagrangianFn = Callable[..., Any]

#: ``L(phi, dphi, x)``: ``phi`` of shape ``(..., n_fields)``, ``dphi`` of shape
#: ``(..., n_fields, n_coords)``, ``x`` of shape ``(..., n_coords)`` -> density
#: of shape ``(...,)``.
LagrangianDensityFn = Callable[[Any, Any, Any], Any]


def _as_names(names: Sequence[str], what: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if not isinstance(n, str) or not n:
            raise ValueError(f"{what} must be non-empty strings, got {n!r}")
        if n in seen:
            raise ValueError(f"duplicate {what} name: {n!r}")
        seen.add(n)
        out.append(n)
    if not out:
        raise ValueError(f"{what} must be non-empty")
    return tuple(out)


@dataclass(frozen=True)
class Lagrangian:
    """A point-mechanics Lagrangian ``L(q, qdot, t)`` (or higher order).

    Parameters
    ----------
    fn
        Callable ``fn(q, qdot, t) -> L`` for the default first order. For
        ``order = n`` the signature is ``fn(q, q^(1), ..., q^(n), t)`` -- that
        is, ``order + 1`` derivative arguments (positions through the ``n``-th
        time derivative) followed by ``t``. Every derivative argument has shape
        ``(..., n_dof)`` (``n_dof == len(dof)``); ``t`` has shape ``(..., 1)``.
        The output is the scalar Lagrangian of shape ``(...,)``. Write it with
        last-axis reductions (e.g. ``(qdot ** 2).sum(-1)``) so it is
        ``vmap``/``jacrev``-compatible on a single sample.
    dof
        Ordered names of the generalized-coordinate components (the field
        components holding ``q(t)``).
    time_axis
        Name of the trajectory's time axis (default ``"t"``); the derivatives
        ``q^(k)`` are taken along it with the closed-form derivative op.
    order
        Highest time-derivative order the Lagrangian depends on (default ``1``).
        ``order = 2`` is an acceleration-dependent Lagrangian (Pais-Uhlenbeck,
        Euler-Bernoulli), whose Euler-Poisson equation is fourth order; the
        closed-form tower supplies the required ``q`` up to order ``2 * order``.
    """

    fn: LagrangianFn
    dof: tuple[str, ...]
    time_axis: str = "t"
    order: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "dof", _as_names(self.dof, "dof"))
        if not callable(self.fn):
            raise TypeError("Lagrangian.fn must be callable")
        if not isinstance(self.time_axis, str) or not self.time_axis:
            raise ValueError("time_axis must be a non-empty string")
        if not isinstance(self.order, int) or isinstance(self.order, bool) or self.order < 1:
            raise ValueError(f"order must be an integer >= 1, got {self.order!r}")

    @property
    def n_dof(self) -> int:
        """Number of generalized coordinates."""
        return len(self.dof)


@dataclass(frozen=True)
class LagrangianDensity:
    """A classical field-theory Lagrangian density ``L(phi, d phi, x)``.

    Parameters
    ----------
    fn
        Callable ``fn(phi, dphi, x) -> L``. ``phi`` has shape
        ``(..., n_fields)``; ``dphi`` has shape ``(..., n_fields, n_coords)``
        with ``dphi[..., a, mu] = d phi_a / d x_mu`` over *all* coordinate axes
        (the metric, if any, is baked into ``fn``); ``x`` has shape
        ``(..., n_coords)``. The output is the density of shape ``(...,)``.
    fields
        Ordered names of the field components ``phi_a``.
    """

    fn: LagrangianDensityFn
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _as_names(self.fields, "fields"))
        if not callable(self.fn):
            raise TypeError("LagrangianDensity.fn must be callable")

    @property
    def n_fields(self) -> int:
        """Number of field components."""
        return len(self.fields)


__all__ = [
    "Lagrangian",
    "LagrangianDensity",
    "LagrangianDensityFn",
    "LagrangianFn",
]
