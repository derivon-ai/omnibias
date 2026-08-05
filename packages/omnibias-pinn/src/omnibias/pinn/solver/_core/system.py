# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The ``System`` abstraction: coupled fields + vector residual + domain + BC/IC.

A :class:`System` is the backend-agnostic statement of a (possibly coupled) PDE
problem. Its residuals are Python callables ``residual(state) -> tensor`` that
compose the closed-form differential operators exposed on a
:class:`omnibias.fields._core.state.FieldState` (``state.ops.laplacian`` etc.).
A coupled system is simply a tuple of such residuals over shared components.

The residual closures execute at *solve* time on whichever backend the driver
chose; nothing here imports torch or jax.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from omnibias.fields._core.components import ComponentSpec
from omnibias.pinn.solver._core.conditions import BoundaryCondition, InitialCondition
from omnibias.pinn.solver._core.domain import Domain
from omnibias.pinn.solver._core.taxonomy import (
    Arity,
    Classification,
    Linearity,
    PDEType,
    ProblemKind,
)
from omnibias.pinn.solver._core.unknowns import Unknown

#: A residual reads a ``FieldState`` and returns a ``(B,)`` tensor to drive to 0.
Residual = Callable[[Any], Any]


@dataclass(frozen=True)
class Field:
    """One scalar field (output component) of a system.

    Grouping several fields into a named vector group (e.g. a velocity) is
    supported via :class:`~omnibias.fields._core.components.ComponentSpec`
    groups; ``group`` records the group this field belongs to, if any.
    """

    name: str
    group: str | None = None


@dataclass(frozen=True)
class System:
    """A coupled PDE system.

    Parameters
    ----------
    domain
        The :class:`Domain` the problem lives on.
    fields
        The scalar fields (output components), in order.
    residuals
        One residual callable per governing equation. Each maps a
        ``FieldState`` to a ``(B,)`` residual tensor to be driven to zero.
    boundary, initial
        Boundary / initial conditions.
    pde_type, linearity
        Declared classification (the canonical builders set these honestly).
    name
        Optional human-readable label.
    unknowns
        Coefficients to be *recovered* rather than supplied (see
        :class:`~omnibias.pinn.solver._core.unknowns.Unknown`). Empty for a forward
        problem, which is the default, so existing call sites are unaffected.
    """

    domain: Domain
    fields: tuple[Field, ...]
    residuals: tuple[Residual, ...]
    boundary: tuple[BoundaryCondition, ...] = ()
    initial: tuple[InitialCondition, ...] = ()
    pde_type: PDEType = PDEType.ELLIPTIC
    linearity: Linearity = Linearity.LINEAR
    name: str = ""
    groups: tuple[tuple[str, tuple[str, ...]], ...] = field(default=())
    unknowns: tuple[Unknown, ...] = ()

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("System requires at least one field")
        if not self.residuals:
            raise ValueError("System requires at least one residual")
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate field names: {names!r}")
        known = set(names)
        for bc in self.boundary:
            if bc.component not in known:
                raise ValueError(
                    f"boundary condition references unknown component {bc.component!r}"
                )
        for ic in self.initial:
            if ic.component not in known:
                raise ValueError(
                    f"initial condition references unknown component {ic.component!r}"
                )
        if self.initial and not self.domain.is_time_dependent:
            raise ValueError(
                "initial conditions given but the domain has no time axis"
            )
        unknown_names = [u.name for u in self.unknowns]
        if len(set(unknown_names)) != len(unknown_names):
            raise ValueError(f"duplicate unknown names: {unknown_names!r}")

    # -- structure ----------------------------------------------------

    def component_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def component_spec(self) -> ComponentSpec:
        """Build the union :class:`ComponentSpec` for the field ansatz."""
        groups: dict[str, tuple[str, ...]] = {}
        for gname, members in self.groups:
            groups[gname] = members
        # also fold in per-field ``group`` memberships
        inferred: dict[str, list[str]] = {}
        for f in self.fields:
            if f.group is not None:
                inferred.setdefault(f.group, []).append(f.name)
        for gname, inferred_members in inferred.items():
            groups.setdefault(gname, tuple(inferred_members))
        return ComponentSpec(
            names=self.component_names(),
            groups=groups or None,
        )

    def is_time_dependent(self) -> bool:
        return self.domain.is_time_dependent

    def is_inverse(self) -> bool:
        """Whether any coefficient is an :class:`Unknown` to be recovered.

        Deliberately *not* folded into
        :class:`~omnibias.pinn.solver._core.taxonomy.ProblemKind`: forward-vs-inverse
        is orthogonal to boundary-value-vs-initial-value (an inverse heat problem
        is still an IVP), so adding an ``INVERSE`` member would mix two
        independent axes into one enum and lose the well-posedness information
        that drives solver dispatch. ``unknowns`` is the single source of truth.
        """
        return bool(self.unknowns)

    def unbound_unknowns(self) -> tuple[Unknown, ...]:
        """Unknowns with no value bound in the current context.

        A forward driver must refuse these: with nothing bound there is no
        coefficient to evaluate, and guessing one would solve a different PDE than
        the caller asked about.
        """
        return tuple(u for u in self.unknowns if not u.is_bound())

    def require_bound_coefficients(self, driver: str) -> None:
        """Guard a *forward* driver against coefficients that have no value.

        Solving at each :attr:`Unknown.initial` guess would return a confident
        answer to a different PDE than the caller described, so the forward
        drivers stop here and name the two honest options.
        """
        missing = self.unbound_unknowns()
        if not missing:
            return
        names = ", ".join(repr(u.name) for u in missing)
        raise ValueError(
            f"{driver} is a forward driver but {names} "
            f"{'is' if len(missing) == 1 else 'are'} unknown; call solve_inverse "
            "with observations to recover them, or wrap the call in "
            "bind_unknowns({...}) to pin them to known values"
        )

    def classify(self) -> Classification:
        kind = (
            ProblemKind.INITIAL_VALUE
            if self.is_time_dependent()
            else ProblemKind.BOUNDARY_VALUE
        )
        arity = Arity.SYSTEM if len(self.fields) > 1 else Arity.SCALAR
        return Classification(
            pde_type=self.pde_type,
            linearity=self.linearity,
            kind=kind,
            arity=arity,
        )

    def __repr__(self) -> str:
        unknowns = (
            f", unknowns={tuple(u.name for u in self.unknowns)!r}"
            if self.unknowns
            else ""
        )
        return (
            f"System(name={self.name!r}, "
            f"fields={self.component_names()!r}, "
            f"class={self.classify()}{unknowns})"
        )


def make_system(
    *,
    domain: Domain,
    fields: Sequence[str | Field],
    residuals: Sequence[Residual],
    boundary: Sequence[BoundaryCondition] = (),
    initial: Sequence[InitialCondition] = (),
    pde_type: PDEType = PDEType.ELLIPTIC,
    linearity: Linearity = Linearity.LINEAR,
    name: str = "",
    groups: dict[str, Sequence[str]] | None = None,
    unknowns: Sequence[Unknown] = (),
) -> System:
    """Ergonomic constructor accepting bare field-name strings."""
    field_objs = tuple(
        f if isinstance(f, Field) else Field(name=f) for f in fields
    )
    group_t = (
        tuple((g, tuple(m)) for g, m in groups.items()) if groups else ()
    )
    return System(
        domain=domain,
        fields=field_objs,
        residuals=tuple(residuals),
        boundary=tuple(boundary),
        initial=tuple(initial),
        pde_type=pde_type,
        linearity=linearity,
        name=name,
        groups=group_t,
        unknowns=tuple(unknowns),
    )


__all__ = ["Field", "Residual", "System", "make_system"]
