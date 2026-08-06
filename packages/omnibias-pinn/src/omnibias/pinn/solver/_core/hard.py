# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Decide which of a system's conditions can be enforced structurally.

:func:`plan_hard_conditions` reads a :class:`~omnibias.pinn.solver._core.system.System`
and returns the subset of its boundary / initial conditions that a
:class:`~omnibias.pinn.torch.cage.constrained.ConstrainedExpressionField` can
enforce *identically*, together with a reason for every condition it declines.

The resulting :class:`HardConditionPlan` is the **single source of truth** for
both halves of the wiring: the field builder wraps the ansatz using it, and the
residual assembler drops exactly the rows it reports absorbed. Any other
arrangement lets the two disagree, and a condition that the loss stops watching
while the architecture does not enforce it would vanish silently. That failure
mode is what the plan object -- and the full-residual guard test -- exist to
prevent.

Absorption is deliberately **partial**: whatever is provably exact becomes
structural, and everything else stays in the loss exactly as before. Declining
is always safe; over-claiming is not.

This module is pure Python: it builds the backend-agnostic condition objects
from :mod:`omnibias.pinn._core.constrained` and never imports torch or jax.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from omnibias.pinn._core.constrained import (
    AxisPlan,
    HardCondition,
    certify_support_matrix,
    derivative_at,
    dirichlet,
    group_hard_conditions,
    neumann,
    periodic,
    robin,
)
from omnibias.pinn.solver._core.sampling import bc_faces
from omnibias.pinn.solver._core.system import System

#: Accepted values of the ``hard_conditions`` solver keyword.
HARD_CONDITION_MODES = ("none", "auto")

#: Derivative orders matched across a periodic seam. A smooth periodic solution
#: matches every derivative, so requiring the value *and* the slope is a subset
#: of the truth rather than an extra assumption -- and matching only the value
#: leaves the solution free to kink exactly where nothing is watching.
PERIODIC_ORDERS = (0, 1)


@dataclass(frozen=True)
class DeclinedCondition:
    """A condition the plan could not absorb, and why."""

    kind: str
    component: str
    index: int
    reason: str

    def __str__(self) -> str:
        return f"{self.kind}[{self.index}] on {self.component!r}: {self.reason}"


@dataclass(frozen=True)
class HardConditionPlan:
    """Which conditions become structural, and the evidence that they can.

    Attributes
    ----------
    conditions
        The absorbed conditions, in the form the cage consumes.
    absorbed_boundary, absorbed_initial
        Indices into ``system.boundary`` / ``system.initial`` that are now
        enforced by construction and must therefore be dropped from the loss.
        A boundary condition is listed only when *every* face it covers was
        absorbed, so partial absorption of one condition never silently drops
        the faces that were not.
    declined
        One entry per condition left in the loss, with a reason.
    certificates
        Sealed support-matrix certificates, one per constrained component-axis.
    """

    conditions: tuple[HardCondition, ...] = ()
    absorbed_boundary: frozenset[int] = frozenset()
    absorbed_initial: frozenset[int] = frozenset()
    declined: tuple[DeclinedCondition, ...] = ()
    certificates: tuple[dict[str, Any], ...] = field(default=())

    def __bool__(self) -> bool:
        return bool(self.conditions)

    @property
    def is_total(self) -> bool:
        """Whether every condition was absorbed, leaving a pure interior loss."""
        return bool(self.conditions) and not self.declined

    def summary(self) -> str:
        """One-line human summary, suitable for solver diagnostics."""
        if not self.conditions:
            return "hard conditions: none absorbed"
        # Two different counts: the linear functionals the cage enforces, and
        # the *declarations* they came from. One `BoundaryCondition` covering a
        # whole axis becomes two functionals, and saying "2 absorbed (1
        # declaration)" is less confusing than reporting either alone.
        head = (
            f"hard conditions: {len(self.conditions)} absorbed from "
            f"{len(self.absorbed_boundary)} boundary + "
            f"{len(self.absorbed_initial)} initial declaration(s)"
        )
        if not self.declined:
            return head + "; interior residual only"
        return head + f"; {len(self.declined)} left soft"


def _face_value(domain: Any, axis: str, side: str) -> float:
    lo, hi = domain.bounds[domain.coordinate_spec.axis_index(axis)]
    return float(lo if side == "lo" else hi)


def _periodic_conditions(
    system: System, bc: Any, orders: Sequence[int]
) -> tuple[list[HardCondition], str | None]:
    """Periodicity as relative constraints ``d^n u(hi) - d^n u(lo) = 0``.

    The mesh-free ansatz is not spectral, so nothing else enforces this: the
    row was dropped from the loss on the grounds that the ansatz carried it,
    which is true of the spectral method-of-lines route and false here. Tying
    the two faces together with a relative functional is what makes the claim
    true for this route as well.
    """
    domain = system.domain
    cs = domain.coordinate_spec
    names = (
        (bc.axis,)
        if bc.axis is not None
        else tuple(a for a in domain.spatial_axes if cs.is_periodic(a))
    )
    if not names:
        return [], (
            "periodic condition names no axis and the domain declares none "
            "periodic, so there is no seam to tie together"
        )
    out: list[HardCondition] = []
    for axis in names:
        index = cs.axis_index(axis)
        lo, hi = domain.bounds[index]
        out.extend(
            HardCondition(
                component=bc.component,
                axis=index,
                constraint=periodic(lo, hi, order=order, label=f"periodic^{order}@{axis}"),
                target=0.0,
            )
            for order in orders
        )
    return out, None


def _boundary_conditions(
    system: System, bc: Any, *, periodic_orders: Sequence[int]
) -> tuple[list[HardCondition], str | None]:
    """The hard conditions one :class:`BoundaryCondition` maps to, or a decline."""
    if bc.kind == "periodic":
        return _periodic_conditions(system, bc, periodic_orders)
    domain = system.domain
    out: list[HardCondition] = []
    for axis, side in bc_faces(domain, bc):
        point = _face_value(domain, axis, side)
        outward = 1.0 if side == "hi" else -1.0
        label = f"{bc.kind}@{axis}={side}"
        if bc.kind == "dirichlet":
            constraint = dirichlet(point, label=label)
        elif bc.kind == "neumann":
            constraint = neumann(point, outward=outward, label=label)
        elif bc.kind == "robin":
            constraint = robin(
                point, alpha=bc.alpha, beta=bc.beta, outward=outward, label=label
            )
        else:  # pragma: no cover -- BoundaryCondition validates its kinds
            return [], f"unsupported boundary kind {bc.kind!r}"
        out.append(
            HardCondition(
                component=bc.component,
                axis=domain.coordinate_spec.axis_index(axis),
                constraint=constraint,
                target=bc.value,
            )
        )
    if not out:
        return [], "covers no non-periodic face"
    return out, None


def _initial_condition(system: System, ic: Any) -> tuple[list[HardCondition], str | None]:
    domain = system.domain
    time_axis = domain.time_axis
    if time_axis is None:  # pragma: no cover -- System validates this pairing
        return [], "no time axis"
    t_lo, _ = domain.time_bounds()
    t0 = float(t_lo if ic.t0 is None else ic.t0)
    label = f"initial(order={ic.order})@t={t0}"
    constraint = (
        dirichlet(t0, label=label)
        if ic.order == 0
        else derivative_at(t0, ic.order, label=label)
    )
    return [
        HardCondition(
            component=ic.component,
            axis=domain.coordinate_spec.axis_index(time_axis),
            constraint=constraint,
            target=ic.value,
        )
    ], None


def plan_hard_conditions(
    system: System,
    *,
    mode: str = "auto",
    condition_limit: float | None = None,
    periodic_orders: Sequence[int] = PERIODIC_ORDERS,
) -> HardConditionPlan:
    """Work out which conditions of ``system`` can be enforced by construction.

    Parameters
    ----------
    system
        The problem whose conditions are being triaged.
    mode
        ``"none"`` returns an empty plan (today's behaviour, unchanged);
        ``"auto"`` absorbs everything it can certify.
    condition_limit
        Optional override of the support-matrix conditioning refusal threshold.
    periodic_orders
        Derivative orders matched across a periodic seam; see
        :data:`PERIODIC_ORDERS`. Pass ``(0,)`` for value-only matching, which is
        what a solution with a genuine kink at the seam needs.

    Returns
    -------
    HardConditionPlan
        Absorbed conditions, the indices to drop from the loss, per-condition
        decline reasons, and the sealed certificates.

    Notes
    -----
    The planner sees only the *structure* of the conditions, never the target
    values, because it is pure Python and the targets are backend callables.
    Whether the data is mutually consistent across axes is therefore checked
    where the cage is built, not here.
    """
    if mode not in HARD_CONDITION_MODES:
        raise ValueError(
            f"hard_conditions must be one of {list(HARD_CONDITION_MODES)}, got {mode!r}"
        )
    if mode == "none":
        return HardConditionPlan()

    candidates: list[tuple[str, int, list[HardCondition]]] = []
    declined: list[DeclinedCondition] = []

    for i, bc in enumerate(system.boundary):
        conds, reason = _boundary_conditions(system, bc, periodic_orders=periodic_orders)
        if reason is not None:
            declined.append(DeclinedCondition("boundary", bc.component, i, reason))
        else:
            candidates.append(("boundary", i, conds))
    for i, ic in enumerate(system.initial):
        conds, reason = _initial_condition(system, ic)
        if reason is not None:
            declined.append(DeclinedCondition("initial", ic.component, i, reason))
        else:
            candidates.append(("initial", i, conds))

    if not candidates:
        return HardConditionPlan(declined=tuple(declined))

    accepted = [c for _, _, group in candidates for c in group]
    try:
        plans = group_hard_conditions(accepted, system.domain.bounds)
        certificates = _certify(plans, condition_limit)
    except ValueError as exc:
        # A condition set that cannot be certified stays soft: the solver keeps
        # working, and the reason travels with the plan instead of an exception.
        return HardConditionPlan(
            declined=tuple(declined)
            + tuple(
                DeclinedCondition(kind, group[0].component, idx, str(exc))
                for kind, idx, group in candidates
            )
        )

    return HardConditionPlan(
        conditions=tuple(accepted),
        absorbed_boundary=frozenset(
            idx for kind, idx, _ in candidates if kind == "boundary"
        ),
        absorbed_initial=frozenset(
            idx for kind, idx, _ in candidates if kind == "initial"
        ),
        declined=tuple(declined),
        certificates=certificates,
    )


def _certify(
    plans: dict[str, tuple[AxisPlan, ...]], condition_limit: float | None
) -> tuple[dict[str, Any], ...]:
    kwargs = {} if condition_limit is None else {"condition_limit": condition_limit}
    return tuple(
        certify_support_matrix(
            step.constraints,
            claim=(
                f"hard conditions on component {name!r}, axis "
                f"{step.constraints.axis}, admit an exact constrained expression"
            ),
            **kwargs,
        )
        for name, steps in plans.items()
        for step in steps
    )


__all__ = [
    "HARD_CONDITION_MODES",
    "PERIODIC_ORDERS",
    "DeclinedCondition",
    "HardConditionPlan",
    "plan_hard_conditions",
]
