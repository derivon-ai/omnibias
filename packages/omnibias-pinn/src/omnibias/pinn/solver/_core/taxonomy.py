# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PDE taxonomy: the type x linearity x problem x arity cross-product.

These enums classify a :class:`~omnibias.pinn.solver._core.system.System` so a driver
can pick a sensible default solve strategy. The type / linearity are *declared*
by the canonical problem builders (inferring them from an opaque residual
closure is not reliable); the problem kind and arity are inferred from the
system's structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PDEType(Enum):
    """Principal-part classification of the (leading) spatial operator."""

    ELLIPTIC = "elliptic"
    PARABOLIC = "parabolic"
    HYPERBOLIC = "hyperbolic"


class Linearity(Enum):
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class ProblemKind(Enum):
    """Steady boundary-value vs time-dependent initial-value problem.

    There is deliberately no ``INVERSE`` member. Forward-vs-inverse is a statement
    about which *coefficients* are known, not about the PDE's structure -- an
    inverse heat problem is still an initial-value problem -- so folding it in here
    would make the enum non-orthogonal and leave "inverse initial-value"
    inexpressible. A system is inverse exactly when it carries unknown
    coefficients, which is what
    :meth:`~omnibias.pinn.solver._core.system.System.is_inverse` reports.
    """

    BOUNDARY_VALUE = "boundary_value"
    INITIAL_VALUE = "initial_value"


class Arity(Enum):
    """Single scalar field vs a coupled system of fields."""

    SCALAR = "scalar"
    SYSTEM = "system"


@dataclass(frozen=True)
class Classification:
    """The full ``(type, linearity, kind, arity)`` tuple for a system."""

    pde_type: PDEType
    linearity: Linearity
    kind: ProblemKind
    arity: Arity

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (
            self.pde_type.value,
            self.linearity.value,
            self.kind.value,
            self.arity.value,
        )

    def __str__(self) -> str:
        return " / ".join(self.as_tuple())


__all__ = [
    "Arity",
    "Classification",
    "Linearity",
    "PDEType",
    "ProblemKind",
]
