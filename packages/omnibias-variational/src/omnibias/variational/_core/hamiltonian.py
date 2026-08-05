# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-agnostic Hamiltonian schema.

A :class:`Hamiltonian` describes a phase-space Hamiltonian ``H(q, p, t)`` by a
*callable* plus the generalized-coordinate names ``dof`` and the time axis. It is
the Legendre dual of :class:`omnibias.variational._core.lagrangian.Lagrangian`:
build one from a Lagrangian with
``omnibias.variational.<backend>.ops.hamiltonian_from_lagrangian`` and generate
its flow with ``canonical_equations`` (``qdot = dH/dp``, ``pdot = -dH/dq``).

Like :class:`Lagrangian` the callable receives backend tensors (torch / jax) and
must be written with last-axis reductions so it works for any leading batch shape
(a single sample ``(n_dof,)`` and a batch ``(B, n_dof)`` alike) -- the backend
ops take its partials with ``vmap`` + ``jacrev``.

This module is pure Python (no torch / jax): it only stores the callable and
metadata, exactly like :class:`Lagrangian` and
:class:`omnibias.core.spec.ActivationSpec`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omnibias.variational._core.lagrangian import _as_names

#: ``H(q, p, t)``: ``q`` (positions) and ``p`` (conjugate momenta) of shape
#: ``(..., n_dof)``; ``t`` of shape ``(..., 1)`` -> Hamiltonian of shape
#: ``(...,)``.
HamiltonianFn = Callable[[Any, Any, Any], Any]


@dataclass(frozen=True)
class Hamiltonian:
    """A phase-space Hamiltonian ``H(q, p, t)``.

    Parameters
    ----------
    fn
        Callable ``fn(q, p, t) -> H``. ``q`` (positions) and ``p`` (conjugate
        momenta) have shape ``(..., n_dof)``; ``t`` has shape ``(..., 1)``. The
        output is the scalar Hamiltonian of shape ``(...,)``. Write it with
        last-axis reductions (e.g. ``(p ** 2).sum(-1)``) so it is
        ``vmap``/``jacrev``-compatible on a single sample.
    dof
        Ordered names of the generalized coordinates (matching the Lagrangian's
        ``dof`` when produced by a Legendre transform).
    time_axis
        Name of the time axis (default ``"t"``).
    """

    fn: HamiltonianFn
    dof: tuple[str, ...]
    time_axis: str = "t"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dof", _as_names(self.dof, "dof"))
        if not callable(self.fn):
            raise TypeError("Hamiltonian.fn must be callable")
        if not isinstance(self.time_axis, str) or not self.time_axis:
            raise ValueError("time_axis must be a non-empty string")

    @property
    def n_dof(self) -> int:
        """Number of generalized coordinates."""
        return len(self.dof)


__all__ = [
    "Hamiltonian",
    "HamiltonianFn",
]
