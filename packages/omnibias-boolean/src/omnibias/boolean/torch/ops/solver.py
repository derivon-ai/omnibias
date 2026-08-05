# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Boolean equation/system solver (torch): propose-and-verify.

A :class:`BooleanSystem` bundles a set of Boolean equations ``phi_k(x) = 0``. Two
solve paths:

* an **exact GF(2) fast-path** (:func:`omnibias.boolean._core.gf2_solve`) used
  automatically when every equation is XOR-linear -- Gaussian elimination is
  polynomial-time and genuinely collapses that search space;
* a **beta-annealed soft search** otherwise: the unknown bits are sigmoid latents
  ``x = sigmoid(beta * theta)`` (the free parameters ``c``), the residual is the
  sum of squared multilinear-extension violations of the gates, and ``beta`` is
  annealed ``beta -> inf`` with :class:`omnibias.binary.BetaAnnealScheduler`.

The soft search is a **heuristic**: it proposes a relaxed assignment, hardens it,
and then *verifies it against the exact Boolean system*. ``verified=False`` means
the heuristic did not find a solution -- not that none exists.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from omnibias.binary.schedule import BetaAnnealScheduler
from omnibias.boolean._core.equations import system_constraint
from omnibias.boolean._core.systems import (
    constraints_are_linear,
    gf2_solve,
    verify_assignment,
)
from omnibias.boolean._core.truth_table import (
    TruthTable,
    all_assignments,
    assignment,
    num_vars,
)
from torch import Tensor


@dataclass(frozen=True)
class SolveResult:
    """Outcome of :func:`solve`: a hardened assignment plus its exact verification."""

    assignment: tuple[int, ...] | None
    verified: bool
    residual: float
    method: str
    relaxed: tuple[float, ...] | None = None


@dataclass(frozen=True)
class BooleanSystem:
    """A system of Boolean equations ``phi_k(x) = 0`` (each ``phi_k`` a truth table)."""

    n: int
    constraints: tuple[TruthTable, ...]

    @classmethod
    def from_constraints(cls, constraints: Sequence[TruthTable]) -> BooleanSystem:
        """Build from constraint truth tables (``1`` = violated)."""
        if not constraints:
            raise ValueError("need at least one constraint")
        n = num_vars(constraints[0])
        for c in constraints:
            if num_vars(c) != n:
                raise ValueError("all constraints must share the same arity")
        return cls(n=n, constraints=tuple(tuple(c) for c in constraints))

    @classmethod
    def from_predicates(
        cls, predicates: Sequence[Callable[..., bool]], n: int
    ) -> BooleanSystem:
        """Build from predicates; ``phi_k`` is ``0`` exactly where ``pred_k`` holds."""
        constraints = [
            tuple(0 if pred(*bits) else 1 for bits in all_assignments(n))
            for pred in predicates
        ]
        return cls.from_constraints(constraints)

    def _constraint_tensors(self, dtype: torch.dtype, device: torch.device) -> list[Tensor]:
        return [
            torch.tensor(c, dtype=dtype, device=device) for c in self.constraints
        ]

    def residual_soft(self, x: Tensor) -> Tensor:
        """Sum of squared multilinear-extension violations at a soft point ``x``."""
        if x.shape[-1] != self.n:
            raise ValueError(f"x must have last dim {self.n}, got {x.shape}")
        total = x.new_zeros(())
        for c in self._constraint_tensors(x.dtype, x.device):
            phi = _multilinear_eval_point(c, x)
            total = total + phi * phi
        return total

    def verify(self, bits: Sequence[int]) -> bool:
        """Exact check: every equation holds at the (hard) assignment ``bits``."""
        return verify_assignment(self.constraints, bits)

    def is_linear(self) -> bool:
        """``True`` iff every equation is XOR-linear (GF(2) fast-path applies)."""
        return constraints_are_linear(self.constraints)

    def combined(self) -> TruthTable:
        """The single OR-combined constraint (``1`` where any equation is violated)."""
        return system_constraint(self.constraints)


def _multilinear_eval_point(values: Tensor, x: Tensor) -> Tensor:
    n = values.shape[0].bit_length() - 1
    total = x.new_zeros(())
    for a in range(1 << n):
        prod = x.new_ones(())
        for i in range(n):
            prod = prod * (x[i] if (a >> i) & 1 else (1.0 - x[i]))
        total = total + values[a] * prod
    return total


def solve(
    system: BooleanSystem,
    *,
    schedule: BetaAnnealScheduler | None = None,
    steps: int = 300,
    lr: float = 0.2,
    restarts: int = 16,
    seed: int = 0,
    use_gf2: bool = True,
    dtype: torch.dtype = torch.float64,
) -> SolveResult:
    """Solve ``system`` (GF(2) fast-path when linear, else annealed soft search)."""
    if use_gf2 and system.is_linear():
        gsol = gf2_solve(system.constraints)
        if gsol is not None:
            if not gsol.consistent:
                return SolveResult(None, False, float("inf"), "gf2")
            bits = gsol.particular
            return SolveResult(bits, system.verify(bits), 0.0, "gf2")

    if schedule is None:
        schedule = BetaAnnealScheduler(1.0, 20.0, steps, "exp")
    device = torch.device("cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    best: SolveResult | None = None
    for _ in range(restarts):
        theta = torch.randn(
            system.n, generator=gen, dtype=dtype, device=device, requires_grad=True
        )
        opt = torch.optim.Adam([theta], lr=lr)
        for t in range(steps):
            beta = schedule.value(t)
            x = torch.sigmoid(beta * theta)
            loss = system.residual_soft(x)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x = torch.sigmoid(schedule.value(steps) * theta)
            bits = tuple(int(v >= 0.5) for v in x)
            res = float(system.residual_soft(x))
        cand = SolveResult(
            bits, system.verify(bits), res, "anneal", tuple(float(v) for v in x)
        )
        if best is None or (cand.verified and not best.verified) or (
            cand.verified == best.verified and cand.residual < best.residual
        ):
            best = cand
        if best.verified:
            break
    assert best is not None
    return best


def brute_force_solutions(system: BooleanSystem) -> list[tuple[int, ...]]:
    """All exact solutions (for tests / verification on small systems)."""
    out = []
    for idx in range(1 << system.n):
        bits = assignment(idx, system.n)
        if system.verify(bits):
            out.append(bits)
    return out


__all__ = [
    "BetaAnnealScheduler",
    "BooleanSystem",
    "SolveResult",
    "brute_force_solutions",
    "solve",
]
