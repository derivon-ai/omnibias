# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable Boolean equation/system solver (jax): propose-and-verify.

Twin of :mod:`omnibias.boolean.torch.ops.solver`. The exact GF(2) fast-path and
the propose-and-verify contract are shared (pure-Python
:mod:`omnibias.boolean._core.systems`); the soft search anneals
``x = sigmoid(beta * theta)`` with a hand-rolled Adam over ``jax.grad`` of the
multilinear-extension residual.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
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

    def residual_soft(self, x: Array) -> Array:
        """Sum of squared multilinear-extension violations at a soft point ``x``."""
        total = jnp.zeros((), dtype=x.dtype)
        for c in self.constraints:
            phi = _multilinear_eval_point(jnp.asarray(c, dtype=x.dtype), x)
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


def _multilinear_eval_point(values: Array, x: Array) -> Array:
    n = values.shape[0].bit_length() - 1
    total = jnp.zeros((), dtype=x.dtype)
    for a in range(1 << n):
        prod = jnp.ones((), dtype=x.dtype)
        for i in range(n):
            prod = prod * (x[i] if (a >> i) & 1 else (1.0 - x[i]))
        total = total + values[a] * prod
    return total


def _adam_solve(
    system: BooleanSystem,
    theta0: Array,
    schedule: BetaAnnealScheduler,
    steps: int,
    lr: float,
) -> Array:
    def loss(theta: Array, beta: Array) -> Array:
        return system.residual_soft(jax.nn.sigmoid(beta * theta))

    grad_fn = jax.grad(loss)
    b1, b2, eps = 0.9, 0.999, 1e-8

    # beta and t are traced (not Python) so the step compiles once and is reused.
    @jax.jit
    def step(theta: Array, m: Array, v: Array, t: Array, beta: Array) -> tuple[Array, Array, Array]:
        g = grad_fn(theta, beta)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        mhat = m / (1.0 - b1 ** (t + 1))
        vhat = v / (1.0 - b2 ** (t + 1))
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)
        return theta, m, v

    theta = theta0
    m = jnp.zeros_like(theta)
    v = jnp.zeros_like(theta)
    for t in range(steps):
        beta = jnp.asarray(schedule.value(t), dtype=theta.dtype)
        theta, m, v = step(theta, m, v, jnp.asarray(t), beta)
    return theta


def solve(
    system: BooleanSystem,
    *,
    schedule: BetaAnnealScheduler | None = None,
    steps: int = 300,
    lr: float = 0.2,
    restarts: int = 16,
    seed: int = 0,
    use_gf2: bool = True,
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
    rng = np.random.default_rng(seed)
    best: SolveResult | None = None
    for _ in range(restarts):
        theta0 = jnp.asarray(rng.standard_normal(system.n), dtype=jnp.float64)
        theta = _adam_solve(system, theta0, schedule, steps, lr)
        x = jax.nn.sigmoid(schedule.value(steps) * theta)
        bits = tuple(int(v >= 0.5) for v in np.asarray(x))
        res = float(system.residual_soft(x))
        cand = SolveResult(
            bits, system.verify(bits), res, "anneal", tuple(float(v) for v in np.asarray(x))
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
