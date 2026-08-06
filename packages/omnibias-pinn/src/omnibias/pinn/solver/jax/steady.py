# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Steady / boundary-value driver (jax): exact-operator linear collocation.

Twin of :func:`omnibias.pinn.solver.torch.steady.solve_least_squares`. The hidden layer
is fixed, so for a linear system the residual is affine in the readout weights;
the collocation matrix is assembled column-by-column from the **closed-form**
operators (no autodiff) and solved with one least-squares solve.
"""

from __future__ import annotations

import jax.numpy as jnp
from omnibias.pinn.solver._core.hard import plan_hard_conditions
from omnibias.pinn.solver._core.sampling import CollocationSpec
from omnibias.pinn.solver._core.system import System
from omnibias.pinn.solver._core.taxonomy import Linearity
from omnibias.pinn.solver.jax._solution import FieldSolution
from omnibias.pinn.solver.jax.assemble import (
    all_rows,
    default_interior,
    residual_norm,
)
from omnibias.pinn.solver.jax.fields import build_field, with_readout


def solve_least_squares(
    system: System,
    *,
    hidden: int = 128,
    activation: str = "tanh",
    weight_init_scale: float | None = 2.0,
    seed: int = 0,
    collocation: CollocationSpec | None = None,
    ridge: float = 1e-8,
    hard_conditions: str = "none",
) -> FieldSolution:
    """Exact-operator linear collocation on jax (one least-squares solve).

    ``hard_conditions="auto"`` embeds every condition it can certify into the
    ansatz and drops those rows from the system. Opt-in: the default ``"none"``
    reproduces the previous behaviour bit for bit.
    """
    if system.linearity is not Linearity.LINEAR:
        raise ValueError(
            "solve_least_squares requires a LINEAR system; jax has no nonlinear "
            "optimiser driver in v1 -- use the torch backend's solve_optimize"
        )
    # Inverse mode is torch-only (there is no jax solve_inverse), so an unbound
    # coefficient here can only be a mistake.
    system.require_bound_coefficients("solve_least_squares")
    spec = collocation or CollocationSpec()
    hard = plan_hard_conditions(system, mode=hard_conditions)
    field = build_field(
        system, hidden=hidden, activation=activation,
        weight_init_scale=weight_init_scale, seed=seed,
        hard_conditions=hard,
    )
    c_shape = field.c.shape
    n_c = int(c_shape[0] * c_shape[1])
    n_unknowns = n_c + int(c_shape[0])
    coords = default_interior(field, system, spec)
    dtype = field.W.dtype

    def rows_for(theta: jnp.ndarray) -> jnp.ndarray:
        c = theta[:n_c].reshape(c_shape)
        b = theta[n_c:]
        return all_rows(with_readout(field, c, b), system, coords, spec, hard)

    r0 = rows_for(jnp.zeros(n_unknowns, dtype=dtype))
    columns = []
    for k in range(n_unknowns):
        ek = jnp.zeros(n_unknowns, dtype=dtype).at[k].set(1.0)
        columns.append(rows_for(ek) - r0)
    mat = jnp.stack(columns, axis=1)
    rhs = -r0
    if ridge > 0.0:
        gram = mat.T @ mat + ridge * jnp.eye(n_unknowns, dtype=dtype)
        theta = jnp.linalg.solve(gram, mat.T @ rhs)
    else:
        theta = jnp.linalg.lstsq(mat, rhs, rcond=None)[0]

    fitted = with_readout(field, theta[:n_c].reshape(c_shape), theta[n_c:])
    return FieldSolution(
        field=fitted,
        system=system,
        residual_norm=residual_norm(fitted, system, spec),
        method="least_squares",
        diagnostics={
            "n_unknowns": n_unknowns,
            "n_rows": int(r0.shape[0]),
            "hard_conditions": hard.summary(),
            "hard_absorbed": len(hard.conditions),
            "hard_declined": tuple(str(d) for d in hard.declined),
        },
    )


def solve_steady(system: System, *, method: str = "auto", **kwargs) -> FieldSolution:
    """Dispatch a steady jax solve (linear collocation only in v1)."""
    if method in ("auto", "least_squares", "lstsq"):
        return solve_least_squares(system, **kwargs)
    raise ValueError(
        f"jax backend supports method='least_squares' in v1, got {method!r}"
    )


__all__ = ["solve_least_squares", "solve_steady"]
