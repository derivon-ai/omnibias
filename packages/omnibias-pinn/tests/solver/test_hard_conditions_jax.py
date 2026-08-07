# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Solver auto-detection of hard conditions on the JAX least-squares driver.

Twin of ``test_hard_conditions.py`` for the JAX path, carrying the same two
guards: the full condition residual re-assembled *ignoring* absorption, and the
bit-for-bit regression check that ``hard_conditions="none"`` changes nothing.
The planner itself is backend-free and is exercised once, on the torch side.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.jax as pj  # noqa: E402
from omnibias.pinn.jax.cage import ConstrainedExpressionField  # noqa: E402
from omnibias.pinn.solver.jax.assemble import condition_residual  # noqa: E402

EXACT = 1e-12
SMALL = pde.CollocationSpec(n_interior=12, n_boundary=12)


def _poisson():
    dom = pde.Domain(("x",), ((0.0, 1.0),))

    def source(c):
        xp = pde.array_namespace(c)
        return -(math.pi**2) * xp.sin(math.pi * c[:, 0])

    return pde.poisson(dom, source=source, boundary=0.0)


def _heat():
    dom = pde.Domain(("t", "x"), ((0.0, 0.2), (0.0, 1.0)), time_axis="t")

    def initial(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 1])

    return pde.heat(dom, diffusivity=0.1, initial=initial, boundary=0.0)


def _full_condition_residual(sol) -> float:
    rows = condition_residual(sol.field, sol.system, SMALL, None)
    return float(np.abs(np.asarray(rows)).max()) if rows.size else 0.0


def test_absorbed_conditions_hold_to_machine_precision() -> None:
    sol = pj.solve_least_squares(
        _poisson(), hidden=48, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    assert isinstance(sol.field, ConstrainedExpressionField)
    assert sol.diagnostics["hard_absorbed"] == 2
    assert _full_condition_residual(sol) < EXACT


def test_the_guard_also_covers_a_time_dependent_system() -> None:
    sol = pj.solve_least_squares(
        _heat(), hidden=48, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    assert sol.diagnostics["hard_absorbed"] == 3
    assert _full_condition_residual(sol) < EXACT


def test_the_absorbed_solve_still_solves_the_interior_equation() -> None:
    sol = pj.solve_least_squares(
        _poisson(),
        hidden=96,
        weight_init_scale=3.0,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=40, n_boundary=8),
        hard_conditions="auto",
    )
    pts = np.linspace(0.02, 0.98, 60).reshape(-1, 1)
    u = np.asarray(sol.evaluate(pts, "u"))
    ustar = np.sin(math.pi * pts[:, 0])
    assert np.linalg.norm(u - ustar) / np.linalg.norm(ustar) < 1e-3


def test_the_default_reproduces_the_previous_solve_bit_for_bit() -> None:
    kw = {"hidden": 32, "seed": 3, "collocation": SMALL}
    a = pj.solve_least_squares(_poisson(), **kw)
    b = pj.solve_least_squares(_poisson(), hard_conditions="none", **kw)
    pts = np.linspace(0.05, 0.95, 25).reshape(-1, 1)
    np.testing.assert_array_equal(
        np.asarray(a.evaluate(pts, "u")), np.asarray(b.evaluate(pts, "u"))
    )
    assert a.residual_norm == b.residual_norm
    assert not isinstance(a.field, ConstrainedExpressionField)


def test_the_two_backends_absorb_the_same_conditions() -> None:
    """The plan is pure Python, so a backend cannot absorb more than the other."""
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    import omnibias.pinn.solver.torch as pt

    j = pj.solve_least_squares(
        _heat(), hidden=32, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    t = pt.solve_least_squares(
        _heat(), hidden=32, seed=0, collocation=SMALL, hard_conditions="auto"
    )
    assert j.diagnostics["hard_conditions"] == t.diagnostics["hard_conditions"]
    assert j.diagnostics["hard_declined"] == t.diagnostics["hard_declined"]


# --------------------------------------------------------------------------- #
# What Stage C added: a second spatial axis, and a periodic seam.
# --------------------------------------------------------------------------- #
def _square():
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))

    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * (math.pi**2) * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    return pde.poisson(dom, source=source, boundary=0.0)


def test_all_four_faces_of_a_square_are_absorbed_and_hold() -> None:
    sol = pj.solve_least_squares(
        _square(),
        hidden=48,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=12, n_boundary=8),
        hard_conditions="auto",
    )
    assert sol.diagnostics["hard_absorbed"] == 4
    assert _full_condition_residual(sol) < EXACT
    rng = np.random.default_rng(4)
    for axis in (0, 1):
        for value in (0.0, 1.0):
            pts = rng.uniform(0.0, 1.0, size=(32, 2))
            pts[:, axis] = value
            assert np.abs(np.asarray(sol.evaluate(pts, "u"))).max() < EXACT


def test_a_periodic_seam_closes_exactly() -> None:
    from dataclasses import replace

    dom = pde.Domain(("x",), ((0.0, 1.0),), periodic=True)

    def source(c):
        xp = pde.array_namespace(c)
        return -((2.0 * math.pi) ** 2) * xp.sin(2.0 * math.pi * c[:, 0])

    system = replace(
        pde.poisson(dom, source=source, boundary=0.0),
        boundary=(pde.BoundaryCondition(component="u", kind="periodic", axis="x"),),
    )
    sol = pj.solve_least_squares(
        system,
        hidden=48,
        seed=0,
        collocation=pde.CollocationSpec(n_interior=48),
        hard_conditions="auto",
    )
    assert sol.diagnostics["hard_absorbed"] == 3  # value, slope, second deriv
    ends = np.array([[0.0], [1.0]])
    value = np.asarray(sol.evaluate(ends, "u"))
    assert abs(value[0] - value[1]) < EXACT
    assert _full_condition_residual(sol) < EXACT
