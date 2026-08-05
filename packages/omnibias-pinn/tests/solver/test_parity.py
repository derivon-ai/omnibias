# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch <-> jax residual-assembly parity.

The two backends share the pure-Python ``_core`` (System / Domain / sampling),
so given *identical* ansatz parameters the assembled residual + BC/IC rows must
agree to double-precision round-off. jax runs in x64 (enabled on import of
``omnibias.pinn.solver.jax``) so the closed-form tower is bit-comparable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")

import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.jax as pj  # noqa: E402
import omnibias.pinn.solver.jax.assemble as ja  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
import omnibias.pinn.solver.torch.assemble as tabm  # noqa: E402


def _matched_fields(system: pde.System, *, hidden: int, seed: int):
    """Build a torch field and a jax field sharing the same parameters."""
    torch.set_default_dtype(torch.float64)
    tfield = pt.build_field(
        system, hidden=hidden, activation="tanh", weight_init_scale=1.5, seed=seed
    )
    w = tfield.W.weight.detach().numpy()
    beta = tfield.W.bias.detach().numpy()
    c = tfield.c.weight.detach().numpy()
    b = tfield.c.bias.detach().numpy()
    jfield = pj.field_from_arrays(
        coordinate_spec=system.domain.coordinate_spec,
        components=system.component_spec(),
        W=w,
        beta=beta,
        c=c,
        b=b,
        activation="tanh",
    )
    return tfield, jfield


def _torch_rows(tfield, system, spec):
    coords = tabm.default_interior(tfield, system, spec)
    with torch.no_grad():
        return tabm.all_rows(tfield, system, coords, spec).detach().numpy()


def _jax_rows(jfield, system, spec):
    coords = ja.default_interior(jfield, system, spec)
    return np.asarray(ja.all_rows(jfield, system, coords, spec))


def test_parity_poisson_scalar() -> None:
    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * math.pi ** 2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    system = pde.poisson(dom, source=source, boundary=0.0)
    spec = pde.CollocationSpec(n_interior=8, n_boundary=8)

    tfield, jfield = _matched_fields(system, hidden=24, seed=1)
    tr = _torch_rows(tfield, system, spec)
    jr = _jax_rows(jfield, system, spec)

    assert tr.shape == jr.shape
    np.testing.assert_allclose(tr, jr, rtol=1e-10, atol=1e-10)


def test_parity_advection_diffusion_coupled() -> None:
    dom = pde.Domain(
        ("x", "t"), ((0.0, 1.0), (0.0, 0.4)), periodic=(False, False)
    )

    def u0(c):
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 0])

    system = pde.advection_diffusion(
        dom,
        velocity=0.7,
        diffusivities=(0.1, 0.05),
        coupling=0.3,
        initial=(u0, 0.0),
        boundary=(0.0, 0.0),
    )
    spec = pde.CollocationSpec(n_interior=8, n_boundary=8)

    tfield, jfield = _matched_fields(system, hidden=24, seed=2)
    tr = _torch_rows(tfield, system, spec)
    jr = _jax_rows(jfield, system, spec)

    assert tr.shape == jr.shape
    np.testing.assert_allclose(tr, jr, rtol=1e-9, atol=1e-10)


def test_parity_linear_solve_converges_on_both_backends() -> None:
    """The linear collocation driver reaches the manufactured solution on both.

    The two backends draw features from *different* RNGs, so residual norms are
    not bit-identical (unlike the assembly parity above); this checks that each
    driver end-to-end fits the same manufactured Poisson solution to comparable
    accuracy.
    """
    def source(c):
        xp = pde.array_namespace(c)
        return -2.0 * math.pi ** 2 * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))
    system = pde.poisson(dom, source=source, boundary=0.0)
    spec = pde.CollocationSpec(n_interior=20, n_boundary=20)

    tsol = pt.solve_least_squares(
        system, hidden=120, weight_init_scale=3.0, seed=0, collocation=spec
    )
    jsol = pj.solve_least_squares(
        system, hidden=120, weight_init_scale=3.0, seed=0, collocation=spec
    )

    grid = np.linspace(0.03, 0.97, 30)
    xx, yy = np.meshgrid(grid, grid, indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    ustar = np.sin(math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])

    ut = tsol.evaluate(pts, "u").detach().numpy()
    uj = np.asarray(jsol.evaluate(pts, "u"))
    rel_t = np.linalg.norm(ut - ustar) / np.linalg.norm(ustar)
    rel_j = np.linalg.norm(uj - ustar) / np.linalg.norm(ustar)
    assert rel_t < 5e-2, f"torch relL2 too large: {rel_t}"
    assert rel_j < 5e-2, f"jax relL2 too large: {rel_j}"
