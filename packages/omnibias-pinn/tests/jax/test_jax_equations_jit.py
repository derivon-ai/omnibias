# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JIT compatibility regression tests for the JAX PINN equation residuals.

Prior to the audit the JAX equation residuals contained
``float(jnp.mean(...))`` calls inside their ``diag`` dictionaries, which
host-converted a traced value and broke ``jax.jit`` compilation of any
training step that wrapped the residual call. This regression test
locks in the fix.

Each equation is exercised via:

1. A direct eager call producing a residual array.
2. ``jax.jit`` of a function that calls the equation, asserting the
   compilation succeeds and returns the same shape.
3. A bit-stable parity check between the eager and jitted residual.

We do *not* test inside ``jax.vmap`` because residuals are computed on
batched coordinate inputs already, so the inner-most reduction inside
the equation is over the batch axis.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax.fields.one_layer import (  # noqa: E402
    make_one_layer_vector_field,
)


def _build_field(comps: tuple[str, ...], spatial_axes: tuple[str, ...],
                 *, hidden: int = 8, seed: int = 0,
                 groups: dict[str, tuple[str, ...]] | None = None):
    domain = tuple((0.0, 1.0) for _ in spatial_axes) + ((0.0, 1.0),)
    coord = CoordinateSpec(
        axes=spatial_axes + ("t",),
        periodicity=tuple(False for _ in spatial_axes) + (False,),
        domain=domain,
        time_axis="t",
    )
    components = ComponentSpec(names=comps, groups=groups or {})
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=components, hidden=hidden,
        base="tanh", seed=seed,
    )


def _grid(D: int, B: int = 6) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(0.1, 0.9, size=(B, D)).astype(np.float64))


def test_heat_residual_jit() -> None:
    from omnibias.pinn.jax.equations.heat import heat

    field = _build_field(("u",), ("x",))
    coords = _grid(D=2)

    def f(field, coords):
        return heat(field(coords)).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    assert val.shape == val_jit.shape == (6,)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_burgers_scalar_residual_jit() -> None:
    from omnibias.pinn.jax.equations.burgers import burgers

    field = _build_field(("u",), ("x",))
    coords = _grid(D=2)

    def f(field, coords):
        return burgers(field(coords), nu=0.05).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_burgers_vector_residual_jit() -> None:
    from omnibias.pinn.jax.equations.burgers import burgers

    field = _build_field(
        ("u", "v"), ("x", "y"),
        groups={"velocity": ("u", "v")},
    )
    coords = _grid(D=3)

    def f(field, coords):
        return burgers(field(coords), nu=0.05, form="vector",
                       velocity=("u", "v")).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_cahn_hilliard_residual_jit() -> None:
    from omnibias.pinn.jax.equations.cahn_hilliard import cahn_hilliard

    field = _build_field(("c",), ("x",))
    coords = _grid(D=2)

    def f(field, coords):
        return cahn_hilliard(field(coords), M=1.0, kappa=1e-3).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_kuramoto_sivashinsky_residual_jit() -> None:
    from omnibias.pinn.jax.equations.kuramoto_sivashinsky import (
        kuramoto_sivashinsky,
    )

    field = _build_field(("u",), ("x",))
    coords = _grid(D=2)

    def f(field, coords):
        return kuramoto_sivashinsky(field(coords)).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_biharmonic_residual_jit() -> None:
    from omnibias.pinn.jax.equations.biharmonic import biharmonic

    field = _build_field(("u",), ("x", "y"))
    coords = _grid(D=3)

    def f(field, coords):
        return biharmonic(field(coords)).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_navier_stokes_primitive_2d_jit() -> None:
    from omnibias.pinn.jax.equations.navier_stokes import navier_stokes

    field = _build_field(
        ("u", "v", "p"), ("x", "y"),
        groups={"velocity": ("u", "v")},
    )
    coords = _grid(D=3)

    def f(field, coords):
        out = navier_stokes(
            field(coords), viscosity=0.05, form="primitive_2d",
            velocity=("u", "v"),
        )
        return out.residual, out.continuity, out.diag["mean_sq_residual"]

    residual, continuity, diag = f(field, coords)
    residual_jit, continuity_jit, diag_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(residual, residual_jit, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(continuity, continuity_jit, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(diag, diag_jit, rtol=1e-12, atol=1e-13)


def test_navier_stokes_primitive_3d_jit() -> None:
    from omnibias.pinn.jax.equations.navier_stokes import navier_stokes

    field = _build_field(
        ("u", "v", "w", "p"), ("x", "y", "z"),
        groups={"velocity": ("u", "v", "w")},
    )
    coords = _grid(D=4, B=4)

    def f(field, coords):
        out = navier_stokes(
            field(coords), viscosity=0.05, form="primitive_3d",
            velocity=("u", "v", "w"),
        )
        return out.residual, out.continuity, out.diag["mean_sq_continuity"]

    residual, continuity, diag = f(field, coords)
    residual_jit, continuity_jit, diag_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(residual, residual_jit, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(continuity, continuity_jit, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(diag, diag_jit, rtol=1e-12, atol=1e-13)


def test_navier_stokes_vorticity_stream_2d_jit() -> None:
    from omnibias.pinn.jax.equations.navier_stokes import navier_stokes

    field = _build_field(("psi",), ("x", "y"), hidden=10)
    coords = _grid(D=3)

    def f(field, coords):
        out = navier_stokes(
            field(coords), viscosity=0.02, form="vorticity_stream_2d",
            streamfunction="psi",
        )
        return out.residual, out.continuity, out.diag["mean_sq_residual"]

    residual, continuity, diag = f(field, coords)
    residual_jit, continuity_jit, diag_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(residual, residual_jit, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(continuity, continuity_jit, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(diag, diag_jit, rtol=1e-12, atol=1e-13)
