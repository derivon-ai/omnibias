# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""QPINN regression tests for the audit (T3).

Locks in three contracts:

1. **Gamma-matrix anticommutator**: both Dirac and Weyl
   representations must satisfy
   ``{gamma^mu, gamma^nu} = 2 eta^{mu nu} I_4`` with the *mostly-minus*
   metric ``eta = diag(+1, -1, -1, -1)`` (Peskin-Schroeder).

2. **Gamma-5 properties**: ``gamma_5`` is hermitian and squares to
   the identity in both representations.

3. **JAX equation residuals are jit-compatible**: the audit fixed
   ``float(jnp.mean(...))`` host-conversions in the JAX equation
   diagnostics. This regression test asserts the residuals (TISE
   / TDSE / NLS / Helmholtz / KleinGordon / Dirac) are jit-traceable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.qpinn._core.spinor import gamma5, gamma_matrices

# ---------------------------------------------------------------------------
# 1. Gamma anticommutator.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rep", ["dirac", "weyl"])
def test_gamma_anticommutator_mostly_minus(rep: str) -> None:
    """``{gamma^mu, gamma^nu} = 2 eta^{mu nu} I`` with mostly-minus metric."""
    g = list(gamma_matrices(rep))
    eta = np.diag([1.0, -1.0, -1.0, -1.0])  # mostly-minus
    for mu in range(4):
        for nu in range(4):
            anticom = g[mu] @ g[nu] + g[nu] @ g[mu]
            expected = 2.0 * eta[mu, nu] * np.eye(4, dtype=np.complex128)
            np.testing.assert_allclose(
                anticom, expected, rtol=0.0, atol=1e-13,
                err_msg=f"{rep} mu={mu} nu={nu}",
            )


@pytest.mark.parametrize("rep", ["dirac", "weyl"])
def test_gamma5_hermitian_and_idempotent(rep: str) -> None:
    """``gamma_5 = gamma_5^dagger`` and ``gamma_5^2 = I``."""
    g5 = gamma5(rep)
    np.testing.assert_allclose(g5, g5.conj().T, rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        g5 @ g5, np.eye(4, dtype=np.complex128), rtol=0.0, atol=1e-13
    )


def test_gamma5_anticommutes_with_gamma_mu() -> None:
    """``{gamma_5, gamma^mu} = 0`` for every mu, both representations."""
    for rep in ("dirac", "weyl"):
        g5 = gamma5(rep)
        for mu, gmu in enumerate(gamma_matrices(rep)):
            anticom = g5 @ gmu + gmu @ g5
            np.testing.assert_allclose(
                anticom, np.zeros((4, 4), dtype=np.complex128),
                rtol=0.0, atol=1e-13, err_msg=f"{rep} mu={mu}",
            )


# ---------------------------------------------------------------------------
# 2. JAX equation jit compatibility.
# ---------------------------------------------------------------------------


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.jax.fields.one_layer import (  # noqa: E402
    make_one_layer_vector_field,
)
from omnibias.qpinn._core.complex import make_psi_components  # noqa: E402


def _psi_field(D_spatial: int = 1, *, hidden: int = 8, seed: int = 0):
    spatial_axes = ("x", "y", "z")[:D_spatial]
    domain = tuple((0.0, 1.0) for _ in spatial_axes) + ((0.0, 1.0),)
    coord = CoordinateSpec(
        axes=spatial_axes + ("t",),
        periodicity=tuple(False for _ in spatial_axes) + (False,),
        domain=domain,
        time_axis="t",
    )
    components = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=components, hidden=hidden,
        base="tanh", seed=seed,
    )


def _grid(D: int, B: int = 6) -> jnp.ndarray:
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(0.1, 0.9, size=(B, D)).astype(np.float64))


def test_tise_residual_jit() -> None:
    from omnibias.qpinn.jax.equations.tise import tise

    field = _psi_field(D_spatial=1)
    coords = _grid(D=2)

    def f(field, coords):
        return tise(field(coords), energy=0.5).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_tdse_residual_jit() -> None:
    from omnibias.qpinn.jax.equations.tdse import tdse

    field = _psi_field(D_spatial=1)
    coords = _grid(D=2)

    def f(field, coords):
        return tdse(field(coords)).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_nls_residual_jit() -> None:
    from omnibias.qpinn.jax.equations.nls import nls

    field = _psi_field(D_spatial=1)
    coords = _grid(D=2)

    def f(field, coords):
        return nls(field(coords), g=0.5).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_helmholtz_residual_jit() -> None:
    from omnibias.qpinn.jax.equations.helmholtz import helmholtz

    field = _psi_field(D_spatial=1)
    coords = _grid(D=2)

    def f(field, coords):
        return helmholtz(field(coords), k=1.0).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)


def test_klein_gordon_residual_jit() -> None:
    from omnibias.qpinn.jax.equations.klein_gordon import klein_gordon

    # KG is a real-scalar field, not complex; use a plain "phi" component.
    coord = CoordinateSpec(
        axes=("x", "t"), periodicity=(False, False),
        domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t",
    )
    components = ComponentSpec(names=("phi",), groups={})
    field = make_one_layer_vector_field(
        coordinate_spec=coord, components=components, hidden=8,
        base="tanh", seed=0,
    )
    coords = _grid(D=2)

    def f(field, coords):
        return klein_gordon(field(coords)).residual

    val = f(field, coords)
    val_jit = jax.jit(f)(field, coords)
    np.testing.assert_allclose(val, val_jit, rtol=1e-12, atol=1e-13)
