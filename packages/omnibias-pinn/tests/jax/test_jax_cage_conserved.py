# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IntegralConservationField and FluxFormField (jax)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax import ops as jops
from omnibias.pinn.jax.cage import (
    make_flux_form_field,
    make_integral_conservation_field,
    make_streamfunction_field,
)
from omnibias.pinn.jax.fields.jet_mlp import make_jet_mlp_vector_field

BOUNDS = ((-3.0, 5.0),)


def _base_1d(names: tuple[str, ...] = ("psi_re", "psi_im")):
    return make_jet_mlp_vector_field(
        coordinate_spec=CoordinateSpec(("x",), domain=BOUNDS),
        components=ComponentSpec(names),
        hidden=16,
        depth=2,
        jet_order=2,
        seed=0,
    )


def _field(axes: tuple[str, ...], names: tuple[str, ...], seed: int = 0, order: int = 3):
    return make_jet_mlp_vector_field(
        coordinate_spec=CoordinateSpec(axes),
        components=ComponentSpec(names),
        hidden=16,
        depth=2,
        jet_order=order,
        seed=seed,
    )


def _coords_1d(n: int = 7):
    return jnp.linspace(-2.0, 4.0, n).reshape(-1, 1)


# ----------------------- integral conservation -------------------------


@pytest.mark.parametrize(("degree", "total"), [(1, 3.0), (2, 1.0), (2, 5.0), (3, 2.0)])
def test_integral_hits_the_target(degree: int, total: float) -> None:
    cage = make_integral_conservation_field(
        base=_base_1d(),
        rule=gauss_legendre(BOUNDS, 200),
        conserved=("psi_re", "psi_im"),
        total=total,
        degree=degree,
    )
    state = cage(_coords_1d())
    assert float(cage.integral(state)) == pytest.approx(total, abs=1e-11)


@pytest.mark.parametrize("order", [1, 2])
def test_every_derivative_is_scaled_by_one_scalar(order: int) -> None:
    cage = make_integral_conservation_field(
        base=_base_1d(),
        rule=gauss_legendre(BOUNDS, 64),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )
    state = cage(_coords_1d(9))
    inner = state.extra["_cage_inner_state"]
    scale = state.extra["_conservation_scale"]
    caged = jops.derivative(state, "psi_re", axis=0, order=order)
    raw = jops.derivative(inner, "psi_re", axis=0, order=order)
    assert np.array_equal(np.asarray(caged), np.asarray(scale * raw))


def test_passthrough_untouched() -> None:
    cage = make_integral_conservation_field(
        base=_base_1d(("psi_re", "psi_im", "p")),
        rule=gauss_legendre(BOUNDS, 64),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )
    state = cage(_coords_1d(5))
    inner = state.extra["_cage_inner_state"]
    assert cage.passthrough_names == ("p",)
    assert np.array_equal(
        np.asarray(jops.value(state, "p")), np.asarray(jops.value(inner, "p"))
    )


def test_survives_jit_and_grad_as_a_pytree() -> None:
    cage = make_integral_conservation_field(
        base=_base_1d(),
        rule=gauss_legendre(BOUNDS, 48),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )

    @jax.jit
    def loss(c) -> jax.Array:
        state = c(_coords_1d())
        return jnp.sum(jops.derivative(state, "psi_re", axis=0) ** 2)

    value = float(loss(cage))
    assert np.isfinite(value)
    grads = jax.grad(loss)(cage)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves and any(float(jnp.abs(g).max()) > 0 for g in leaves)


def test_rescaling_is_idempotent() -> None:
    rule = gauss_legendre(BOUNDS, 96)
    once = make_integral_conservation_field(
        base=_base_1d(), rule=rule, conserved=("psi_re", "psi_im"), total=1.0, degree=2
    )
    twice = make_integral_conservation_field(
        base=once, rule=rule, conserved=("psi_re", "psi_im"), total=1.0, degree=2
    )
    assert float(twice(_coords_1d()).extra["_conservation_scale"]) == pytest.approx(
        1.0, abs=1e-12
    )


def test_factory_validation() -> None:
    base = _base_1d()
    rule = gauss_legendre(BOUNDS, 16)
    with pytest.raises(ValueError, match="at least one component"):
        make_integral_conservation_field(base=base, rule=rule, conserved=())
    with pytest.raises(ValueError, match="not in base components"):
        make_integral_conservation_field(base=base, rule=rule, conserved=("nope",))
    with pytest.raises(ValueError, match="degree must be"):
        make_integral_conservation_field(
            base=base, rule=rule, conserved=("psi_re",), degree=0
        )
    with pytest.raises(ValueError, match="quadrature dim"):
        make_integral_conservation_field(
            base=base,
            rule=gauss_legendre(((0.0, 1.0), (0.0, 1.0)), 4),
            conserved=("psi_re",),
        )


# --------------------------- flux form ---------------------------------


def test_two_axes_reproduce_the_streamfunction_cage() -> None:
    base = _field(("x", "y"), ("psi",))
    stream = make_streamfunction_field(base=base, psi="psi", velocity_names=("u", "v"))
    flux = make_flux_form_field(
        base=base, potential_names=("psi",), flux_names=("u", "v")
    )
    coords = jnp.asarray(np.random.RandomState(0).rand(9, 2) * 2 - 1)
    s_stream, s_flux = stream(coords), flux(coords)
    for name in ("u", "v"):
        assert np.array_equal(
            np.asarray(jops.value(s_stream, name)), np.asarray(jops.value(s_flux, name))
        )
        for axis in (0, 1):
            assert np.array_equal(
                np.asarray(jops.derivative(s_stream, name, axis=axis)),
                np.asarray(jops.derivative(s_flux, name, axis=axis)),
            )


@pytest.mark.parametrize("n_axes", [2, 3, 4])
def test_divergence_vanishes_in_any_dimension(n_axes: int) -> None:
    axes = ("t", "x", "y", "z")[:n_axes]
    potentials = tuple(f"A{i}" for i in range(n_axes * (n_axes - 1) // 2))
    fluxes = ("rho", "fx", "fy", "fz")[:n_axes]
    flux = make_flux_form_field(
        base=_field(axes, potentials, seed=n_axes, order=2),
        potential_names=potentials,
        flux_names=fluxes,
    )
    coords = jnp.asarray(np.random.RandomState(n_axes).rand(7, n_axes) * 2 - 1)
    state = flux(coords)
    divergence = sum(
        jops.derivative(state, name, axis=i) for i, name in enumerate(fluxes)
    )
    assert float(jnp.abs(divergence).max()) < 1e-11


def test_flux_is_the_signed_sum_of_potential_derivatives() -> None:
    base = _field(("t", "x", "y"), ("A01", "A02", "A12"), order=2)
    flux = make_flux_form_field(
        base=base,
        potential_names=("A01", "A02", "A12"),
        flux_names=("rho", "fx", "fy"),
    )
    coords = jnp.asarray(np.random.RandomState(3).rand(6, 3) * 2 - 1)
    state = flux(coords)
    inner = state.extra["_cage_inner_state"]
    expected_rho = jops.derivative(inner, "A01", axis=1) + jops.derivative(
        inner, "A02", axis=2
    )
    expected_fx = -jops.derivative(inner, "A01", axis=0) + jops.derivative(
        inner, "A12", axis=2
    )
    assert np.allclose(np.asarray(jops.value(state, "rho")), np.asarray(expected_rho))
    assert np.allclose(np.asarray(jops.value(state, "fx")), np.asarray(expected_fx))


def test_flux_form_survives_jit_and_grad() -> None:
    flux = make_flux_form_field(
        base=_field(("t", "x"), ("A",)),
        potential_names=("A",),
        flux_names=("rho", "f"),
    )
    coords = jnp.asarray(np.random.RandomState(4).rand(6, 2))

    @jax.jit
    def residual(f) -> jax.Array:
        state = f(coords)
        div = jops.derivative(state, "rho", axis=0) + jops.derivative(
            state, "f", axis=1
        )
        return jnp.sum(div**2) + jnp.sum(jops.value(state, "rho") ** 2)

    assert np.isfinite(float(residual(flux)))
    grads = jax.grad(residual)(flux)
    leaves = [g for g in jax.tree_util.tree_leaves(grads) if g is not None]
    assert leaves and any(float(jnp.abs(g).max()) > 0 for g in leaves)


def test_factory_validation_flux() -> None:
    base = _field(("x", "y"), ("psi", "p"))
    with pytest.raises(ValueError, match="one flux name per axis"):
        make_flux_form_field(base=base, potential_names=("psi",), flux_names=("u",))
    with pytest.raises(ValueError, match="independent components"):
        make_flux_form_field(
            base=base, potential_names=("psi", "p"), flux_names=("u", "v")
        )
    with pytest.raises(ValueError, match="not in base components"):
        make_flux_form_field(
            base=base, potential_names=("nope",), flux_names=("u", "v")
        )
