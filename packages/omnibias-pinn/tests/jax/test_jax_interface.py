# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""XPINN interface residuals (JAX).

Mirrors :mod:`tests.torch.test_torch_interface` claim for claim, and adds the
JAX-only one: the whole interface loss goes through ``jit`` and ``grad`` with
the interface itself -- a frozen numpy dataclass -- riding along as static
geometry rather than as a traced array.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn._core.interface import (  # noqa: E402
    Interface,
    InterfaceSpec,
    interface_points,
)
from omnibias.pinn.jax.fields import (  # noqa: E402
    make_jet_mlp_vector_field,
    make_one_layer_vector_field,
)
from omnibias.pinn.jax.losses import (  # noqa: E402
    flux_jump,
    interface_loss,
    interface_residual,
    normal_derivative,
    normal_flux,
)

CS = CoordinateSpec(("x", "y"))
COMPS = ComponentSpec(("u", "v"))
IFACE = Interface(normal=(1.0, 2.0), offset=0.5, label="seam")


@pytest.fixture
def points():
    x = interface_points(IFACE, ((-1.0, 1.0), (-1.0, 1.0)), n_points=12, seed=4)
    return jnp.asarray(np.asarray(x))


def _field(seed: int, depth: int = 2):
    return make_jet_mlp_vector_field(
        coordinate_spec=CS,
        components=COMPS,
        hidden=6,
        depth=depth,
        base="tanh",
        jet_order=2,
        seed=seed,
    )


# ------------------------------------------------- the fictitious seam ---


def test_one_field_split_by_a_fake_seam_has_no_jump_at_all(points) -> None:
    field = _field(0)
    out = interface_residual(field(points), field(points), IFACE)
    assert float(jnp.abs(out.value_jump).max()) == 0.0
    assert float(jnp.abs(out.flux_jump).max()) == 0.0
    assert float(interface_loss(out)) == 0.0


def test_two_different_fields_do_jump(points) -> None:
    out = interface_residual(_field(0)(points), _field(1)(points), IFACE)
    assert out.diag["max_abs_value_jump"] > 1e-3
    assert out.diag["max_abs_flux_jump"] > 1e-3


# -------------------------------------------------------- the normal ----


def test_the_normal_derivative_is_the_gradient_contracted_with_n(points) -> None:
    state = _field(2)(points)
    g = state.ops.gradient(state, "u", axes=(0, 1))
    n = jnp.asarray(IFACE.unit_normal)
    assert jnp.allclose(normal_derivative(state, "u", normal=IFACE), g @ n, atol=1e-15)


def test_the_normal_derivative_matches_a_directional_finite_difference(
    points,
) -> None:
    field = _field(3)
    n = jnp.asarray(IFACE.unit_normal)
    h = 1e-6
    fd = (
        field.net.value(points + h * n)[:, 0] - field.net.value(points - h * n)[:, 0]
    ) / (2 * h)
    got = normal_derivative(field(points), "u", normal=IFACE)
    assert jnp.allclose(got, fd, rtol=1e-7, atol=1e-8)


def test_flipping_the_interface_flips_every_jump(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    fwd = interface_residual(a, b, IFACE)
    rev = interface_residual(a, b, IFACE.flip())
    assert jnp.allclose(rev.flux_jump, -fwd.flux_jump, atol=1e-15)
    assert jnp.allclose(rev.value_jump, fwd.value_jump)
    assert float(interface_loss(rev)) == pytest.approx(float(interface_loss(fwd)))


def test_a_raw_vector_normal_is_accepted_and_normalised(points) -> None:
    state = _field(4)(points)
    scaled = normal_derivative(state, "u", normal=[3.0, 6.0])
    assert jnp.allclose(scaled, normal_derivative(state, "u", normal=IFACE), atol=1e-15)


# --------------------------------------------------- material contrast ---


def test_matched_media_reduce_the_flux_jump_to_a_derivative_jump(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    names = ("u", "v")
    raw = jnp.stack(
        [
            normal_derivative(a, nm, normal=IFACE)
            - normal_derivative(b, nm, normal=IFACE)
            for nm in names
        ],
        axis=-1,
    )
    assert jnp.allclose(flux_jump(a, b, names, normal=IFACE), raw, atol=1e-15)


def test_a_contrast_makes_a_continuous_field_fail_the_flux_condition(
    points,
) -> None:
    field = _field(5)
    a = b = field(points)
    out = interface_residual(a, b, InterfaceSpec(IFACE, conductivity=(3.0, 1.0)))
    assert float(jnp.abs(out.value_jump).max()) == 0.0
    assert out.diag["max_abs_flux_jump"] > 1e-3
    expected = 2.0 * normal_flux(a, ("u", "v"), normal=IFACE)
    assert jnp.allclose(out.flux_jump, expected, atol=1e-15)


def test_an_explicit_conductivity_overrides_the_spec(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    spec = InterfaceSpec(IFACE, conductivity=(3.0, 1.0))
    got = interface_residual(a, b, spec, conductivity=(1.0, 1.0))
    assert jnp.allclose(
        got.flux_jump, interface_residual(a, b, IFACE).flux_jump, atol=1e-15
    )


# ------------------------------------------------------------- XPINN -----


def test_the_residual_jump_is_carried_when_supplied(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    ra, rb = jnp.ones(12), jnp.zeros(12)
    plain = interface_residual(a, b, IFACE)
    assert plain.residual_jump is None
    assert "mean_sq_residual_jump" not in plain.diag

    out = interface_residual(a, b, IFACE, residuals=(ra, rb))
    assert jnp.allclose(out.residual_jump, ra - rb)
    assert out.diag["mean_sq_residual_jump"] == pytest.approx(1.0)
    assert float(interface_loss(out, residual_weight=2.0)) == pytest.approx(
        float(interface_loss(out, residual_weight=0.0)) + 2.0
    )


# -------------------------------------------------------------- loss -----


def test_the_loss_is_the_weighted_mean_square_of_the_two_jumps(points) -> None:
    out = interface_residual(_field(0)(points), _field(1)(points), IFACE)
    ref = 2.0 * (out.value_jump**2).mean() + 0.5 * (out.flux_jump**2).mean()
    assert jnp.allclose(interface_loss(out, weights=(2.0, 0.5)), ref, atol=1e-15)


def test_negative_weights_are_rejected(points) -> None:
    out = interface_residual(_field(0)(points), _field(1)(points), IFACE)
    with pytest.raises(ValueError, match="non-negative"):
        interface_loss(out, weights=(1.0, -1.0))
    with pytest.raises(ValueError, match="non-negative"):
        interface_loss(out, residual_weight=-1.0)


# ------------------------------------------------- heterogeneous sides ---


def test_the_two_sides_may_be_different_field_types(points) -> None:
    deep = _field(0, depth=3)
    shallow = make_one_layer_vector_field(
        coordinate_spec=CS, components=COMPS, hidden=5, base="tanh", seed=9
    )
    out = interface_residual(deep(points), shallow(points), IFACE)
    assert out.value_jump.shape == (12, 2)
    assert bool(jnp.isfinite(out.flux_jump).all())


# ---------------------------------------------------------- jit / grad ---


def test_the_seam_loss_is_jittable_and_differentiable(points) -> None:
    """The interface is static geometry; the two fields are the pytrees."""
    a0, b0 = _field(0), _field(1)

    @jax.jit
    def loss(a, b):
        return interface_loss(interface_residual(a(points), b(points), IFACE))

    assert float(loss(a0, b0)) == pytest.approx(
        float(interface_loss(interface_residual(a0(points), b0(points), IFACE)))
    )
    grads = jax.grad(loss, argnums=(0, 1))(a0, b0)
    flat = jax.tree_util.tree_leaves(grads)
    assert flat and all(bool(jnp.isfinite(g).all()) for g in flat)
    assert max(float(jnp.abs(g).max()) for g in flat) > 0.0


def test_the_loss_trains_the_seam_shut(points) -> None:
    a, b = _field(0), _field(1)

    def loss(pair):
        fa, fb = pair
        return interface_loss(interface_residual(fa(points), fb(points), IFACE))

    grad_fn = jax.jit(jax.value_and_grad(loss))
    pair = (a, b)
    before = float(loss(pair))
    for _ in range(200):
        _, g = grad_fn(pair)
        pair = jax.tree_util.tree_map(lambda p, d: p - 0.05 * d, pair, g)
    assert float(loss(pair)) < 0.02 * before


# ------------------------------------------------------- validation -----


def test_mismatched_point_sets_are_caught_not_broadcast(points) -> None:
    with pytest.raises(ValueError, match="same interface points"):
        interface_residual(_field(0)(points), _field(1)(points[:5]), IFACE)


def test_a_normal_of_the_wrong_dimension_is_rejected(points) -> None:
    with pytest.raises(ValueError, match="coordinate spec is 2-D"):
        normal_derivative(_field(0)(points), "u", normal=[1.0, 0.0, 0.0])


def test_a_zero_normal_is_rejected(points) -> None:
    with pytest.raises(ValueError, match="non-zero"):
        normal_derivative(_field(0)(points), "u", normal=[0.0, 0.0])


def test_names_default_to_every_component(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    full = interface_residual(a, b, IFACE)
    one = interface_residual(a, b, IFACE, names=("u",))
    assert full.value_jump.shape == (12, 2)
    assert one.value_jump.shape == (12, 1)
    assert jnp.allclose(one.value_jump[:, 0], full.value_jump[:, 0])
