# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""XPINN interface residuals (torch).

The claims worth pinning down are physical, not numerical:

* a single smooth field cut by a *fictitious* seam has zero jump in both value
  and flux -- so a non-zero jump means a real defect, never a bookkeeping error;
* the flux condition is the one that carries the material contrast, and a field
  that is perfectly continuous can still fail it, which is the entire reason
  the second condition exists;
* the ops read the field through ``state.ops``, so either side may be any field
  type -- including a *different* one from its neighbour.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.interface import Interface, InterfaceSpec, interface_points
from omnibias.pinn.torch.fields import (
    OneLayerVectorField,
    build_jet_mlp_vector_field,
)
from omnibias.pinn.torch.losses import (
    flux_jump,
    interface_loss,
    interface_residual,
    normal_derivative,
    normal_flux,
    value_jump,
)

CS = CoordinateSpec(("x", "y"))
COMPS = ComponentSpec(("u", "v"))
IFACE = Interface(normal=(1.0, 2.0), offset=0.5, label="seam")


@pytest.fixture
def points() -> torch.Tensor:
    x = interface_points(IFACE, ((-1.0, 1.0), (-1.0, 1.0)), n_points=12, seed=4)
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def _field(seed: int, depth: int = 2):
    return build_jet_mlp_vector_field(
        coordinate_spec=CS,
        components=COMPS,
        hidden=6,
        depth=depth,
        base="tanh",
        jet_order=2,
        seed=seed,
    )


def _one_layer(seed: int) -> OneLayerVectorField:
    torch.manual_seed(seed)
    return OneLayerVectorField(
        coordinate_spec=CS, components=COMPS, hidden=5, base="tanh"
    )


def _loss(*args, **kwargs) -> float:
    return float(interface_loss(*args, **kwargs).detach())


# ------------------------------------------------- the fictitious seam ---


def test_one_field_split_by_a_fake_seam_has_no_jump_at_all(points) -> None:
    field = _field(0)
    out = interface_residual(field(points), field(points), IFACE)
    assert out.diag["max_abs_value_jump"] == 0.0
    assert out.diag["max_abs_flux_jump"] == 0.0
    assert _loss(out) == 0.0


def test_two_different_fields_do_jump(points) -> None:
    out = interface_residual(_field(0)(points), _field(1)(points), IFACE)
    assert out.diag["max_abs_value_jump"] > 1e-3
    assert out.diag["max_abs_flux_jump"] > 1e-3


# -------------------------------------------------------- the normal ----


def test_the_normal_derivative_is_the_gradient_contracted_with_n(points) -> None:
    field = _field(2)
    state = field(points)
    g = state.ops.gradient(state, "u", axes=(0, 1))
    n = torch.as_tensor(IFACE.unit_normal, dtype=torch.float64)
    got = normal_derivative(state, "u", normal=IFACE)
    assert torch.allclose(got, g @ n, rtol=0.0, atol=1e-15)


def test_the_normal_derivative_matches_a_directional_finite_difference(
    points,
) -> None:
    """A sanity anchor outside the jet machinery entirely."""
    field = _field(3)
    n = torch.as_tensor(IFACE.unit_normal, dtype=torch.float64)
    h = 1e-6
    plus = field.forward_values(points + h * n)[:, 0]
    minus = field.forward_values(points - h * n)[:, 0]
    fd = (plus - minus) / (2 * h)
    got = normal_derivative(field(points), "u", normal=IFACE)
    assert torch.allclose(got, fd, rtol=1e-7, atol=1e-8)


def test_flipping_the_interface_flips_every_jump(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    fwd = interface_residual(a, b, IFACE)
    rev = interface_residual(a, b, IFACE.flip())
    assert torch.allclose(rev.flux_jump, -fwd.flux_jump, rtol=0.0, atol=1e-15)
    assert torch.allclose(rev.value_jump, fwd.value_jump)  # value has no normal
    assert _loss(rev) == pytest.approx(_loss(fwd))


def test_a_raw_vector_normal_is_accepted_and_normalised(points) -> None:
    state = _field(4)(points)
    scaled = normal_derivative(state, "u", normal=[3.0, 6.0])
    unit = normal_derivative(state, "u", normal=IFACE)
    assert torch.allclose(scaled, unit, rtol=0.0, atol=1e-15)


# --------------------------------------------------- material contrast ---


def test_matched_media_reduce_the_flux_jump_to_a_derivative_jump(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    names = ("u", "v")
    raw = torch.stack(
        [
            normal_derivative(a, nm, normal=IFACE)
            - normal_derivative(b, nm, normal=IFACE)
            for nm in names
        ],
        dim=-1,
    )
    assert torch.allclose(flux_jump(a, b, names, normal=IFACE), raw, atol=1e-15)


def test_a_contrast_makes_a_continuous_field_fail_the_flux_condition(
    points,
) -> None:
    """The reason the second condition is not redundant.

    Both sides are the *same* field, so the value jump is identically zero;
    with ``k_+ != k_-`` the flux jump is not, and the loss sees it.
    """
    field = _field(5)
    a = b = field(points)
    spec = InterfaceSpec(IFACE, conductivity=(3.0, 1.0))
    out = interface_residual(a, b, spec)
    assert out.diag["max_abs_value_jump"] == 0.0
    assert out.diag["max_abs_flux_jump"] > 1e-3
    expected = 2.0 * normal_flux(a, ("u", "v"), normal=IFACE)
    assert torch.allclose(out.flux_jump, expected, rtol=0.0, atol=1e-15)


def test_an_explicit_conductivity_overrides_the_spec(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    spec = InterfaceSpec(IFACE, conductivity=(3.0, 1.0))
    got = interface_residual(a, b, spec, conductivity=(1.0, 1.0))
    ref = interface_residual(a, b, IFACE)
    assert torch.allclose(got.flux_jump, ref.flux_jump, rtol=0.0, atol=1e-15)


# ------------------------------------------------------------- XPINN -----


def test_the_residual_jump_is_carried_when_supplied(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    ra, rb = torch.ones(12, dtype=torch.float64), torch.zeros(12, dtype=torch.float64)
    plain = interface_residual(a, b, IFACE)
    assert plain.residual_jump is None
    assert "mean_sq_residual_jump" not in plain.diag

    out = interface_residual(a, b, IFACE, residuals=(ra, rb))
    assert torch.allclose(out.residual_jump, ra - rb)
    assert out.diag["mean_sq_residual_jump"] == pytest.approx(1.0)
    assert _loss(out, residual_weight=2.0) == pytest.approx(
        _loss(out, residual_weight=0.0) + 2.0
    )


# -------------------------------------------------------------- loss -----


def test_the_loss_is_the_weighted_mean_square_of_the_two_jumps(points) -> None:
    out = interface_residual(_field(0)(points), _field(1)(points), IFACE)
    got = interface_loss(out, weights=(2.0, 0.5))
    ref = 2.0 * (out.value_jump**2).mean() + 0.5 * (out.flux_jump**2).mean()
    assert torch.allclose(got, ref, rtol=0.0, atol=1e-15)


def test_the_loss_trains_the_seam_shut(points) -> None:
    """The end-to-end claim: gradients flow and the jump actually falls."""
    a, b = _field(0), _field(1)
    params = list(a.parameters()) + list(b.parameters())
    opt = torch.optim.Adam(params, lr=0.05)
    before = _loss(interface_residual(a(points), b(points), IFACE))
    for _ in range(120):
        opt.zero_grad()
        loss = interface_loss(interface_residual(a(points), b(points), IFACE))
        loss.backward()
        opt.step()
    after = _loss(interface_residual(a(points), b(points), IFACE))
    assert after < 0.02 * before


def test_negative_weights_are_rejected(points) -> None:
    out = interface_residual(_field(0)(points), _field(1)(points), IFACE)
    with pytest.raises(ValueError, match="non-negative"):
        interface_loss(out, weights=(1.0, -1.0))
    with pytest.raises(ValueError, match="non-negative"):
        interface_loss(out, residual_weight=-1.0)


# ------------------------------------------------- heterogeneous sides ---


def test_the_two_sides_may_be_different_field_types(points) -> None:
    """Decomposition's whole selling point: a stiff patch can be bigger."""
    deep, shallow = _field(0, depth=3), _one_layer(9)
    out = interface_residual(deep(points), shallow(points), IFACE)
    assert out.value_jump.shape == (12, 2)
    assert torch.isfinite(out.flux_jump).all()
    loss = interface_loss(out)
    loss.backward()
    assert all(p.grad is not None for p in shallow.parameters())


# ------------------------------------------------------- validation -----


def test_mismatched_point_sets_are_caught_not_broadcast(points) -> None:
    a = _field(0)(points)
    b = _field(1)(points[:5])
    with pytest.raises(ValueError, match="same interface points"):
        interface_residual(a, b, IFACE)


def test_a_normal_of_the_wrong_dimension_is_rejected(points) -> None:
    state = _field(0)(points)
    with pytest.raises(ValueError, match="coordinate spec is 2-D"):
        normal_derivative(state, "u", normal=[1.0, 0.0, 0.0])


def test_a_zero_normal_is_rejected(points) -> None:
    state = _field(0)(points)
    with pytest.raises(ValueError, match="non-zero"):
        normal_derivative(state, "u", normal=[0.0, 0.0])


def test_names_default_to_every_component(points) -> None:
    a, b = _field(0)(points), _field(1)(points)
    full = interface_residual(a, b, IFACE)
    one = interface_residual(a, b, IFACE, names=("u",))
    assert full.value_jump.shape == (12, 2)
    assert one.value_jump.shape == (12, 1)
    assert torch.allclose(one.value_jump[:, 0], full.value_jump[:, 0])
