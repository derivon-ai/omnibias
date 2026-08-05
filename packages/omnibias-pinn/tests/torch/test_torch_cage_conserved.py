# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IntegralConservationField and FluxFormField (torch)."""

from __future__ import annotations

import pytest
import torch
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.cage import (
    FluxFormField,
    IntegralConservationField,
    StreamfunctionField,
    VectorPotentialField,
)
from omnibias.pinn.torch.fields import JetMLPVectorField

DTYPE = torch.float64
BOUNDS = ((-3.0, 5.0),)


@pytest.fixture(autouse=True)
def _float64() -> None:
    torch.set_default_dtype(DTYPE)


def _base_1d(names: tuple[str, ...] = ("psi_re", "psi_im")) -> JetMLPVectorField:
    torch.manual_seed(0)
    return JetMLPVectorField(
        coordinate_spec=CoordinateSpec(("x",), domain=BOUNDS),
        components=ComponentSpec(names),
        hidden=16,
        depth=2,
        jet_order=2,
    )


def _field(axes: tuple[str, ...], names: tuple[str, ...], seed: int = 0, order: int = 3):
    torch.manual_seed(seed)
    return JetMLPVectorField(
        coordinate_spec=CoordinateSpec(axes),
        components=ComponentSpec(names),
        hidden=16,
        depth=2,
        jet_order=order,
    )


# ----------------------- integral conservation -------------------------


@pytest.mark.parametrize(("degree", "total"), [(1, 3.0), (2, 1.0), (2, 5.0), (3, 2.0)])
def test_integral_hits_the_target_to_quadrature_accuracy(
    degree: int, total: float
) -> None:
    base = _base_1d()
    cage = IntegralConservationField(
        base=base,
        rule=gauss_legendre(BOUNDS, 200),
        conserved=("psi_re", "psi_im"),
        total=total,
        degree=degree,
    )
    state = cage(torch.linspace(-2.0, 4.0, 7, dtype=DTYPE).reshape(-1, 1))
    assert float(cage.integral(state).detach()) == pytest.approx(total, abs=1e-11)


def test_degree_two_reproduces_the_qpinn_norm_cage_formula() -> None:
    """``degree=2`` is exactly ``psi / ||psi||``, the NormConservationField rule."""
    base = _base_1d()
    rule = gauss_legendre(BOUNDS, 128)
    cage = IntegralConservationField(
        base=base, rule=rule, conserved=("psi_re", "psi_im"), total=1.0, degree=2
    )
    nodes = torch.as_tensor(rule.nodes, dtype=DTYPE)
    weights = torch.as_tensor(rule.weights, dtype=DTYPE)
    inner = base(nodes)
    density = ops.value(inner, "psi_re") ** 2 + ops.value(inner, "psi_im") ** 2
    expected = 1.0 / torch.sqrt((weights * density).sum())
    state = cage(torch.linspace(-2.0, 4.0, 5, dtype=DTYPE).reshape(-1, 1))
    assert torch.allclose(state.extra["_conservation_scale"], expected, atol=1e-14)


@pytest.mark.parametrize("order", [1, 2])
def test_every_derivative_is_the_base_derivative_times_one_scalar(order: int) -> None:
    base = _base_1d()
    cage = IntegralConservationField(
        base=base,
        rule=gauss_legendre(BOUNDS, 64),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )
    state = cage(torch.linspace(-2.0, 4.0, 9, dtype=DTYPE).reshape(-1, 1))
    inner = state.extra["_cage_inner_state"]
    scale = state.extra["_conservation_scale"]
    caged = ops.derivative(state, "psi_re", axis=0, order=order)
    raw = ops.derivative(inner, "psi_re", axis=0, order=order)
    # Bit-identical, not merely close: the cage multiplies, it does not re-derive.
    assert torch.equal(caged, scale * raw)


def test_passthrough_components_are_untouched() -> None:
    base = _base_1d(("psi_re", "psi_im", "p"))
    cage = IntegralConservationField(
        base=base,
        rule=gauss_legendre(BOUNDS, 64),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )
    state = cage(torch.linspace(-2.0, 4.0, 5, dtype=DTYPE).reshape(-1, 1))
    inner = state.extra["_cage_inner_state"]
    assert cage.passthrough_names == ("p",)
    assert torch.equal(ops.value(state, "p"), ops.value(inner, "p"))
    assert torch.equal(
        ops.derivative(state, "p", axis=0), ops.derivative(inner, "p", axis=0)
    )


def test_rescaling_is_idempotent_on_an_already_conserved_field() -> None:
    """Caging a caged field changes nothing: ``lambda`` becomes 1."""
    base = _base_1d()
    rule = gauss_legendre(BOUNDS, 96)
    once = IntegralConservationField(
        base=base, rule=rule, conserved=("psi_re", "psi_im"), total=1.0, degree=2
    )
    twice = IntegralConservationField(
        base=once, rule=rule, conserved=("psi_re", "psi_im"), total=1.0, degree=2
    )
    coords = torch.linspace(-2.0, 4.0, 7, dtype=DTYPE).reshape(-1, 1)
    assert float(twice(coords).extra["_conservation_scale"].detach()) == pytest.approx(
        1.0, abs=1e-12
    )
    assert torch.allclose(
        ops.value(twice(coords), "psi_re"), ops.value(once(coords), "psi_re"), atol=1e-14
    )


def test_gradients_reach_the_base_parameters_through_the_cage() -> None:
    base = _base_1d()
    cage = IntegralConservationField(
        base=base,
        rule=gauss_legendre(BOUNDS, 64),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )
    state = cage(torch.linspace(-2.0, 4.0, 7, dtype=DTYPE).reshape(-1, 1))
    ops.value(state, "psi_re").pow(2).sum().backward()
    grads = [p.grad for p in cage.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)


def test_a_vanishing_integral_is_refused_not_silently_nan() -> None:
    base = _base_1d()
    cage = IntegralConservationField(
        base=base,
        rule=gauss_legendre(BOUNDS, 32),
        conserved=("psi_re", "psi_im"),
        total=1.0,
        degree=2,
    )
    with torch.no_grad():
        for p in cage.base.parameters():
            p.zero_()
    with pytest.raises(ValueError, match="no real rescaling"):
        cage(torch.linspace(-2.0, 4.0, 5, dtype=DTYPE).reshape(-1, 1))


def test_constructor_validation() -> None:
    base = _base_1d()
    rule = gauss_legendre(BOUNDS, 16)
    with pytest.raises(ValueError, match="at least one component"):
        IntegralConservationField(base=base, rule=rule, conserved=())
    with pytest.raises(ValueError, match="not in base components"):
        IntegralConservationField(base=base, rule=rule, conserved=("nope",))
    with pytest.raises(ValueError, match="unique"):
        IntegralConservationField(base=base, rule=rule, conserved=("psi_re", "psi_re"))
    with pytest.raises(ValueError, match="degree must be"):
        IntegralConservationField(
            base=base, rule=rule, conserved=("psi_re",), degree=0
        )
    with pytest.raises(ValueError, match="total must be"):
        IntegralConservationField(base=base, rule=rule, conserved=("psi_re",), total=0.0)
    with pytest.raises(ValueError, match="quadrature dim"):
        IntegralConservationField(
            base=base,
            rule=gauss_legendre(((0.0, 1.0), (0.0, 1.0)), 4),
            conserved=("psi_re",),
        )


# --------------------------- flux form ---------------------------------


def test_two_axes_reproduce_the_streamfunction_cage_bit_for_bit() -> None:
    base = _field(("x", "y"), ("psi",))
    stream = StreamfunctionField(base=base, psi="psi", velocity_names=("u", "v"))
    flux = FluxFormField(base=base, potential_names=("psi",), flux_names=("u", "v"))
    coords = torch.rand(9, 2, dtype=DTYPE) * 2 - 1
    s_stream, s_flux = stream(coords), flux(coords)
    for name in ("u", "v"):
        assert torch.equal(ops.value(s_stream, name), ops.value(s_flux, name))
        for axis in (0, 1):
            assert torch.equal(
                ops.derivative(s_stream, name, axis=axis),
                ops.derivative(s_flux, name, axis=axis),
            )
            assert torch.equal(
                ops.mixed_partial(s_stream, name, (0, 1), (1, 1)),
                ops.mixed_partial(s_flux, name, (0, 1), (1, 1)),
            )


@pytest.mark.parametrize("n_axes", [2, 3, 4])
def test_divergence_vanishes_identically_in_any_dimension(n_axes: int) -> None:
    axes = ("t", "x", "y", "z")[:n_axes]
    n_pot = n_axes * (n_axes - 1) // 2
    potentials = tuple(f"A{i}" for i in range(n_pot))
    fluxes = ("rho", "fx", "fy", "fz")[:n_axes]
    base = _field(axes, potentials, seed=n_axes, order=2)
    flux = FluxFormField(base=base, potential_names=potentials, flux_names=fluxes)
    state = flux(torch.rand(7, n_axes, dtype=DTYPE) * 2 - 1)
    divergence = sum(
        ops.derivative(state, name, axis=i) for i, name in enumerate(fluxes)
    )
    assert float(divergence.abs().max().detach()) < 1e-11


def test_three_axes_are_divergence_free_like_the_vector_potential_cage() -> None:
    """Both are curls, so both kill the divergence -- with a different basis."""
    base = _field(("x", "y", "z"), ("A1", "A2", "A3"), order=2)
    curl = VectorPotentialField(
        base=base, A_components=("A1", "A2", "A3"), velocity_names=("u", "v", "w")
    )
    flux = FluxFormField(
        base=base, potential_names=("A1", "A2", "A3"), flux_names=("u", "v", "w")
    )
    coords = torch.rand(9, 3, dtype=DTYPE) * 2 - 1
    for state in (curl(coords), flux(coords)):
        divergence = sum(
            ops.derivative(state, name, axis=i)
            for i, name in enumerate(("u", "v", "w"))
        )
        assert float(divergence.abs().max().detach()) < 1e-12


def test_flux_is_the_signed_sum_of_potential_derivatives() -> None:
    """``G^0 = d_1 A^{01} + d_2 A^{02}`` read straight off the definition."""
    base = _field(("t", "x", "y"), ("A01", "A02", "A12"), order=2)
    flux = FluxFormField(
        base=base,
        potential_names=("A01", "A02", "A12"),
        flux_names=("rho", "fx", "fy"),
    )
    coords = torch.rand(6, 3, dtype=DTYPE) * 2 - 1
    state = flux(coords)
    inner = state.extra["_cage_inner_state"]
    expected_rho = ops.derivative(inner, "A01", axis=1) + ops.derivative(
        inner, "A02", axis=2
    )
    expected_fx = -ops.derivative(inner, "A01", axis=0) + ops.derivative(
        inner, "A12", axis=2
    )
    assert torch.allclose(ops.value(state, "rho"), expected_rho, atol=1e-14)
    assert torch.allclose(ops.value(state, "fx"), expected_fx, atol=1e-14)


def test_derivatives_of_the_flux_stay_exact() -> None:
    """``d_a G^i`` must equal the same signed sum one order higher."""
    base = _field(("t", "x"), ("A",), order=3)
    flux = FluxFormField(base=base, potential_names=("A",), flux_names=("rho", "f"))
    coords = torch.rand(6, 2, dtype=DTYPE) * 2 - 1
    state = flux(coords)
    inner = state.extra["_cage_inner_state"]
    # rho = d_x A, so d_x rho = d_x^2 A and d_t rho = d_t d_x A.
    assert torch.allclose(
        ops.derivative(state, "rho", axis=1),
        ops.derivative(inner, "A", axis=1, order=2),
        atol=1e-14,
    )
    assert torch.allclose(
        ops.derivative(state, "rho", axis=0),
        ops.mixed_partial(inner, "A", (0, 1), (1, 1)),
        atol=1e-14,
    )


def test_passthrough_and_validation() -> None:
    base = _field(("x", "y"), ("psi", "p"))
    flux = FluxFormField(
        base=base,
        potential_names=("psi",),
        flux_names=("u", "v"),
        passthrough_names=("p",),
    )
    coords = torch.rand(5, 2, dtype=DTYPE)
    state = flux(coords)
    assert torch.equal(
        ops.value(state, "p"), ops.value(state.extra["_cage_inner_state"], "p")
    )
    with pytest.raises(ValueError, match="one flux name per axis"):
        FluxFormField(base=base, potential_names=("psi",), flux_names=("u",))
    with pytest.raises(ValueError, match="independent components"):
        FluxFormField(
            base=base, potential_names=("psi", "p"), flux_names=("u", "v")
        )
    with pytest.raises(ValueError, match="not in base components"):
        FluxFormField(base=base, potential_names=("nope",), flux_names=("u", "v"))
    with pytest.raises(ValueError, match="unique"):
        FluxFormField(base=base, potential_names=("psi",), flux_names=("u", "u"))


def test_flux_form_gradients_reach_the_base() -> None:
    base = _field(("t", "x"), ("A",))
    flux = FluxFormField(base=base, potential_names=("A",), flux_names=("rho", "f"))
    state = flux(torch.rand(6, 2, dtype=DTYPE))
    ops.value(state, "rho").pow(2).sum().backward()
    grads = [p.grad for p in flux.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().max()) > 0 for g in grads)
