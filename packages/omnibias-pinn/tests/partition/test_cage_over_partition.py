# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A flux-form cage wrapped **around** a partitioned potential still conserves.

Representing a shock needs a sharp seam; representing a conservation law needs
``div G = 0``. The obvious way to get both -- a :class:`PartitionedField` whose
patches are each a :class:`FluxFormField` -- does **not** work, and the reason is
worth stating because it is invisible until measured. Blending divergence-free
fluxes gives

    div (sum_l w_l G_l) = sum_l (grad w_l . G_l + w_l div G_l)
                        = sum_l grad w_l . G_l

which vanishes only where the gates are saturated. Conservation breaks exactly
at the seam, which is the one place a shock needs it.

Inverting the nesting fixes it. ``FluxFormField`` builds ``G^i = sum_j d_j
P^{ij}`` from an antisymmetric potential, so ``div G = d_i d_j P^{ij} = 0`` by
symmetry of mixed partials -- for **any** twice-differentiable ``P``, however
sharp. Putting the partition *inside*, as the potential, buys an arbitrarily
sharp front at no cost to the conservation law.

That is also the right representation rather than a convenient one: with axes
``(t, x)`` the cage gives ``rho = d_x P``, so ``P`` is the cumulative mass, and
integrating a viscous shock profile shows ``P -> c0 x - a |x - c0 t|`` as
``nu -> 0``. The potential of a shock is a *kink*, and a partition of unity over
smooth patches is what represents kinks well.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.partition.torch.field import PartitionedField, build_partitioned_field
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.cage import FluxFormField
from omnibias.pinn.torch.fields import JetMLPVectorField

DTYPE = torch.float64

AXES = ("t", "x")
POTENTIAL = ComponentSpec(("P",))
FLUX = ComponentSpec(("rho", "F"))

#: Relative bound on ``|div G|``. Gate derivatives scale like ``beta`` and the
#: mixed partials like ``beta**2``, so absolute round-off grows with seam
#: sharpness and an absolute threshold would be wrong. Measured range is 2e-16
#: (beta=2) to 3.4e-14 (beta=200), so this keeps ~1.5 orders of headroom.
MAX_RELATIVE_DIVERGENCE = 1e-12


@pytest.fixture(autouse=True)
def _float64() -> None:
    torch.set_default_dtype(DTYPE)


def _coords(n: int = 64, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.rand(n, 2, dtype=DTYPE) * 2.0 - 1.0).requires_grad_(True)


def _patch(index: int) -> JetMLPVectorField:
    """A deep patch carrying the closed-form tower to third order.

    ``jet_order=3`` because the flux needs ``d_x rho = d_x d_x P``.
    """
    torch.manual_seed(100 + index)
    return JetMLPVectorField(
        coordinate_spec=CoordinateSpec(AXES),
        components=POTENTIAL,
        hidden=16,
        depth=2,
        jet_order=3,
    )


def _cage_over_partition(beta: float = 8.0) -> FluxFormField:
    """The chosen nesting: the potential is partitioned, the cage wraps it."""
    field = build_partitioned_field(
        coordinate_spec=CoordinateSpec(AXES),
        components=POTENTIAL,
        # Tilted so the seam can track a *moving* front, not just a fixed one.
        split_dirs=torch.tensor([[-0.5, 1.0]], dtype=DTYPE),
        split_thresh=torch.tensor([0.0], dtype=DTYPE),
        beta=beta,
        trainable_partition=True,
        seed=0,
        subfield_factory=_patch,
    )
    return FluxFormField(base=field, potential_names=("P",), flux_names=("rho", "F"))


def _partition_over_cage(beta: float = 8.0) -> PartitionedField:
    """The rejected nesting, kept so the failure stays demonstrated."""

    def caged_patch(index: int) -> FluxFormField:
        return FluxFormField(base=_patch(index), potential_names=("P",), flux_names=("rho", "F"))

    return PartitionedField(
        coordinate_spec=CoordinateSpec(AXES),
        components=FLUX,
        subfields=[caged_patch(0), caged_patch(1)],
        split_dirs=torch.tensor([[-0.5, 1.0]], dtype=DTYPE),
        split_thresh=torch.tensor([0.0], dtype=DTYPE),
        beta=beta,
        trainable_partition=True,
    )


def _relative_divergence(field: object, coords: torch.Tensor) -> float:
    """``max|d_t rho + d_x F|`` scaled by the flux magnitude it is measured against."""
    state = field(coords)
    divergence = ops.derivative(state, "rho", axis=0) + ops.derivative(state, "F", axis=1)
    scale = torch.maximum(ops.value(state, "rho").abs().max(), ops.value(state, "F").abs().max())
    return float((divergence.abs().max() / scale).detach())


# --------------------------- the chosen nesting ---------------------------


@pytest.mark.parametrize("beta", [2.0, 8.0, 20.0, 50.0, 200.0])
def test_the_cage_over_a_partitioned_potential_conserves_at_machine_epsilon(
    beta: float,
) -> None:
    """Sharpening the seam does not degrade the conservation law."""
    assert _relative_divergence(_cage_over_partition(beta), _coords()) < (MAX_RELATIVE_DIVERGENCE)


def test_conservation_holds_at_a_seam_so_sharp_the_gate_has_saturated() -> None:
    """The regime a shock actually lives in: points straddling a hard front."""
    cage = _cage_over_partition(beta=200.0)
    # Straddle the seam -0.5 t + x = 0 tightly, where the gate swings fastest.
    t = torch.linspace(-1.0, 1.0, 64, dtype=DTYPE)
    x = 0.5 * t + torch.linspace(-1e-3, 1e-3, 64, dtype=DTYPE)
    coords = torch.stack((t, x), dim=-1).requires_grad_(True)
    assert _relative_divergence(cage, coords) < MAX_RELATIVE_DIVERGENCE


# ------------------------- the rejected nesting ---------------------------


@pytest.mark.parametrize("beta", [8.0, 20.0])
def test_the_rejected_nesting_really_does_destroy_conservation(beta: float) -> None:
    """Blending divergence-free fluxes is not divergence-free.

    Without this the passing test above would look like a property of cages in
    general rather than of *this* nesting. Measured relative divergence is order
    one -- the error is as large as the flux itself.
    """
    assert _relative_divergence(_partition_over_cage(beta), _coords()) > 0.1


def test_the_two_nestings_differ_by_many_orders_of_magnitude() -> None:
    """The whole architectural argument, as one number."""
    coords = _coords()
    good = _relative_divergence(_cage_over_partition(8.0), coords)
    bad = _relative_divergence(_partition_over_cage(8.0), coords)
    assert bad / max(good, 1e-300) > 1e8


# ------------------------- the seam has to move ---------------------------


def test_the_seam_parameters_are_trainable() -> None:
    """A shock position that cannot move is not a shock model."""
    cage = _cage_over_partition()
    seam = [
        p
        for name, p in cage.named_parameters()
        if name.endswith("split_W") or name.endswith("split_t")
    ]
    assert len(seam) == 2, [n for n, _ in cage.named_parameters()]
    assert all(p.requires_grad for p in seam)


def test_the_seam_receives_gradient_from_a_flux_loss() -> None:
    """Gradient must reach the seam through the cage, or it never learns."""
    cage = _cage_over_partition()
    state = cage(_coords())
    ops.value(state, "rho").pow(2).mean().backward()
    grads = {
        name: p.grad
        for name, p in cage.named_parameters()
        if name.endswith("split_W") or name.endswith("split_t")
    }
    assert grads, "no seam parameters found"
    for name, grad in grads.items():
        assert grad is not None, f"{name} got no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient not finite"
        assert float(grad.abs().max()) > 0.0, f"{name} gradient is identically zero"


def test_conservation_survives_after_the_seam_has_moved() -> None:
    """Conservation is structural, so training must not be able to break it."""
    cage = _cage_over_partition(beta=20.0)
    coords = _coords()
    before = _relative_divergence(cage, coords)

    optimiser = torch.optim.Adam(cage.parameters(), lr=5e-2)
    seam_start = next(
        p.detach().clone() for n, p in cage.named_parameters() if n.endswith("split_t")
    )
    for _ in range(15):
        optimiser.zero_grad()
        state = cage(coords)
        # Any objective at all; the point is that the seam relocates.
        (ops.value(state, "rho") - 1.0).pow(2).mean().backward()
        optimiser.step()
    seam_end = next(p.detach().clone() for n, p in cage.named_parameters() if n.endswith("split_t"))

    assert not torch.allclose(seam_start, seam_end), "seam never moved; test is vacuous"
    assert before < MAX_RELATIVE_DIVERGENCE
    assert _relative_divergence(cage, _coords()) < MAX_RELATIVE_DIVERGENCE


# ------------------- the finite-volume reading of the potential -------------


def test_the_potential_is_the_cumulative_mass() -> None:
    """``integral_a^b rho dx = P(b, t) - P(a, t)``: the finite-volume statement.

    Exact as an identity; here it is limited by the quadrature rule, so the
    tolerance is a quadrature tolerance and is labelled as one.
    """
    cage = _cage_over_partition(beta=8.0)
    inner = cage.base
    lo, hi, t0 = -0.9, 0.8, 0.25

    rule = gauss_legendre(((lo, hi),), 64)
    nodes = torch.as_tensor(rule.nodes, dtype=DTYPE).reshape(-1, 1)
    weights = torch.as_tensor(rule.weights, dtype=DTYPE).reshape(-1)
    coords = torch.cat((torch.full_like(nodes, t0), nodes), dim=-1).requires_grad_(True)
    quadrature = float((weights * ops.value(cage(coords), "rho")).sum().detach())

    ends = torch.tensor([[t0, lo], [t0, hi]], dtype=DTYPE, requires_grad=True)
    potential = ops.value(inner(ends), "P").detach()
    endpoints = float(potential[1] - potential[0])

    assert quadrature == pytest.approx(endpoints, abs=1e-9)
