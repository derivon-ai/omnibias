# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Domain decomposition and stiff time stepping -- omnibias.pinn.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_xpinn_stiff.py

Two problems a single global smooth network handles badly, and the primitives
that fix them.

Part 1 -- **the seam**. Steady conduction ``-(k u')' = 0`` on ``(0, 1)`` with
``k = 1`` on the left half and ``k = 20`` on the right, ``u(0) = 0``,
``u(1) = 1``. The exact solution is piecewise linear with a *kink* at ``x = 1/2``
-- the flux ``k u'`` is continuous, so ``u'`` jumps by the conductivity ratio.
A single smooth network has no kink to give. Two patches plus the interface
conditions do, and the part worth watching is that continuity of the *value*
alone is not one of them: it is satisfiable, it looks converged, and it gets the
wrong answer, because it quietly solves the constant-``k`` problem instead.

Part 2 -- **heterogeneous patches**. ``PartitionedField`` no longer forces every
region to be the same network. A deep ``JetMLPVectorField`` where the solution is
hard, a four-unit one-layer patch where it is not, in one blended field.

Part 3 -- **stiff time stepping**. Kuramoto-Sivashinsky, whose fourth-order term
makes explicit stepping hopeless: RK4 blows up at a step size where ETDRK4 is
already converged. Plus the two supporting pieces -- the ``phi`` functions
evaluated where the textbook formula loses every digit, and a Jacobian read off
one closed-form jet.

Honesty: the seam ops add no approximation (``d/dn`` contracts the gradient the
substrate already provides), and the ``jet_mlp`` derivatives are exact closed
form; the *solutions* in Part 1 are trained, so those numbers are training
outcomes on a fixed seed. The ``phi`` functions and the scheme coefficients are
numerical (a truncated series with a bounded argument, then exact squaring), and
their orders below are measured, not asserted.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.partition.torch import build_partitioned_field
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.fields import (
    OneLayerVectorField,
    build_jet_mlp_vector_field,
)
from omnibias.pinn.torch.losses import (
    Interface,
    InterfaceSpec,
    interface_loss,
    interface_points,
    interface_residual,
    split_by_interface,
)

DTYPE = torch.float64
COORDS = CoordinateSpec(("x",))
COMPS = ComponentSpec(("u",))

K_LEFT, K_RIGHT = 1.0, 20.0
# k u' is constant; matching u(0) = 0, u(1) = 1 fixes the flux.
FLUX = 1.0 / (0.5 / K_LEFT + 0.5 / K_RIGHT)


def exact(x: torch.Tensor) -> torch.Tensor:
    """The piecewise-linear two-material solution."""
    left = FLUX * x / K_LEFT
    right = FLUX * 0.5 / K_LEFT + FLUX * (x - 0.5) / K_RIGHT
    return torch.where(x < 0.5, left, right)


def patch(seed: int, *, hidden: int = 16, depth: int = 2):
    torch.manual_seed(seed)
    return build_jet_mlp_vector_field(
        coordinate_spec=COORDS,
        components=COMPS,
        hidden=hidden,
        depth=depth,
        base="tanh",
        jet_order=2,
        seed=seed,
    )


# ---------------------------------------------------------------- part 1 --


def solve_two_patch(*, flux_weight: float) -> tuple[float, float, float]:
    """Train a two-patch XPINN; return (sup error, left slope, right slope)."""
    seam = Interface.from_spec(COORDS, axis="x", value=0.5)
    on_seam = torch.as_tensor(
        interface_points(seam, ((0.0, 1.0),), n_points=1), dtype=DTYPE
    )
    # The geometry routes collocation: each patch trains on the half it owns.
    plus, minus = split_by_interface(seam, np.linspace(0.0, 1.0, 101).reshape(-1, 1))
    x_plus = torch.as_tensor(plus, dtype=DTYPE)
    x_minus = torch.as_tensor(minus, dtype=DTYPE)
    x_lo = torch.zeros(1, 1, dtype=DTYPE)
    x_hi = torch.ones(1, 1, dtype=DTYPE)

    right, left = patch(0), patch(1)
    # Orientation: the normal is +x, so "plus" is the right patch and the
    # conductivity pair is (k_+, k_-) = (k_right, k_left).
    spec = InterfaceSpec(seam, conductivity=(K_RIGHT, K_LEFT))
    params = list(left.parameters()) + list(right.parameters())

    def total_loss() -> torch.Tensor:
        # k is constant inside each patch, so -(k u')' = 0 reduces to u'' = 0.
        res_l = ops.derivative(left(x_minus), "u", axis=0, order=2)
        res_r = ops.derivative(right(x_plus), "u", axis=0, order=2)
        bc = (ops.value(left(x_lo), "u") ** 2).mean() + (
            (ops.value(right(x_hi), "u") - 1.0) ** 2
        ).mean()
        seam_out = interface_residual(right(on_seam), left(on_seam), spec)
        return (
            (res_l**2).mean()
            + (res_r**2).mean()
            + 100.0 * bc
            + 100.0 * interface_loss(seam_out, weights=(1.0, flux_weight))
        )

    adam = torch.optim.Adam(params, lr=1e-2)
    for _ in range(300):
        adam.zero_grad()
        loss = total_loss()
        loss.backward()
        adam.step()
    lbfgs = torch.optim.LBFGS(params, max_iter=300, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        lbfgs.zero_grad()
        loss = total_loss()
        loss.backward()
        return loss

    lbfgs.step(closure)

    grid = torch.linspace(0.0, 1.0, 201, dtype=DTYPE).reshape(-1, 1)
    slope_l = float(ops.derivative(left(on_seam), "u", axis=0, order=1).detach())
    slope_r = float(ops.derivative(right(on_seam), "u", axis=0, order=1).detach())
    with torch.no_grad():
        blended = torch.where(
            grid < 0.5,
            left.forward_values(grid),
            right.forward_values(grid),
        )[:, 0]
        err = float((blended - exact(grid[:, 0])).abs().max())
    return err, slope_l, slope_r


def part_one() -> None:
    print("\n[1] The seam: value continuity is not the interface condition")

    with_flux = solve_two_patch(flux_weight=1.0)
    value_only = solve_two_patch(flux_weight=0.0)
    exact_slopes = (FLUX / K_LEFT, FLUX / K_RIGHT)

    print(
        f"    exact slopes about the seam: {exact_slopes[0]:.5f} (left, k=1) and "
        f"{exact_slopes[1]:.5f} (right, k=20)\n"
        f"    value + flux : sup error {with_flux[0]:.2e}, slopes "
        f"{with_flux[1]:.5f} / {with_flux[2]:.5f}\n"
        f"    value only   : sup error {value_only[0]:.2e}, slopes "
        f"{value_only[1]:.5f} / {value_only[2]:.5f}"
    )
    ratio = value_only[0] / with_flux[0]
    print(
        f"    Enforcing only continuity is {ratio:.0f}x worse, and the reason is "
        "visible in the slopes: it drove\n    them together, which is the "
        "constant-k solution. Nothing about that run looks like failure --\n"
        "    the loss converges, the field is continuous, the answer is wrong. "
        "The flux condition is what\n    ties the two patches to the same "
        "physics."
    )
    assert with_flux[0] < 1e-3, with_flux
    assert abs(with_flux[1] - exact_slopes[0]) < 1e-3
    assert abs(with_flux[2] - exact_slopes[1]) < 1e-3
    assert value_only[0] > 50.0 * with_flux[0]

    # A fictitious seam through one field jumps by exactly zero -- so a non-zero
    # jump is a real defect, never bookkeeping.
    seam = Interface.from_spec(COORDS, axis="x", value=0.5)
    on_seam = torch.as_tensor(
        interface_points(seam, ((0.0, 1.0),), n_points=8, method="grid"), dtype=DTYPE
    )
    same = patch(3)
    self_jump = interface_residual(same(on_seam), same(on_seam), seam)
    print(
        "    Sanity: cutting one field with a fictitious seam gives value jump "
        f"{float(self_jump.diag['max_abs_value_jump']):.1e} and flux jump "
        f"{float(self_jump.diag['max_abs_flux_jump']):.1e}."
    )
    assert float(self_jump.diag["max_abs_flux_jump"]) == 0.0

    # And the sampler puts the points *on* the seam, not near it.
    offsets = np.abs(seam.signed_distance(on_seam.numpy()))
    print(
        f"    Sampled seam points are on it to {offsets.max():.1e} -- 'near the "
        "interface' would put a\n    floor under the residual that no amount of "
        "training removes."
    )
    assert offsets.max() < 1e-14


# ---------------------------------------------------------------- part 2 --


def part_two() -> None:
    print("\n[2] Heterogeneous patches: spend the capacity where it is needed")

    def factory(region: int):
        if region == 0:  # the hard side gets the depth
            return patch(7, hidden=16, depth=3)
        return OneLayerVectorField(
            coordinate_spec=COORDS, components=COMPS, hidden=4, base="tanh"
        )

    field = build_partitioned_field(
        coordinate_spec=COORDS,
        components=COMPS,
        split_dirs=torch.tensor([[1.0]], dtype=DTYPE),
        split_thresh=torch.tensor([0.0], dtype=DTYPE),
        beta=6.0,
        subfield_factory=factory,
    )
    kinds = [type(sub).__name__ for sub in field.subfields]
    sizes = [sum(p.numel() for p in sub.parameters()) for sub in field.subfields]
    print(f"    patches: {kinds[0]} ({sizes[0]} params) | {kinds[1]} ({sizes[1]})")
    assert kinds[0] != kinds[1]

    # Far from the seam the gate is saturated, so the blend *is* the deep patch:
    # its exact closed-form derivatives survive into the composite untouched.
    far = torch.tensor([[-6.0], [-7.0]], dtype=DTYPE)
    weight = field.partition_weights(far)[:, 0].detach()
    deep = field.subfields[0]
    gaps = [
        float(
            (
                ops.derivative(field(far), "u", axis=0, order=k)
                - ops.derivative(deep(far), "u", axis=0, order=k)
            )
            .detach()
            .abs()
            .max()
        )
        for k in (1, 2)
    ]
    print(
        f"    at x = -6 the gate weight is 1 to {float((1 - weight).abs().max()):.0e}, "
        f"and the blend's d/dx and d2/dx2 match\n    the deep patch alone to "
        f"{max(gaps):.0e} -- capacity bought on one side is not diluted there."
    )
    assert max(gaps) < 1e-11

    # Both patch types still receive a gradient: a patch that never does is a
    # patch that is not being trained.
    x = torch.linspace(-1.5, 1.5, 11, dtype=DTYPE).reshape(-1, 1)
    ops.derivative(field(x), "u", axis=0, order=1).pow(2).mean().backward()
    reach = [
        max(float(p.grad.abs().max()) for p in sub.parameters())
        for sub in field.subfields
    ]
    print(f"    largest gradient reaching each patch: {reach[0]:.2e} / {reach[1]:.2e}")
    assert min(reach) > 0.0


# ---------------------------------------------------------------- part 3 --


def part_three() -> None:
    import omnibias.pinn.solver.torch as pt
    from omnibias.pinn.solver.torch import stiff
    from omnibias.torch.activations.registry import get_activation

    print("\n[3] Stiff time stepping: where explicit stepping runs out")

    # -- Kuramoto-Sivashinsky: u_t = -u u_x - u_xx - u_xxxx -----------------
    grid = pt.SpectralGrid1D(128, 32.0 * math.pi)
    x = grid.points()
    u0 = torch.cos(x / 16.0) * (1.0 + torch.sin(x / 16.0))
    semi = pt.kuramoto_sivashinsky_semidiscrete(grid)
    stiffness = float(semi.symbol.abs().max())

    dt, horizon = 0.25, 5.0
    n_steps = int(round(horizon / dt))
    u_etd = u0
    for _ in range(n_steps):
        u_etd = stiff.etdrk4_step(semi.symbol, semi.nonlinear, u_etd, dt)
    u_ref = u0
    for _ in range(8 * n_steps):
        u_ref = stiff.etdrk4_step(semi.symbol, semi.nonlinear, u_ref, dt / 8.0)
    u_rk4 = u0
    for _ in range(n_steps):
        u_rk4 = pt.rk4_step(semi.rhs, u_rk4, dt)

    print(
        f"    KS on 128 modes: the linear symbol reaches {stiffness:.0f}, so the "
        f"explicit stability limit is\n    about {2.0 / stiffness:.1e} -- far "
        f"below the dt = {dt} the solution itself needs.\n"
        f"    ETDRK4 at dt = {dt}: |u - u_fine| = "
        f"{float((u_etd - u_ref).abs().max()):.2e}\n"
        f"    RK4    at dt = {dt}: max |u| = {float(u_rk4.abs().max()):.2e}"
    )
    assert float((u_etd - u_ref).abs().max()) < 1e-4
    assert not bool(torch.isfinite(u_rk4).all())

    # -- ROS2: L-stability on a dense stiff system --------------------------
    a = torch.tensor([[-1000.0, 1.0], [0.0, -1.0]], dtype=DTYPE)
    u_start = torch.tensor([1.0, 1.0], dtype=DTYPE)
    landed = stiff.rosenbrock_step(lambda u: a @ u, u_start, 1e6, jacobian=a)
    print(
        f"    ROS2 across a step of 1e6 (the fast mode's own scale is 1e-3) lands "
        f"at {float(landed.abs().max()):.1e}\n    -- L-stable, so an infinitely "
        "stiff decaying mode is annihilated rather than left ringing."
    )
    assert float(landed.abs().max()) < 1e-5

    # -- phi functions where the textbook formula cancels away --------------
    z = torch.tensor([1e-14, 1e-8, 1.0], dtype=DTYPE)
    naive = (torch.exp(z) - 1.0) / z
    got = stiff.phi_diagonal(z, 1)[1]
    series = 1.0 + z / 2.0 + z**2 / 6.0  # phi_1(z) = 1 + z/2 + z^2/6 + ...
    print(
        f"    phi_1(1e-14) = 1 + 5e-15 exactly; computed {got[0]:.16f}, and the "
        f"(exp(z)-1)/z formula\n    gives {naive[0]:.16f}. The formula's numerator "
        "cancelled every significant digit; the\n    scaled-and-squared series "
        f"keeps them, to {float((got[0] - series[0]).abs()):.0e}."
    )
    assert float((got[:2] - series[:2]).abs().max()) < 1e-15
    assert abs(float(naive[0]) - 1.0) > 1e-4

    # -- a Jacobian off one closed-form jet ---------------------------------
    spec = get_activation("tanh")
    torch.manual_seed(0)
    w1 = torch.randn(5, 3, dtype=DTYPE)
    b1 = torch.randn(5, dtype=DTYPE)
    w2 = torch.randn(3, 5, dtype=DTYPE)
    b2 = torch.randn(3, dtype=DTYPE)

    def rhs(u: torch.Tensor) -> torch.Tensor:
        return w2 @ spec.forward(w1 @ u + b1) + b2

    state = torch.tensor([0.3, -0.2, 0.7], dtype=DTYPE)
    closed = stiff.closed_form_jacobian([(w1, b1, spec), (w2, b2, None)], state)
    auto = stiff.dense_jacobian(rhs, state)
    print(
        "    Jacobian from one order-1 jet vs forward-over-reverse autodiff: "
        f"max |difference| {float((closed - auto).abs().max()):.1e}\n    -- the "
        "stiff step's linearisation is closed form, not a graph and not a "
        "finite difference."
    )
    assert float((closed - auto).abs().max()) < 1e-13
    stepped = stiff.rosenbrock_step(rhs, state, 0.05, jacobian=closed)
    assert bool(torch.isfinite(stepped).all())


def main() -> None:
    part_one()
    part_two()
    part_three()
    print("\nSeams glued, patches mixed, stiffness stepped; all checks passed.")


if __name__ == "__main__":
    main()
