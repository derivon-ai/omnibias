# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Conservation by construction, and a non-local field -- omnibias.pinn.

Run:

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_conservation_attention.py

A PINN normally enforces a conservation law the way it enforces everything else:
as one more penalty in the loss. That makes conservation a *training outcome* --
it is satisfied as well as the optimiser happened to do, which on a stiff or
under-trained problem is not very well. The two cages here make it a *structural*
property instead, and the third part shows the non-local field type that the same
closed-form machinery unlocks.

Part 1 -- flux form vs a penalty, on 1+1D advection ``d_t rho + d_x F = 0`` with
``F = v rho``. The baseline puts the conservation law in the loss. The caged field
writes the space-time flux as ``G^i = sum_j d_j A^ij`` with ``A`` antisymmetric,
which makes ``div G = 0`` an *identity* (``d_i d_j`` is symmetric, ``A^ij`` is
not, so the double sum cancels term by term) and leaves the constitutive relation
``F = v rho`` as the thing to fit. We then check the finite-volume cell balance
``d/dt int_a^b rho dx + F(b) - F(a) = 0`` that every conservative scheme is built
to respect.

Part 2 -- total mass. ``IntegralConservationField`` rescales globally by
``lambda = (C / I)^(1/p)``, so ``int rho dx = 1`` holds at *every* optimiser step
rather than approximately at the end. Honest limitation: to quadrature accuracy,
since ``I`` is a finite sum.

Part 3 -- ``AttentionVectorField``, a softmax mixture over a trainable memory. It
is the first field on the substrate whose value at ``x`` depends on the whole
memory at once, and its *coordinate* derivatives are still exactly closed form
(omnibias.hopfield differentiates the same block, but with respect to the scores,
which is the wrong variable for a PDE).

Honesty: the derivatives are exact (closed-form jets, no ``autograd`` in the
operator) and the cages hold by construction; the *solutions* are trained, not
certified, and the numbers below are training outcomes on a fixed seed.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.pinn import ComponentSpec, CoordinateSpec, FieldState
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.cage import FluxFormField, IntegralConservationField
from omnibias.pinn.torch.fields import (
    AttentionVectorField,
    JetMLPVectorField,
)

DTYPE = torch.float64
VELOCITY = 1.0
T_MAX = 1.0
X_LO, X_HI = -3.0, 3.0
SIGMA = 0.5


def rho0(x: torch.Tensor) -> torch.Tensor:
    """Unit-mass Gaussian bump, the initial density."""
    return torch.exp(-0.5 * (x / SIGMA) ** 2) / (SIGMA * math.sqrt(2.0 * math.pi))


def collocation(n_t: int = 24, n_x: int = 40) -> torch.Tensor:
    """``(B, 2)`` space-time points in ``t`` then ``x`` axis order."""
    t = torch.linspace(0.0, T_MAX, n_t, dtype=DTYPE)
    x = torch.linspace(X_LO, X_HI, n_x, dtype=DTYPE)
    grid_t, grid_x = torch.meshgrid(t, x, indexing="ij")
    return torch.stack([grid_t.reshape(-1), grid_x.reshape(-1)], dim=-1)


def initial_points(n_x: int = 80) -> torch.Tensor:
    x = torch.linspace(X_LO, X_HI, n_x, dtype=DTYPE)
    return torch.stack([torch.zeros_like(x), x], dim=-1)


# --------------------------------------------------------------------------
# Part 1: the conservation law as a penalty vs as an identity
# --------------------------------------------------------------------------


def build_penalty_field() -> JetMLPVectorField:
    """Plain deep field over ``(t, x)`` predicting ``rho``; ``F = v rho`` by fiat."""
    torch.manual_seed(0)
    return JetMLPVectorField(
        coordinate_spec=CoordinateSpec(("t", "x")),
        components=ComponentSpec(("rho",)),
        hidden=32,
        depth=3,
        jet_order=1,
    )


def build_caged_field() -> FluxFormField:
    """Flux-form cage: one antisymmetric potential ``A^tx`` exposes ``(rho, F)``.

    ``rho = d_x A`` and ``F = -d_t A``, so ``d_t rho + d_x F = d_t d_x A - d_x d_t A``
    vanishes identically. Nothing about that depends on training.
    """
    torch.manual_seed(0)
    potential = JetMLPVectorField(
        coordinate_spec=CoordinateSpec(("t", "x")),
        components=ComponentSpec(("A",)),
        hidden=32,
        depth=3,
        jet_order=3,  # reading d^2 G needs one more order than the potential
    )
    return FluxFormField(
        base=potential, potential_names=("A",), flux_names=("rho", "F")
    )


def train_penalty(field: JetMLPVectorField, *, steps: int = 400) -> None:
    """Loss = conservation residual + initial condition."""
    interior, ic = collocation(), initial_points()
    ic_target = rho0(ic[:, 1])
    opt = torch.optim.Adam(field.parameters(), lr=5e-3)
    for _ in range(steps):
        opt.zero_grad()
        state = field(interior)
        # F = v rho, so the law reads d_t rho + v d_x rho = 0.
        residual = ops.derivative(state, "rho", axis=0) + VELOCITY * ops.derivative(
            state, "rho", axis=1
        )
        loss = residual.pow(2).mean() + (
            ops.value(field(ic), "rho") - ic_target
        ).pow(2).mean()
        loss.backward()
        opt.step()


def train_caged(field: FluxFormField, *, steps: int = 400) -> None:
    """Loss = constitutive relation + initial condition. The law itself is free."""
    interior, ic = collocation(), initial_points()
    ic_target = rho0(ic[:, 1])
    opt = torch.optim.Adam(field.parameters(), lr=5e-3)
    for _ in range(steps):
        opt.zero_grad()
        state = field(interior)
        constitutive = (
            ops.value(state, "F") - VELOCITY * ops.value(state, "rho")
        ).pow(2).mean()
        loss = constitutive + (ops.value(field(ic), "rho") - ic_target).pow(2).mean()
        loss.backward()
        opt.step()


def divergence_error(field, *, penalty: bool) -> float:
    """``max |d_t rho + d_x F|`` on a fresh grid."""
    pts = collocation(n_t=17, n_x=29)
    with torch.no_grad():
        state = field(pts)
        d_t_rho = ops.derivative(state, "rho", axis=0)
        if penalty:
            d_x_flux = VELOCITY * ops.derivative(state, "rho", axis=1)
        else:
            d_x_flux = ops.derivative(state, "F", axis=1)
        return float((d_t_rho + d_x_flux).abs().max())


def cell_balance_error(field, *, penalty: bool, a: float = -1.0, b: float = 1.5) -> float:
    """``|d/dt int_a^b rho dx + F(b) - F(a)|``, the finite-volume cell balance.

    The integral is a Gauss-Legendre sum, so ``d/dt`` of it is the same sum over
    ``d_t rho`` -- which the closed-form tower supplies directly.
    """
    rule = gauss_legendre(((a, b),), 48)
    nodes = torch.as_tensor(rule.nodes, dtype=DTYPE).reshape(-1)
    weights = torch.as_tensor(rule.weights, dtype=DTYPE).reshape(-1)
    worst = 0.0
    with torch.no_grad():
        for t in (0.15, 0.5, 0.85):
            times = torch.full_like(nodes, t)
            interior = torch.stack([times, nodes], dim=-1)
            faces = torch.tensor([[t, a], [t, b]], dtype=DTYPE)
            d_dt_mass = float((weights * ops.derivative(field(interior), "rho", axis=0)).sum())
            face_state = field(faces)
            if penalty:
                flux = VELOCITY * ops.value(face_state, "rho")
            else:
                flux = ops.value(face_state, "F")
            worst = max(worst, abs(d_dt_mass + float(flux[1] - flux[0])))
    return worst


def solution_error(field) -> float:
    """Relative L2 error against the exact solution ``rho0(x - v t)``."""
    pts = collocation(n_t=21, n_x=61)
    with torch.no_grad():
        got = ops.value(field(pts), "rho")
        want = rho0(pts[:, 1] - VELOCITY * pts[:, 0])
        return float((got - want).pow(2).mean().sqrt() / want.pow(2).mean().sqrt())


def part_one() -> None:
    print("=== 1. conservation as a penalty vs as an identity ===")
    penalty = build_penalty_field()
    train_penalty(penalty)
    caged = build_caged_field()
    train_caged(caged)

    rows = [
        ("penalty (law in the loss)", penalty, True),
        ("FluxFormField (law by construction)", caged, False),
    ]
    results = {}
    for label, field, is_penalty in rows:
        div = divergence_error(field, penalty=is_penalty)
        bal = cell_balance_error(field, penalty=is_penalty)
        err = solution_error(field)
        results[label] = (div, bal, err)
        print(
            f"    {label:36s}: max|div G| = {div:8.2e}   "
            f"cell balance = {bal:8.2e}   rel L2 = {err:.4f}"
        )

    pen_div = results["penalty (law in the loss)"][0]
    cage_div = results["FluxFormField (law by construction)"][0]
    cage_bal = results["FluxFormField (law by construction)"][1]
    print(
        f"\n    Both fields fit the solution about equally well, but the penalised one "
        f"violates its own conservation law by {pen_div:.1e}\n    while the cage is at "
        "round-off -- and would stay there with the optimiser switched off entirely, "
        "because nothing about\n    the identity depends on training. The cell balance "
        "holds for every control volume at once, which is what a conservative\n    "
        "finite-volume scheme buys, here on a mesh-free field."
    )
    assert cage_div < 1e-10, cage_div
    assert cage_bal < 1e-9, cage_bal
    assert pen_div > 1e-4, pen_div  # the penalty really is only trained-accurate


# --------------------------------------------------------------------------
# Part 2: a conserved integral, held at every step
# --------------------------------------------------------------------------


def part_two() -> None:
    print("\n=== 2. total mass: rescaled, not penalised ===")
    bounds = ((X_LO, X_HI),)
    cspec = CoordinateSpec(("x",), domain=bounds)
    comp = ComponentSpec(("rho",))
    rule = gauss_legendre(bounds, 96)

    def fresh() -> JetMLPVectorField:
        torch.manual_seed(1)
        return JetMLPVectorField(
            coordinate_spec=cspec, components=comp, hidden=32, depth=2, jet_order=1
        )

    free = fresh()
    inner = fresh()
    caged = IntegralConservationField(
        base=inner, rule=rule, conserved=("rho",), total=1.0, degree=1, dtype=DTYPE
    )

    x = torch.linspace(X_LO, X_HI, 96, dtype=DTYPE).reshape(-1, 1)
    target = rho0(x[:, 0])
    nodes = torch.as_tensor(rule.nodes, dtype=DTYPE)
    weights = torch.as_tensor(rule.weights, dtype=DTYPE).reshape(-1)

    def mass_of(field: Callable[[torch.Tensor], FieldState]) -> float:
        with torch.no_grad():
            return float((weights * ops.value(field(nodes), "rho")).sum())

    def fit_of(field: Callable[[torch.Tensor], FieldState]) -> float:
        with torch.no_grad():
            return float((ops.value(field(x), "rho") - target).pow(2).mean().sqrt())

    opts = (
        torch.optim.Adam(free.parameters(), lr=5e-3),
        torch.optim.Adam(inner.parameters(), lr=5e-3),
    )
    header = f"{'step':>6s}  {'free mass':>12s}  {'caged mass':>13s}"
    print(f"    {header}  {'free fit':>9s}  {'caged fit':>9s}")
    for step in range(301):
        if step % 100 == 0:
            m_free, m_caged = mass_of(free), mass_of(caged)
            row = f"{step:6d}  {m_free:12.9f}  {m_caged:13.9f}"
            print(f"    {row}  {fit_of(free):9.2e}  {fit_of(caged):9.2e}")
            assert abs(m_caged - 1.0) < 1e-9, m_caged
        for field, opt in zip((free, caged), opts, strict=True):
            opt.zero_grad()
            (ops.value(field(x), "rho") - target).pow(2).mean().backward()
            opt.step()

    print(
        "\n    Both fields are fitting the same unit-mass target from the same "
        "initialisation, and both fit it about\n    equally well -- but only the caged "
        "one has the right total mass, and it has it at step 0, before any\n    "
        "training at all. The rescaling is a single scalar with no x dependence, so "
        "D^alpha rho = lambda D^alpha rho~:\n    the closed-form tower survives the "
        "cage intact."
    )
    assert abs(mass_of(free) - 1.0) > 1e-4, mass_of(free)


# --------------------------------------------------------------------------
# Part 3: the non-local field
# --------------------------------------------------------------------------


def part_three() -> None:
    print("\n=== 3. a non-local field with closed-form d/dx ===")
    field = AttentionVectorField(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        hidden=16,
        depth=2,
        memory=6,
        beta=2.0,
        jet_order=3,
    )
    coords = torch.randn(64, 2, dtype=DTYPE)

    # Exactness: third-order coordinate derivatives against nested autograd.
    xr = coords.clone().requires_grad_(True)
    u = field.forward_values(xr)[:, 0]
    d1 = torch.autograd.grad(u.sum(), xr, create_graph=True)[0][:, 0]
    d2 = torch.autograd.grad(d1.sum(), xr, create_graph=True)[0][:, 0]
    d3 = torch.autograd.grad(d2.sum(), xr)[0][:, 0].detach()
    with torch.no_grad():
        closed = ops.derivative(field(coords), "u", axis=0, order=3)
    rel = float((closed - d3).abs().max() / d3.abs().max())
    print(f"    closed-form d^3u/dx^3 vs autograd: max rel error {rel:.2e}")
    assert rel < 1e-10, rel

    # Interpretability: the softmax weights are a partition of unity over the
    # memory, so they say which slots the field consults at each point.
    with torch.no_grad():
        weights = field.attention_weights(coords)
        row_sums = weights.sum(-1)
        concentration = float(weights.max(dim=-1).values.mean())
    print(
        f"    attention weights: {tuple(weights.shape)}, rows sum to "
        f"{float(row_sums.min()):.12f}..{float(row_sums.max()):.12f}, "
        f"mean peak {concentration:.3f}"
    )
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-14)

    # Non-locality: perturbing one memory slot moves every collocation point.
    with torch.no_grad():
        before = field.forward_values(coords).clone()
        field.net.values[0] += 1.0
        moved = (field.forward_values(coords) - before).abs().min()
        field.net.values[0] -= 1.0
    print(
        f"    perturbing one memory slot moves every point by at least {float(moved):.2e} "
        "-- the softmax\n    denominator couples the whole memory, which is what "
        "'non-local' means here."
    )
    assert float(moved) > 1e-9

    # Sharpening the mixture: temperature collapse in the feasibility sense.
    peaks = []
    for beta in (0.5, 4.0, 32.0):
        sharp = AttentionVectorField(
            coordinate_spec=CoordinateSpec(("x", "t")),
            components=ComponentSpec(("u",)),
            hidden=16,
            depth=2,
            memory=6,
            beta=beta,
            jet_order=1,
        )
        with torch.no_grad():
            peaks.append(float(sharp.attention_weights(coords).max(dim=-1).values.mean()))
    print(
        "    mean peak weight at beta = 0.5 / 4 / 32: "
        + " / ".join(f"{p:.3f}" for p in peaks)
        + "\n    -- the mixture hardens towards a crisp partition (the feasibility "
        "sense of collapse, not\n    the founding delta -> 0 one)."
    )
    assert peaks[0] < peaks[1] < peaks[2]


def main() -> None:
    part_one()
    part_two()
    part_three()
    print("\nConservation structural, tower exact throughout; all checks passed.")


if __name__ == "__main__":
    main()
