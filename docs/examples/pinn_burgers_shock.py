# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""A shock-capturing Burgers PINN that conserves by construction.

Run::

    pip install "omnibias-pinn[torch]" omnibias-partition
    python docs/examples/pinn_burgers_shock.py

Viscous Burgers in **conservation form**, ``d_t u + d_x(u^2/2 - nu u_x) = 0``,
has the exact travelling wave

    u(x, t) = c0 - a tanh(a (x - c0 t) / (2 nu))

with ``u_L = c0 + a``, ``u_R = c0 - a`` and speed ``s = c0 = (u_L + u_R)/2`` --
the Rankine-Hugoniot speed for flux ``u^2/2``. Ground truth is analytic, so
nothing here is compared against another solver.

The architecture is a flux-form cage wrapped **around** a partitioned scalar
potential. ``FluxFormField`` writes ``G^i = sum_j d_j P^{ij}`` with ``P``
antisymmetric, so ``div G = d_i d_j P^{ij} = 0`` by symmetry of mixed partials
-- for *any* twice-differentiable ``P``, however sharp. The partition supplies
the sharp, movable front; the cage supplies the conservation law; neither costs
the other anything.

The nesting matters and the reverse does not work. Blending divergence-free
fluxes gives ``div (sum_l w_l G_l) = sum_l grad w_l . G_l``, which vanishes only
where the gates are saturated -- conservation would break exactly at the seam.
Part 1 measures both nestings so the difference is a number, not an assertion.

Why a partition is the *right* potential: with axes ``(t, x)`` the cage gives
``rho = d_x P``, so ``P`` is the cumulative mass, and integrating the profile
above gives ``P -> c0 x - a |x - c0 t|`` as ``nu -> 0``. The potential of a
shock is a **kink**, and a partition of unity over smooth patches is what
represents kinks well.

What the sweep in ``benchmarks/burgers_shock_conservation.py`` measured, over
six viscosities and five seeds against the same non-conservative baseline: the
conservative arm holds global mass balance tighter at **every** viscosity, and
the margin widens as the layer goes under-resolved -- 1.3x at ``nu = 2e-2``,
2.9x at ``nu = 2e-3``, winning 5 seeds out of 5 once ``nu <= 3e-3``. That is the
finite-volume result carried onto a mesh-free field: a conservative scheme is
not more accurate on a well-resolved smooth problem, it is more *robust* when
the feature is under-resolved. As ``nu`` falls tenfold the baseline's mass error
grows 4.5x while the cage's grows 2.0x.

Honesty, in three parts.

*Structural.* ``div G = 0`` holds pointwise everywhere, to machine epsilon
(3.4e-15 was the worst value over all 30 sweep cells), with no training, no
quadrature and no tolerance. Nothing about it is optimised.

*Measured.* Solution accuracy is optimised, not proven. On shock speed and
relative L2 the ordering **reverses** with resolution: the non-conservative arm
is up to 16x better where the layer is resolved, the conservative arm better
once it is not. Only the mass-balance result is seed-robust, so only it is
asserted below; the rest is printed and left to the reader.

*Out of scope.* Conservation pins the Rankine-Hugoniot jump condition; it does
**not** select the entropy solution, which needs a further ingredient
(``omnibias.pinn.torch.losses.entropy_consistent_residual``). The ``beta -> inf``
gate hardening is **temperature collapse** (the feasibility sense), not the
founding ``delta -> 0`` bias collapse (see ``docs/theory.md``).
"""

from __future__ import annotations

import math

import torch
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.partition import certify_partition_gap
from omnibias.partition._core.config import PartitionConfig
from omnibias.partition._core.params import PartitionParams
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.partition.torch import build_partitioned_field
from omnibias.pinn.partition.torch.field import PartitionedField
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.cage import FluxFormField
from omnibias.pinn.torch.fields import JetMLPVectorField

DTYPE = torch.float64

# Riemann data: u_L = 1, u_R = 0, so the exact shock speed is (1 + 0)/2 = 1/2.
C0 = 0.5
AMP = 0.5
#: Chosen so the viscous layer (half-width ``2 nu / a = 0.012``) is *under*-
#: resolved by a uniform draw of ``N_INTERIOR`` points -- the regime where a
#: conservative scheme separates from a non-conservative one.
NU = 3e-3

T_END = 0.8
X_LO, X_HI = -1.0, 1.0

AXES = ("t", "x")
POTENTIAL = ComponentSpec(("P",))
SOLUTION = ComponentSpec(("u",))

N_INTERIOR = 900
N_EDGE = 60
#: How many layer half-widths the refined ``t = 0`` grid spans.
LAYER_SPANS = 6.0
STEPS = 900
LR = 6e-3
CONDITION_WEIGHT = 20.0
BETA = 24.0
SEED = 0


# --------------------------------------------------------------- exact ------


def exact_u(coords: torch.Tensor, nu: float | None = None) -> torch.Tensor:
    """The travelling-wave solution; ``coords`` is ``(n, 2)`` ordered ``(t, x)``.

    ``nu`` resolves against the module global at call time rather than at
    definition time, so a sweep can retune the viscosity in one assignment.
    """
    viscosity = NU if nu is None else nu
    t, x = coords[:, 0], coords[:, 1]
    return C0 - AMP * torch.tanh(AMP * (x - C0 * t) / (2.0 * viscosity))


def _log_cosh(z: torch.Tensor) -> torch.Tensor:
    """``log cosh`` without overflowing: ``|z| + log1p(exp(-2|z|)) - log 2``."""
    a = z.abs()
    return a + torch.log1p(torch.exp(-2.0 * a)) - math.log(2.0)


def exact_potential(coords: torch.Tensor, nu: float | None = None) -> torch.Tensor:
    """The exact cumulative mass ``P``, integrating the wave in ``x``.

    Integrating ``u`` gives ``c0 x - 2 nu log cosh(theta)`` with ``theta =
    a (x - c0 t) / 2 nu``, and the linear-in-``t`` term below fixes the one
    remaining constant so that ``-d_t P`` is the flux rather than the flux plus
    an offset. The pair the cage needs then holds exactly:

        d_x P = u        and        -d_t P = u^2/2 - nu u_x

    both verified to machine epsilon in :func:`part_one`. Note the shape: as
    ``nu -> 0`` this tends to ``c0 x - (c0^2 + a^2) t / 2 - a |x - c0 t|``. The
    potential of a shock is a **kink**, which is precisely what a partition of
    unity over smooth patches represents well.
    """
    viscosity = NU if nu is None else nu
    t, x = coords[:, 0], coords[:, 1]
    theta = AMP * (x - C0 * t) / (2.0 * viscosity)
    return C0 * x - 0.5 * (C0**2 + AMP**2) * t - 2.0 * viscosity * _log_cosh(theta)


# ---------------------------------------------------------------- fields ----


def _patch_factory(hidden: int, components: ComponentSpec):
    def make(index: int) -> JetMLPVectorField:
        torch.manual_seed(SEED * 97 + index)
        return JetMLPVectorField(
            coordinate_spec=CoordinateSpec(AXES),
            components=components,
            hidden=hidden,
            depth=2,
            jet_order=3,
        )

    return make


def _partition(components: ComponentSpec, *, beta: float, hidden: int = 16):
    """Two patches split by a *tilted* seam, so the front can move with time."""
    return build_partitioned_field(
        coordinate_spec=CoordinateSpec(AXES),
        components=components,
        split_dirs=torch.tensor([[-C0, 1.0]], dtype=DTYPE),
        split_thresh=torch.tensor([0.0], dtype=DTYPE),
        beta=beta,
        trainable_partition=True,
        seed=SEED,
        subfield_factory=_patch_factory(hidden, components),
    )


def build_cage_field(*, beta: float = BETA) -> FluxFormField:
    """Conservative arm: ``rho = d_x P``, ``F = -d_t P``, so ``d_t rho + d_x F = 0``."""
    return FluxFormField(
        base=_partition(POTENTIAL, beta=beta),
        potential_names=("P",),
        flux_names=("rho", "F"),
    )


def build_plain_field(*, beta: float = BETA) -> PartitionedField:
    """Baseline arm: identical architecture, non-conservative residual."""
    return _partition(SOLUTION, beta=beta)


def predict(field: object, coords: torch.Tensor) -> torch.Tensor:
    """``u`` from either arm: the cage exposes it as ``rho``."""
    state = field(coords)
    name = "rho" if isinstance(field, FluxFormField) else "u"
    return ops.value(state, name)


# ------------------------------------------------------------- sampling -----


def layer_width(nu: float | None = None) -> float:
    """Half-width of the viscous layer: ``tanh(a (x - c0 t) / 2 nu)`` turns over
    on a scale ``2 nu / a``."""
    return 2.0 * (NU if nu is None else nu) / AMP


def _sample(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interior points drawn **uniformly**; initial data resolved on the front.

    Uniform interior sampling is the point of the experiment, not a shortcut. A
    layer of half-width ``2 nu / a`` catches uniform points only ``O(nu)`` of the
    time, so as ``nu`` falls the front becomes under-resolved -- which is exactly
    the regime where a conservative finite-volume scheme separates from a
    non-conservative one. Concentrating points on the front would resolve away
    the very stress being measured.

    The initial condition is different: it is *given data*, and a finite-volume
    scheme starts from resolved cell averages, so the ``t = 0`` grid is refined
    across the layer. Both arms receive identical points either way.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(N_INTERIOR, 1, generator=g, dtype=DTYPE) * T_END
    x = X_LO + torch.rand(N_INTERIOR, 1, generator=g, dtype=DTYPE) * (X_HI - X_LO)
    interior = torch.cat((t, x), dim=-1).requires_grad_(True)

    span = LAYER_SPANS * layer_width()
    xs = torch.cat(
        (
            torch.linspace(X_LO, X_HI, N_EDGE, dtype=DTYPE),
            torch.linspace(-span, span, N_EDGE, dtype=DTYPE),
        )
    ).reshape(-1, 1)
    initial = torch.cat((torch.zeros_like(xs), xs), dim=-1).requires_grad_(True)

    ts = torch.linspace(0.0, T_END, N_EDGE // 2, dtype=DTYPE).reshape(-1, 1)
    edges = torch.cat(
        (
            torch.cat((ts, torch.full_like(ts, X_LO)), dim=-1),
            torch.cat((ts, torch.full_like(ts, X_HI)), dim=-1),
        )
    ).requires_grad_(True)
    return interior, initial, edges


# ------------------------------------------------------------- training -----


def _condition_loss(field: object, initial: torch.Tensor, edges: torch.Tensor):
    """Initial and boundary data, expressed in each arm's own state variable.

    The baseline parameterises ``u`` and is supervised on ``u``. The cage
    parameterises the potential ``P`` and is supervised on **both** ``P`` and
    ``rho = d_x P``. That is the same data, not more of it: ``P`` is the running
    integral of ``u``, which is exactly the primitive-variable reconstruction a
    finite-volume scheme does. It matters because supervising a potential only
    through its own derivative is badly conditioned -- the network is asked to
    place a feature it can only see one differentiation away.
    """
    ic = (predict(field, initial) - exact_u(initial)).pow(2).mean()
    bc = (predict(field, edges) - exact_u(edges)).pow(2).mean()
    if not isinstance(field, FluxFormField):
        return ic + bc
    inner = field.base
    ic_p = (ops.value(inner(initial), "P") - exact_potential(initial)).pow(2).mean()
    bc_p = (ops.value(inner(edges), "P") - exact_potential(edges)).pow(2).mean()
    return ic + bc + ic_p + bc_p


def cage_residual(field: FluxFormField, coords: torch.Tensor) -> torch.Tensor:
    """Only the constitutive law is a residual; conservation is free."""
    state = field(coords)
    rho = ops.value(state, "rho")
    flux = ops.value(state, "F")
    constitutive = 0.5 * rho.pow(2) - NU * ops.derivative(state, "rho", axis=1)
    return flux - constitutive


def plain_residual(field: PartitionedField, coords: torch.Tensor) -> torch.Tensor:
    """The usual non-conservative form ``u_t + u u_x - nu u_xx``."""
    state = field(coords)
    u = ops.value(state, "u")
    u_t = ops.derivative(state, "u", axis=0)
    u_x = ops.derivative(state, "u", axis=1)
    u_xx = ops.derivative(state, "u", axis=1, order=2)
    return u_t + u * u_x - NU * u_xx


def train(field: object, *, seed: int = SEED, steps: int = STEPS) -> float:
    residual = cage_residual if isinstance(field, FluxFormField) else plain_residual
    interior, initial, edges = _sample(seed)
    optimiser = torch.optim.Adam(field.parameters(), lr=LR)
    loss = torch.zeros((), dtype=DTYPE)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = residual(field, interior).pow(2).mean() + CONDITION_WEIGHT * (
            _condition_loss(field, initial, edges)
        )
        loss.backward()
        optimiser.step()
    return float(loss.detach())


# ------------------------------------------------------------- measuring ----


def structural_divergence(field: FluxFormField, coords: torch.Tensor) -> float:
    """``max|d_t rho + d_x F|`` relative to the flux it is measured against."""
    state = field(coords)
    divergence = ops.derivative(state, "rho", axis=0) + ops.derivative(state, "F", axis=1)
    scale = torch.maximum(ops.value(state, "rho").abs().max(), ops.value(state, "F").abs().max())
    return float((divergence.abs().max() / scale).detach())


def mass_balance_error(field: object, *, n_t: int = 20, n_q: int = 192) -> float:
    """``max_t |dM/dt + Phi(1,t) - Phi(-1,t)|``: the finite-volume mass check.

    The integral form of the conservation law over the whole domain, with
    ``M(t) = integral u dx`` and the *physical* flux ``Phi = u^2/2 - nu u_x``
    rebuilt from each arm's own ``u``. Applied identically to both arms, so
    neither gets it for free -- the cage's structural guarantee is about its own
    ``(rho, F)`` pair, and this asks the harder question of whether that pair
    also satisfies the physics. ``dM/dt`` is exact (autodiff of the quadrature),
    not a finite difference.
    """
    rule = gauss_legendre(((X_LO, X_HI),), n_q)
    nodes = torch.as_tensor(rule.nodes, dtype=DTYPE).reshape(-1)
    weights = torch.as_tensor(rule.weights, dtype=DTYPE).reshape(-1)
    worst = 0.0
    for t_value in torch.linspace(0.0, T_END, n_t, dtype=DTYPE):
        t = t_value.clone().requires_grad_(True)
        column = torch.stack((t.expand(n_q), nodes), dim=-1)
        mass = (weights * predict(field, column)).sum()
        d_mass = torch.autograd.grad(mass, t, create_graph=True)[0]

        ends = torch.tensor(
            [[float(t_value), X_LO], [float(t_value), X_HI]],
            dtype=DTYPE,
            requires_grad=True,
        )
        u = predict(field, ends)
        u_x = torch.autograd.grad(u.sum(), ends, create_graph=True)[0][:, 1]
        flux = 0.5 * u.pow(2) - NU * u_x
        worst = max(worst, abs(float((d_mass + flux[1] - flux[0]).detach())))
    return worst


def shock_speed(field: object, *, n_t: int = 24, n_x: int = 401) -> float:
    """Fit ``x(t)`` through the ``u = c0`` level set and return its slope.

    ``c0`` is the midpoint of the two states, so its level set is the shock
    locus. A straight-line fit of that locus is the propagation speed, which
    Rankine-Hugoniot pins at ``c0``.
    """
    ts = torch.linspace(0.0, T_END, n_t, dtype=DTYPE)
    xs = torch.linspace(X_LO, X_HI, n_x, dtype=DTYPE)
    hits_t: list[float] = []
    hits_x: list[float] = []
    for t in ts:
        coords = torch.stack((torch.full_like(xs, float(t)), xs), dim=-1).requires_grad_(True)
        u = predict(field, coords).detach() - C0
        sign = torch.sign(u)
        change = (sign[:-1] * sign[1:] < 0).nonzero().flatten()
        if change.numel() == 0:
            continue
        i = int(change[0])
        u0, u1 = float(u[i]), float(u[i + 1])
        x0, x1 = float(xs[i]), float(xs[i + 1])
        hits_t.append(float(t))
        hits_x.append(x0 - u0 * (x1 - x0) / (u1 - u0))
    if len(hits_t) < 3:
        return float("nan")
    t_arr = torch.tensor(hits_t, dtype=DTYPE)
    x_arr = torch.tensor(hits_x, dtype=DTYPE)
    t_c = t_arr - t_arr.mean()
    return float((t_c * (x_arr - x_arr.mean())).sum() / (t_c * t_c).sum())


def solution_error(field: object, *, n: int = 120) -> float:
    """Relative L2 error of ``u`` against the exact wave on a dense grid."""
    ts = torch.linspace(0.0, T_END, n, dtype=DTYPE)
    xs = torch.linspace(X_LO, X_HI, n, dtype=DTYPE)
    grid_t, grid_x = torch.meshgrid(ts, xs, indexing="ij")
    coords = torch.stack((grid_t.reshape(-1), grid_x.reshape(-1)), dim=-1).requires_grad_(True)
    pred = predict(field, coords).detach()
    truth = exact_u(coords).detach()
    return float((pred - truth).norm() / truth.norm())


# ---------------------------------------------------------------- part 1 ----


def _check_analytic_ground_truth() -> None:
    """The exact potential really does generate the exact flux pair."""
    coords = torch.tensor(
        [[0.0, -0.3], [0.3, -0.2], [0.5, 0.1], [T_END, 0.6]],
        dtype=DTYPE,
        requires_grad=True,
    )
    grad_p = torch.autograd.grad(exact_potential(coords).sum(), coords, create_graph=True)
    rho, flux = grad_p[0][:, 1], -grad_p[0][:, 0]
    u = exact_u(coords)
    u_x = torch.autograd.grad(u.sum(), coords, create_graph=True)[0][:, 1]
    constitutive = 0.5 * u.pow(2) - NU * u_x
    d_rho = float((rho - u).abs().max().detach())
    d_flux = float((flux - constitutive).abs().max().detach())
    print(f"    analytic check: |d_x P - u| = {d_rho:.1e}, |-d_t P - flux| = {d_flux:.1e}")
    assert d_rho < 1e-12 and d_flux < 1e-12, (d_rho, d_flux)


def part_one() -> None:
    print("[1] conservation is structural, and the nesting is what makes it so")
    _check_analytic_ground_truth()
    torch.manual_seed(SEED)
    coords = (
        torch.stack(
            (
                torch.rand(256, dtype=DTYPE) * T_END,
                X_LO + torch.rand(256, dtype=DTYPE) * (X_HI - X_LO),
            ),
            dim=-1,
        )
    ).requires_grad_(True)

    cage = build_cage_field()
    fresh = structural_divergence(cage, coords)
    print(f"    cage over partition, untrained : rel |div G| = {fresh:.2e}")

    train(cage, steps=60)
    trained = structural_divergence(cage, coords)
    print(f"    cage over partition, trained   : rel |div G| = {trained:.2e}")

    # The reverse nesting: divergence-free patches blended by a partition.
    def caged_patch(index: int) -> FluxFormField:
        return FluxFormField(
            base=_patch_factory(16, POTENTIAL)(index),
            potential_names=("P",),
            flux_names=("rho", "F"),
        )

    reversed_nesting = PartitionedField(
        coordinate_spec=CoordinateSpec(AXES),
        components=ComponentSpec(("rho", "F")),
        subfields=[caged_patch(0), caged_patch(1)],
        split_dirs=torch.tensor([[-C0, 1.0]], dtype=DTYPE),
        split_thresh=torch.tensor([0.0], dtype=DTYPE),
        beta=BETA,
        trainable_partition=True,
    )
    broken = structural_divergence(reversed_nesting, coords)
    print(f"    partition over cage (rejected) : rel |div G| = {broken:.2e}")
    print(f"    -> the chosen nesting is {broken / max(trained, 1e-300):.1e}x tighter")

    assert fresh < 1e-12, fresh
    assert trained < 1e-12, trained
    assert broken > 1e-2, broken


# ---------------------------------------------------------------- part 2 ----


def part_two() -> dict[str, float]:
    print("\n[2] the physics, at equal architecture / parameters / seed / budget")
    print(f"    nu = {NU:g}; layer half-width 2nu/a = {layer_width():.3f}")
    print(f"    exact speed s = (u_L + u_R)/2 = {C0} (Rankine-Hugoniot for flux u^2/2)")

    measured: dict[str, float] = {}
    for label, build in (("conservative", build_cage_field), ("baseline", build_plain_field)):
        torch.manual_seed(SEED)
        field = build()
        train(field)
        speed = shock_speed(field)
        measured[f"{label}_speed_error"] = abs(speed - C0)
        measured[f"{label}_mass_error"] = mass_balance_error(field)
        measured[f"{label}_l2"] = solution_error(field)
        print(
            f"    {label:13s}: speed = {speed:.5f} (|err| "
            f"{measured[f'{label}_speed_error']:.2e})   "
            f"mass |err| = {measured[f'{label}_mass_error']:.2e}   "
            f"rel-L2 = {measured[f'{label}_l2']:.2e}"
        )

    ratio = measured["baseline_mass_error"] / measured["conservative_mass_error"]
    print(f"    -> conservative arm holds global mass balance {ratio:.1f}x tighter")
    assert math.isfinite(measured["conservative_speed_error"]), "no front formed"
    return measured


# ---------------------------------------------------------------- part 3 ----


def part_three() -> None:
    print("\n[3] sharpening the front (temperature collapse) with a sound gap")
    torch.manual_seed(SEED)
    cage = build_cage_field()
    train(cage, steps=300)

    field = cage.base
    grid = torch.stack(
        (
            torch.linspace(0.0, T_END, 60, dtype=DTYPE).repeat_interleave(60),
            torch.linspace(X_LO, X_HI, 60, dtype=DTYPE).repeat(60),
        ),
        dim=-1,
    )
    params = PartitionParams(
        PartitionConfig(n_features=2, depth=1, split_kind="oblique"),
        field.split_W.detach().numpy(),
        field.split_t.detach().numpy(),
    )
    coords = grid.clone().requires_grad_(True)
    gaps: list[float] = []
    for beta in (BETA, 4.0 * BETA, 16.0 * BETA):
        certificate = certify_partition_gap(params, grid.numpy(), beta=beta)
        gaps.append(certificate.mean_gap)
        print(
            f"    beta = {beta:6.1f}  certified soft->hard L1 gap: mean <= "
            f"{certificate.mean_gap:.3e}, max <= {certificate.max_gap:.3e}  "
            f"(sound={certificate.is_sound})  rel |div G| = "
            f"{structural_divergence(cage, coords):.2e}"
        )
        assert certificate.is_sound

    # The max gap stays near 1 at every beta and that is correct, not a failure:
    # a dense grid always contains a point *on* the seam, where soft and hard
    # genuinely disagree by a full unit. Sharpening shrinks the width of that
    # region, which is what the mean sees.
    assert gaps[-1] < gaps[0], gaps
    print("    the mean gap shrinks with beta; conservation does not move at all")


def main() -> None:
    torch.set_default_dtype(DTYPE)
    print("=== Conservative shock-capturing Burgers PINN ===")
    part_one()
    measured = part_two()
    part_three()

    # Every threshold below is read off benchmarks/burgers_shock_conservation.py
    # (6 viscosities x 5 seeds), not guessed. At nu = 3e-3 the conservative arm
    # held tighter mass balance on 5 seeds out of 5, by a factor of 1.5x to 3.8x;
    # the worst single-seed values were 1.01e-1 (mass) and 4.14e-2 (speed).
    assert measured["conservative_mass_error"] < measured["baseline_mass_error"], measured
    assert measured["conservative_mass_error"] < 1.5e-1, measured
    assert measured["conservative_speed_error"] < 6.0e-2, measured

    # Deliberately *not* asserted: that the conservative arm also wins on shock
    # speed and relative L2. It does at this viscosity, but only on 3 and 4 seeds
    # out of 5, and the ordering reverses entirely once the layer is well
    # resolved. Asserting it would be asserting noise.
    print("\nConservation is exact by construction; all checks passed.")


if __name__ == "__main__":
    main()
