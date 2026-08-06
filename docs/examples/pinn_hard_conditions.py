# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard Neumann / Robin / initial conditions, and the solver that finds them.

Run::

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_hard_conditions.py

A PINN usually meets its boundary and initial conditions the way it meets
everything else: as penalty terms, weighted against the interior residual. That
makes the condition a *training outcome*, and it is the term that most often
loses -- the gradients of a boundary penalty and a PDE residual live on
different scales, which is the multi-term stiffness that ``condition_weight``
exists to paper over.

``ConstrainedExpressionField`` removes the term instead of weighting it. For
linear conditions ``C_k[u] = t_k`` pick support functions ``s_j``, form the
support matrix ``M_kj = C_k[s_j]``, and define switching functions
``phi_i = sum_j (M^-1)_ji s_j``, which satisfy ``C_k[phi_i] = delta_ki``. Then

    u = g + sum_k phi_k (t_k - C_k[g])

satisfies every condition for **any** free function ``g``. This is the Theory of
Functional Connections (Mortari 2017; Leake and Mortari, *Mathematics* 8(8):1303,
2020). Nothing is fitted: it is an algebraic identity, so it survives an
untrained network, a randomised one, and every optimiser step in between.

Part 1 shows that on the four condition kinds. Part 2 shows the precondition
being *certified* rather than assumed, and refused when it fails. Part 3 hands
the whole thing to the solver, which works out which conditions it can absorb
and deletes those rows from the loss. Part 4 goes to three axes and to a
periodic seam, which the same recursion covers without a special case.

Honesty, in three parts.

*Structural.* The conditions hold to machine epsilon with no training and no
tolerance, and the support-matrix invertibility carries a hash-sealed
certificate over a finite rational obligation.

*Measured.* Interior accuracy is optimised, not proven. What
``benchmarks/hard_conditions_solver.py`` measured over five seeds at equal
architecture, parameter count and collocation budget: the hard arm's worst
boundary violation over every cell was ``1.4e-14``, and its median interior
relative L2 was better on Poisson (``3.5e-07`` vs ``1.6e-06``), heat
(``3.8e-06`` vs ``1.2e-02``), wave (``1.1e-06`` vs ``3.1e-03``) and the 2-D
square (``2.8e-05`` vs ``7.3e-02``), winning 5 seeds out of 5 on all four. The
parabolic and hyperbolic gaps are the large ones because that is where the soft
arm has an initial condition competing with the interior residual. **On the
periodic seam it lost**: the seam closes exactly, but the interior fit was ~3x
worse on every seed, because two degrees of freedom go into tying the ends
together. Absorption buys a guarantee, and on that problem the guarantee is not
free.

*Out of scope.* The domain must be an axis-aligned box; arbitrary geometry is
what the distance-function ``HardBoundaryField`` is for. A condition whose
support matrix will not certify is declined with a reason and stays soft, which
is exactly what the solver does today.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.constrained import (
    HardCondition,
    derivative_at,
    dirichlet,
    neumann,
    periodic,
    robin,
)
from omnibias.pinn.solver import CollocationSpec, Domain, array_namespace, heat, poisson
from omnibias.pinn.solver._core.hard import plan_hard_conditions
from omnibias.pinn.solver.torch import solve_least_squares
from omnibias.pinn.solver.torch.assemble import condition_residual
from omnibias.pinn.torch.cage import ConstrainedExpressionField
from omnibias.pinn.torch.fields import OneLayerVectorField

DTYPE = torch.float64
DOMAIN = ((0.0, 1.0), (0.0, 1.0))
SEED = 0
HIDDEN = 32

#: What "exact" means here: float64 round-off, not a tuned tolerance.
EXACT = 1e-13


# ------------------------------------------------------------------ set-up ---


def build_base(seed: int = SEED, hidden: int = HIDDEN) -> OneLayerVectorField:
    """The free function ``g``. Nothing about it knows the conditions exist."""
    torch.manual_seed(seed)
    return OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("t", "x"), domain=DOMAIN, time_axis="t"),
        components=ComponentSpec(("u",)),
        hidden=hidden,
        base="tanh",
        dtype=DTYPE,
    )


def face(axis: int, value: float, n: int = 64, seed: int = 3) -> torch.Tensor:
    """``n`` random points on one face of the box."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(n, 2, generator=g, dtype=DTYPE)
    pts[:, axis] = value
    return pts


def value_at(field: object, coords: torch.Tensor) -> torch.Tensor:
    state = field(coords)
    return state.ops.value(state, "u")


def derivative_at_points(
    field: object, coords: torch.Tensor, axis: int, order: int
) -> torch.Tensor:
    state = field(coords)
    return state.ops.derivative(state, "u", axis=axis, order=order)


def randomise(field: torch.nn.Module, seed: int = 11) -> None:
    """Move every parameter far from its initialisation.

    Exactness that survives this is structural rather than fitted -- which is
    the entire claim, so it is worth checking twice rather than once.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in field.parameters():
            p.add_(torch.randn(p.shape, generator=g, dtype=p.dtype) * 0.75)


def worst(got: torch.Tensor, want: torch.Tensor | float) -> float:
    return float((got - want).detach().abs().max())


# ---------------------------------------------------------------- part 1 -----


def initial_value(c: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * c[:, 1] / 2.0)


def initial_velocity(c: torch.Tensor) -> torch.Tensor:
    r"""``cos(pi x / 2)^2``, and the shape is forced rather than chosen.

    The Dirichlet data ``u(t, 0) = sin t`` has ``u_t(0, 0) = 1``, so the initial
    velocity must be ``1`` at ``x = 0``; the Neumann data ``u_x(t, 1) = 0`` is
    constant in ``t``, so ``d_x u_t(0, 1)`` must vanish. Both hold here. Data
    that misses either is not something an ansatz can repair -- part 2 measures
    what happens when it does.
    """
    return torch.cos(math.pi * c[:, 1] / 2.0) ** 2


def part_one() -> None:
    print("[1] four condition kinds, all exact, none of them trained")
    # Dirichlet and Neumann on x; value and velocity at t = 0. A target may be
    # a constant or a callable of the *other* coordinates, which is how the
    # Dirichlet face carries time-dependent data.
    cage = ConstrainedExpressionField(
        base=build_base(),
        conditions=[
            HardCondition("u", 1, dirichlet(0.0), lambda c: torch.sin(c[:, 0])),
            HardCondition("u", 1, neumann(1.0), 0.0),
            HardCondition("u", 0, dirichlet(0.0), initial_value),
            HardCondition("u", 0, derivative_at(0.0, 1), initial_velocity),
        ],
    )
    lo, hi, t0 = face(1, 0.0), face(1, 1.0), face(0, 0.0)

    for label in ("untrained", "randomised"):
        checks = {
            "dirichlet u(t,0) = sin t": worst(
                value_at(cage, lo), torch.sin(lo[:, 0])
            ),
            "neumann  u_x(t,1) = 0": worst(
                derivative_at_points(cage, hi, 1, 1), 0.0
            ),
            "initial  u(0,x) = sin(pi x/2)": worst(
                value_at(cage, t0), initial_value(t0)
            ),
            "velocity u_t(0,x) = cos^2": worst(
                derivative_at_points(cage, t0, 0, 1), initial_velocity(t0)
            ),
        }
        print(f"    {label}:")
        for name, err in checks.items():
            print(f"      {name:32s} |err| = {err:.1e}")
            assert err < EXACT, (name, err)
        randomise(cage)

    # The corner is where a naive per-axis ansatz goes wrong: both axes claim
    # the point (t=0, x=0), and the recursion's cross term is what reconciles
    # them.
    corner = torch.zeros(1, 2, dtype=DTYPE)
    corner_err = abs(float(value_at(cage, corner).detach()[0]))
    print(f"    corner (t=0, x=0) claimed by both axes: |err| = {corner_err:.1e}")
    assert corner_err < EXACT, corner_err

    g = torch.Generator().manual_seed(5)
    sample = torch.rand(64, 2, generator=g, dtype=DTYPE)
    print(f"    cross-axis data agreement: residual {cage.compatibility_residual(sample):.1e}")
    cage.check_compatibility(sample)


# ---------------------------------------------------------------- part 2 -----


def part_two() -> None:
    print("\n[2] the two preconditions, certified and refused rather than assumed")

    # (a) The support matrix must be invertible. That is a finite rational
    # obligation, so it seals into a hash-verifiable certificate.
    cage = ConstrainedExpressionField(
        base=build_base(),
        conditions=[
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 1, robin(1.0, alpha=1.5, beta=0.25), -0.7),
        ],
    )
    cert = cage.support_certificates()["u"][0]
    lam_lo = float.fromhex(cert["payload"]["gram_lambda_min"]["lo"])
    kappa_hi = float.fromhex(cert["payload"]["condition_number"]["hi"])
    print(f"    lambda_min(M^T M) >= {lam_lo:.3e},  kappa(M) <= {kappa_hi:.2f}")
    print(f"    certificate digest verifies: {verify_certificate_digest(cert)}")
    assert verify_certificate_digest(cert)
    assert lam_lo > 0.0

    # ... and a dependent set is refused, not silently approximated.
    try:
        ConstrainedExpressionField(
            base=build_base(),
            conditions=[
                HardCondition("u", 1, dirichlet(0.0), 0.0),
                HardCondition("u", 1, dirichlet(0.0), 1.0),
            ],
        )
    except ValueError as exc:
        print(f"    two conditions at one point are refused: {str(exc)[:58]}...")
    else:  # pragma: no cover -- the refusal is the point
        raise AssertionError("a singular condition set must be refused")

    # (b) Data on different axes must agree where those axes meet. That is a
    # statement about the *data*, not the method, and it is not repairable, so
    # construction refuses it outright. The second case is the one worth knowing
    # about -- the values agree perfectly and only the *derivatives* clash,
    # which is easy to write by accident and impossible to see by inspection.
    g = torch.Generator().manual_seed(5)
    sample = torch.rand(64, 2, generator=g, dtype=DTYPE)
    clashes = {
        "values: u(t,0) = 0 vs u(0,x) = 1 + x^2": [
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 0, dirichlet(0.0), lambda c: 1.0 + c[:, 1] ** 2),
        ],
        "slopes: u(t,0) = sin t vs u_t(0,x) = 0": [
            HardCondition("u", 1, dirichlet(0.0), lambda c: torch.sin(c[:, 0])),
            HardCondition("u", 0, dirichlet(0.0), initial_value),
            HardCondition("u", 0, derivative_at(0.0, 1), 0.0),
        ],
    }
    for label, conditions in clashes.items():
        try:
            ConstrainedExpressionField(base=build_base(), conditions=conditions)
        except ValueError:
            pass
        else:  # pragma: no cover -- the refusal is the point
            raise AssertionError(f"incompatible data must be refused: {label}")
        # Building with the gate off keeps the residual visible as a live
        # falsifier: order one when the clash is real, round-off when it is not.
        bad = ConstrainedExpressionField(
            base=build_base(), conditions=conditions, check_data=False
        )
        residual = bad.compatibility_residual(sample)
        print(f"    {label:40s} -> refused, residual {residual:.2e}")
        assert residual > 0.9, (label, residual)


# ---------------------------------------------------------------- part 3 -----


def _poisson_system():
    """``u'' = -pi^2 sin(pi x)`` on ``[0, 1]`` with zero ends."""
    def source(c):  # noqa: ANN001, ANN202
        xp = array_namespace(c)
        return -(math.pi**2) * xp.sin(math.pi * c[:, 0])

    return poisson(Domain(("x",), ((0.0, 1.0),)), source=source, boundary=0.0)


def _heat_system():
    """``u_t = 0.25 u_xx``, ``u(x,0) = sin(pi x)``, zero ends."""
    def initial(c):  # noqa: ANN001, ANN202
        xp = array_namespace(c)
        return xp.sin(math.pi * c[:, 1])

    dom = Domain(("t", "x"), ((0.0, 0.3), (0.0, 1.0)), time_axis="t")
    return heat(dom, diffusivity=0.25, initial=initial, boundary=0.0)


def part_three() -> None:
    print("\n[3] the solver finds them itself, and drops the rows from the loss")
    spec = CollocationSpec(n_interior=48, n_boundary=16)

    plan = plan_hard_conditions(_heat_system())
    print(f"    heat: {plan.summary()}")
    assert plan.is_total, plan.declined

    for name, build, exact in (
        ("poisson", _poisson_system, lambda p: np.sin(math.pi * p[:, 0])),
        (
            "heat",
            _heat_system,
            lambda p: np.exp(-0.25 * math.pi**2 * p[:, 0])
            * np.sin(math.pi * p[:, 1]),
        ),
    ):
        system = build()
        measured: dict[str, tuple[float, float]] = {}
        for arm, mode in (("hard", "auto"), ("soft", "none")):
            sol = solve_least_squares(
                system,
                hidden=96,
                weight_init_scale=3.0,
                seed=SEED,
                collocation=spec,
                hard_conditions=mode,
            )
            # Re-assembled *ignoring* what the plan absorbed. If the plan ever
            # claimed a condition the cage does not enforce, the loss would
            # stop watching it and nothing else would notice; this is what
            # turns that silence into a number.
            rows = condition_residual(sol.field, sol.system, spec, None)
            violation = float(rows.detach().abs().max()) if rows.numel() else 0.0
            pts = _interior_grid(system)
            u = sol.evaluate(pts, "u").detach().numpy()
            truth = exact(pts)
            rel = float(np.linalg.norm(u - truth) / np.linalg.norm(truth))
            measured[arm] = (violation, rel)
            print(
                f"    {name:8s} {arm}: boundary |err| = {violation:.2e}   "
                f"interior rel-L2 = {rel:.2e}"
            )
        assert measured["hard"][0] < EXACT, measured
        assert measured["hard"][0] < measured["soft"][0], measured


def _interior_grid(system: object, n: int = 40) -> np.ndarray:
    axes = [
        np.linspace(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo), n)
        for lo, hi in system.domain.bounds  # type: ignore[attr-defined]
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


# ---------------------------------------------------------------- part 4 -----

CUBE = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))


def part_four() -> None:
    print("\n[4] three axes, and a seam -- same recursion, no special cases")
    torch.manual_seed(SEED)
    base = OneLayerVectorField(
        coordinate_spec=CoordinateSpec(("t", "x", "y"), domain=CUBE, time_axis="t"),
        components=ComponentSpec(("u",)),
        hidden=HIDDEN,
        base="tanh",
        dtype=DTYPE,
    )
    # An initial value on t, Dirichlet + Neumann on x, and a *periodic seam* on
    # y. The seam is a relative constraint -- it ties u(y=0) to u(y=1) without
    # pinning either -- which the same switching form absorbs because a linear
    # functional may reference more than one point.
    cage = ConstrainedExpressionField(
        base=base,
        conditions=[
            HardCondition("u", 0, dirichlet(0.0), 0.0),
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 1, neumann(1.0), 0.0),
            HardCondition("u", 2, periodic(0.0, 1.0, order=0), 0.0),
            HardCondition("u", 2, periodic(0.0, 1.0, order=1), 0.0),
        ],
    )
    randomise(cage, seed=17)

    gen = torch.Generator().manual_seed(23)
    pts = torch.rand(48, 3, generator=gen, dtype=DTYPE)
    t0, x0, x1 = pts.clone(), pts.clone(), pts.clone()
    t0[:, 0], x0[:, 1], x1[:, 1] = 0.0, 0.0, 1.0
    lo, hi = pts.clone(), pts.clone()
    lo[:, 2], hi[:, 2] = 0.0, 1.0

    checks = {
        "initial u(0,x,y) = 0": worst(value_at(cage, t0), 0.0),
        "dirichlet u(t,0,y) = 0": worst(value_at(cage, x0), 0.0),
        "neumann u_x(t,1,y) = 0": worst(derivative_at_points(cage, x1, 1, 1), 0.0),
        "seam u(.,.,1) - u(.,.,0) = 0": worst(
            value_at(cage, hi), value_at(cage, lo)
        ),
        "seam slope matches too": worst(
            derivative_at_points(cage, hi, 2, 1),
            derivative_at_points(cage, lo, 2, 1),
        ),
    }
    for name, err in checks.items():
        print(f"      {name:32s} |err| = {err:.1e}")
        assert err < EXACT, (name, err)

    # The edge where two constrained axes meet, and the corner where three do.
    edge = torch.rand(16, 3, generator=gen, dtype=DTYPE)
    edge[:, 0], edge[:, 1] = 0.0, 0.0
    print(f"      edge (t=0, x=0)                  |err| = {worst(value_at(cage, edge), 0.0):.1e}")
    assert worst(value_at(cage, edge), 0.0) < EXACT

    # And the cost, stated rather than implied: the recursion multiplies.
    print(
        f"    base evaluations per pass: {cage.projection_cost} "
        f"(product over axes of 1 + #projection points)"
    )

    # The solver reaches the same conclusion on its own, on a square whose four
    # faces all get absorbed -- two constrained spatial axes, no time.
    def source(c):  # noqa: ANN001, ANN202
        xp = array_namespace(c)
        return -2.0 * (math.pi**2) * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    square = poisson(
        Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0))), source=source, boundary=0.0
    )
    plan = plan_hard_conditions(square)
    print(f"    2-D square: {plan.summary()}")
    assert plan.is_total, plan.declined


def main() -> None:
    torch.set_default_dtype(DTYPE)
    print("=== Hard boundary / initial conditions ===")
    part_one()
    part_two()
    part_three()
    part_four()
    print("\nThe conditions are algebra, not training; all checks passed.")


if __name__ == "__main__":
    main()
