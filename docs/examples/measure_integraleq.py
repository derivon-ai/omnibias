# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation smoke for Fredholm / Volterra integral equations on omnibias-measure.

Run:

    pip install "omnibias-measure[torch,jax,test]" "omnibias-pinn[torch,integral]"
    JAX_PLATFORMS=cpu python docs/examples/measure_integraleq.py

An integral equation of the second kind

    u(x) = f(x) + lam * int_Omega K(x, t) u(t) dmu(t)

is the mirror image of a differential equation: a derivative reads the solution
locally, an integral operator reads all of it at once. Nystrom discretisation
turns it into ``(I - lam K W) u = f``, and because a ``Measure`` already *is*
nodes and weights, that system is one outer product away from the measure
integral the rest of the package is built on.

This smoke exercises the ``omnibias-dev-empirical-validation`` gates:

* **analytic oracle** -- a separable kernel has a closed-form solution, and
  ``degenerate_kernel_solve`` reproduces it exactly (the kernel is never
  discretised, only scalar moments are);
* **best-in-class** -- Gauss-Legendre Nystrom versus the named classical
  baseline, trapezoid Nystrom, at the *same node budget*;
* **honest failure** -- a Neumann series outside its convergence radius, and a
  solve at a Fredholm alternative, must both say so rather than return a
  plausible number;
* **train-through** -- a PINN drives the ``Fredholm`` residual to zero and
  recovers the analytic solution as a differentiable function;
* **parity** -- the numpy, torch and jax twins agree to float64 tolerance.

Honesty labels: ``degenerate_kernel_solve`` is **exact in the kernel**;
``nystrom_solve`` is **numerical** at the measure's own quadrature order (so
spectral for Gauss-Legendre on a smooth kernel); ``volterra_solve`` is
**numerical**, second order, set by its cumulative rule rather than the measure.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from omnibias.measure._core import integraleq as C
from omnibias.measure._core.measure import Measure, lebesgue
from omnibias.measure.torch import integraleq as T

torch.set_default_dtype(torch.float64)

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.measure.jax import integraleq as J

    _HAS_JAX = True
except ImportError:  # pragma: no cover - the [jax] extra ships jax
    _HAS_JAX = False


def check(name: str, got: float, want: float, *, tol: float = 1e-9) -> None:
    err = abs(got - want)
    status = "ok" if err <= tol else "FAIL"
    print(f"  [{status}] {name:46s} got={got:+.10f} want={want:+.10f} err={err:.2e}")
    if err > tol:
        raise SystemExit(f"{name}: error {err:.2e} exceeds tol {tol:.1e}")


def trapezoid_measure(a: float, b: float, n: int) -> Measure:
    """The named classical baseline: a composite trapezoid rule as a Measure."""
    nodes = np.linspace(a, b, n)
    h = (b - a) / (n - 1)
    weights = np.full(n, h)
    weights[0] = weights[-1] = h / 2.0
    return Measure(nodes=nodes.reshape(-1, 1), weights=weights)


def main() -> None:
    print("omnibias-measure :: Fredholm / Volterra integral equations\n")

    # ------------------------------------------------------------------ #
    # 1. Analytic oracle: a separable kernel collapses to a closed form
    # ------------------------------------------------------------------ #
    # u = 1 + lam int_0^1 x t u(t) dt has u = 1 + c x with
    # c = lam int_0^1 t (1 + c t) dt  =>  c = (lam/2) / (1 - lam/3).
    print("separable kernel K = x t on [0,1], f = 1  (closed form u = 1 + c x):")
    lam = 0.7
    c = (lam / 2.0) / (1.0 - lam / 3.0)
    mu = lebesgue([(0.0, 1.0)], 24)
    exact = 1.0 + c * mu.nodes[:, 0]

    def k_sep(x, t):
        return x[:, :1] * t[:, 0][None, :]

    def f_one(p):
        return np.ones(p.shape[0])

    deg = C.degenerate_kernel_solve(
        [(lambda p: p[:, 0], lambda p: p[:, 0])], f_one, mu, lam=lam
    )
    nys = C.nystrom_solve(k_sep, f_one, mu, lam=lam)
    check("degenerate solve (exact in kernel)", float(np.abs(deg - exact).max()), 0.0, tol=1e-14)
    check("Nystrom solve (numerical)", float(np.abs(nys - exact).max()), 0.0, tol=1e-13)

    # ------------------------------------------------------------------ #
    # 2. Best-in-class: Gauss-Legendre vs trapezoid at equal node budget
    # ------------------------------------------------------------------ #
    # A smooth non-separable kernel, so neither rule is exact by construction.
    print("\nbest-in-class vs named baseline (trapezoid Nystrom, equal node budget):")

    def k_smooth(x, t):
        return np.cos(x[:, :1] - t[:, 0][None, :])

    def f_smooth(p):
        return np.exp(-p[:, 0])

    def nystrom_at(measure, values, xs, lam):
        """The Nystrom interpolant ``u(x) = f(x) + lam int K(x,t) u(t) dmu``.

        The right way to read a Nystrom solution off its nodes: it is exact given
        the nodal values, so comparing two rules through it measures the
        quadrature and nothing else. Linear interpolation would instead measure
        how unevenly Gauss-Legendre spaces its nodes, which is not the question.
        """
        pts = xs.reshape(-1, 1)
        k = k_smooth(pts, measure.nodes)
        return f_smooth(pts) + lam * (k * measure.weights[None, :]) @ values

    n = 16
    lam_s = 0.5
    reference = lebesgue([(0.0, 1.0)], 200)
    probe = np.linspace(0.05, 0.95, 21)
    ref_probe = nystrom_at(
        reference, C.nystrom_solve(k_smooth, f_smooth, reference, lam=lam_s), probe, lam_s
    )

    gl = lebesgue([(0.0, 1.0)], n)
    tz = trapezoid_measure(0.0, 1.0, n)
    err_gl = float(
        np.abs(nystrom_at(gl, C.nystrom_solve(k_smooth, f_smooth, gl, lam=lam_s), probe, lam_s) - ref_probe).max()
    )
    err_tz = float(
        np.abs(nystrom_at(tz, C.nystrom_solve(k_smooth, f_smooth, tz, lam=lam_s), probe, lam_s) - ref_probe).max()
    )
    print(f"  trapezoid      (n={n}) err = {err_tz:.3e}")
    print(f"  Gauss-Legendre (n={n}) err = {err_gl:.3e}   ({err_tz / max(err_gl, 1e-300):.1e}x tighter)")
    if not err_gl < 1e-6 * err_tz:
        raise SystemExit("Gauss-Legendre did not beat the trapezoid baseline")

    # ------------------------------------------------------------------ #
    # 3. Volterra: causal, always solvable, second order
    # ------------------------------------------------------------------ #
    print("\ncausal Volterra  u(x) = 1 + int_0^x u(t) dt  (exact u = e^x):")
    orders = []
    prev = None
    for m in (64, 128, 256):
        nodes = np.linspace(0.0, 1.0, m).reshape(-1, 1)
        got = C.volterra_solve(
            lambda x, t: np.ones((x.shape[0], t.shape[0])),
            lambda p: np.ones(p.shape[0]),
            Measure(nodes=nodes, weights=np.full(m, 1.0 / m)),
            lam=1.0,
        )
        err = float(np.abs(got - np.exp(nodes[:, 0])).max())
        if prev is not None:
            orders.append(math.log2(prev / err))
        print(f"  n={m:4d}  max err = {err:.3e}" + (f"   order ~ {orders[-1]:.2f}" if orders else ""))
        prev = err
    if not all(o > 1.8 for o in orders):
        raise SystemExit(f"Volterra convergence order below the claimed 2: {orders}")

    # ------------------------------------------------------------------ #
    # 4. Honest failure I: a Neumann series outside its radius
    # ------------------------------------------------------------------ #
    print("\nNeumann series -- converged flag is earned by the measured residual:")
    inside = C.neumann_series(k_sep, f_one, mu, lam=0.5)
    outside = C.neumann_series(k_sep, f_one, mu, lam=6.0, max_terms=60)
    print(f"  lam=0.5  rho={inside.spectral_radius:.4f}  converged={inside.converged}  residual={inside.residual:.2e}")
    print(f"  lam=6.0  rho={outside.spectral_radius:.4f}  converged={outside.converged}  residual={outside.residual:.2e}")
    if not inside.converged:
        raise SystemExit("Neumann series failed inside its convergence radius")
    if outside.converged:
        raise SystemExit("Neumann series claimed convergence outside its radius")
    check("converged series matches the oracle", float(np.abs(inside.solution - C.nystrom_solve(k_sep, f_one, mu, lam=0.5)).max()), 0.0, tol=1e-9)

    # ------------------------------------------------------------------ #
    # 5. Honest failure II: a Fredholm alternative
    # ------------------------------------------------------------------ #
    # int_0^1 t^2 dt = 1/3, so 1/lam is an eigenvalue of the operator at lam = 3.
    print("\nFredholm alternative at lam = 3 (1/lam is an eigenvalue of K W):")
    try:
        C.nystrom_solve(k_sep, f_one, mu, lam=3.0)
    except ValueError as exc:
        print(f"  [ok] refused: {str(exc)[:88]}...")
    else:
        raise SystemExit("a solve at a Fredholm alternative was not refused")

    unguarded = C.nystrom_solve(k_sep, f_one, mu, lam=3.0, check_conditioning=False)
    print(f"  without the screen it returns a finite vector of size {np.abs(unguarded).max():.2e} -- the quiet wrong answer")
    margin = C.solvability_margin(
        np.linalg.svd(
            np.eye(mu.n_nodes) - 3.0 * (mu.nodes[:, :1] * (mu.nodes[:, 0] * mu.weights)),
            compute_uv=False,
        )
    )
    print(f"  solvability margin at lam=3: {margin:.2e}  (floor {C.SINGULAR_RCOND:g})")

    # ------------------------------------------------------------------ #
    # 6. Train-through: a PINN drives the Fredholm residual to zero
    # ------------------------------------------------------------------ #
    print("\ntrain-through (PINN Fredholm residual -> 0, solution as a function):")
    from omnibias.pinn._core.components import ComponentSpec
    from omnibias.pinn._core.coords import CoordinateSpec
    from omnibias.pinn.torch import equations as teq
    from omnibias.pinn.torch import ops as tops
    from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField

    collocation = torch.linspace(0.0, 1.0, 41).reshape(-1, 1)
    quadrature = lebesgue([(0.0, 1.0)], 20)
    truth = 1.0 + c * collocation[:, 0]

    def fresh_field() -> OneLayerVectorField:
        torch.manual_seed(0)
        return OneLayerVectorField(
            coordinate_spec=CoordinateSpec(
                axes=("x",), periodicity=(False,), time_axis=None
            ),
            components=ComponentSpec(names=("u",), groups={}),
            hidden=24,
            base="tanh",
        )

    def train(field, loss_fn, steps: int = 400):
        """Adam for the shape, then L-BFGS for the digits."""
        adam = torch.optim.Adam(field.parameters(), lr=0.02)
        for _ in range(steps):
            adam.zero_grad()
            loss_fn().backward()
            adam.step()
        lbfgs = torch.optim.LBFGS(
            field.parameters(), max_iter=150, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        lbfgs.step(closure)

    solved = fresh_field()

    def residual_loss() -> torch.Tensor:
        out = teq.fredholm(
            solved(collocation),
            kernel=lambda x, t: x[:, :1] * t[:, 0][None, :],
            measure=quadrature,
            lam=lam,
            source=lambda state: torch.ones(state.coords.shape[0]),
        )
        return (out.residual**2).mean()

    loss0 = float(residual_loss().detach())
    train(solved, residual_loss)
    loss1 = float(residual_loss().detach())
    sup_err = float(
        (tops.value(solved(collocation), "u").detach() - truth).abs().max()
    )

    # The calibration that makes the number mean something: hand a second, identical
    # field the answer and fit it directly. That is the best this architecture can do
    # on this target, so it is the floor the residual-trained one is judged against --
    # a raw sup error would be measuring tanh's difficulty with a straight line.
    supervised = fresh_field()
    train(
        supervised,
        lambda: ((tops.value(supervised(collocation), "u") - truth) ** 2).mean(),
    )
    floor = float(
        (tops.value(supervised(collocation), "u").detach() - truth).abs().max()
    )

    print(f"  residual loss {loss0:.3e} -> {loss1:.3e}")
    print(f"  sup|u - (1 + {c:.4f} x)|:  from the residual {sup_err:.3e}")
    print(f"                               supervised floor {floor:.3e}  ({sup_err / floor:.2f}x)")
    if not (loss1 < 1e-4 * loss0):
        raise SystemExit("the PINN did not drive the Fredholm residual down")
    if not sup_err < 3.0 * floor:
        raise SystemExit(
            f"solving from the residual ({sup_err:.2e}) was much worse than being "
            f"handed the answer ({floor:.2e})"
        )

    # ------------------------------------------------------------------ #
    # 7. Cross-backend parity
    # ------------------------------------------------------------------ #
    print("\nnumpy <-> torch <-> jax parity:")
    tor = T.nystrom_solve(
        lambda x, t: x[:, :1] * t[:, 0][None, :],
        lambda p: torch.ones(p.shape[0], dtype=p.dtype),
        mu,
        lam=lam,
    )
    check("torch Nystrom", float(np.abs(tor.numpy() - nys).max()), 0.0, tol=1e-13)
    if _HAS_JAX:
        jx = J.nystrom_solve(
            lambda x, t: x[:, :1] * t[:, 0][None, :],
            lambda p: jnp.ones(p.shape[0], dtype=p.dtype),
            mu,
            lam=lam,
        )
        check("jax Nystrom", float(np.abs(np.asarray(jx) - nys).max()), 0.0, tol=1e-13)
    else:  # pragma: no cover
        print("  (jax not installed -- skipping)")

    # Gradients flow into the kernel, which is why a learned kernel is possible.
    theta = torch.tensor(0.8, requires_grad=True)
    u = T.nystrom_solve(
        lambda x, t: theta * x[:, :1] * t[:, 0][None, :],
        lambda p: torch.ones(p.shape[0], dtype=p.dtype),
        mu,
        lam=lam,
    )
    u.sum().backward()
    print(f"  d(sum u)/d(kernel scale) = {float(theta.grad):+.6f}  (differentiable through the solve)")
    if not abs(float(theta.grad)) > 0.0:
        raise SystemExit("no gradient reached the kernel parameter")

    print("\nall integral-equation checks passed.")


if __name__ == "__main__":
    main()
