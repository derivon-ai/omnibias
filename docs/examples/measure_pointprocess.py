# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation smoke for omnibias-measure temporal point processes & survival.

Run:

    pip install "omnibias-measure[torch,jax,test]"
    JAX_PLATFORMS=cpu python docs/examples/measure_pointprocess.py

The intensity / hazard of a temporal point process is integrated into its
**compensator** ``Lambda = int lambda`` -- the term normally Monte-Carlo /
quadrature approximated. This smoke exercises the ``omnibias-dev-empirical-
validation`` gates on that integral:

* **analytic oracle** -- the compensator, Poisson log-likelihood and the
  right-censored survival log-likelihood are checked against closed-form
  answers (homogeneous Poisson, exponential and polynomial-Weibull hazards);
* **best-in-class** -- the omnibias routes (the closed-form antiderivative
  window and the Gauss-Legendre measure quadrature) are compared to the named
  classical baseline: a left-Riemann compensator at the *same node budget*;
* **train-through** -- a log-linear intensity is fit by maximum likelihood and
  the negative log-likelihood must strictly drop;
* **parity** -- the torch and jax twins agree to float64 tolerance.

Honesty labels: :func:`closed_form_compensator` is **closed form** (the exact
antiderivative ``S`` with ``S' = sigma`` from the activation registry -- e.g.
``sigma``'s antiderivative is ``softplus``); :func:`compensator` is a
**numerical** Gauss-Legendre quadrature (exact only for polynomial intensities).
"""

from __future__ import annotations

import math

import torch
from omnibias.measure.torch import pointprocess as T

torch.set_default_dtype(torch.float64)

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.measure.jax import pointprocess as J

    _HAS_JAX = True
except ImportError:  # pragma: no cover - the [jax] extra ships jax
    _HAS_JAX = False


def check(name: str, got: float, want: float, *, tol: float = 1e-9) -> None:
    err = abs(got - want)
    status = "ok" if err <= tol else "FAIL"
    print(f"  [{status}] {name:44s} got={got:+.10f} want={want:+.10f} err={err:.2e}")
    if err > tol:
        raise SystemExit(f"{name}: error {err:.2e} exceeds tol {tol:.1e}")


def main() -> None:
    print("omnibias-measure :: temporal point process / survival validation\n")

    # ------------------------------------------------------------------ #
    # 1. Closed-form compensator vs analytic (closed form -- no quadrature)
    # ------------------------------------------------------------------ #
    print("closed-form antiderivative compensator (S' = sigma):")
    w, b, t0, t1 = 0.7, -0.3, 0.0, 2.5
    exp_ana = (math.exp(w * t1 + b) - math.exp(w * t0 + b)) / w
    check("exp intensity  int exp(wt+b)", float(T.closed_form_compensator("exp", w, b, t0, t1)), exp_ana, tol=1e-12)

    # sigma intensity: antiderivative is softplus; cross-check vs high-order GL.
    cf_sig = float(T.closed_form_compensator("sigmoid", 1.3, -0.5, 0.0, 3.0, scale=2.0))
    gl_sig = float(T.compensator(lambda t: 2.0 * torch.sigmoid(1.3 * t - 0.5), 0.0, 3.0, num=128))
    check("sigma intensity  closed vs GL", cf_sig, gl_sig, tol=1e-10)

    # ------------------------------------------------------------------ #
    # 2. Best-in-class: GL / closed-form vs coarse left-Riemann baseline
    # ------------------------------------------------------------------ #
    print("\nbest-in-class vs named baseline (left-Riemann, equal node budget):")
    n = 16
    grid = torch.linspace(t0, t1, n + 1)[:-1]
    dx = (t1 - t0) / n
    riemann = float((torch.exp(w * grid + b) * dx).sum())
    gl = float(T.compensator(lambda t: torch.exp(w * t + b), t0, t1, num=n))
    err_riemann, err_gl = abs(riemann - exp_ana), abs(gl - exp_ana)
    print(f"  left-Riemann (n={n}) err = {err_riemann:.3e}")
    print(f"  Gauss-Legendre (n={n}) err = {err_gl:.3e}   (>= {err_riemann / max(err_gl, 1e-300):.1e}x tighter)")
    if not err_gl < 1e-6 * err_riemann:
        raise SystemExit("Gauss-Legendre did not beat the Riemann baseline")

    # ------------------------------------------------------------------ #
    # 3. Poisson process log-likelihood vs analytic (homogeneous)
    # ------------------------------------------------------------------ #
    print("\ninhomogeneous Poisson NLL (homogeneous oracle mu*T - n*log mu):")
    mu, horizon = 1.7, 4.0
    events = torch.tensor([0.3, 1.1, 2.7, 3.5])
    nll = float(T.poisson_nll(lambda t: mu * torch.ones_like(t), events, 0.0, horizon, num=32))
    check("homogeneous NLL", nll, mu * horizon - len(events) * math.log(mu), tol=1e-9)

    # ------------------------------------------------------------------ #
    # 4. Survival / hazard log-likelihood vs analytic
    # ------------------------------------------------------------------ #
    print("\nright-censored survival NLL (cumulative hazard = survival compensator):")
    lam = 0.9
    d = torch.tensor([0.5, 1.2, 2.0, 0.8])
    obs = torch.tensor([1.0, 0.0, 1.0, 1.0])
    snll = float(T.survival_nll(lambda t: lam * torch.ones_like(t), d, obs, num=16))
    exp_surv = -float((obs * math.log(lam) - lam * d).sum())
    check("exponential hazard", snll, exp_surv, tol=1e-9)
    # Weibull shape k=2: h(t)=2t, H(t)=t^2 (polynomial -> GL exact).
    dw = torch.tensor([0.5, 1.5, 2.0])
    ow = torch.tensor([1.0, 1.0, 0.0])
    snll_w = float(T.survival_nll(lambda t: 2.0 * t, dw, ow, num=16))
    ana_w = -float((ow * torch.log(2.0 * dw) - dw**2).sum())
    check("polynomial-Weibull hazard", snll_w, ana_w, tol=1e-9)

    # ------------------------------------------------------------------ #
    # 5. Train-through: fit a log-linear intensity by maximum likelihood
    # ------------------------------------------------------------------ #
    print("\ntrain-through (fit softplus(a t + c) intensity, NLL must drop):")
    torch.manual_seed(0)

    class Intensity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.0))
            self.c = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.a * t + self.c)

    tpp = T.TemporalPointProcess(Intensity(), num=64)
    late_events = torch.tensor([0.4, 1.3, 1.9, 2.2, 2.6, 2.8])
    opt = torch.optim.Adam(tpp.parameters(), lr=0.1)
    nll0 = float(tpp.nll(late_events, 0.0, 3.0).detach())
    for _ in range(80):
        opt.zero_grad()
        loss = tpp(late_events, 0.0, 3.0)
        loss.backward()
        opt.step()
    nll1 = float(tpp.nll(late_events, 0.0, 3.0).detach())
    slope = float(tpp.intensity.a.detach())
    print(f"  NLL {nll0:+.5f} -> {nll1:+.5f}   recovered slope a = {slope:+.4f} (events cluster late)")
    if not (nll1 < nll0 - 1e-3 and slope > 0.0):
        raise SystemExit("train-through did not improve / recover an increasing intensity")

    # ------------------------------------------------------------------ #
    # 6. Cross-backend parity (torch <-> jax)
    # ------------------------------------------------------------------ #
    if _HAS_JAX:
        print("\ntorch <-> jax parity:")
        cj = float(J.closed_form_compensator("sigmoid", 1.3, -0.5, 0.0, 3.0, scale=2.0))
        check("closed-form compensator parity", cf_sig, cj, tol=1e-11)
        ev_j = jnp.array([0.3, 1.1, 2.7, 3.5])
        nj = float(J.poisson_nll(lambda t: mu * jnp.ones_like(t), ev_j, 0.0, horizon, num=32))
        check("Poisson NLL parity", nll, nj, tol=1e-9)
    else:  # pragma: no cover
        print("\n(jax not installed -- skipping parity)")

    print("\nall temporal-point-process / survival checks passed.")


if __name__ == "__main__":
    main()
