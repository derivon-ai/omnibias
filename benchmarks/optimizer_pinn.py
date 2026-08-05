# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""1-D Poisson PINN: exact-curvature optimisers vs Adam / L-BFGS.

Self-contained (uses ``omnibias.torch`` only). The residual Laplacian is the
closed-form ``σ''(z)`` of tanh, so Gauss-Newton / Newton products are
autodiff-exact over a smooth closed-form operator. Run::

    uv run python benchmarks/optimizer_pinn.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from _common import provenance, write_json  # noqa: E402
from omnibias.torch.optim import (  # noqa: E402
    CubicRegularizedGaussNewton,
    GaussNewton,
    TrustRegionNewtonCG,
    functional_residual_fn,
)

torch.set_default_dtype(torch.float64)

H = 16
N_INT = 62
SEEDS = (0, 1, 2, 3, 4)
ADAM_STEPS = 800
ADAM_LR = 1e-2
LBFGS_STEPS = 60
CURVATURE_STEPS = 40
BC_WEIGHT = 20.0


class PoissonResidual(nn.Module):
    """Stacked residual of ``-u'' = π² sin(πx)`` with Dirichlet BCs on [0, 1].

    ``u(x) = c · tanh(W x + β) + b``. The second derivative uses the closed-form
    tanh identity ``σ'' = -2 t (1 - t²)``.
    """

    def __init__(self, hidden: int = H) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden, 1) * 0.5)
        self.beta = nn.Parameter(torch.randn(hidden) * 0.1)
        self.c = nn.Parameter(torch.randn(hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(()))

    def value(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W.T + self.beta
        return (torch.tanh(z) * self.c).sum(-1) + self.b

    def laplacian(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.W.T + self.beta
        t = torch.tanh(z)
        sigma_pp = -2.0 * t * (1.0 - t * t)
        return (sigma_pp * self.c * (self.W[:, 0] ** 2)).sum(-1)

    def forward(
        self, x_int: torch.Tensor, x_bc: torch.Tensor, f_int: torch.Tensor
    ) -> torch.Tensor:
        pde = -self.laplacian(x_int) - f_int
        bc = BC_WEIGHT * self.value(x_bc)
        return torch.cat([pde, bc], dim=0)


def _points() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_int = torch.linspace(0.0, 1.0, N_INT + 2)[1:-1].reshape(-1, 1)
    x_bc = torch.tensor([[0.0], [1.0]])
    f_int = (math.pi**2) * torch.sin(math.pi * x_int).reshape(-1)
    return x_int, x_bc, f_int


def _rel_l2(model: PoissonResidual, n: int = 200) -> float:
    xs = torch.linspace(0.0, 1.0, n).unsqueeze(-1)
    with torch.no_grad():
        pred = model.value(xs)
        exact = torch.sin(math.pi * xs.squeeze(-1))
        return float(torch.linalg.norm(pred - exact) / torch.linalg.norm(exact))


def _fresh(seed: int) -> tuple[PoissonResidual, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    model = PoissonResidual(H)
    return model, *_points()


def run_adam(seed: int) -> dict[str, float]:
    model, x_int, x_bc, f_int = _fresh(seed)
    opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    t0 = time.perf_counter()
    loss = torch.tensor(0.0)
    for _ in range(ADAM_STEPS):
        opt.zero_grad(set_to_none=True)
        r = model(x_int, x_bc, f_int)
        loss = 0.5 * (r * r).mean()
        loss.backward()
        opt.step()
    return {
        "wall_s": time.perf_counter() - t0,
        "steps": float(ADAM_STEPS),
        "rel_l2": _rel_l2(model),
        "final_loss": float(loss.detach()),
    }


def run_lbfgs(seed: int) -> dict[str, float]:
    model, x_int, x_bc, f_int = _fresh(seed)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=20, history_size=20, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        r = model(x_int, x_bc, f_int)
        loss = 0.5 * (r * r).mean()
        loss.backward()
        return loss

    t0 = time.perf_counter()
    for _ in range(LBFGS_STEPS):
        opt.step(closure)
    r = model(x_int, x_bc, f_int)
    return {
        "wall_s": time.perf_counter() - t0,
        "steps": float(LBFGS_STEPS),
        "rel_l2": _rel_l2(model),
        "final_loss": float(0.5 * (r * r).mean().detach()),
    }


def run_gauss_newton(seed: int) -> dict[str, float]:
    model, x_int, x_bc, f_int = _fresh(seed)
    flat0, residual_fn = functional_residual_fn(model, x_int, x_bc, f_int)
    opt = GaussNewton(damping=1e-3, solver="qr", damping_strategy="nielsen")
    t0 = time.perf_counter()
    theta, hist = opt.minimize(residual_fn, flat0, steps=CURVATURE_STEPS)
    torch.nn.utils.vector_to_parameters(theta.detach(), model.parameters())
    return {
        "wall_s": time.perf_counter() - t0,
        "steps": float(CURVATURE_STEPS),
        "rel_l2": _rel_l2(model),
        "final_loss": float(hist[-1]) if hist else float("nan"),
    }


def run_cubic_gauss_newton(seed: int) -> dict[str, float]:
    model, x_int, x_bc, f_int = _fresh(seed)
    flat0, residual_fn = functional_residual_fn(model, x_int, x_bc, f_int)
    opt = CubicRegularizedGaussNewton(sigma=1.0, krylov_dim=12)
    t0 = time.perf_counter()
    theta, hist = opt.minimize(residual_fn, flat0, steps=CURVATURE_STEPS)
    torch.nn.utils.vector_to_parameters(theta.detach(), model.parameters())
    return {
        "wall_s": time.perf_counter() - t0,
        "steps": float(CURVATURE_STEPS),
        "rel_l2": _rel_l2(model),
        "final_loss": float(hist[-1]) if hist else float("nan"),
    }


def run_trust_newton(seed: int) -> dict[str, float]:
    model, x_int, x_bc, f_int = _fresh(seed)
    opt = TrustRegionNewtonCG(model.parameters(), cg_max_iter=25)

    def closure() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        r = model(x_int, x_bc, f_int)
        loss = 0.5 * (r * r).mean()
        loss.backward(create_graph=True)
        return loss

    t0 = time.perf_counter()
    for _ in range(CURVATURE_STEPS):
        opt.step(closure)
    r = model(x_int, x_bc, f_int)
    return {
        "wall_s": time.perf_counter() - t0,
        "steps": float(CURVATURE_STEPS),
        "rel_l2": _rel_l2(model),
        "final_loss": float(0.5 * (r * r).mean().detach()),
    }


METHODS = {
    "adam": run_adam,
    "lbfgs": run_lbfgs,
    "gauss_newton": run_gauss_newton,
    "cubic_gauss_newton": run_cubic_gauss_newton,
    "trust_region_newton_cg": run_trust_newton,
}


def _aggregate(samples: list[dict[str, float]]) -> dict[str, object]:
    rel = [s["rel_l2"] for s in samples if not math.isnan(s["rel_l2"])]
    wall = [s["wall_s"] for s in samples if not math.isnan(s["wall_s"])]
    return {
        "seeds": len(samples),
        "ok_seeds": len(rel),
        "rel_l2_median": float(np.median(rel)) if rel else float("nan"),
        "rel_l2_mean": float(np.mean(rel)) if rel else float("nan"),
        "rel_l2_std": float(np.std(rel)) if rel else float("nan"),
        "wall_s_median": float(np.median(wall)) if wall else float("nan"),
        "wall_s_mean": float(np.mean(wall)) if wall else float("nan"),
        "per_seed": samples,
    }


def main() -> None:
    results: dict[str, object] = {}
    for name, fn in METHODS.items():
        print(f"--- {name} ---", flush=True)
        samples: list[dict[str, float]] = []
        for seed in SEEDS:
            try:
                row = fn(seed)
                print(f"  seed={seed}: rel_l2={row['rel_l2']:.3e}  wall={row['wall_s']:.2f}s")
                samples.append(row)
            except Exception as exc:  # noqa: BLE001
                print(f"  seed={seed}: ERROR {type(exc).__name__}: {exc}")
                samples.append(
                    {
                        "wall_s": float("nan"),
                        "steps": float("nan"),
                        "rel_l2": float("nan"),
                        "final_loss": float("nan"),
                    }
                )
        results[name] = _aggregate(samples)

    payload = provenance(
        schema="omnibias/optimizer-pinn/v1",
        config={
            "problem": "1d_poisson",
            "exact": "sin(pi x)",
            "hidden": H,
            "n_interior": N_INT,
            "seeds": list(SEEDS),
            "adam_steps": ADAM_STEPS,
            "lbfgs_steps": LBFGS_STEPS,
            "curvature_steps": CURVATURE_STEPS,
            "bc_weight": BC_WEIGHT,
            "dtype": "float64",
            "note": (
                "Self-contained one-layer PINN; residual Laplacian is closed-form "
                "tanh sigma^(2). Equal-step budgets favour first-order methods on raw "
                "step count; curvature methods use fewer, cheaper-to-compare steps."
            ),
        },
    )
    payload["results"] = results
    path = write_json("optimizer_pinn.json", payload)
    print(f"wrote {path}")
    print(f"{'method':<28} {'rel_l2 med':>12} {'wall med':>10}")
    for name, agg in results.items():
        a = agg  # type: ignore[assignment]
        print(f"{name:<28} {a['rel_l2_median']:12.3e} {a['wall_s_median']:10.2f}s")  # type: ignore[index]


if __name__ == "__main__":
    main()
