# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 10-02: jet-adjoint exactness (G1), analytic LQR reference (G2),
and torch/jax full-pipeline structural bit-parity (G9).

G1 checks two independent things and reports both: (a) the closed-form
``dpi/dy`` from the directional jet tower matches ordinary reverse-mode
autodiff through the same forward pass, on both backends; (b) the
full-horizon exact-adjoint policy gradient matches plain backprop-through-
time on a small closed-loop rollout, on both backends. G2 checks the
backend-free recursion against the analytic finite-horizon discrete-LQR
costate ``lambda_k = P_k y_k`` on a random-system sweep. G9 is structural,
not numerical: both backend twins call the identical ``omnibias.core.adjoint``
functions, so bit-parity is true by construction, not merely measured to a
tolerance.

Not a claim about G3/G4/G8 (control performance / sample efficiency / cost
parity vs named baselines); those need a measured Phase-A curve and are
tracked separately (see ``theory/10-control/02-jet-adjoint-policy-optimization.md``
section 8).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.adjoint import DISCLAIMER, adjoint_skill, worked_example  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

DPI_DY_ATOL = 1e-10
FULL_PIPELINE_ATOL = 1e-8
G2_MEDIAN_ERR_TOL = 1e-10


def _small_layers_jax(key: Any) -> list[Any]:
    import jax
    import jax.numpy as jnp

    k1, k2 = jax.random.split(key)
    w1 = 0.1 * jax.random.normal(k1, (6, 2))
    b1 = jnp.zeros(6)
    w2 = 0.1 * jax.random.normal(k2, (2, 6))
    b2 = jnp.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def _small_layers_torch(seed: int = 0) -> list[Any]:
    import torch

    g = torch.Generator().manual_seed(seed)
    w1 = 0.1 * torch.randn(6, 2, generator=g)
    b1 = torch.zeros(6)
    w2 = 0.1 * torch.randn(2, 6, generator=g)
    b2 = torch.zeros(2)
    return [(w1, b1, "tanh"), (w2, b2, None)]


def _g1_dpi_dy_jax() -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    from omnibias.control.jax.adjoint import policy_forward, policy_jacobian_dy

    layers = _small_layers_jax(jax.random.PRNGKey(0))
    y = jax.numpy.array([0.3, -0.2])
    dy_closed = np.asarray(policy_jacobian_dy(layers, y))
    dy_autodiff = np.asarray(jax.jacfwd(lambda yy: policy_forward(layers, yy))(y))
    max_abs = float(np.max(np.abs(dy_closed - dy_autodiff)))
    return {
        "name": "g1_dpi_dy_jax",
        "passed": bool(max_abs < DPI_DY_ATOL),
        "max_abs_diff": max_abs,
        "atol": DPI_DY_ATOL,
    }


def _g1_dpi_dy_torch() -> dict[str, Any]:
    import torch
    from omnibias.control.torch.adjoint import policy_forward, policy_jacobian_dy

    torch.set_default_dtype(torch.float64)
    layers = _small_layers_torch(0)
    y = torch.tensor([0.3, -0.2])
    dy_closed = policy_jacobian_dy(layers, y).detach().numpy()
    dy_autodiff = torch.func.jacfwd(lambda yy: policy_forward(layers, yy))(y).detach().numpy()
    max_abs = float(np.max(np.abs(dy_closed - dy_autodiff)))
    return {
        "name": "g1_dpi_dy_torch",
        "passed": bool(max_abs < DPI_DY_ATOL),
        "max_abs_diff": max_abs,
        "atol": DPI_DY_ATOL,
    }


def _g1_full_pipeline_jax() -> dict[str, Any]:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.control.jax.adjoint import actor_adjoint_gradient, flatten_layers, policy_forward
    from omnibias.control.jax.envs import DoubleGyrePointMass

    env = DoubleGyrePointMass()
    layers = _small_layers_jax(jax.random.PRNGKey(1))
    y0 = jnp.array([0.3, 0.4])
    horizon = 4
    result = actor_adjoint_gradient(env, layers, y0, horizon)

    theta, unravel = flatten_layers(layers)

    def rollout_cost(th: Any) -> Any:
        ll = unravel(th)
        y = y0
        total = 0.0
        for _ in range(horizon):
            u = policy_forward(ll, y)
            total = total + env.cost(y, u)
            y = env.step(y, u)
        return total + env.terminal_cost(y)

    bptt_grad = np.asarray(jax.grad(rollout_cost)(theta))
    max_abs = float(np.max(np.abs(result.grad_theta - bptt_grad)))
    return {
        "name": "g1_full_pipeline_jax",
        "passed": bool(max_abs < FULL_PIPELINE_ATOL),
        "max_abs_diff": max_abs,
        "atol": FULL_PIPELINE_ATOL,
    }


def _g1_full_pipeline_torch() -> dict[str, Any]:
    import torch
    from omnibias.control.torch.adjoint import (
        actor_adjoint_gradient,
        flatten_layers,
        policy_forward,
    )
    from omnibias.control.torch.envs import DoubleGyrePointMass

    torch.set_default_dtype(torch.float64)
    env = DoubleGyrePointMass()
    layers = _small_layers_torch(1)
    y0 = torch.tensor([0.3, 0.4])
    horizon = 4
    result = actor_adjoint_gradient(env, layers, y0, horizon)

    theta, structure = flatten_layers(layers)
    theta_req = theta.detach().clone().requires_grad_(True)
    ll = structure.unflatten(theta_req)
    y = y0
    total = torch.zeros(())
    for _ in range(horizon):
        u = policy_forward(ll, y)
        total = total + env.cost(y, u)
        y = env.step(y, u)
    total = total + env.terminal_cost(y)
    (bptt_grad,) = torch.autograd.grad(total, theta_req)
    max_abs = float(np.max(np.abs(result.grad_theta - bptt_grad.detach().numpy())))
    return {
        "name": "g1_full_pipeline_torch",
        "passed": bool(max_abs < FULL_PIPELINE_ATOL),
        "max_abs_diff": max_abs,
        "atol": FULL_PIPELINE_ATOL,
    }


def _g2_analytic_lqr(n: int) -> dict[str, Any]:
    ex = worked_example()
    skill = adjoint_skill(n=n, seed=0)
    passed = bool(
        ex["g2_earned"]
        and skill["g2_earned"]
        and float(skill["median_err"]) < G2_MEDIAN_ERR_TOL
    )
    return {
        "name": "g2_analytic_lqr_costate",
        "passed": passed,
        "worked_example_max_abs_err": ex["max_abs_err"],
        "skill_n": n,
        "skill_median_err": skill["median_err"],
        "skill_all_finite": skill["all_finite"],
        "skill_singular_raised": skill["singular_raised"],
        "tol": G2_MEDIAN_ERR_TOL,
    }


def _g9_structural_parity() -> dict[str, Any]:
    from omnibias.control.jax import adjoint as jax_adjoint
    from omnibias.control.torch import adjoint as torch_adjoint

    checks = {
        "adjoint_recursion": jax_adjoint.adjoint_recursion is torch_adjoint.adjoint_recursion,
        "closed_loop_jacobian": jax_adjoint.closed_loop_jacobian is torch_adjoint.closed_loop_jacobian,
        "policy_gradient": jax_adjoint.policy_gradient is torch_adjoint.policy_gradient,
    }
    return {
        "name": "g9_full_pipeline_structural_parity",
        "passed": bool(all(checks.values())),
        "shared_object_checks": checks,
        "basis": "both backend twins call the identical omnibias.core.adjoint "
        "function objects; bit-parity is structural, not measured to a tolerance",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()

    g2_n = 1000 if full else 40
    entries = [
        _g1_dpi_dy_jax(),
        _g1_dpi_dy_torch(),
        _g1_full_pipeline_jax(),
        _g1_full_pipeline_torch(),
        _g2_analytic_lqr(g2_n),
        _g9_structural_parity(),
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")

    payload = {
        **provenance(
            schema="omnibias.benchmarks.adjoint_exactness.v1",
            config={"family": "adjoint_exactness", "full": full, "g2_n": g2_n},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "honesty": {
            "g3_control_performance": "not measured here; needs a Phase-A curve",
            "g4_sample_efficiency": "not measured here; needs named baselines",
            "g8_cost_parity": "not measured here; needs Brax/MJX installed",
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "adjoint_exactness.json" if full else "adjoint_exactness_smoke.json"
    if full:
        dest = SCRATCH / "control" / "adjoint"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
