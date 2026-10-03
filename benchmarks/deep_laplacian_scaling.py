# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deep-network Laplacian accuracy, scaling and cross-backend checks.

Compares the direct Laplacian recursion against a small mixed-jet oracle and
nested Hessian traces. Tests a dimension beyond the mixed-jet allocation budget,
float64 torch/JAX numerical parity, and the sampled estimator against the exact
supported value across independent seeds. The measured backend tolerance is
numerical agreement, not universal bit identity.

Usage::

    uv run python benchmarks/deep_laplacian_scaling.py
    uv run python benchmarks/deep_laplacian_scaling.py --full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import median_time_ms, provenance, rss_mb, write_json  # noqa: E402
from _gates import (  # noqa: E402
    gates_block,
    require_all_seeds,
    require_cost_parity,
    require_within_stderr,
)
from omnibias.core.multi_index import index_position, multi_index_factorial  # noqa: E402
from omnibias.torch.jet_mv import mlp_jet_mv  # noqa: E402
from omnibias.torch.laplacian import (  # noqa: E402
    deep_field_laplacian,
    deep_field_polylaplacian,
)

torch.set_default_dtype(torch.float64)

BASELINE_NAME = "torch.func.hessian nested AD"
SMALL_DIMS = (2, 4, 8)
HIDDEN = 6
DEPTH = 3
CEILING_DIM = 5000
CEILING_HIDDEN = 4
CEILING_DEPTH = 2
G4_BACKEND_TOL = 5e-9


def _make_layers_np(dim: int, hidden: int, depth: int, activation: str, seed: int):
    """A small random deep MLP's ``(W, b, spec)`` layer list, plain numpy."""
    rng = np.random.default_rng(seed)
    dims = [dim] + [hidden] * depth + [1]
    layers = []
    for i in range(len(dims) - 1):
        scale = 1.0 / np.sqrt(dims[i])
        W = rng.normal(scale=scale, size=(dims[i + 1], dims[i]))
        b = rng.normal(scale=0.1, size=(dims[i + 1],))
        spec = None if i == len(dims) - 2 else activation
        layers.append((W, b, spec))
    return layers


def _to_torch(layers_np):
    return [
        (
            torch.as_tensor(W, dtype=torch.float64),
            torch.as_tensor(b, dtype=torch.float64),
            spec,
        )
        for W, b, spec in layers_np
    ]


def _oracle_laplacian_via_mlp_jet_mv(x: torch.Tensor, layers) -> torch.Tensor:
    """``Delta f(x)`` by reading the order-2 rows off the full multivariate jet."""
    dim = x.shape[-1]
    jet = mlp_jet_mv(x, layers, 2)
    pos = index_position(dim, 2)
    total = None
    for i in range(dim):
        alpha = tuple(2 if j == i else 0 for j in range(dim))
        term = jet[pos[alpha]] * multi_index_factorial(alpha)
        total = term if total is None else total + term
    return total


def _run_g1_exactness_vs_mlp_jet_mv() -> dict[str, Any]:
    max_err = 0.0
    for dim in SMALL_DIMS:
        layers_np = _make_layers_np(dim, HIDDEN, DEPTH, "tanh", seed=0)
        layers = _to_torch(layers_np)
        x = torch.as_tensor(
            np.random.default_rng(1).normal(scale=0.3, size=(dim,)), dtype=torch.float64
        )
        got = deep_field_laplacian(x, layers)[0]
        want = _oracle_laplacian_via_mlp_jet_mv(x, layers)[0]
        err = float(torch.abs(got - want))
        max_err = max(max_err, err)
    passed = bool(max_err <= 1e-9)
    verdict = {
        "name": "exactness_vs_mlp_jet_mv",
        "dims": list(SMALL_DIMS),
        "max_abs_error": max_err,
        "max_abs_error_cap": 1e-9,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"exactness_vs_mlp_jet_mv: max_abs_error={max_err:.4e} exceeds 1e-9"
        )
    return verdict


def _run_g2_no_ceiling() -> dict[str, Any]:
    layers_np = _make_layers_np(CEILING_DIM, CEILING_HIDDEN, CEILING_DEPTH, "tanh", seed=2)
    layers = _to_torch(layers_np)
    x = torch.as_tensor(
        np.random.default_rng(3).normal(scale=0.1, size=(2, CEILING_DIM)),
        dtype=torch.float64,
    )
    lap = deep_field_laplacian(x, layers)
    fast_lane_ok = bool(torch.all(torch.isfinite(lap)))
    ceiling_hit = False
    ceiling_message = ""
    try:
        mlp_jet_mv(x[0], layers, 2)
    except ValueError as exc:
        ceiling_hit = True
        ceiling_message = str(exc)
    passed = fast_lane_ok and ceiling_hit
    verdict = {
        "name": "no_ceiling_d5000",
        "dim": CEILING_DIM,
        "fast_lane_succeeded": fast_lane_ok,
        "mlp_jet_mv_raised_ceiling_error": ceiling_hit,
        "mlp_jet_mv_error": ceiling_message,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(f"no_ceiling_d5000 failed: {verdict}")
    return verdict


def _run_g3_cost_parity(*, cost_dim: int, hidden: int, depth: int, repeats: int) -> dict[str, Any]:
    layers_np = _make_layers_np(cost_dim, hidden, depth, "tanh", seed=4)
    layers = _to_torch(layers_np)
    x = torch.as_tensor(
        np.random.default_rng(5).normal(scale=0.1, size=(8, cost_dim)), dtype=torch.float64
    )

    def fast() -> torch.Tensor:
        return deep_field_laplacian(x, layers)

    def value(xi: torch.Tensor) -> torch.Tensor:
        a = xi
        for W, b, spec in layers:
            u = a @ W.t() + b
            a = u if spec is None else torch.tanh(u)
        return a[0]

    def nested() -> torch.Tensor:
        return torch.func.vmap(lambda xi: torch.func.hessian(value)(xi).trace())(x)

    lap_fast = fast()
    lap_nested = nested()
    agree = float(torch.max(torch.abs(lap_fast[:, 0] - lap_nested)))
    t_fast = median_time_ms(fast, warmup=2, repeats=repeats)
    t_nested = median_time_ms(nested, warmup=2, repeats=repeats)
    cost = require_cost_parity(
        t_fast, t_nested, max_ratio=0.5, name="deep_laplacian_vs_nested_ad"
    )
    passed = bool(cost["passed"]) and agree <= 1e-8
    return {
        "name": "cost_parity_vs_nested_ad",
        "dim": cost_dim,
        "hidden": hidden,
        "depth": depth,
        "fast_ms": t_fast,
        "nested_ad_ms": t_nested,
        "speedup": float(t_nested / t_fast) if t_fast > 0.0 else float("inf"),
        "max_abs_agreement": agree,
        "passed": passed,
    }


def _run_g4_backend_parity() -> dict[str, Any]:
    """Torch vs JAX numerical parity -- honestly *not* bit-exact.

    See the module docstring: the shared pure-Python fastpath polynomial
    evaluation is bit-exact between backends in isolation, but
    ``deep_field_laplacian``'s intermediate tensordot / sum steps pick up a
    1-2 ULP gap even at (D=1, H=1). ``require_backend_parity`` (exact
    equality) would fail honestly here rather than pass; this gate reports
    the measured gap against a tight float64 tolerance instead of forcing a
    false bit-identity claim.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax.laplacian import deep_field_laplacian as jax_deep_field_laplacian

    max_err = 0.0
    configs = ((1, 1, 2), (3, 4, 2), (8, 6, 3))
    for dim, hidden, depth in configs:
        layers_np = _make_layers_np(dim, hidden, depth, "tanh", seed=6)
        x_np = np.random.default_rng(7).normal(scale=0.3, size=(4, dim))
        torch_lap = deep_field_laplacian(
            torch.as_tensor(x_np, dtype=torch.float64), _to_torch(layers_np)
        )
        jax_layers = [
            (jnp.asarray(W, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64), spec)
            for W, b, spec in layers_np
        ]
        jax_lap = jax_deep_field_laplacian(jnp.asarray(x_np, dtype=jnp.float64), jax_layers)
        err = float(np.max(np.abs(torch_lap.detach().numpy() - np.asarray(jax_lap))))
        max_err = max(max_err, err)
    passed = bool(max_err <= G4_BACKEND_TOL)
    verdict = {
        "name": "backend_numerical_parity_torch_vs_jax",
        "bit_exact": False,
        "honesty_note": (
            "not require_backend_parity (bit-exact); the fastpath polynomial "
            "evaluation is bit-exact in isolation but the multi-step "
            "tensordot/sum recursion picks up a 1-2 ULP gap between backends "
            "even at trivial shapes -- this is a measured finding, not a "
            "shortfall"
        ),
        "configs": [list(c) for c in configs],
        "max_abs_error": max_err,
        "max_abs_error_cap": G4_BACKEND_TOL,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(
            f"backend_numerical_parity_torch_vs_jax: max_abs_error={max_err:.4e} "
            f"exceeds {G4_BACKEND_TOL:.4e}"
        )
    return verdict


def _run_g5_estimator_unbiased(*, n_seeds: int, n_directions: int) -> dict[str, Any]:
    dim, k = 4, 2
    layers_np = _make_layers_np(dim, 5, 2, "tanh", seed=8)
    layers = _to_torch(layers_np)
    x = torch.as_tensor(
        np.random.default_rng(9).normal(scale=0.2, size=(dim,)), dtype=torch.float64
    )
    exact = float(deep_field_polylaplacian(x, layers, k, mode="support")[0])

    per_seed = []
    for seed in range(n_seeds):
        est = float(
            deep_field_polylaplacian(
                x, layers, k, mode="estimator", n_directions=n_directions, seed=seed
            )[0]
        )
        per_seed.append({"seed": seed, "estimate": est})
    values = np.asarray([r["estimate"] for r in per_seed])
    grand_mean = float(values.mean())
    # Standard error of the grand mean across independent seeds (each seed's
    # estimate is itself an average of n_directions i.i.d. directional draws).
    stderr = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    within = require_within_stderr(
        grand_mean, exact, stderr, max_sigmas=4.0, name="polylaplacian_estimator_unbiased"
    )
    seeds_ok = require_all_seeds(
        [
            {"seed": r["seed"], "finite": 1.0 if np.isfinite(r["estimate"]) else 0.0}
            for r in per_seed
        ],
        key="finite",
        expected=1.0,
        tol=0.0,
        min_seeds=5,
        name="polylaplacian_estimator_all_seeds_finite",
    )
    passed = bool(within["passed"]) and bool(seeds_ok["passed"])
    return {
        "name": "estimator_unbiasedness",
        "dim": dim,
        "k": k,
        "exact_support": exact,
        "n_directions": n_directions,
        "per_seed": per_seed,
        "grand_mean": grand_mean,
        "stderr": stderr,
        "sigmas": within["sigmas"],
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "deep_laplacian_scaling.json" if full else "deep_laplacian_scaling_smoke.json"
    t0 = time.perf_counter()

    g1 = _run_g1_exactness_vs_mlp_jet_mv()
    g2 = _run_g2_no_ceiling()
    g3 = _run_g3_cost_parity(
        cost_dim=2048 if full else 256,
        hidden=32 if full else 16,
        depth=3,
        repeats=7 if full else 3,
    )
    g4 = _run_g4_backend_parity()
    g5 = _run_g5_estimator_unbiased(
        n_seeds=8 if full else 5,
        n_directions=8000 if full else 3000,
    )
    entries = [g1, g2, g3, g4, g5]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")

    payload = {
        **provenance(
            schema="omnibias.benchmarks.deep_laplacian_scaling.v1",
            config={
                "family": "deep_laplacian_scaling",
                "full": full,
                "small_dims": list(SMALL_DIMS),
                "ceiling_dim": CEILING_DIM,
                "cost_dim": g3["dim"],
                "backend_parity_bit_exact": False,
                "public_primitive": (
                    "omnibias.torch.laplacian.deep_field_laplacian / "
                    "omnibias.jax.laplacian.deep_field_laplacian"
                ),
            },
        ),
        "baseline": {"name": BASELINE_NAME},
        "rss_mb": rss_mb(),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
