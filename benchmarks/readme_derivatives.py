# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded CPU comparison: neural jets, nested AD and JAX Taylor-mode AD.

Run ``uv run python benchmarks/readme_derivatives.py``. Writes JSON to the
standard scratch directory. Times derivative evaluation, not training or PDE solving.
"""

from __future__ import annotations

import hashlib
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _common import provenance, write_json  # noqa: E402
from _reference import error_metrics, tanh_network_reference  # noqa: E402
from jax.experimental.jet import jet as jax_taylor_jet  # noqa: E402
from omnibias.jax.jet import jet_to_tower as jax_tower  # noqa: E402
from omnibias.jax.jet import mlp_jet as jax_mlp_jet  # noqa: E402
from omnibias.torch.jet import jet_to_tower as torch_tower  # noqa: E402
from omnibias.torch.jet import mlp_jet as torch_mlp_jet  # noqa: E402

SEED = 2026
DIMS = (1, 8, 8, 1)
POINTS = 128
ORDERS = (2, 4, 6)
REPEATS = 9
TAYLOR_JET = cast(Callable[..., Any], jax_taylor_jet)


def timed(function: Callable[[], Any], *, synchronize: bool) -> dict[str, Any]:
    """Record synchronized first execution separately from warmed steady-state calls."""
    def evaluate() -> None:
        result = function()
        if synchronize:
            result.block_until_ready()

    start = time.perf_counter()
    evaluate()
    first_ms = (time.perf_counter() - start) * 1000
    for _ in range(2):
        evaluate()
    samples = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        evaluate()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "first_execution_ms": first_ms,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> None:
    torch.set_num_threads(1)
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    rng = np.random.default_rng(SEED)
    arrays = [
        (rng.normal(0, 0.4, (out_dim, in_dim)), rng.normal(0, 0.1, out_dim))
        for in_dim, out_dim in zip(DIMS[:-1], DIMS[1:], strict=True)
    ]
    torch_layers = [
        (torch.tensor(w), torch.tensor(b), None if i == len(arrays) - 1 else "tanh")
        for i, (w, b) in enumerate(arrays)
    ]
    jax_layers = [
        (jnp.asarray(w), jnp.asarray(b), None if i == len(arrays) - 1 else "tanh")
        for i, (w, b) in enumerate(arrays)
    ]
    xs = np.linspace(-0.7, 0.7, POINTS).reshape(-1, 1)
    tx, jx = torch.from_numpy(xs), jnp.asarray(xs)
    reference_indices = np.linspace(0, POINTS - 1, 9, dtype=int)

    def torch_forward(x: torch.Tensor) -> torch.Tensor:
        for w, b, activation in torch_layers:
            x = x @ w.T + b
            if activation:
                x = torch.tanh(x)
        return x

    def jax_forward(x: Any) -> Any:
        for w, b, activation in jax_layers:
            x = x @ w.T + b
            if activation:
                x = jnp.tanh(x)
        return x

    rows = []
    for order in ORDERS:
        reference = tanh_network_reference(arrays, xs[reference_indices, 0], order)

        def torch_nested(order: int = order) -> torch.Tensor:
            x = tx.detach().requires_grad_(True)
            value = torch_forward(x)
            for k in range(order):
                value = torch.autograd.grad(value.sum(), x, create_graph=k < order - 1)[0]
            return value

        def torch_jet(order: int = order) -> torch.Tensor:
            return torch_tower(
                torch_mlp_jet(tx, torch.ones_like(tx), torch_layers, order, riccati=True)
            )[order]

        def nested(x: Any) -> Any:
            return jax_forward(x).sum()

        for _ in range(order):
            derivative = jax.grad(nested)

            def summed(x: Any, derivative: Callable[[Any], Any] = derivative) -> Any:
                return derivative(x).sum()

            nested = summed

        def taylor(x: Any, order: int = order) -> Any:
            series = (jnp.ones_like(x),) + (jnp.zeros_like(x),) * (order - 1)
            return TAYLOR_JET(jax_forward, (x,), (series,))[1][-1]

        def omnibias(x: Any, order: int = order) -> Any:
            return jax_tower(
                jax_mlp_jet(x, jnp.ones_like(x), jax_layers, order, riccati=True)
            )[order]

        methods: dict[str, Any] = {}
        for name, function in (("torch_nested_ad", torch_nested), ("torch_omnibias", torch_jet)):
            result = timed(function, synchronize=False)
            result["compile_ms"] = None
            values = function().detach().numpy().reshape(-1)[reference_indices]
            result["error_vs_mpmath"] = error_metrics(values, reference)
            methods[name] = result
        for name, jax_function in (
            ("jax_nested_ad", derivative),
            ("jax_taylor_ad", taylor),
            ("jax_omnibias", omnibias),
        ):
            start = time.perf_counter()
            compiled = jax.jit(jax_function).lower(jx).compile()
            compile_ms = (time.perf_counter() - start) * 1000

            def execute(compiled: Any = compiled) -> Any:
                return compiled(jx)

            result = timed(execute, synchronize=True)
            result["compile_ms"] = compile_ms
            values = np.asarray(compiled(jx)).reshape(-1)[reference_indices]
            result["error_vs_mpmath"] = error_metrics(values, reference)
            methods[name] = result
        for name, result in methods.items():
            if not result["error_vs_mpmath"]["max_abs"] <= 1e-9:
                raise AssertionError(f"{name} order={order}: independent accuracy check failed")
        rows.append({"order": order, "methods": methods})
        print(f"order={order}: " + ", ".join(
            f"{name}={result['median_ms']:.4f}ms" for name, result in methods.items()
        ), flush=True)

    root = Path(__file__).resolve().parents[1]
    source_paths = [
        "benchmarks/readme_derivatives.py", "benchmarks/_reference.py",
        "packages/omnibias-torch/src/omnibias/torch/jet.py",
        "packages/omnibias-jax/src/omnibias/jax/jet.py",
    ]
    payload = provenance(schema="omnibias/readme-derivatives/v1", config={
        "seed": SEED, "dimensions": DIMS, "batch_size": POINTS, "orders": ORDERS,
        "activation": "tanh", "dtype": "float64", "device": "cpu",
        "torch_threads": torch.get_num_threads(), "jax_devices": [d.platform for d in jax.devices()],
        "repeats": REPEATS, "warmup": 2, "riccati": True,
        "reference": "independent mpmath.diff of the MLP, 80 decimal digits",
        "reference_points": len(reference_indices), "max_abs_error_cap": 1e-9,
        "relative_error_floor": 1e-3,
        "timing_scope": "input derivatives only; excludes parameter backward and model training",
        "execution": "Torch eager (one thread); JAX JIT (CPU runtime default threads)",
        "compilation": "JAX lowering plus compile, separate from synchronized steady-state execution",
    })
    payload["source_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    payload["source_worktree_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
    ).strip())
    payload["source_revision_kind"] = (
        "base_commit" if payload["source_worktree_dirty"] else "exact_commit"
    )
    payload["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths
    }
    payload["rows"] = rows
    print(f"wrote {write_json('readme_derivatives.json', payload)}")


if __name__ == "__main__":
    main()
