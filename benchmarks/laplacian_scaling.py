# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Laplacian cost for a runtime-input ridge field, by input dimension.

Run ``uv run --with folx python benchmarks/laplacian_scaling.py``.
JAX methods receive X, W, beta and c as runtime arguments. Compilation,
first execution and synchronized steady-state samples are recorded separately.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import enable_x64, measure, provenance, source_provenance, write_json  # noqa: E402
from _operator_reference import tanh_field_laplacian_reference
from _reference import error_metrics  # noqa: E402

enable_x64()

import folx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.jax import neural_field_laplacian  # noqa: E402

H, B = 32, 64
DS = (3, 12, 30, 60)
SEED = 0
REPEATS = 9


def _params(d: int, key: jax.Array) -> tuple[Any, Any, Any, Any]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w = jax.random.normal(k1, (H, d), dtype=jnp.float64) * 0.3
    beta = jax.random.normal(k2, (H,), dtype=jnp.float64) * 0.1
    c = jax.random.normal(k3, (H,), dtype=jnp.float64)
    return w, beta, c / jnp.linalg.norm(c), jax.random.normal(k4, (B, d), dtype=jnp.float64)


def omni(x: Any, w: Any, beta: Any, c: Any) -> Any:
    return neural_field_laplacian(x, w, beta, c, "tanh")


def dense(x: Any, w: Any, beta: Any, c: Any) -> Any:
    return jax.vmap(lambda point: jnp.trace(jax.hessian(
        lambda y: jnp.dot(c, jnp.tanh(w @ y + beta))
    )(point)))(x)


def forward(x: Any, w: Any, beta: Any, c: Any) -> Any:
    field = folx.forward_laplacian(lambda y: jnp.dot(c, jnp.tanh(w @ y + beta)))
    return jax.vmap(lambda point: field(point).laplacian)(x)


def _run_d(d: int) -> dict[str, Any]:
    w, beta, c, x = _params(d, jax.random.PRNGKey(SEED + d))
    args = (x, w, beta, c)
    indices = np.array([0, B // 2, B - 1])
    reference = tanh_field_laplacian_reference(
        np.asarray(x)[indices], np.asarray(w), np.asarray(beta), np.asarray(c), 1
    )
    methods = {}
    for name, function in (("omnibias", omni), ("jax_hessian", dense), ("folx", forward)):
        start = time.perf_counter()
        compiled = jax.jit(function).lower(*args).compile()
        compile_ms = (time.perf_counter() - start) * 1000
        def execute(compiled: Any = compiled) -> Any:
            return compiled(*args)

        result = measure(execute, repeats=REPEATS)
        result["compile_ms"] = compile_ms
        values = np.asarray(compiled(*args))[indices]
        np.testing.assert_allclose(values, reference, rtol=2e-10, atol=1e-9)
        result["error_vs_mpmath"] = error_metrics(values, reference)
        methods[name] = result

    xt, wt, bt, ct = [torch.tensor(np.asarray(arg), dtype=torch.float64) for arg in args]

    def torch_dense() -> torch.Tensor:
        return cast(torch.Tensor, torch.vmap(lambda point: torch.func.hessian(
            lambda y: torch.dot(ct, torch.tanh(wt @ y + bt))
        )(point).trace())(xt))

    result = measure(torch_dense, repeats=REPEATS)
    result["compile_ms"] = None
    values = torch_dense().detach().numpy()[indices]
    np.testing.assert_allclose(values, reference, rtol=2e-10, atol=1e-9)
    result["error_vs_mpmath"] = error_metrics(values, reference)
    methods["torch_func_hessian"] = result
    row = {"D": d, "methods": methods, "time_ms": {
        name: result["median_ms"] for name, result in methods.items()
    }}
    print(f"D={d}: " + ", ".join(f"{name}={result['median_ms']:.4f} ms"
          for name, result in methods.items()), flush=True)
    return row


def constant_folding_audit() -> dict[str, Any]:
    """Compare the old zero-argument closure with an equivalent runtime-input JIT."""
    w, beta, c, x = _params(60, jax.random.PRNGKey(60))
    old = jax.jit(lambda: omni(x, w, beta, c)).lower().compile()
    dynamic = jax.jit(omni).lower(x, w, beta, c).compile()
    old_hlo, dynamic_hlo = old.as_text(), dynamic.as_text()
    assert old_hlo is not None and dynamic_hlo is not None
    return {
        "description": "Optimized HLO of the former zero-argument benchmark and the corrected runtime-input benchmark",
        "zero_argument_entry_hlo": old_hlo.split("ENTRY ")[-1],
        "runtime_argument_entry_hlo": dynamic_hlo.split("ENTRY ")[-1],
        "zero_argument_tanh_operations": old_hlo.count("tanh("),
        "runtime_argument_tanh_operations": dynamic_hlo.count("tanh("),
        "compiler_version": jax.__version__,
    }


def main() -> None:
    torch.set_num_threads(1)
    rows = [_run_d(d) for d in DS]
    payload = provenance(schema="omnibias/laplacian-scaling/v2", config={
        "H": H, "B": B, "D": list(DS), "activation": "tanh", "dtype": "float64",
        "seed": SEED, "repeats": REPEATS, "warmup": 2, "torch_threads": 1,
        "runtime_arguments": ["X", "W", "beta", "c"],
        "execution": "JAX JIT CPU defaults; torch.func.hessian eager, one CPU thread",
        "timing_scope": "Laplacian evaluation only; no parameter backward or training",
        "reference": "mpmath.diff(tanh), full ridge-field dot products at 80 decimal digits",
        "reference_points": 3, "relative_error_floor": 1e-3,
        "accuracy_rtol": 2e-10, "accuracy_atol": 1e-9,
    })
    payload.update(source_provenance([
        "benchmarks/laplacian_scaling.py", "benchmarks/_common.py", "benchmarks/_reference.py", "benchmarks/_operator_reference.py",
        "packages/omnibias-jax/src/omnibias/jax/laplacian.py",
        "packages/omnibias-jax/src/omnibias/jax/_fastpath.py",
        "packages/omnibias-core/src/omnibias/core/polynomials.py",
    ]))
    payload["rows"] = rows
    payload["constant_folding_audit"] = constant_folding_audit()
    print(f"wrote {write_json('laplacian_scaling.json', payload)}")


if __name__ == "__main__":
    main()
