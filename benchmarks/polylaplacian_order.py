# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Runtime-input iterated Laplacians with isolated, resource-bounded baselines.

Run ``uv run --with folx python benchmarks/polylaplacian_order.py``.
Each method/order runs in a subprocess. The parent enforces a wall-clock and
RSS budget; a terminated run is a budget result, not a proof of impossibility.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import enable_x64, provenance, source_provenance, write_json  # noqa: E402

enable_x64()

H, D, B = 16, 16, 32
KS = (1, 2, 3, 4)
SEED, REPEATS = 0, 9
TIMEOUT_S, RSS_LIMIT_MIB = 120, 3072


def _worker(method: str, order: int) -> dict[str, Any]:
    import folx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from _common import measure
    from _operator_reference import tanh_field_laplacian_reference
    from _reference import error_metrics
    from omnibias.jax.laplacian import neural_field_polylaplacian

    k1, _, k3, k4 = jax.random.split(jax.random.PRNGKey(SEED), 4)
    w = jax.random.normal(k1, (H, D), dtype=jnp.float64) * 0.3
    beta = jnp.zeros(H, dtype=jnp.float64)
    c = jax.random.normal(k3, (H,), dtype=jnp.float64)
    c = c / jnp.linalg.norm(c)
    x = jax.random.normal(k4, (B, D), dtype=jnp.float64)
    args = (x, w, beta, c)

    def evaluate(x: Any, w: Any, beta: Any, c: Any) -> Any:
        if method == "omnibias":
            return neural_field_polylaplacian(x, w, beta, c, "tanh", k=order)
        def point(y: Any) -> Any:
            return jnp.dot(c, jnp.tanh(w @ y + beta))

        function: Callable[[Any], Any] = point
        for _ in range(order):
            if method == "dense_nested":
                hessian = jax.hessian(function)

                def dense_laplacian(y: Any, hessian: Callable[[Any], Any] = hessian) -> Any:
                    return jnp.trace(hessian(y))

                function = dense_laplacian
            else:
                forward = cast(Callable[[Any], Any], folx.forward_laplacian(function))

                def forward_laplacian(y: Any, forward: Callable[[Any], Any] = forward) -> Any:
                    return forward(y).laplacian

                function = forward_laplacian
        return jax.vmap(function)(x)

    start = time.perf_counter()
    compiled = jax.jit(evaluate).lower(*args).compile()
    compile_ms = (time.perf_counter() - start) * 1000
    def execute() -> Any:
        return compiled(*args)

    result = measure(execute, repeats=REPEATS)
    result.update(status="ok", compile_ms=compile_ms, time_ms=result["median_ms"])
    indices = np.array([0, B // 2, B - 1])
    reference = tanh_field_laplacian_reference(
        np.asarray(x)[indices], np.asarray(w), np.asarray(beta), np.asarray(c), order
    )
    values = np.asarray(compiled(*args))[indices]
    np.testing.assert_allclose(values, reference, rtol=2e-10, atol=1e-9)
    result["error_vs_mpmath"] = error_metrics(values, reference)
    result["value_sample"] = float(values[0])
    return result


def _rss_mib(pid: int) -> float:
    """Linux RSS, including JAX's threads, without imposing a virtual-address cap."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def _isolated(method: str, order: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="omnibias-polylap-") as directory:
        output = Path(directory) / "result.json"
        with (Path(directory) / "log.txt").open("w+") as log:
            start = time.monotonic()
            process = subprocess.Popen([
                sys.executable, str(Path(__file__).resolve()), "--worker", method,
                "--order", str(order), "--output", str(output),
            ], stdout=log, stderr=log, start_new_session=True)
            peak, status = 0.0, None
            while process.poll() is None:
                peak = max(peak, _rss_mib(process.pid))
                if peak > RSS_LIMIT_MIB:
                    status = "memory_budget"
                elif time.monotonic() - start > TIMEOUT_S:
                    status = "timeout"
                if status:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.025)
            process.wait()
            result: dict[str, Any]
            if status:
                result = {"status": status, "detail": "parent terminated the subprocess at its configured budget"}
            elif process.returncode != 0:
                log.seek(0)
                result = {"status": "error", "detail": log.read()[-1000:]}
            else:
                result = json.loads(output.read_text())
            result["observed_peak_rss_mib"] = peak
            result["subprocess_wall_s"] = time.monotonic() - start
            return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["omnibias", "dense_nested", "folx_nested"])
    parser.add_argument("--order", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.order is None or args.output is None:
            parser.error("--worker requires --order and --output")
        args.output.write_text(json.dumps(_worker(args.worker, args.order)))
        return
    if not Path("/proc/self/status").exists():
        parser.error("the bounded benchmark requires Linux /proc RSS monitoring")
    rows = []
    for order in KS:
        row: dict[str, Any] = {"k": order}
        for method in ("omnibias", "folx_nested", "dense_nested"):
            row[method] = _isolated(method, order)
            result = row[method]
            print(f"k={order} {method}: {result['status']} {result.get('median_ms', '')}", flush=True)
        rows.append(row)
    payload = provenance(schema="omnibias/polylaplacian-order/v2", config={
        "H": H, "D": D, "B": B, "k": list(KS), "activation": "tanh",
        "dtype": "float64", "seed": SEED, "repeats": REPEATS, "warmup": 2,
        "timeout_s": TIMEOUT_S, "rss_limit_mib": RSS_LIMIT_MIB,
        "budget_scope": "per subprocess, including imports, compilation and accuracy validation",
        "execution": "JAX JIT CPU runtime defaults; methods run sequentially in fresh processes",
        "runtime_arguments": ["X", "W", "beta", "c"],
        "timing_scope": "iterated Laplacian evaluation only; no parameter backward or training",
        "reference": "mpmath.diff(tanh), full ridge-field dot products at 80 decimal digits",
        "reference_points": 3, "relative_error_floor": 1e-3,
        "accuracy_rtol": 2e-10, "accuracy_atol": 1e-9,
    })
    payload.update(source_provenance([
        "benchmarks/polylaplacian_order.py", "benchmarks/_common.py", "benchmarks/_reference.py", "benchmarks/_operator_reference.py",
        "packages/omnibias-jax/src/omnibias/jax/laplacian.py",
        "packages/omnibias-jax/src/omnibias/jax/_fastpath.py",
        "packages/omnibias-core/src/omnibias/core/polynomials.py",
    ]))
    payload["rows"] = rows
    print(f"wrote {write_json('polylaplacian_order.json', payload)}")


if __name__ == "__main__":
    main()
