# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Iterated Laplacian ``Δᵏ`` cost vs order ``k``.

Compares omnibias closed-form ``neural_field_polylaplacian`` against nested
``jax.hessian`` (dense) and nested folx. OOM / timeout are recorded as status
strings rather than silently omitted. Run::

    uv run python benchmarks/polylaplacian_order.py
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import enable_x64, median_time_ms, provenance, write_json  # noqa: E402

enable_x64()

import folx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import vmap  # noqa: E402
from omnibias.jax.laplacian import neural_field_polylaplacian  # noqa: E402

H = 16
D = 16
B = 32
KS = (1, 2, 3, 4)
SEED = 0
TIMEOUT_S = 120.0
WARMUP = 1
REPEATS = 3


def _params(key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W = jax.random.normal(k1, (H, D), dtype=jnp.float64) * 0.3
    beta = jnp.zeros(H, dtype=jnp.float64)
    c = jax.random.normal(k3, (H,), dtype=jnp.float64)
    c = c / jnp.linalg.norm(c)
    X = jax.random.normal(k4, (B, D), dtype=jnp.float64)
    return W, beta, c, X


def _nested_hessian_lap(f, x: jnp.ndarray, k: int) -> jnp.ndarray:
    """Apply ``trace(hessian(·))`` ``k`` times to a scalar function of ``x``."""

    def lap_once(g):
        return lambda y: jnp.trace(jax.hessian(g)(y))

    g = f
    for _ in range(k):
        g = lap_once(g)
    return g(x)


def _try_time(label: str, build_fn) -> dict[str, object]:
    """Build + time a callable; catch OOM / timeout / compile blow-ups."""
    t_compile0 = time.perf_counter()
    try:
        fn = build_fn()
        # force first evaluation (JIT compile)
        out0 = fn()
        if hasattr(out0, "block_until_ready"):
            out0.block_until_ready()
        compile_s = time.perf_counter() - t_compile0
        if compile_s > TIMEOUT_S:
            return {"status": "timeout", "detail": f"compile+first-eval {compile_s:.1f}s > {TIMEOUT_S}s"}

        def run() -> None:
            o = fn()
            if hasattr(o, "block_until_ready"):
                o.block_until_ready()

        t_ms = median_time_ms(run, warmup=WARMUP, repeats=REPEATS)
        val = fn()
        if hasattr(val, "block_until_ready"):
            val.block_until_ready()
        arr = np.asarray(val, dtype=np.float64)
        return {
            "status": "ok",
            "time_ms": round(float(t_ms), 4),
            "value_sample": float(arr.reshape(-1)[0]),
            "compile_s": round(compile_s, 3),
        }
    except MemoryError as exc:
        return {"status": "oom", "detail": str(exc)[:200]}
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        # Nested autodiff often dies with a resource / tracing error rather than MemoryError.
        low = msg.lower()
        status = "oom" if ("memory" in low or "oom" in low or "resource" in low) else "error"
        return {"status": status, "detail": msg[:300], "traceback": traceback.format_exc()[-400:]}


def main() -> None:
    key = jax.random.PRNGKey(SEED)
    W, beta, c, X = _params(key)

    def f_point(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(c, jnp.tanh(W @ x + beta))

    rows: list[dict[str, object]] = []
    for k in KS:
        print(f"--- k={k} ---", flush=True)

        def build_omni(kk: int = k):
            @jax.jit
            def fn() -> jnp.ndarray:
                return neural_field_polylaplacian(X, W, beta, c, "tanh", k=kk)

            return fn

        omni = _try_time("omnibias", build_omni)

        def build_dense(kk: int = k):
            @jax.jit
            def fn() -> jnp.ndarray:
                return vmap(lambda x: _nested_hessian_lap(f_point, x, kk))(X)

            return fn

        dense = _try_time("dense_nested", build_dense)

        def build_folx(kk: int = k):
            # Nest forward_laplacian kk times on the scalar field.
            g = f_point
            for _ in range(kk):
                fwd = folx.forward_laplacian(g)

                def g(x, _fwd=fwd):  # noqa: B023
                    return _fwd(x).laplacian

            @jax.jit
            def fn() -> jnp.ndarray:
                return vmap(g)(X)

            return fn

        folx_row = _try_time("folx_nested", build_folx)

        row: dict[str, object] = {"k": k, "omnibias": omni, "folx_nested": folx_row, "dense_nested": dense}
        if omni.get("status") == "ok" and dense.get("status") == "ok":
            # Recompute values for an accuracy check (cheap for omnibias; dense already ok).
            def _dense_k(kk: int = k) -> jnp.ndarray:
                return vmap(lambda x: _nested_hessian_lap(f_point, x, kk))(X)

            def _omni_k(kk: int = k) -> jnp.ndarray:
                return neural_field_polylaplacian(X, W, beta, c, "tanh", k=kk)

            ref = np.asarray(jax.jit(_dense_k)())
            pred = np.asarray(jax.jit(_omni_k)())
            row["max_abs_diff_omnibias_vs_dense"] = float(np.max(np.abs(pred - ref)))
        rows.append(row)

    payload = provenance(
        schema="omnibias/polylaplacian-order/v1",
        config={
            "H": H,
            "D": D,
            "B": B,
            "k": list(KS),
            "activation": "tanh",
            "dtype": "float64",
            "seed": SEED,
            "timeout_s": TIMEOUT_S,
        },
    )
    payload["rows"] = rows
    path = write_json("polylaplacian_order.json", payload)
    print(f"wrote {path}")
    for r in rows:
        def _fmt(cell: dict) -> str:
            if cell.get("status") != "ok":
                return str(cell.get("status"))
            return f"{cell['time_ms']:.3f} ms"

        print(f"k={r['k']}: omni={_fmt(r['omnibias'])}  folx={_fmt(r['folx_nested'])}  dense={_fmt(r['dense_nested'])}")


if __name__ == "__main__":
    main()
