# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Laplacian cost vs input dimension ``D``.

Compares four methods on the one-layer field

    f(x) = c · tanh(W x + β)

for a batch of ``B`` points in ``R^D``:

* omnibias closed-form ``neural_field_laplacian``
* folx ``forward_laplacian`` (vmap'd)
* ``jax.hessian`` trace (vmap'd)
* ``torch.func.hessian`` trace (vmap'd)

Accuracy is max |Δ| against the ``jax.hessian`` reference. Run::

    uv run python benchmarks/laplacian_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import enable_x64, median_time_ms, provenance, write_json  # noqa: E402

enable_x64()

import folx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from jax import vmap  # noqa: E402
from omnibias.jax import neural_field_laplacian  # noqa: E402

torch.set_default_dtype(torch.float64)

H = 32
B = 64
DS = (3, 12, 30, 60)
SEED = 0
WARMUP = 2
REPEATS = 5


def _params(d: int, key: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    W = jax.random.normal(k1, (H, d), dtype=jnp.float64) * 0.3
    beta = jax.random.normal(k2, (H,), dtype=jnp.float64) * 0.1
    c = jax.random.normal(k3, (H,), dtype=jnp.float64)
    c = c / jnp.linalg.norm(c)
    X = jax.random.normal(k4, (B, d), dtype=jnp.float64)
    return W, beta, c, X


def _run_d(d: int) -> dict[str, object]:
    key = jax.random.PRNGKey(SEED + d)
    W, beta, c, X = _params(d, key)

    # ---- omnibias ----------------------------------------------------------
    def omni() -> jnp.ndarray:
        return neural_field_laplacian(X, W, beta, c, "tanh")

    omni_jit = jax.jit(omni)
    omni_jit().block_until_ready()
    t_omni = median_time_ms(lambda: omni_jit().block_until_ready(), warmup=WARMUP, repeats=REPEATS)
    lap_omni = np.asarray(omni_jit())

    # ---- jax.hessian (reference) -------------------------------------------
    def f_point(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(c, jnp.tanh(W @ x + beta))

    def jax_hess() -> jnp.ndarray:
        return vmap(lambda x: jnp.trace(jax.hessian(f_point)(x)))(X)

    jax_hess_jit = jax.jit(jax_hess)
    jax_hess_jit().block_until_ready()
    t_jax = median_time_ms(lambda: jax_hess_jit().block_until_ready(), warmup=WARMUP, repeats=REPEATS)
    lap_jax = np.asarray(jax_hess_jit())

    # ---- folx --------------------------------------------------------------
    fwd = folx.forward_laplacian(f_point)

    def folx_lap() -> jnp.ndarray:
        return vmap(lambda x: fwd(x).laplacian)(X)

    folx_jit = jax.jit(folx_lap)
    folx_jit().block_until_ready()
    t_folx = median_time_ms(lambda: folx_jit().block_until_ready(), warmup=WARMUP, repeats=REPEATS)
    lap_folx = np.asarray(folx_jit())

    # ---- torch.func.hessian ------------------------------------------------
    Wt = torch.tensor(np.asarray(W), dtype=torch.float64)
    betat = torch.tensor(np.asarray(beta), dtype=torch.float64)
    ct = torch.tensor(np.asarray(c), dtype=torch.float64)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float64)

    def f_torch(x: torch.Tensor) -> torch.Tensor:
        return torch.dot(ct, torch.tanh(Wt @ x + betat))

    def torch_hess() -> torch.Tensor:
        return torch.vmap(lambda x: torch.func.hessian(f_torch)(x).trace())(Xt)

    # warmup
    torch_hess()
    t_torch = median_time_ms(torch_hess, warmup=WARMUP, repeats=REPEATS)
    lap_torch = torch_hess().detach().numpy()

    ref = lap_jax
    return {
        "D": d,
        "time_ms": {
            "omnibias": round(t_omni, 4),
            "folx": round(t_folx, 4),
            "jax_hessian": round(t_jax, 4),
            "torch_func_hessian": round(t_torch, 4),
        },
        "speedup_vs_omnibias": {
            "folx": round(t_folx / t_omni, 3),
            "jax_hessian": round(t_jax / t_omni, 3),
            "torch_func_hessian": round(t_torch / t_omni, 3),
        },
        "max_abs_diff_vs_jax_hessian": {
            "omnibias": float(np.max(np.abs(lap_omni - ref))),
            "folx": float(np.max(np.abs(lap_folx - ref))),
            "torch_func_hessian": float(np.max(np.abs(lap_torch - ref))),
        },
    }


def main() -> None:
    rows = [_run_d(d) for d in DS]
    payload = provenance(
        schema="omnibias/laplacian-scaling/v1",
        config={"H": H, "B": B, "D": list(DS), "activation": "tanh", "dtype": "float64", "seed": SEED},
    )
    payload["rows"] = rows
    path = write_json("laplacian_scaling.json", payload)
    print(f"wrote {path}")
    print(f"{'D':>4}  {'omni':>10}  {'folx':>10}  {'jax.H':>10}  {'torch.H':>10}  {'Δomni':>10}")
    for r in rows:
        t = r["time_ms"]  # type: ignore[index]
        dlt = r["max_abs_diff_vs_jax_hessian"]  # type: ignore[index]
        print(
            f"{r['D']:4d}  {t['omnibias']:10.3f}  {t['folx']:10.3f}  "
            f"{t['jax_hessian']:10.3f}  {t['torch_func_hessian']:10.3f}  "
            f"{dlt['omnibias']:10.2e}"
        )


if __name__ == "__main__":
    main()
