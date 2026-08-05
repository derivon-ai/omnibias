# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``σ^(n)`` cost and accuracy vs nested autodiff and finite differences.

Run::

    uv run python benchmarks/derivative_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import enable_x64, median_time_ms, provenance, write_json  # noqa: E402

enable_x64()

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get  # noqa: E402
from omnibias.torch.activations.registry import get_activation as torch_get  # noqa: E402

torch.set_default_dtype(torch.float64)

N_POINTS = 20_000
ORDERS = list(range(1, 9))
H_FD = 5e-2
WARMUP = 2
REPEATS = 5


def _fd_nth(values_on_pad: np.ndarray, n: int, h: float) -> np.ndarray:
    """Apply the central-difference operator ``n`` times along axis 1."""
    d = values_on_pad
    for _ in range(n):
        d = (d[:, 2:] - d[:, :-2]) / (2.0 * h)
    return d[:, d.shape[1] // 2]


def main() -> None:
    spec_t = torch_get("tanh")
    spec_j = jax_get("tanh")
    z_np = np.linspace(-3.0, 3.0, N_POINTS, dtype=np.float64)
    zt = torch.from_numpy(z_np)
    zj = jnp.asarray(z_np)

    rows: list[dict[str, object]] = []
    for n in ORDERS:
        # closed-form (torch)
        def cf(nn: int = n) -> torch.Tensor:
            return spec_t.fastpath(zt, nn)

        t_cf = median_time_ms(cf, warmup=WARMUP, repeats=REPEATS)
        truth = cf().detach().numpy()

        # nested torch autograd
        def nested(nn: int = n) -> torch.Tensor:
            zz = zt.clone().requires_grad_(True)
            y = spec_t.forward(zz).sum()
            g = torch.autograd.grad(y, zz, create_graph=True)[0]
            for _ in range(nn - 1):
                g = torch.autograd.grad(g.sum(), zz, create_graph=True)[0]
            return g

        t_ag = median_time_ms(nested, warmup=max(1, WARMUP - 1), repeats=3)
        ag = nested().detach().numpy()

        # jax closed-form (parity)
        jv = np.asarray(spec_j.fastpath(zj, n), dtype=np.float64)

        # finite differences
        half = n + 2
        pad = np.arange(-half, half + 1, dtype=np.float64) * H_FD
        grid = z_np[:, None] + pad[None, :]
        base = np.tanh(grid)
        fd = _fd_nth(base, n, H_FD)

        denom = np.maximum(np.abs(truth), 1e-3)
        rows.append(
            {
                "n": n,
                "time_ms": {
                    "closed_form": round(t_cf, 4),
                    "nested_autograd": round(t_ag, 4),
                },
                "speedup_vs_closed_form": round(t_ag / t_cf, 3) if t_cf > 0 else None,
                "median_rel_error": {
                    "closed_form_vs_self": 0.0,
                    "nested_autograd": float(np.median(np.abs(ag - truth) / denom)),
                    "finite_difference": float(np.median(np.abs(fd - truth) / denom)),
                    "jax_vs_torch": float(np.median(np.abs(jv - truth) / denom)),
                },
                "max_abs_jax_vs_torch": float(np.max(np.abs(jv - truth))),
            }
        )
        print(
            f"n={n}: cf={t_cf:.3f} ms  nested={t_ag:.3f} ms  "
            f"FD_err={rows[-1]['median_rel_error']['finite_difference']:.2e}  "  # type: ignore[index]
            f"jaxΔ={rows[-1]['max_abs_jax_vs_torch']:.2e}"
        )

    payload = provenance(
        schema="omnibias/derivative-order/v1",
        config={
            "activation": "tanh",
            "n_points": N_POINTS,
            "orders": ORDERS,
            "fd_step": H_FD,
            "dtype": "float64",
        },
    )
    payload["rows"] = rows
    path = write_json("derivative_order.json", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
