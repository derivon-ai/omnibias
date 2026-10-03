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

from _common import enable_x64, measure, provenance, source_provenance, write_json  # noqa: E402
from _reference import derivative_reference, error_metrics  # noqa: E402

enable_x64()

import jax.numpy as jnp  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get  # noqa: E402
from omnibias.torch.activations.registry import get_activation as torch_get  # noqa: E402

torch.set_default_dtype(torch.float64)

N_POINTS = 20_000
ORDERS = list(range(1, 9))
H_FD = 5e-2
WARMUP = 2
REPEATS = 9
REFERENCE_POINTS = 33


def _fd_nth(values_on_pad: np.ndarray, n: int, h: float) -> np.ndarray:
    """Apply the central-difference operator ``n`` times along axis 1."""
    d = values_on_pad
    for _ in range(n):
        d = (d[:, 2:] - d[:, :-2]) / (2.0 * h)
    return d[:, d.shape[1] // 2]


def main() -> None:
    torch.set_num_threads(1)
    spec_t = torch_get("tanh")
    spec_j = jax_get("tanh")
    torch_fastpath = spec_t.fastpath
    jax_fastpath = spec_j.fastpath
    assert torch_fastpath is not None and jax_fastpath is not None
    z_np = np.linspace(-3.0, 3.0, N_POINTS, dtype=np.float64)
    zt = torch.from_numpy(z_np)
    zj = jnp.asarray(z_np)
    reference_indices = np.linspace(0, N_POINTS - 1, REFERENCE_POINTS, dtype=int)

    rows: list[dict[str, object]] = []
    for n in ORDERS:
        # closed-form (torch)
        def cf(nn: int = n) -> torch.Tensor:
            return torch_fastpath(zt, nn)

        measured_cf = measure(cf, warmup=WARMUP, repeats=REPEATS)
        t_cf = measured_cf["median_ms"]
        closed_form = cf().detach().numpy()
        reference = derivative_reference(mp.tanh, z_np[reference_indices], n)

        # nested torch autograd
        def nested(nn: int = n) -> torch.Tensor:
            zz = zt.clone().requires_grad_(True)
            y = spec_t.forward(zz).sum()
            g = torch.autograd.grad(y, zz, create_graph=nn > 1)[0]
            for depth in range(nn - 1):
                g = torch.autograd.grad(g.sum(), zz, create_graph=depth < nn - 2)[0]
            return g

        measured_ag = measure(nested, warmup=WARMUP, repeats=REPEATS)
        t_ag = measured_ag["median_ms"]
        ag = nested().detach().numpy()

        # jax closed-form (parity)
        jv = np.asarray(jax_fastpath(zj, n), dtype=np.float64)

        for values in (closed_form, ag, jv):
            np.testing.assert_allclose(values[reference_indices], reference, rtol=2e-10, atol=1e-9)

        # finite differences
        half = n + 2
        pad = np.arange(-half, half + 1, dtype=np.float64) * H_FD
        grid = z_np[:, None] + pad[None, :]
        base = np.tanh(grid)
        fd = _fd_nth(base, n, H_FD)

        rows.append(
            {
                "n": n,
                "measurements": {"closed_form": measured_cf, "nested_autograd": measured_ag},
                "time_ms": {
                    "closed_form": round(t_cf, 4),
                    "nested_autograd": round(t_ag, 4),
                },
                "speedup_vs_closed_form": round(t_ag / t_cf, 3) if t_cf > 0 else None,
                "error_vs_mpmath": {
                    "closed_form": error_metrics(closed_form[reference_indices], reference),
                    "nested_autograd": error_metrics(ag[reference_indices], reference),
                    "finite_difference": error_metrics(fd[reference_indices], reference),
                    "jax_closed_form": error_metrics(jv[reference_indices], reference),
                },
                "max_abs_jax_vs_torch": float(np.max(np.abs(jv - closed_form))),
            }
        )
        print(
            f"n={n}: cf={t_cf:.3f} ms  nested={t_ag:.3f} ms  "
            f"FD_max_abs={error_metrics(fd[reference_indices], reference)['max_abs']:.2e}  "
            f"jaxΔ={rows[-1]['max_abs_jax_vs_torch']:.2e}"
        )

    payload = provenance(
        schema="omnibias/derivative-order/v3",
        config={
            "activation": "tanh",
            "n_points": N_POINTS,
            "orders": ORDERS,
            "fd_step": H_FD,
            "dtype": "float64",
            "reference": "mpmath.diff(tanh), 80 decimal digits, rounded to float64",
            "reference_points": REFERENCE_POINTS,
            "relative_error_floor": 1e-3,
            "torch_threads": 1, "repeats": REPEATS, "warmup": WARMUP,
            "execution": "PyTorch eager, one CPU thread; JAX closed form for accuracy only",
            "timing_scope": "activation derivative evaluation only; final derivative graph not retained",
            "accuracy_rtol": 2e-10, "accuracy_atol": 1e-9,
        },
    )
    payload.update(source_provenance([
        "benchmarks/derivative_order.py", "benchmarks/_common.py", "benchmarks/_reference.py",
        "packages/omnibias-torch/src/omnibias/torch/fastpath/legendre.py",
        "packages/omnibias-jax/src/omnibias/jax/_fastpath.py",
        "packages/omnibias-core/src/omnibias/core/polynomials.py",
    ]))
    payload["rows"] = rows
    path = write_json("derivative_order.json", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
