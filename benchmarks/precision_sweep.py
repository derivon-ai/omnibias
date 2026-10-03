# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded float64 accuracy probe against independent 80-digit derivatives.

Run ``uv run python benchmarks/precision_sweep.py``. The default four cases
show how input and derivative order affect polynomial cancellation. This is
an accuracy probe, not a timing comparison or an exhaustive stability bound.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from typing import Any, cast

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mpmath as mp  # type: ignore[import-untyped]  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from _common import provenance, source_provenance, write_json  # noqa: E402
from _reference import derivative_reference  # noqa: E402
from jax.experimental.jet import jet  # noqa: E402
from omnibias.jax._fastpath import tanh_nth_derivative as jax_closed  # noqa: E402
from omnibias.torch.fastpath.legendre import tanh_nth_derivative as torch_closed  # noqa: E402

TAYLOR_JET = cast(Callable[..., Any], jet)


def collect(orders: list[int], points: list[float]) -> dict[str, Any]:
    """Compare actual candidate values with independently differentiated tanh."""
    if not orders or len(orders) > 8 or any(n < 1 or n > 32 for n in orders):
        raise ValueError("choose 1 to 8 derivative orders, each between 1 and 32")
    if not points or len(points) > 16 or not np.isfinite(points).all():
        raise ValueError("choose 1 to 16 finite input points")
    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    xs = np.asarray(points, dtype=np.float64)
    tx, jx = torch.from_numpy(xs), jnp.asarray(xs)
    rows = []
    for order in orders:
        reference = derivative_reference(mp.tanh, xs, order)
        series = (jnp.ones_like(jx),) + (jnp.zeros_like(jx),) * (order - 1)
        candidates = {
            "torch_closed_form": torch_closed(tx, order).detach().numpy(),
            "jax_closed_form": np.asarray(jax_closed(jx, order)),
            "jax_taylor_ad": np.asarray(TAYLOR_JET(jnp.tanh, (jx,), (series,))[1][-1]),
        }
        for index, point in enumerate(points):
            truth = float(reference[index])
            methods = {}
            for name, values in candidates.items():
                value = float(values[index])
                absolute = abs(value - truth)
                methods[name] = {
                    "value": value,
                    "absolute_error": absolute,
                    "relative_error": absolute / abs(truth) if truth else None,
                }
            rows.append({"order": order, "point": point, "reference": truth,
                         "methods": methods})
    payload = provenance(schema="omnibias/precision-sweep/v1", config={
        "activation": "tanh", "dtype": "float64", "device": "cpu",
        "orders": orders, "points": points,
        "reference": "mpmath.diff(mp.tanh), 80 decimal digits, rounded once to float64",
        "reference_inputs": "exact binary64 inputs, not decimal-rounded replacements",
        "execution": "eager closed-form fastpaths; JAX experimental.jet Taylor-mode AD",
        "scope": "pointwise accuracy only; no timings, training, or exhaustive stability claim",
        "relative_error": "absolute_error / abs(reference); null when reference is zero",
    })
    payload["versions"]["mpmath"] = mp.__version__
    payload.update(source_provenance([
        "benchmarks/precision_sweep.py", "benchmarks/_common.py", "benchmarks/_reference.py",
        "packages/omnibias-core/src/omnibias/core/polynomials.py",
        "packages/omnibias-torch/src/omnibias/torch/fastpath/legendre.py",
        "packages/omnibias-jax/src/omnibias/jax/_fastpath.py",
    ]))
    payload["results"] = rows
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[12, 16])
    parser.add_argument("--points", type=float, nargs="+", default=[0.1, 3.0])
    args = parser.parse_args()
    print(write_json("precision_sweep.json", collect(args.orders, args.points)))


if __name__ == "__main__":
    main()
