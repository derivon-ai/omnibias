# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared-grid DeepONet residual scaling: one trunk jet across F inputs.

Times a closed-form heat residual over ``F`` input functions via
``field.on_grid(query)`` against a per-sample loop, for
``F in {1, 2, 4, 8, 16, 32}``. Every timing is paired with an accuracy check
that both paths agree to float64 round-off.

Run::

    uv run python benchmarks/operator_shared_grid.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from _common import median_time_ms, provenance, write_json  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn.operator.torch import build_deeponet  # noqa: E402
from omnibias.pinn.torch import ops as tops  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUT_NAME = os.environ.get("OP_SG_OUT", "operator_shared_grid.json")
SEED = int(os.environ.get("OP_SG_SEED", "0"))
F_VALUES = tuple(
    int(x) for x in os.environ.get("OP_SG_F", "1,2,4,8,16,32").split(",")
)
Q = int(os.environ.get("OP_SG_Q", "64"))
DIFFUSIVITY = 0.1
WARMUP = 2
REPEATS = 5


def _residual(state: Any) -> torch.Tensor:
    u_t = tops.derivative(state, "u", axis=1, order=1)
    u_xx = tops.derivative(state, "u", axis=0, order=2)
    return u_t - DIFFUSIVITY * u_xx


def main() -> None:
    t0 = time.perf_counter()
    torch.manual_seed(SEED)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        n_sensors=16,
        trunk_width=16,
        trunk_hidden=32,
        trunk_depth=2,
        branch_hidden=32,
        branch_depth=2,
        jet_order=2,
    )
    query = torch.randn(Q, 2)

    rows: list[dict[str, Any]] = []
    for F in F_VALUES:
        sensors = torch.randn(F, 16)

        def shared(sensors_f: torch.Tensor = sensors) -> torch.Tensor:
            field = op.condition(sensors_f)
            return _residual(field.on_grid(query))

        def looped(
            sensors_f: torch.Tensor = sensors, n_f: int = F
        ) -> torch.Tensor:
            outs = []
            for f in range(n_f):
                field = op.condition(sensors_f[f : f + 1])
                outs.append(_residual(field(query)))
            return torch.cat(outs, dim=0)

        # Accuracy pairing (house rule: never time without checking agreement).
        with torch.no_grad():
            a = shared()
            b = looped()
            max_abs = float((a - b).abs().max())
        assert max_abs < 1e-12, f"F={F}: shared vs loop disagree by {max_abs}"

        t_shared = median_time_ms(shared, warmup=WARMUP, repeats=REPEATS)
        t_loop = median_time_ms(looped, warmup=WARMUP, repeats=REPEATS)
        rows.append(
            {
                "F": F,
                "shared_median_ms": round(t_shared, 4),
                "loop_median_ms": round(t_loop, 4),
                "speedup": round(t_loop / t_shared, 4) if t_shared > 0 else None,
                "max_abs_diff": max_abs,
            }
        )
        print(
            f"F={F:2d}  shared={t_shared:.2f}ms  loop={t_loop:.2f}ms  "
            f"speedup={t_loop / t_shared:.2f}x  err={max_abs:.2e}",
            flush=True,
        )

    payload = provenance(
        schema="operator_shared_grid/v1",
        config={
            "seed": SEED,
            "F_values": list(F_VALUES),
            "Q": Q,
            "diffusivity": DIFFUSIVITY,
            "warmup": WARMUP,
            "repeats": REPEATS,
            "trunk_width": 16,
            "jet_order": 2,
        },
    )
    payload["rows"] = rows
    payload["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
