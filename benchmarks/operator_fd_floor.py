# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Deterministic FD accuracy floor vs closed-form DeepONet derivatives.

For a fixed conditioned DeepONet field, sweep the finite-difference step ``h``
across many decades and compare:

* 5-point central FD ``u_xxxx`` against the closed-form trunk-jet ``u_xxxx``
* 2-point central FD ``u_t`` against the closed-form ``u_t``

The closed form is itself validated against four nested ``torch.autograd``
passes. The FD curve is U-shaped (truncation vs round-off); its floor sits
orders of magnitude above machine epsilon, while the closed-form error does
not. Measured truncation slope in the truncation-dominated regime is checked
with a ``log2`` rate chain (target ≈ 2 for the 5-point 4th-derivative stencil).

Run::

    uv run python benchmarks/operator_fd_floor.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from _common import provenance, write_json  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn.operator.torch import build_deeponet  # noqa: E402
from omnibias.pinn.torch import ops as tops  # noqa: E402

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

OUT_NAME = os.environ.get("OP_FD_OUT", "operator_fd_floor.json")
SEED = int(os.environ.get("OP_FD_SEED", "0"))
# Geometric h sweep covering truncation and round-off regimes.
H_VALUES = tuple(
    float(h)
    for h in os.environ.get(
        "OP_FD_H",
        ",".join(str(10.0**e) for e in range(-1, -9, -1)),
    ).split(",")
)
QUERY = (0.3, 0.4)  # (x, t) interior point
MACHINE_EPS = float(torch.finfo(torch.float64).eps)


def _build_field() -> Any:
    torch.manual_seed(SEED)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=8,
        trunk_hidden=16,
        trunk_depth=2,
        branch_hidden=16,
        branch_depth=2,
        jet_order=4,
    )
    sensors = torch.randn(1, 8)
    return op.condition(sensors)


def _value_at(field: Any, x: float, t: float) -> float:
    coords = torch.tensor([[x, t]], dtype=torch.float64)
    return float(field.forward_values(coords)[0, 0].detach())


def _closed_form(field: Any, x: float, t: float) -> tuple[float, float]:
    coords = torch.tensor([[x, t]], dtype=torch.float64)
    state = field(coords)
    u_xxxx = float(tops.derivative(state, "u", axis=0, order=4)[0].detach())
    u_t = float(tops.derivative(state, "u", axis=1, order=1)[0].detach())
    return u_xxxx, u_t


def _autograd_u_xxxx(field: Any, x: float, t: float) -> float:
    coords = torch.tensor([[x, t]], dtype=torch.float64, requires_grad=True)
    u = field.forward_values(coords)[:, 0]
    v = u
    for _ in range(4):
        (g,) = torch.autograd.grad(v.sum(), coords, create_graph=True)
        v = g[:, 0]
    return float(v.detach()[0])


def _fd_u_xxxx(field: Any, x: float, t: float, h: float) -> float:
    # 5-point stencil: (u_{-2} - 4 u_{-1} + 6 u_0 - 4 u_1 + u_2) / h^4
    um2 = _value_at(field, x - 2 * h, t)
    um1 = _value_at(field, x - h, t)
    u0 = _value_at(field, x, t)
    up1 = _value_at(field, x + h, t)
    up2 = _value_at(field, x + 2 * h, t)
    return (um2 - 4.0 * um1 + 6.0 * u0 - 4.0 * up1 + up2) / (h**4)


def _fd_u_t(field: Any, x: float, t: float, h: float) -> float:
    # 2-point central: (u(t+h) - u(t-h)) / (2h)
    return (_value_at(field, x, t + h) - _value_at(field, x, t - h)) / (2.0 * h)


def main() -> None:
    t0 = time.perf_counter()
    field = _build_field()
    x, t = QUERY
    closed_xxxx, closed_t = _closed_form(field, x, t)
    ad_xxxx = _autograd_u_xxxx(field, x, t)
    closed_vs_ad = abs(closed_xxxx - ad_xxxx)

    rows: list[dict[str, Any]] = []
    for h in H_VALUES:
        fd4 = _fd_u_xxxx(field, x, t, h)
        fd1 = _fd_u_t(field, x, t, h)
        rows.append(
            {
                "h": h,
                "fd_u_xxxx_abs_err": abs(fd4 - closed_xxxx),
                "fd_u_t_abs_err": abs(fd1 - closed_t),
                "closed_u_xxxx": closed_xxxx,
                "closed_u_t": closed_t,
            }
        )

    # Truncation-dominated regime for order-4 5-point: largest two h with
    # decreasing error as h shrinks (before the round-off upturn).
    errs4 = [r["fd_u_xxxx_abs_err"] for r in rows]
    rates: list[float] = []
    for i in range(len(errs4) - 1):
        if errs4[i] > 0.0 and errs4[i + 1] > 0.0 and errs4[i + 1] < errs4[i]:
            # h halves? our sweep is decade (×0.1), so log10; convert to order.
            # err ~ C h^p  =>  log(e_i/e_{i+1}) / log(h_i/h_{i+1}) = p
            hi, hj = H_VALUES[i], H_VALUES[i + 1]
            p = math.log(errs4[i] / errs4[i + 1]) / math.log(hi / hj)
            rates.append(p)
        else:
            break
    # Prefer the first two successive rates in the truncation chain (log2 sense
    # after normalising to factor-of-two steps for the assertion).
    measured_order = float(sum(rates) / len(rates)) if rates else float("nan")
    # Also report a log2 rate between the first pair of decade steps by
    # rescaling: if decade ratio is 10, log2-equivalent order uses same p.
    fd_floor = min(errs4)
    closed_floor = closed_vs_ad

    # Assertions that gate the artifact.
    assert closed_vs_ad < 1e-10, f"closed form vs nested AD: {closed_vs_ad}"
    assert fd_floor > 1e3 * MACHINE_EPS, (
        f"FD floor {fd_floor} not bounded away from machine eps {MACHINE_EPS}"
    )
    assert closed_floor < 1e2 * MACHINE_EPS or closed_floor < 1e-12, (
        f"closed-form vs AD {closed_floor} not near machine precision"
    )
    # 5-point 4th-derivative truncation is O(h^2); accept [1.5, 2.5] on the
    # measured chain in the truncation regime.
    assert rates, "no truncation-dominated h pairs found"
    assert 1.5 <= measured_order <= 2.5, f"measured order {measured_order} not ~2"

    payload = provenance(
        schema="operator_fd_floor/v1",
        config={
            "seed": SEED,
            "h_values": list(H_VALUES),
            "query": list(QUERY),
            "jet_order": 4,
            "trunk_width": 8,
            "stencil_u_xxxx": "5-point central / h^4",
            "stencil_u_t": "2-point central / (2h)",
        },
    )
    payload["closed_form_vs_autograd_u_xxxx"] = closed_vs_ad
    payload["machine_eps"] = MACHINE_EPS
    payload["fd_u_xxxx_floor"] = fd_floor
    payload["measured_truncation_order_u_xxxx"] = measured_order
    payload["truncation_rates"] = rates
    payload["rows"] = rows
    payload["elapsed_seconds"] = round(time.perf_counter() - t0, 3)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path}")
    print(f"FD floor={fd_floor:.3e}  closed_vs_AD={closed_vs_ad:.3e}")
    print(f"measured truncation order={measured_order:.3f}  rates={rates}")


if __name__ == "__main__":
    main()
