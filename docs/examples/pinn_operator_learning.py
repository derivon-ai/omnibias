# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator learning with closed-form query-coordinate derivatives.

Run::

    pip install "omnibias-pinn[torch]"
    python docs/examples/pinn_operator_learning.py

A DeepONet ``G(u)(y) = b_0 + sum_k b_k(u) t_k(y)`` is linear in the trunk
basis, so every mixed partial in the query coordinate is

    d^alpha G(u)(y) = sum_k b_k(u) d^alpha t_k(y)

The trunk is an omnibias jet network: one multivariate trunk jet yields every
``d^alpha t_k`` up to a chosen order. The shipped Kuramoto-Sivashinsky residual
(order 4) therefore runs unchanged on a conditioned DeepONet -- no finite
difference, no re-derived residual.

This example pins deterministic structural claims only. Measured operator
accuracy lives in ``benchmarks/operator_*.py`` / ``docs/benchmarks.md``.

Honesty

*Structural.* Query-coordinate derivatives of ``G(u)`` are exact through order
4 (closed-form trunk jet vs nested autograd); a 4th-order residual costs
exactly one trunk jet; the residual enclosure over a family of inputs is sound
interval arithmetic, not a solution-error bound.

*Measured.* Operator accuracy is optimised, not proven -- see the bake-off /
FD-floor / shared-grid artifacts.

*Out of scope.* Nothing Clay-scale. FNO derivatives stay FFT-based; the
closed-form claim does not transfer. "Operator" here is neural-operator
learning (sense 3 of ``docs/operator-surface.md``), not ``OperatorBlock``.
"""

from __future__ import annotations

import torch
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator import (
    branch_coefficient_box,
    certify_heat_residual,
    enclose_heat_residual,
)
from omnibias.pinn.operator.torch import build_deeponet
from omnibias.pinn.operator.torch.deeponet import TRUNK_JET_CACHE_KEY
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.equations.kuramoto_sivashinsky import KuramotoSivashinsky

DTYPE = torch.float64
SEED = 0
DIFFUSIVITY = 0.1


def part_one_order4_exactness() -> None:
    print("[1] Order-4 closed-form exactness vs nested autograd")
    torch.manual_seed(SEED)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=6,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        jet_order=4,
        dtype=DTYPE,
    )
    field = op.condition(torch.randn(1, 8, dtype=DTYPE))
    coords = torch.randn(6, 2, dtype=DTYPE, requires_grad=True)
    state = field(coords.detach())
    closed = tops.derivative(state, "u", axis=0, order=4)
    u = field.forward_values(coords)[:, 0]
    v = u
    for _ in range(4):
        (g,) = torch.autograd.grad(v.sum(), coords, create_graph=True)
        v = g[:, 0]
    err = float((closed.detach() - v.detach()).abs().max())
    print(f"    max |u_xxxx closed - nested AD| = {err:.3e}")
    assert err < 1e-10


def part_two_ks_one_jet() -> None:
    print("[2] Shipped KS residual on a DeepONet: exactly one order-4 trunk jet")
    torch.manual_seed(SEED)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=6,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        jet_order=4,
        dtype=DTYPE,
    )
    field = op.condition(torch.randn(1, 8, dtype=DTYPE))
    state = field(torch.randn(8, 2, dtype=DTYPE))
    out = KuramotoSivashinsky()(state)
    assert torch.isfinite(out.residual).all()
    cached = state.extra[TRUNK_JET_CACHE_KEY]
    print(f"    residual finite; trunk-jet cache orders = {sorted(cached)}")
    assert sorted(cached) == [4]


def part_three_fd_floor_smoke() -> None:
    print("[3] FD floor smoke: 5-point u_xxxx vs closed form at two h values")
    torch.manual_seed(SEED)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t"), time_axis="t"),
        components=ComponentSpec(("u",)),
        n_sensors=8,
        trunk_width=6,
        trunk_hidden=12,
        trunk_depth=2,
        branch_hidden=12,
        branch_depth=2,
        jet_order=4,
        dtype=DTYPE,
    )
    field = op.condition(torch.randn(1, 8, dtype=DTYPE))
    x, t = 0.3, 0.4

    def value(xx: float, tt: float) -> float:
        return float(
            field.forward_values(torch.tensor([[xx, tt]], dtype=DTYPE))[0, 0].detach()
        )

    state = field(torch.tensor([[x, t]], dtype=DTYPE))
    closed = float(tops.derivative(state, "u", axis=0, order=4)[0].detach())

    def fd5(h: float) -> float:
        return (
            value(x - 2 * h, t)
            - 4 * value(x - h, t)
            + 6 * value(x, t)
            - 4 * value(x + h, t)
            + value(x + 2 * h, t)
        ) / (h**4)

    # Stay in the truncation-dominated regime (see operator_fd_floor.json):
    # for order-4 the round-off upturn arrives near h ~ 1e-3.
    e_coarse = abs(fd5(1e-1) - closed)
    e_fine = abs(fd5(1e-2) - closed)
    print(f"    |FD-closed| at h=1e-1: {e_coarse:.3e}; at h=1e-2: {e_fine:.3e}")
    assert e_fine < e_coarse
    assert e_coarse > 1e-8  # FD is not at machine epsilon in this regime


def part_four_family_certificate() -> None:
    print("[4] Family certificate: sensor box -> coeffs -> residual enclosure")
    # Affine+tanh branch and nonlinear trunk over a small box.
    W1 = [[0.3, -0.1, 0.2, 0.05], [0.1, 0.2, -0.15, 0.1]]
    b1 = [0.0, 0.0]
    W2 = [[0.4, -0.2], [-0.1, 0.3]]
    b2 = [0.0, 0.0]
    sensors_box = [(-0.1, 0.1)] * 4
    coeff_box, _ = branch_coefficient_box(
        sensors_box, [(W1, b1, "tanh"), (W2, b2, None)]
    )
    trunk = [
        ([[0.4, 0.1], [-0.2, 0.3], [0.1, -0.15]], [0.0, 0.0, 0.0], "tanh"),
        ([[0.5, -0.2, 0.1], [0.1, 0.4, -0.3]], [0.0, 0.0], None),
    ]
    enclosure = enclose_heat_residual(
        trunk_layers=trunk,
        coeffs=coeff_box,
        query_box=[(-0.05, 0.05), (0.0, 0.1)],
        diffusivity=DIFFUSIVITY,
    )
    cert = certify_heat_residual(
        trunk_layers=trunk,
        coeffs=coeff_box,
        query_box=[(-0.05, 0.05), (0.0, 0.1)],
        diffusivity=DIFFUSIVITY,
    )
    assert verify_certificate_digest(cert)
    assert enclosure.hi > enclosure.lo
    print(
        f"    residual enclosure [{enclosure.lo:.4e}, {enclosure.hi:.4e}]; "
        "digest ok"
    )


def main() -> None:
    torch.set_default_dtype(DTYPE)
    part_one_order4_exactness()
    part_two_ks_one_jet()
    part_three_fd_floor_smoke()
    part_four_family_certificate()
    print("ok")


if __name__ == "__main__":
    main()
