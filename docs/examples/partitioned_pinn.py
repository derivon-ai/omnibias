# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Discontinuity-capturing PINN -- omnibias.pinn.partition (a bridge on omnibias-partition).

Run:

    pip install "omnibias-pinn[torch]" omnibias-partition
    python docs/examples/partitioned_pinn.py

A single smooth activation network cannot represent a **kink** (a derivative jump); a
partition of smooth sub-solutions can. We solve a 1D Poisson interface problem whose exact
solution has a kink at ``x = 0``:

    u''(x) = 2   on (-1, 0) u (0, 1),     u(-1) = u(1) = 0,     u(0) = 0,
    exact:  u*(x) = x^2 - |x|             (slopes +/-1 at 0 -> a derivative jump of 2).

A plain ``OneLayerVectorField`` is globally smooth: forced ``u'' ~ 2`` everywhere with
``u(+/-1) = 0`` its best is ~ ``x^2 - 1`` (``u(0) ~ -1``), so it *cannot* also hit the
interface value ``u(0) = 0``. The :class:`PartitionedField` runs a **conservative-PINN**:
each region's residual ``u_l'' = 2`` is the exact closed-form ``sigma``-tower second
derivative of that region's own :class:`OneLayerVectorField`, and the two are glued by an
interface condition ``u_L(0) = u_R(0) = 0``. The deploy-time field is the soft blend
``u = w_L u_L + w_R u_R`` which hardens as ``beta -> inf``; we report its *sound* certified
soft->hard partition gap.

Honesty: the physics residual is *minimized* (not proven); the partition gap is *sound*
(via ``omnibias.partition.certify_partition_gap``). The ``beta -> inf`` gate hardening is
**temperature collapse** (the feasibility sense), not the founding ``delta -> 0`` bias
collapse (the multi-bias limit to ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

import torch
from omnibias.partition import certify_partition_gap
from omnibias.partition._core.config import PartitionConfig
from omnibias.partition._core.params import PartitionParams
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.partition.torch import build_partitioned_field
from omnibias.pinn.torch import ops
from omnibias.pinn.torch.fields import OneLayerVectorField


def _exact(x: torch.Tensor) -> torch.Tensor:
    return x[:, 0] ** 2 - x[:, 0].abs()


def _residual(field: object, x: torch.Tensor) -> torch.Tensor:
    r"""MSE of ``u'' - 2`` on ``x`` (closed-form 2nd derivative of a OneLayerVectorField)."""
    return (ops.derivative(field(x), "u", axis=0, order=2) - 2.0).pow(2).mean()


def _value_at(field: object, x: torch.Tensor) -> torch.Tensor:
    return ops.value(field(x), "u")


def _train_plain(field: object, steps: int = 900, lr: float = 5e-3) -> None:
    x_int = torch.linspace(-1.0, 1.0, 81, dtype=torch.float64).reshape(-1, 1)
    x_bc = torch.tensor([[-1.0], [1.0], [0.0]], dtype=torch.float64)  # u=0 at all three
    opt = torch.optim.Adam(field.parameters(), lr=lr)  # type: ignore[attr-defined]
    for _ in range(steps):
        opt.zero_grad()
        loss = _residual(field, x_int) + 10.0 * _value_at(field, x_bc).pow(2).mean()
        loss.backward()
        opt.step()


def _train_partitioned(field: object, steps: int = 900, lr: float = 5e-3) -> None:
    xL = torch.linspace(-1.0, 0.0, 41, dtype=torch.float64).reshape(-1, 1)  # left region
    xR = torch.linspace(0.0, 1.0, 41, dtype=torch.float64).reshape(-1, 1)  # right region
    x0 = torch.zeros(1, 1, dtype=torch.float64)
    subL, subR = field.subfields[0], field.subfields[1]  # region 0 = x<0, region 1 = x>0
    xm1 = torch.tensor([[-1.0]], dtype=torch.float64)
    xp1 = torch.tensor([[1.0]], dtype=torch.float64)
    opt = torch.optim.Adam(field.parameters(), lr=lr)  # type: ignore[attr-defined]
    for _ in range(steps):
        opt.zero_grad()
        # each region's PDE on its OWN sub-solution (exact closed-form u''), glued at x=0
        pde = _residual(subL, xL) + _residual(subR, xR)
        bc = _value_at(subL, xm1).pow(2).mean() + _value_at(subR, xp1).pow(2).mean()
        uL0, uR0 = _value_at(subL, x0), _value_at(subR, x0)
        interface = (uL0.pow(2) + uR0.pow(2) + (uL0 - uR0).pow(2)).mean()
        (pde + 10.0 * bc + 10.0 * interface).backward()
        opt.step()


def _errors(field: object) -> tuple[float, float]:
    xg = torch.linspace(-1.0, 1.0, 201, dtype=torch.float64).reshape(-1, 1)
    with torch.no_grad():
        l2 = float((_value_at(field, xg) - _exact(xg)).pow(2).mean().sqrt())
        interface = float(_value_at(field, torch.zeros(1, 1, dtype=torch.float64)).abs().item())
    return l2, interface


def main() -> None:
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    print("=== discontinuity-capturing PINN: u''=2, u(+/-1)=0, u(0)=0 (kink at x=0) ===")

    torch.manual_seed(0)
    plain = OneLayerVectorField(
        coordinate_spec=cs, components=comp, hidden=32, base="tanh", dtype=torch.float64
    )
    _train_plain(plain)
    plain_l2, plain_iface = _errors(plain)
    print(f"    plain OneLayerVectorField  : L2={plain_l2:.4f}  interface |u(0)|={plain_iface:.4f}")

    torch.manual_seed(0)
    field = build_partitioned_field(
        coordinate_spec=cs, components=comp,
        split_dirs=torch.tensor([[1.0]]), split_thresh=torch.tensor([0.0]),
        hidden=16, base="tanh", beta=20.0, trainable_partition=False, seed=0,
    )
    _train_partitioned(field)
    part_l2, part_iface = _errors(field)
    print(f"    PartitionedField (2 regions): L2={part_l2:.4f}  interface |u(0)|={part_iface:.4f}")

    xg = torch.linspace(-1.0, 1.0, 201, dtype=torch.float64).reshape(-1, 1).numpy()
    params = PartitionParams(
        PartitionConfig(n_features=1, depth=1, split_kind="axis"),
        field.split_W.detach().numpy(), field.split_t.detach().numpy(),
    )
    cert = certify_partition_gap(params, xg, beta=20.0)
    print(f"    certified partition gap: max L1 <= {cert.max_gap:.4f} "
          f"(measured {cert.measured_max:.4f}, sound={cert.is_sound})")

    assert part_iface < 0.25 * plain_iface + 1e-3, (part_iface, plain_iface)
    assert part_l2 < plain_l2
    assert cert.is_sound
    print("\nPartitioned PINN captures the interface; all checks passed.")


if __name__ == "__main__":
    main()
