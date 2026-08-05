# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The discontinuity-capturing PartitionedField bridge (omnibias.pinn.partition)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omnibias.partition")  # the optional 'partition' extra (alpha keystone)

from omnibias.partition import certify_partition_gap  # noqa: E402
from omnibias.partition._core.config import PartitionConfig  # noqa: E402
from omnibias.partition._core.params import PartitionParams  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec  # noqa: E402
from omnibias.pinn.partition.torch import PartitionedField, build_partitioned_field  # noqa: E402
from omnibias.pinn.torch import ops  # noqa: E402
from omnibias.pinn.torch.fields import OneLayerVectorField  # noqa: E402


def _field(depth_dirs, depth_thresh, *, beta=8.0, trainable=True, seed=0):
    return build_partitioned_field(
        coordinate_spec=CoordinateSpec(("x",)),
        components=ComponentSpec(("u",)),
        split_dirs=torch.tensor(depth_dirs, dtype=torch.float64),
        split_thresh=torch.tensor(depth_thresh, dtype=torch.float64),
        hidden=8, base="tanh", beta=beta, trainable_partition=trainable, seed=seed,
    )


def test_partition_weights_are_a_partition_of_unity() -> None:
    field = _field([[1.0]], [0.0], beta=6.0)
    x = torch.linspace(-2.0, 2.0, 40, dtype=torch.float64).reshape(-1, 1)
    w = field.partition_weights(x)
    assert w.shape == (40, 2)
    assert bool((w >= 0.0).all())
    assert torch.allclose(w.sum(dim=1), torch.ones(40, dtype=torch.float64), atol=1e-12)


def test_forward_is_the_pou_blend_of_subfields() -> None:
    field = _field([[1.0]], [0.0], beta=5.0)
    x = torch.randn(13, 1, dtype=torch.float64)
    w = field.partition_weights(x)  # (n, 2)
    with torch.no_grad():
        subvals = torch.stack(
            [ops.value(sub(x), "u") for sub in field.subfields], dim=1
        )  # (n, 2)
        blend = (w * subvals).sum(dim=1)
        got = field.forward_values(x)[:, 0]
    assert torch.allclose(got, blend, atol=1e-12)


def test_derivative_ops_match_finite_difference() -> None:
    field = _field([[1.0]], [0.0], beta=4.0)
    x = torch.linspace(-1.0, 1.0, 9, dtype=torch.float64).reshape(-1, 1)
    ux = ops.derivative(field(x), "u", axis=0, order=1)
    # independent central finite difference on the blended forward
    h = 1e-5
    with torch.no_grad():
        fp = field.forward_values(x + h)[:, 0]
        fm = field.forward_values(x - h)[:, 0]
        fd = (fp - fm) / (2 * h)
    assert torch.allclose(ux, fd, atol=1e-6)


def test_second_derivative_and_laplacian_agree() -> None:
    field = _field([[1.0]], [0.0], beta=4.0)
    x = torch.linspace(-1.0, 1.0, 7, dtype=torch.float64).reshape(-1, 1)
    uxx = ops.derivative(field(x), "u", axis=0, order=2)
    lap = ops.laplacian(field(x), "u")  # 1D spatial laplacian == d2/dx2
    assert torch.allclose(uxx, lap, atol=1e-10)


def test_gradient_op_routes_through_partitioned_branch() -> None:
    field = _field([[1.0]], [0.0])
    x = torch.randn(5, 1, dtype=torch.float64)
    g = ops.gradient(field(x), "u", axes=("x",))
    assert g.shape == (5, 1)
    assert bool(torch.isfinite(g).all())


def test_depth2_has_four_regions() -> None:
    # two gates on the same 1D axis (x < -0.5, x < 0.5) -> 4 (some empty) regions
    field = _field([[1.0], [1.0]], [-0.5, 0.5])
    assert field.n_regions == 4 and len(field.subfields) == 4
    x = torch.linspace(-2.0, 2.0, 6, dtype=torch.float64).reshape(-1, 1)
    w = field.partition_weights(x)
    assert w.shape == (6, 4)
    assert torch.allclose(w.sum(dim=1), torch.ones(6, dtype=torch.float64), atol=1e-12)


def test_wrong_subfield_count_raises() -> None:
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    sub = OneLayerVectorField(coordinate_spec=cs, components=comp, hidden=4, dtype=torch.float64)
    with pytest.raises(ValueError, match="subfields"):
        PartitionedField(
            coordinate_spec=cs, components=comp, subfields=[sub],  # depth-1 needs 2
            split_dirs=torch.tensor([[1.0]]), split_thresh=torch.tensor([0.0]),
        )


def _residual(field: object, x: torch.Tensor) -> torch.Tensor:
    return (ops.derivative(field(x), "u", axis=0, order=2) - 2.0).pow(2).mean()


def _val(field: object, x: torch.Tensor) -> torch.Tensor:
    return ops.value(field(x), "u")


def test_partitioned_captures_kink_better_than_plain() -> None:
    r"""u''=2, u(+/-1)=0, u(0)=0 -> exact x^2-|x| (kink). Partition beats a single smooth net."""
    cs, comp = CoordinateSpec(("x",)), ComponentSpec(("u",))
    x0 = torch.zeros(1, 1, dtype=torch.float64)

    torch.manual_seed(0)
    plain = OneLayerVectorField(
        coordinate_spec=cs, components=comp, hidden=24, base="tanh", dtype=torch.float64
    )
    x_int = torch.linspace(-1.0, 1.0, 61, dtype=torch.float64).reshape(-1, 1)
    x_bc = torch.tensor([[-1.0], [1.0], [0.0]], dtype=torch.float64)
    opt = torch.optim.Adam(plain.parameters(), lr=6e-3)
    for _ in range(350):
        opt.zero_grad()
        (_residual(plain, x_int) + 10.0 * _val(plain, x_bc).pow(2).mean()).backward()
        opt.step()
    plain_iface = float(_val(plain, x0).abs().item())

    torch.manual_seed(0)
    field = build_partitioned_field(
        coordinate_spec=cs, components=comp,
        split_dirs=torch.tensor([[1.0]]), split_thresh=torch.tensor([0.0]),
        hidden=12, base="tanh", beta=20.0, trainable_partition=False, seed=0,
    )
    xL = torch.linspace(-1.0, 0.0, 31, dtype=torch.float64).reshape(-1, 1)
    xR = torch.linspace(0.0, 1.0, 31, dtype=torch.float64).reshape(-1, 1)
    subL, subR = field.subfields[0], field.subfields[1]
    xm1 = torch.tensor([[-1.0]], dtype=torch.float64)
    xp1 = torch.tensor([[1.0]], dtype=torch.float64)
    opt = torch.optim.Adam(field.parameters(), lr=6e-3)
    for _ in range(350):
        opt.zero_grad()
        pde = _residual(subL, xL) + _residual(subR, xR)
        bc = _val(subL, xm1).pow(2).mean() + _val(subR, xp1).pow(2).mean()
        uL0, uR0 = _val(subL, x0), _val(subR, x0)
        interface = (uL0.pow(2) + uR0.pow(2) + (uL0 - uR0).pow(2)).mean()
        (pde + 10.0 * bc + 10.0 * interface).backward()
        opt.step()
    part_iface = float(_val(field, x0).abs().item())

    assert part_iface < 0.25 * plain_iface + 1e-3, (part_iface, plain_iface)


def test_field_split_certificate_is_sound() -> None:
    field = _field([[1.0]], [0.0], beta=16.0, trainable=False)
    import numpy as np

    xg = np.linspace(-1.0, 1.0, 101).reshape(-1, 1)
    params = PartitionParams(
        PartitionConfig(n_features=1, depth=1, split_kind="axis"),
        field.split_W.detach().numpy(), field.split_t.detach().numpy(),
    )
    cert = certify_partition_gap(params, xg, beta=16.0)
    assert cert.is_sound
