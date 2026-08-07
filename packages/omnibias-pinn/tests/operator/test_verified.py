# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Operator residual enclosure: soundness against a dense grid + random sample."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omnibias.core.multi_index import index_position, multi_index_factorial
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.operator import (
    branch_coefficient_box,
    certify_heat_residual,
    enclose_heat_residual,
)
from omnibias.pinn.operator.torch import build_deeponet
from omnibias.pinn.torch import ops as tops
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet_mv import mlp_jet_mv

DTYPE = torch.float64
DIFFUSIVITY = 0.1


def _tanh_trunk(*, p: int = 2, hidden: int = 3):
    """Nonlinear trunk ``affine -> tanh -> affine`` of width ``p``."""
    rng = np.random.default_rng(11)
    W1 = rng.normal(scale=0.4, size=(hidden, 2)).tolist()
    b1 = rng.normal(scale=0.1, size=(hidden,)).tolist()
    W2 = rng.normal(scale=0.4, size=(p, hidden)).tolist()
    b2 = rng.normal(scale=0.1, size=(p,)).tolist()
    return [(W1, b1, "tanh"), (W2, b2, None)], W1, b1, W2, b2


def _true_heat_residual(
    x: float,
    t: float,
    W1: list,
    b1: list,
    W2: list,
    b2: list,
    coeffs: list[float],
    diffusivity: float,
) -> float:
    """Closed-form residual via the torch jet kernel at a single query point."""
    spec = get_activation("tanh")
    layers = [
        (
            torch.tensor(W1, dtype=DTYPE),
            torch.tensor(b1, dtype=DTYPE),
            spec,
        ),
        (
            torch.tensor(W2, dtype=DTYPE),
            torch.tensor(b2, dtype=DTYPE),
            None,
        ),
    ]
    xi = torch.tensor([x, t], dtype=DTYPE)
    jet = mlp_jet_mv(xi, layers, 2)  # (M, p)
    pos = index_position(2, 2)
    c = torch.tensor(coeffs, dtype=DTYPE)
    scale_t = float(multi_index_factorial((0, 1)))
    scale_xx = float(multi_index_factorial((2, 0)))
    u_t = float(torch.dot(c, jet[pos[(0, 1)]] * scale_t))
    u_xx = float(torch.dot(c, jet[pos[(2, 0)]] * scale_xx))
    return u_t - float(diffusivity) * u_xx


def _branch_true(sensors: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    h = np.tanh(W1 @ sensors + b1)
    return W2 @ h + b2


def test_enclose_heat_residual_nonlinear_trunk_contains_grid_and_sample() -> None:
    layers, W1, b1, W2, b2 = _tanh_trunk(p=2, hidden=3)
    coeffs = [0.5, -0.25]
    box = [(-0.15, 0.15), (0.0, 0.2)]
    enclosure = enclose_heat_residual(
        trunk_layers=layers,
        coeffs=coeffs,
        query_box=box,
        diffusivity=DIFFUSIVITY,
        order=2,
    )
    assert enclosure.hi > enclosure.lo, "nonlinear trunk must yield a nontrivial box"
    xs = np.linspace(box[0][0], box[0][1], 7)
    ts = np.linspace(box[1][0], box[1][1], 7)
    rng = np.random.default_rng(0)
    samples = rng.uniform(
        [box[0][0], box[1][0]], [box[0][1], box[1][1]], size=(25, 2)
    )
    points = [(float(x), float(t)) for x in xs for t in ts] + [
        (float(s[0]), float(s[1])) for s in samples
    ]
    for x, t in points:
        residual = _true_heat_residual(
            x, t, W1, b1, W2, b2, coeffs, DIFFUSIVITY
        )
        assert enclosure.lo <= residual <= enclosure.hi, (
            x,
            t,
            residual,
            enclosure,
        )


def test_enclose_heat_residual_tightens_with_smaller_box() -> None:
    layers, *_ = _tanh_trunk()
    coeffs = [0.4, -0.3]
    wide = enclose_heat_residual(
        trunk_layers=layers,
        coeffs=coeffs,
        query_box=[(-0.3, 0.3), (0.0, 0.4)],
        diffusivity=DIFFUSIVITY,
    )
    narrow = enclose_heat_residual(
        trunk_layers=layers,
        coeffs=coeffs,
        query_box=[(-0.05, 0.05), (0.1, 0.15)],
        diffusivity=DIFFUSIVITY,
    )
    assert (narrow.hi - narrow.lo) < (wide.hi - wide.lo)


def test_certify_heat_residual_seals_and_is_honest() -> None:
    layers, *_ = _tanh_trunk()
    cert = certify_heat_residual(
        trunk_layers=layers,
        coeffs=[0.5, -0.25],
        query_box=[(-0.1, 0.1), (0.0, 0.2)],
        diffusivity=DIFFUSIVITY,
    )
    assert verify_certificate_digest(cert)
    assert cert["honesty"]["residual_enclosure_not_solution_error"] is True
    assert cert["meta"]["kind"] == "operator_heat_residual"


def test_branch_coefficient_box_contains_true_outputs() -> None:
    rng = np.random.default_rng(3)
    m, h, p = 4, 5, 3
    W1 = rng.normal(scale=0.3, size=(h, m))
    b1 = rng.normal(scale=0.05, size=(h,))
    W2 = rng.normal(scale=0.3, size=(p, h))
    b2 = rng.normal(scale=0.05, size=(p,))
    sensors_box = [(-0.2, 0.2)] * m
    coeff_box, trailing = branch_coefficient_box(
        sensors_box,
        [(W1.tolist(), b1.tolist(), "tanh"), (W2.tolist(), b2.tolist(), None)],
    )
    assert trailing is None
    assert len(coeff_box) == p
    # Dense corners of the sensor hypercube (2^m is small) + random samples.
    corners = np.array(np.meshgrid(*[[-0.2, 0.2]] * m)).T.reshape(-1, m)
    samples = rng.uniform(-0.2, 0.2, size=(30, m))
    for s in np.vstack([corners, samples]):
        true = _branch_true(s, W1, b1, W2, b2)
        for k in range(p):
            assert coeff_box[k].lo <= true[k] <= coeff_box[k].hi, (
                k,
                true[k],
                coeff_box[k],
            )


def test_branch_coefficient_box_rejects_unsupported_activation() -> None:
    with pytest.raises(ValueError, match="supports tanh/sigmoid/affine"):
        branch_coefficient_box(
            [(-0.1, 0.1), (-0.1, 0.1)],
            [([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], "gelu")],
        )


def test_family_certificate_contains_real_deeponet_residuals() -> None:
    """Sensor box -> coefficient box -> residual enclosure vs a live DeepONet."""
    torch.manual_seed(0)
    op = build_deeponet(
        coordinate_spec=CoordinateSpec(("x", "t")),
        components=ComponentSpec(("u",)),
        n_sensors=4,
        trunk_width=3,
        trunk_hidden=4,
        trunk_depth=1,
        branch_hidden=4,
        branch_depth=1,
        jet_order=2,
        per_sample_bias=False,
        dtype=DTYPE,
    )
    # Extract branch layers for interval propagation.
    branch_linears = list(op.branch.linears)
    assert len(branch_linears) == 2
    W1 = branch_linears[0].weight.detach().cpu().numpy()
    b1 = branch_linears[0].bias.detach().cpu().numpy()
    W2 = branch_linears[1].weight.detach().cpu().numpy()
    b2 = branch_linears[1].bias.detach().cpu().numpy()
    sensors_box = [(-0.15, 0.15)] * 4
    coeff_box, _ = branch_coefficient_box(
        sensors_box,
        [(W1.tolist(), b1.tolist(), "tanh"), (W2.tolist(), b2.tolist(), None)],
    )
    # Trunk layers: depth-1 JetMLP is hidden affine+tanh then affine readout.
    trunk_linears = list(op.core.trunk.linears)
    assert len(trunk_linears) == 2
    tw1 = trunk_linears[0].weight.detach().cpu().numpy()
    tb1 = trunk_linears[0].bias.detach().cpu().numpy()
    tw2 = trunk_linears[1].weight.detach().cpu().numpy()
    tb2 = trunk_linears[1].bias.detach().cpu().numpy()
    trunk_layers = [
        (tw1.tolist(), tb1.tolist(), "tanh"),
        (tw2.tolist(), tb2.tolist(), None),
    ]
    query_box = [(-0.1, 0.1), (0.0, 0.15)]
    cert = certify_heat_residual(
        trunk_layers=trunk_layers,
        coeffs=coeff_box,
        query_box=query_box,
        diffusivity=DIFFUSIVITY,
    )
    assert verify_certificate_digest(cert)
    enclosure = enclose_heat_residual(
        trunk_layers=trunk_layers,
        coeffs=coeff_box,
        query_box=query_box,
        diffusivity=DIFFUSIVITY,
    )
    rng = np.random.default_rng(5)
    for _ in range(12):
        sensors = torch.tensor(
            rng.uniform(-0.15, 0.15, size=(1, 4)), dtype=DTYPE
        )
        field = op.condition(sensors)
        qx = float(rng.uniform(query_box[0][0], query_box[0][1]))
        qt = float(rng.uniform(query_box[1][0], query_box[1][1]))
        coords = torch.tensor([[qx, qt]], dtype=DTYPE)
        state = field(coords)
        u_t = float(tops.derivative(state, "u", axis=1, order=1)[0].detach())
        u_xx = float(tops.derivative(state, "u", axis=0, order=2)[0].detach())
        residual = u_t - DIFFUSIVITY * u_xx
        assert enclosure.lo <= residual <= enclosure.hi, (
            residual,
            enclosure,
            qx,
            qt,
        )
