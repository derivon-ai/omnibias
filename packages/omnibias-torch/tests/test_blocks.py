# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the block layer: :class:`OperatorBlock`, :class:`cmbLinear`,
:class:`cmbConv1d`, :class:`cmbConv2d`."""

from __future__ import annotations

import pytest
import torch
from omnibias.torch import ActivationSpec, OperatorBlock, cmbConv1d, cmbConv2d, cmbLinear


def test_identity_block_matches_base() -> None:
    block = OperatorBlock(op="identity", base="softplus", channels=4)
    z = torch.randn(3, 4)
    out = block(z)
    ref = torch.nn.functional.softplus(z)
    assert torch.allclose(out, ref, atol=1e-6)


def test_grad_block_matches_first_derivative() -> None:
    block = OperatorBlock(op="grad", base="sigmoid", channels=4)
    z = torch.randn(3, 4)
    out = block(z)
    s = torch.sigmoid(z)
    ref = s * (1.0 - s)
    assert torch.allclose(out, ref, atol=1e-6)


def test_laplacian_block_matches_second_derivative() -> None:
    block = OperatorBlock(op="laplacian", base="gaussian", channels=4)
    z = torch.randn(3, 4)
    out = block(z)
    ref = (z * z - 1.0) * torch.exp(-0.5 * z * z)
    assert torch.allclose(out, ref, atol=1e-6)


def test_derivative_block_matches_high_order_fastpath() -> None:
    block = OperatorBlock(op="derivative", derivative_order=4, base="gaussian", channels=4)
    z = torch.randn(3, 4)
    out = block(z)
    ref = block.ombu.spec.fastpath(z, 4)
    assert block.K == 5
    assert block.derivative_order == 4
    assert torch.allclose(out, ref, atol=1e-6)


def test_derivative_block_supports_k7_order6() -> None:
    block = OperatorBlock(op="derivative", derivative_order=6, base="tanh", channels=2)
    z = torch.randn(5, 2)
    out = block(z)
    ref = block.ombu.spec.fastpath(z, 6)
    assert block.K == 7
    assert block.derivative_order == 6
    assert torch.allclose(out, ref, atol=1e-6)


def test_integral_block_zero_delta_is_blank_window() -> None:
    """A zero-width strict integral window initializes as a blank feature."""
    block = OperatorBlock(op="integral", base="sigmoid", channels=3, init_delta=0.0)
    z = torch.randn(2, 3)
    out = block(z)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_band_block_default_delta_nonzero() -> None:
    block = OperatorBlock(op="band", base="sigmoid", channels=3)
    z = torch.randn(2, 3)
    out = block(z)
    expected = torch.sigmoid(z + 0.5) - torch.sigmoid(z - 0.5)
    assert torch.allclose(out, expected, atol=1e-6)


def test_integral_block_default_delta_nonzero() -> None:
    block = OperatorBlock(op="integral", base="sigmoid", channels=3)  # default init_delta=1.0
    z = torch.randn(2, 3)
    out = block(z)
    # output = S(z + b_hi) - S(z + b_lo), S' = sigmoid.
    expected = torch.nn.functional.softplus(z + 0.5) - torch.nn.functional.softplus(z - 0.5)
    assert torch.allclose(out, expected, atol=1e-6)


def test_integral_block_defaults_to_fixed_signs() -> None:
    block = OperatorBlock(op="integral", base="sigmoid", channels=3)
    assert not isinstance(block.ombu.signs, torch.nn.Parameter)


def test_integral_block_learnable_signs_is_expert_option() -> None:
    block = OperatorBlock(op="integral", base="sigmoid", channels=3, learnable_signs=True)
    assert isinstance(block.ombu.signs, torch.nn.Parameter)


def test_grad_block_rejects_no_fastpath() -> None:
    """``op='laplacian'`` with a base that has no fast-path kernel raises.

    (``relu`` now carries an all-orders a.e. tower, so we use a deliberately
    fastpath-less spec to exercise the guard.)
    """
    no_fastpath = ActivationSpec(name="_no_fastpath_probe", forward=torch.relu, fastpath=None)
    with pytest.raises(TypeError):
        OperatorBlock(op="laplacian", base=no_fastpath, channels=3)


def test_derivative_block_requires_order() -> None:
    with pytest.raises(ValueError):
        OperatorBlock(op="derivative", base="sigmoid", channels=3)


def test_derivative_order_only_valid_for_derivative_op() -> None:
    with pytest.raises(ValueError):
        OperatorBlock(op="grad", derivative_order=4, base="sigmoid", channels=3)


def test_derivative_block_rejects_unsupported_order() -> None:
    """A partial fast path that raises beyond its max order is rejected."""

    def _partial_fastpath(z: torch.Tensor, n: int) -> torch.Tensor:
        if n <= 1:
            return torch.ones_like(z)
        raise NotImplementedError(f"order {n} not supported")

    partial = ActivationSpec(
        name="_partial_fastpath_probe",
        forward=torch.relu,
        derivative=lambda z: torch.ones_like(z),
        fastpath=_partial_fastpath,
    )
    with pytest.raises(TypeError):
        OperatorBlock(op="derivative", derivative_order=2, base=partial, channels=3)


def test_integral_block_rejects_no_integral() -> None:
    with pytest.raises(TypeError):
        OperatorBlock(op="integral", base="softplus", channels=3)


def test_invalid_op_raises() -> None:
    with pytest.raises(ValueError):
        OperatorBlock(op="curl", base="sigmoid", channels=2)


def test_cmblinear_forward_shape() -> None:
    layer = cmbLinear(5, 3, op="identity", base="tanh")
    out = layer(torch.randn(2, 5))
    assert out.shape == (2, 3)


def test_cmblinear_grad_op_matches_chain() -> None:
    """``cmbLinear(op='grad')`` returns ``sigma'(W x + b_lin + b_omb)``."""
    layer = cmbLinear(5, 3, op="grad", base="sigmoid")
    x = torch.randn(2, 5)
    z_lin = layer.linear(x)  # (2, 3)
    out = layer(x)
    s = torch.sigmoid(z_lin)
    ref = s * (1.0 - s)
    assert torch.allclose(out, ref, atol=1e-6)


def test_cmblinear_derivative_op_accepts_order() -> None:
    layer = cmbLinear(
        5,
        3,
        op="derivative",
        base="gaussian",
        block_kwargs={"derivative_order": 4},
    )
    x = torch.randn(2, 5)
    z_lin = layer.linear(x)
    out = layer(x)
    ref = layer.block.ombu.spec.fastpath(z_lin, 4)
    assert layer.block.K == 5
    assert torch.allclose(out, ref, atol=1e-6)


def test_cmbconv1d_shape() -> None:
    layer = cmbConv1d(2, 4, kernel_size=3, padding=1, op="grad", base="tanh")
    out = layer(torch.randn(1, 2, 16))
    assert out.shape == (1, 4, 16)


def test_cmbconv2d_shape_and_chain_rule() -> None:
    layer = cmbConv2d(2, 4, kernel_size=3, padding=1, op="laplacian", base="gaussian")
    x = torch.randn(2, 2, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)
    # Independently apply Hermite-2 * exp to the conv output
    z = layer.conv(x)
    ref = (z * z - 1.0) * torch.exp(-0.5 * z * z)
    assert torch.allclose(out, ref, atol=1e-5)


def test_block_backward_through_grad() -> None:
    """Backprop works through a grad-block (closed-form derivative path)."""
    block = OperatorBlock(op="grad", base="sigmoid", channels=4)
    z = torch.randn(3, 4, requires_grad=True)
    out = block(z)
    out.pow(2).sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
