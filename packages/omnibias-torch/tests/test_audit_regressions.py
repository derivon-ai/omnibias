# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for fixes from the enterprise audit (T1.b).

Locks down two contracts:

1. **Default-dtype propagation**: when the user has set
   ``torch.set_default_dtype(torch.float64)`` (the standard pattern for
   scientific work), the OMBU / OperatorBlock / cmbLinear / cmbConv1d /
   cmbConv2d / GrowableOMBU init helpers must produce float64
   parameters by default. Prior to the fix the init helpers in
   ``omnibias.torch.identity_init`` and ``omnibias.torch.stencil``
   defaulted to ``torch.float32``, silently demoting biases / signs to
   float32 even in float64-default sessions and breaking dtype
   homogeneity inside the model.

2. **Negative-order validation**: every public ``ActivationSpec.fastpath``
   kernel must reject ``order < 0`` with ``ValueError`` rather than
   falling through to ``NotImplementedError`` with a misleading
   "got -1" message.

Both regressions are bit-stable and fast; they run on CPU in float64.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.torch import (  # noqa: E402
    OMBU,
    GrowableOMBU,
    OperatorBlock,
    cmbConv1d,
    cmbConv2d,
    cmbLinear,
)
from omnibias.torch.activations.registry import get_activation, list_activations  # noqa: E402
from omnibias.torch.identity_init import identity_init_biases, identity_init_signs  # noqa: E402
from omnibias.torch.stencil import (  # noqa: E402
    central_bias_offsets,
    forward_bias_offsets,
    forward_difference_signs,
    identity_signs,
    stencil_offsets,
    stencil_signs,
)

# ---------------------------------------------------------------------------
# 1. Default-dtype propagation.
# ---------------------------------------------------------------------------


@pytest.fixture
def torch_default_float64():
    """Set default dtype to float64 for the duration of the test."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def test_identity_init_biases_follows_default_dtype(torch_default_float64) -> None:
    biases = identity_init_biases(num_channels=4, K=2)
    assert biases.dtype == torch.float64


def test_identity_init_signs_follows_default_dtype(torch_default_float64) -> None:
    signs = identity_init_signs(num_channels=4, K=2)
    assert signs.dtype == torch.float64


def test_identity_signs_follows_default_dtype(torch_default_float64) -> None:
    s = identity_signs(K=3)
    assert s.dtype == torch.float64


@pytest.mark.parametrize("K", [1, 2, 3, 5])
def test_forward_difference_signs_follows_default_dtype(
    torch_default_float64, K
) -> None:
    s = forward_difference_signs(K, delta=0.1)
    assert s.dtype == torch.float64


def test_forward_bias_offsets_follows_default_dtype(torch_default_float64) -> None:
    b = forward_bias_offsets(K=3, delta=0.5)
    assert b.dtype == torch.float64


def test_central_bias_offsets_follows_default_dtype(torch_default_float64) -> None:
    b = central_bias_offsets(K=3, delta=0.5)
    assert b.dtype == torch.float64


def test_stencil_offsets_follows_default_dtype(torch_default_float64) -> None:
    for stencil in ("forward", "central"):
        b = stencil_offsets(K=3, delta=0.5, stencil=stencil)
        assert b.dtype == torch.float64, stencil


def test_stencil_signs_follows_default_dtype(torch_default_float64) -> None:
    for stencil in ("forward", "central"):
        s = stencil_signs(K=3, delta=0.5, stencil=stencil)
        assert s.dtype == torch.float64, stencil


def test_ombu_uses_default_dtype(torch_default_float64) -> None:
    ombu = OMBU(num_channels=4, K=2, base="tanh")
    assert ombu.biases.dtype == torch.float64
    assert ombu.signs.dtype == torch.float64
    out = ombu(torch.zeros(8, 4))
    assert out.dtype == torch.float64


@pytest.mark.parametrize(
    "op,base",
    [
        ("identity", "tanh"),
        ("grad", "tanh"),
        ("laplacian", "gaussian"),
        ("derivative", "sigmoid"),
        ("band", "gaussian"),
        ("integral", "tanh"),
    ],
)
def test_operator_block_uses_default_dtype(torch_default_float64, op, base) -> None:
    kwargs = {}
    if op == "derivative":
        kwargs["derivative_order"] = 2
    block = OperatorBlock(op=op, base=base, channels=4, **kwargs)
    assert block.ombu.biases.dtype == torch.float64
    assert block.ombu.signs.dtype == torch.float64


def test_cmblinear_uses_default_dtype(torch_default_float64) -> None:
    fc = cmbLinear(in_features=8, out_features=4, op="grad", base="tanh")
    assert fc.linear.weight.dtype == torch.float64
    assert fc.block.ombu.biases.dtype == torch.float64


def test_cmbconv1d_uses_default_dtype(torch_default_float64) -> None:
    conv = cmbConv1d(
        in_channels=2, out_channels=4, kernel_size=3, op="grad", base="tanh"
    )
    assert conv.conv.weight.dtype == torch.float64
    assert conv.block.ombu.biases.dtype == torch.float64


def test_cmbconv2d_uses_default_dtype(torch_default_float64) -> None:
    conv = cmbConv2d(
        in_channels=2,
        out_channels=4,
        kernel_size=3,
        op="laplacian",
        base="gaussian",
    )
    assert conv.conv.weight.dtype == torch.float64
    assert conv.block.ombu.biases.dtype == torch.float64


def test_growable_ombu_uses_default_dtype(torch_default_float64) -> None:
    g = GrowableOMBU(num_channels=4, init_K=1, K_max=4, base="sigmoid")
    assert g.biases.dtype == torch.float64
    assert g.signs.dtype == torch.float64
    out = g(torch.zeros(8, 4))
    assert out.dtype == torch.float64


def test_dtype_default_passthrough_does_not_break_float32(monkeypatch) -> None:
    """When default dtype is float32 (the global default), helpers still
    produce float32 (no regression)."""
    assert torch.get_default_dtype() == torch.float32
    biases = identity_init_biases(num_channels=4, K=2)
    assert biases.dtype == torch.float32
    s = identity_signs(K=3)
    assert s.dtype == torch.float32


# ---------------------------------------------------------------------------
# 2. Negative-order validation.
# ---------------------------------------------------------------------------


_FASTPATH_ACTIVATIONS = [
    name for name in list_activations() if get_activation(name).fastpath is not None
]


@pytest.mark.parametrize("name", _FASTPATH_ACTIVATIONS)
def test_fastpath_rejects_negative_order(name: str) -> None:
    """Every spec.fastpath must raise ValueError on n < 0, not
    NotImplementedError with a misleading 'only implements n in
    {0, 1, 2}, got -1' message."""
    spec = get_activation(name)
    fp = spec.fastpath
    assert fp is not None
    with pytest.raises(ValueError, match="order n must be"):
        fp(torch.zeros(1), -1)
    with pytest.raises(ValueError, match="order n must be"):
        fp(torch.zeros(1), -7)
