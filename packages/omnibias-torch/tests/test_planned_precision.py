# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The precision policy in ``compile_planned_operator`` and the Torch ``compile_residual``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from omnibias.core.planner import PlanCache  # noqa: E402
from omnibias.core.precision import PrecisionPolicy  # noqa: E402
from omnibias.torch.experimental import (  # noqa: E402
    Residual,
    compile_planned_operator,
    compile_residual,
)
from omnibias.torch.experimental.planned import matmul_precision  # noqa: E402
from torch import nn  # noqa: E402

D, H = 6, 8


def _net(dtype=torch.float32):
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(D, H), nn.Tanh(), nn.Linear(H, H), nn.Tanh(), nn.Linear(H, 1)
    ).to(dtype)


def _x(dtype=torch.float32, n=5):
    torch.manual_seed(3)
    return torch.randn(n, D, dtype=dtype)


def _mode():
    """The process's float32 matrix-product setting, as ``"ieee"``/``"tf32"`` (whichever API this PyTorch has)."""
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        return str(matmul.fp32_precision)
    return "ieee" if torch.get_float32_matmul_precision() == "highest" else "tf32"


def _set_mode(value):
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        matmul.fp32_precision = value
    else:
        torch.set_float32_matmul_precision("highest" if value == "ieee" else "high")


def test_the_policy_is_applied_inside_the_call_and_the_process_setting_is_restored():
    seen = {}

    class Probe(nn.Module):
        def forward(self, x):
            seen["during"] = _mode()
            return x.sum(-1)

    before = _mode()
    _set_mode("tf32")
    try:
        with matmul_precision(PrecisionPolicy("highest")):
            assert _mode() == "ieee"
        assert _mode() == "tf32"
        with matmul_precision(PrecisionPolicy("default")):
            assert _mode() == "tf32"
        _set_mode("ieee")
        with matmul_precision(PrecisionPolicy("tf32")):
            assert _mode() == "tf32"
        assert _mode() == "ieee"
        _set_mode("tf32")
        op = compile_planned_operator(Probe(), _x(), operator="laplacian", cache=PlanCache())
        with torch.no_grad():
            op(_x())
        assert seen["during"] == "ieee"  # the default policy turns TF32 off for the call
        assert _mode() == "tf32"  # and the process setting is back afterwards
    finally:
        _set_mode(before if before in ("ieee", "tf32") else "ieee")


def test_the_default_policy_is_reported_and_tf32_is_opt_in():
    net, x = _net(), _x()
    op = compile_planned_operator(net, x, cache=PlanCache())
    assert op.explain()["precision"] == {
        "matmul": "highest",
        "allow_low_precision": False,
        "notes": [],
    }
    tf = compile_planned_operator(net, x, cache=PlanCache(), precision="tf32")
    entry = tf.explain()["precision"]
    assert entry["matmul"] == "tf32" and any("TF32" in n for n in entry["notes"])
    torch.testing.assert_close(op(x), tf(x), rtol=1e-3, atol=1e-4)


def test_half_precision_is_refused_for_second_order_operators_unless_allowed():
    net, x = _net(torch.bfloat16), _x(torch.bfloat16)
    with pytest.raises(ValueError, match="refused for derivative order 2"):
        compile_planned_operator(net, x, cache=PlanCache())
    op = compile_planned_operator(net, x, cache=PlanCache(), allow_low_precision=True)
    assert any("not bounded" in n for n in op.explain()["precision"]["notes"])


def test_residual_programs_take_the_order_from_the_demand():
    net, x = _net(), _x()
    blocks = [(x, Residual(lambda q: q.lap + q.u, ("grad", "lap")))]
    op = compile_residual(net, blocks, cache=PlanCache(), precision="tf32")
    assert op.explain()["precision"]["matmul"] == "tf32"
    net16, x16 = _net(torch.bfloat16), _x(torch.bfloat16)
    with pytest.raises(ValueError, match="refused for derivative order 2"):
        compile_residual(
            net16,
            [(x16, Residual(lambda q: q.lap + q.u, ("grad", "lap")))],
            cache=PlanCache(),
        )
    first_order = compile_residual(
        net16,
        [(x16, Residual(lambda q: q.grad.sum(-1), ("grad",)))],
        quantity="ntk_diagonal",
        cache=PlanCache(),
    )
    assert any("order 1" in n for n in first_order.explain()["precision"]["notes"])
