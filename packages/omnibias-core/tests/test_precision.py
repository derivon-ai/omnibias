# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The precision policy: defaults, the half-precision refusal and derivative orders."""

from __future__ import annotations

import pytest
from omnibias.core.precision import (
    MATMUL_MODES,
    PrecisionPolicy,
    request_order,
    resolve_precision,
)


def test_the_default_is_full_float32_products_and_no_low_precision():
    policy = resolve_precision(None)
    assert policy.matmul == "highest" and policy.allow_low_precision is False
    assert policy.jax_name == "highest"
    assert policy.check("float32", 2) == []
    assert policy.check("float64", 4) == []


def test_tf32_is_opt_in_and_is_reported():
    policy = resolve_precision("tf32")
    assert policy.jax_name == "tensorfloat32"
    notes = policy.check("float32", 2)
    assert len(notes) == 1 and "TF32" in notes[0]
    assert resolve_precision("default").jax_name is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "torch.bfloat16", "half"])
def test_half_precision_is_refused_from_order_two_unless_allowed(dtype):
    policy = PrecisionPolicy()
    with pytest.raises(ValueError, match="refused for derivative order 2"):
        policy.check(dtype, 2)
    with pytest.raises(ValueError, match="allow_low_precision"):
        policy.check(dtype, 4, what="a biharmonic residual")
    allowed = resolve_precision(None, allow_low_precision=True)
    notes = allowed.check(dtype, 2)
    assert notes and "not bounded" in notes[0]


def test_first_order_quantities_admit_half_precision_with_a_note():
    notes = PrecisionPolicy().check("bfloat16", 1)
    assert notes and "derivative order 1" in notes[0]


def test_orders_follow_the_request_kind_and_a_residual_demand():
    assert request_order("laplacian") == 2
    assert request_order("laplacian", power=3) == 6
    assert request_order("divergence") == 1 and request_order("logdet") == 1
    assert request_order("gauss_newton", demand_order=4) == 4
    assert request_order("gram", demand_order=0) == 0


def test_unknown_modes_are_rejected_and_policies_pass_through():
    with pytest.raises(ValueError, match="one of"):
        resolve_precision("fp8")
    base = PrecisionPolicy("tf32")
    assert resolve_precision(base) is base
    widened = resolve_precision(base, allow_low_precision=True)
    assert widened.matmul == "tf32" and widened.allow_low_precision
    assert set(MATMUL_MODES) == {"highest", "tf32", "default"}
    assert base.to_config() == {"matmul": "tf32", "allow_low_precision": False}
