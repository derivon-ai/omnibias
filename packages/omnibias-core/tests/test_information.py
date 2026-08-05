# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-core information-theory primitives + exponential-family metadata."""

from __future__ import annotations

import math

import pytest
from omnibias.core.information import (
    binary_entropy,
    has_cumulant_tower,
    is_log_partition_activation,
)
from omnibias.core.spec import ActivationSpec


def _spec(name: str, *, noise_model: str = "none", fastpath: object = None) -> ActivationSpec[float]:
    return ActivationSpec(
        name=name,
        forward=lambda z: z,
        noise_model=noise_model,
        fastpath=fastpath,  # type: ignore[arg-type]
    )


def test_binary_entropy_half_is_ln2() -> None:
    assert binary_entropy(0.5) == pytest.approx(math.log(2.0))


def test_binary_entropy_in_bits_is_one() -> None:
    assert binary_entropy(0.5, base=2.0) == pytest.approx(1.0)


def test_binary_entropy_endpoints_are_zero() -> None:
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0


def test_binary_entropy_is_symmetric() -> None:
    assert binary_entropy(0.2) == pytest.approx(binary_entropy(0.8))


def test_binary_entropy_is_maximal_at_half() -> None:
    assert binary_entropy(0.5) > binary_entropy(0.3) > binary_entropy(0.05)


@pytest.mark.parametrize("p", [-0.01, 1.01, 2.0])
def test_binary_entropy_rejects_out_of_range(p: float) -> None:
    with pytest.raises(ValueError, match="p must be in"):
        binary_entropy(p)


def test_binary_entropy_rejects_bad_base() -> None:
    with pytest.raises(ValueError, match="base must be > 1"):
        binary_entropy(0.5, base=1.0)


def test_is_log_partition_activation() -> None:
    assert is_log_partition_activation(_spec("softplus", noise_model="bernoulli"))
    assert not is_log_partition_activation(_spec("relu", noise_model="none"))
    assert not is_log_partition_activation(_spec("x", noise_model=""))


def test_has_cumulant_tower() -> None:
    assert has_cumulant_tower(_spec("a", fastpath=lambda z, n: z))
    assert not has_cumulant_tower(_spec("b", fastpath=None))
