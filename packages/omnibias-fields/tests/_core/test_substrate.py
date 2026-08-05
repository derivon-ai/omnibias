# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Phase 1 substrate tests for omnibias-fields (pure Python; no torch / jax).

Covers the extraction invariants:

- the public import surface is present,
- the dispatch marker name is stable,
- :class:`SigmaCache` evaluates each order exactly once and reuses it,
- the op-extension registry round-trips,
- the omnibias-pinn back-compat shims point at the very same objects.
"""

from __future__ import annotations

import omnibias.fields as fields
import pytest
from omnibias.fields._core import (
    DISPATCH_ATTR,
    ComponentSpec,
    CoordinateSpec,
    FieldState,
    SigmaCache,
    ops_registry,
)


def test_public_surface() -> None:
    assert fields.__version__ == "0.1.0"
    for name in ("FieldState", "ComponentSpec", "CoordinateSpec",
                 "ComponentView", "VectorView", "SigmaCache", "ops_registry"):
        assert hasattr(fields, name)


def test_dispatch_attr_name_is_stable() -> None:
    assert DISPATCH_ATTR == "_omnibias_dispatch"


def test_sigma_cache_single_evaluation() -> None:
    calls: list[int] = []

    def build(n: int) -> str:
        calls.append(n)
        return f"sigma^{n}"

    cache: SigmaCache[str] = SigmaCache(z="z")
    assert cache.get_or_compute(2, build) == "sigma^2"
    assert cache.get_or_compute(2, build) == "sigma^2"
    assert cache.get_or_compute(0, build) == "sigma^0"
    # Order 2 evaluated once despite two accesses.
    assert calls == [2, 0]
    assert cache.orders() == (0, 2)


def test_sigma_cache_rejects_negative_order() -> None:
    cache: SigmaCache[str] = SigmaCache(z="z")
    with pytest.raises(ValueError):
        cache.get_or_compute(-1, lambda n: "x")


def test_ops_registry_roundtrip() -> None:
    @ops_registry.register("phase1_probe_op")
    def _probe(state: object, name: str) -> str:
        return f"probe::{name}"

    assert ops_registry.lookup("phase1_probe_op") is _probe
    assert ops_registry.lookup("definitely_not_registered") is None


def test_pinn_shims_are_the_same_objects() -> None:
    # omnibias-pinn re-exports the moved substrate via transparent shims.
    from omnibias.pinn._core import FieldState as PinnFieldState
    from omnibias.pinn._core.ops_registry import register as pinn_register

    assert PinnFieldState is FieldState
    assert pinn_register is ops_registry.register


def test_specs_construct() -> None:
    coords = CoordinateSpec(axes=("t", "x"))
    comps = ComponentSpec(names=("u",))
    assert coords.ndim == 2
    assert comps.is_component("u")
    assert coords.axis_index("x") == 1
