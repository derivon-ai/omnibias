# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the third-party ops extension registry."""

from __future__ import annotations

import pytest
from omnibias.pinn._core import ops_registry


def setup_function(function):
    ops_registry.clear()


def teardown_function(function):
    ops_registry.clear()


def test_register_and_lookup():
    @ops_registry.register("symmetric_laplacian")
    def sym_lap(state, name):
        return f"sym_lap({name})"

    fn = ops_registry.lookup("symmetric_laplacian")
    assert fn is not None
    assert fn(None, "u") == "sym_lap(u)"
    assert "symmetric_laplacian" in ops_registry.list_registered()


def test_register_invalid_names():
    with pytest.raises(ValueError):
        ops_registry.register("")(lambda *a: None)
    with pytest.raises(ValueError):
        ops_registry.register("Has Space")(lambda *a: None)
    with pytest.raises(ValueError):
        ops_registry.register("UPPER")(lambda *a: None)


def test_register_duplicate_raises():
    @ops_registry.register("foo")
    def f1(state, name):
        return 1

    with pytest.raises(ValueError):
        @ops_registry.register("foo")
        def f2(state, name):
            return 2


def test_unregister_and_clear():
    @ops_registry.register("ephemeral")
    def f(state, name):
        return None

    ops_registry.unregister("ephemeral")
    assert ops_registry.lookup("ephemeral") is None
    # Re-register should now succeed.
    ops_registry.register("ephemeral")(f)
    ops_registry.clear()
    assert ops_registry.list_registered() == ()
