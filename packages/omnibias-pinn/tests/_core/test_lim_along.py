# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Backend-neutral tests for the ``lim_along`` registry extension.

The adaptor reads a ``{component: callable}`` mapping from ``state.extra`` and
calls the closure; here we drive it with a lightweight stub state so no torch /
jax backend is required (the end-to-end ``state.<comp>.lim_along`` path is
exercised against a real jax field in ``tests/jax/test_jax_lim_along.py``).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from omnibias.fields import ops_registry
from omnibias.pinn.extensions import (
    LIM_ALONG_KEY,
    register_lim_along,
    unregister_lim_along,
)
from omnibias.pinn.extensions.lim_along import _lim_along_op


def setup_function(function):
    ops_registry.clear()


def teardown_function(function):
    ops_registry.clear()


def _stub(extra):
    return SimpleNamespace(extra=extra)


def test_register_and_unregister() -> None:
    name = register_lim_along()
    assert name == LIM_ALONG_KEY
    assert LIM_ALONG_KEY in ops_registry.list_registered()
    assert ops_registry.lookup(LIM_ALONG_KEY) is _lim_along_op
    unregister_lim_along()
    assert ops_registry.lookup(LIM_ALONG_KEY) is None


def test_double_register_requires_overwrite() -> None:
    register_lim_along()
    with pytest.raises(ValueError, match="already registered"):
        register_lim_along()
    # overwrite=True replaces the existing registration cleanly
    assert register_lim_along(overwrite=True) == LIM_ALONG_KEY


def test_adaptor_calls_component_closure() -> None:
    state = _stub({LIM_ALONG_KEY: {"u": lambda: 42.0, "v": lambda: -1.5}})
    assert _lim_along_op(state, "u") == 42.0
    assert _lim_along_op(state, "v") == -1.5


def test_adaptor_missing_mapping_raises() -> None:
    with pytest.raises(KeyError, match="to be a"):
        _lim_along_op(_stub({}), "u")


def test_adaptor_missing_component_raises() -> None:
    state = _stub({LIM_ALONG_KEY: {"u": lambda: 1.0}})
    with pytest.raises(KeyError, match="no lim_along closure"):
        _lim_along_op(state, "missing")


def test_adaptor_noncallable_raises() -> None:
    state = _stub({LIM_ALONG_KEY: {"u": 3.0}})
    with pytest.raises(TypeError, match="must be callable"):
        _lim_along_op(state, "u")


def test_custom_name_binding() -> None:
    register_lim_along(name="jet_limit")
    assert ops_registry.lookup("jet_limit") is _lim_along_op
    assert ops_registry.lookup(LIM_ALONG_KEY) is None
