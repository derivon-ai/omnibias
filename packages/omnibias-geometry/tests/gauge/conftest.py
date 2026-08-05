# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared gauge test fixtures: backend adapters and sample points.

The analytic inputs (the 't Hooft symbol and the BPST instanton) live in
``_gauge_helpers`` so test modules can import them directly; a ``conftest`` is
auto-loaded per directory and must not be imported by name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


# ----------------------------------------------------------------------------
# backend adapter (torch / jax)
# ----------------------------------------------------------------------------
@dataclass
class Backend:
    name: str
    ops: Any
    asarray: Callable[[np.ndarray], Any]
    tonumpy: Callable[[Any], np.ndarray]


def _torch_backend() -> Backend:
    import torch

    torch.set_default_dtype(torch.float64)
    import omnibias.geometry.gauge.torch.ops as ops

    return Backend(
        name="torch",
        ops=ops,
        asarray=lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64),
        tonumpy=lambda t: np.asarray(t.detach().cpu().numpy()),
    )


def _jax_backend() -> Backend:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.geometry.gauge.jax.ops as ops

    return Backend(
        name="jax",
        ops=ops,
        asarray=lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64),
        tonumpy=lambda t: np.asarray(t),
    )


_BACKEND_FACTORIES = {"torch": _torch_backend, "jax": _jax_backend}


def _available_backends() -> list[str]:
    names = []
    for name in ("torch", "jax"):
        try:
            __import__(name)
            names.append(name)
        except ModuleNotFoundError:  # pragma: no cover
            pass
    return names


@pytest.fixture(params=_available_backends())
def backend(request: pytest.FixtureRequest) -> Backend:
    return _BACKEND_FACTORIES[request.param]()


@pytest.fixture
def sample_points() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(-2.0, 2.0, size=(48, 4))
