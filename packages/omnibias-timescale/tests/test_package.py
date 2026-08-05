# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package invariants: version, sorted __all__, backend twins, founding-sense honesty."""

from __future__ import annotations

from pathlib import Path

import omnibias.timescale as ts
import pytest

# The founding modules must state the derivative sense (mirrors core's terminology guard).
_FOUNDING_MODULES = [
    "src/omnibias/timescale/__init__.py",
    "src/omnibias/timescale/_core/derivative.py",
]


def _pkg_root() -> Path:
    return Path(ts.__file__).resolve().parents[3]


def test_version() -> None:
    assert ts.__version__ == "0.1.0a1"


def test_all_sorted_and_exported() -> None:
    assert ts.__all__ == sorted(ts.__all__)
    for name in ts.__all__:
        assert hasattr(ts, name), name


def test_docstring_states_mu_to_zero_founding_sense() -> None:
    doc = ts.__doc__ or ""
    assert "mu -> 0" in doc
    assert "founding" in doc.lower()
    assert "distinct" in doc.lower()  # distinct from q -> 1 / beta -> inf


@pytest.mark.parametrize("rel", _FOUNDING_MODULES)
def test_founding_modules_name_the_derivative_sense(rel: str) -> None:
    text = (_pkg_root() / rel).read_text(encoding="utf-8")
    assert "founding bias collapse" in text, f"{rel} no longer names the founding sense"
    assert "sigma^(K-1)" in text, f"{rel} no longer states the derivative output"


def test_torch_twin_mu_to_zero_collapse() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.timescale.torch import delta_derivative, delta_derivative_limit

    z = torch.linspace(0.5, 2.0, 16, dtype=torch.float64)
    limit = delta_derivative_limit("tanh", z)
    prev = float("inf")
    for h in (0.4, 0.1, 0.01):
        res = (delta_derivative("tanh", z, ts.h_integers(h)) - limit).abs().max().item()
        assert res < prev
        prev = res
    assert prev < 1e-2
    # reals dispatch returns the closed-form derivative exactly.
    on_reals = delta_derivative("tanh", z, ts.reals())
    assert torch.allclose(on_reals, limit)


def test_jax_twin_matches_torch_twin() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.timescale.jax import delta_derivative as jdd
    from omnibias.timescale.torch import delta_derivative as tdd

    xs = [0.5, 0.9, 1.4, 1.9]
    for scale in (ts.h_integers(0.3), ts.quantum(1.5)):
        t = tdd("sigmoid", torch.tensor(xs, dtype=torch.float64), scale).tolist()
        j = jdd("sigmoid", jnp.asarray(xs, dtype=jnp.float64), scale).tolist()
        for a, b in zip(t, j, strict=True):
            assert a == pytest.approx(b, rel=1e-10, abs=1e-12)
