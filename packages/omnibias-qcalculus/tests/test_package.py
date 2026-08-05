# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Package-level invariants: version, sorted __all__, backend twins, q -> 1 honesty."""

from __future__ import annotations

from fractions import Fraction

import omnibias.qcalculus as qc
import pytest


def test_version() -> None:
    assert qc.__version__ == "0.1.0a1"


def test_all_sorted_and_exported() -> None:
    assert qc.__all__ == sorted(qc.__all__)
    for name in qc.__all__:
        assert hasattr(qc, name), name


def test_docstring_states_q_to_one_limit_not_delta_or_beta() -> None:
    doc = qc.__doc__ or ""
    assert "q -> 1" in doc
    # Honesty guard: the q-limit is distinct and must not be conflated.
    assert "distinct" in doc.lower()


def test_q_to_one_matches_difference_special_numbers() -> None:
    # The q -> 1 q-Bernoulli / q-Euler must agree with the founding difference register.
    from omnibias.difference import bernoulli_number, euler_number

    for n in range(9):
        assert qc.q_bernoulli(n, 1) == bernoulli_number(n)
    for n in range(0, 9, 2):
        assert qc.q_euler(n, 1) == euler_number(n)


def test_torch_twin_q_to_one_collapse() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.qcalculus.torch import q_derivative, q_derivative_limit, q_derivative_residual

    z = torch.linspace(0.5, 2.0, 16, dtype=torch.float64)
    limit = q_derivative_limit("tanh", z)
    prev = float("inf")
    for q in (0.9, 0.99, 0.999):
        res = q_derivative_residual("tanh", z, q).abs().max().item()
        assert res < prev  # monotone collapse toward sigma'
        prev = res
    assert prev < 1e-2
    # q_derivative is the plain Jackson quotient.
    got = q_derivative("tanh", z, 0.999)
    assert got.shape == z.shape and limit.shape == z.shape


def test_jax_twin_matches_torch_twin() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.qcalculus.jax import q_derivative as jq
    from omnibias.qcalculus.torch import q_derivative as tq

    xs = [0.5, 0.9, 1.4, 1.9]
    q = 0.95
    t = tq("sigmoid", torch.tensor(xs, dtype=torch.float64), q).tolist()
    j = jq("sigmoid", jnp.asarray(xs, dtype=jnp.float64), q).tolist()
    for a, b in zip(t, j, strict=False):
        assert a == pytest.approx(b, rel=1e-10, abs=1e-12)


def test_gaussian_binomial_polynomial_is_exact_object() -> None:
    # Exposed as an exact integer-polynomial (closed-form), not a float.
    poly = qc.q_binomial_poly(6, 3)
    assert all(isinstance(c, int) for c in poly)
    assert qc.q_binomial(6, 3, Fraction(1)) == sum(poly)
