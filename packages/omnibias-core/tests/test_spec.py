# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the backend-agnostic ActivationSpec.

These confirm the dataclass shape; backend registry tests live in the
torch / jax packages.
"""

from __future__ import annotations

import math

import pytest
from omnibias.core.spec import ActivationSpec, make_tempered_fastpath, tempered


def _identity(x: float) -> float:
    return x


def _exp_fastpath(z: float, n: int) -> float:
    """``exp^(n)(z) = exp(z)`` for every order -- a convenient all-orders base."""
    return math.exp(z)


def _exp_base() -> ActivationSpec[float]:
    return ActivationSpec(
        name="exp",
        forward=math.exp,
        derivative=math.exp,
        fastpath=_exp_fastpath,
        integral=math.exp,
        limit_neg_inf=0.0,
    )


def test_minimal_spec_has_defaults() -> None:
    spec = ActivationSpec(name="id", forward=_identity)
    assert spec.name == "id"
    assert spec.forward is _identity
    assert spec.derivative is None
    assert spec.fastpath is None
    assert spec.integral is None
    assert spec.riccati_polynomial is None
    assert spec.noise_model == "none"
    assert spec.operator_role == ""
    assert spec.aliases == ()
    assert spec.transforms is None


def test_full_spec_round_trip() -> None:
    spec = ActivationSpec(
        name="sigmoid",
        forward=_identity,
        derivative=_identity,
        fastpath=lambda z, n: z,
        integral=_identity,
        riccati_polynomial=(0.0, 1.0, -1.0),
        noise_model="bernoulli",
        operator_role="logistic IRLS",
        aliases=("logistic", "expit"),
    )
    assert spec.name == "sigmoid"
    assert spec.integral is _identity
    assert spec.riccati_polynomial == (0.0, 1.0, -1.0)
    assert spec.noise_model == "bernoulli"
    assert spec.operator_role == "logistic IRLS"
    assert spec.aliases == ("logistic", "expit")


def test_spec_is_frozen() -> None:
    spec = ActivationSpec(name="id", forward=_identity)
    try:
        spec.name = "other"  # type: ignore[misc]
    except Exception as exc:
        msg = str(exc).lower()
        assert "frozen" in msg or "immutable" in msg or "cannot assign" in msg
    else:  # pragma: no cover
        raise AssertionError("frozen dataclass should reject attribute writes")


# --- tempered / beta-scaled combinator -------------------------------------


def test_make_tempered_fastpath_unit_scale() -> None:
    """``g(beta z)`` has tower ``beta**n * g^(n)(beta z)``."""
    fp = make_tempered_fastpath(_exp_fastpath, 2.0, scale_power=0)
    for n in range(5):
        assert fp(0.3, n) == pytest.approx(2.0**n * math.exp(2.0 * 0.3))


def test_make_tempered_fastpath_one_over_beta_scale() -> None:
    """``g(beta z) / beta`` has tower ``beta**(n-1) * g^(n)(beta z)``."""
    fp = make_tempered_fastpath(_exp_fastpath, 2.0, scale_power=1)
    for n in range(5):
        assert fp(0.3, n) == pytest.approx(2.0 ** (n - 1) * math.exp(2.0 * 0.3))
    # n = 0 reproduces the surrogate forward g(beta z) / beta.
    assert fp(0.3, 0) == pytest.approx(math.exp(2.0 * 0.3) / 2.0)


def test_tempered_spec_forward_derivative_integral() -> None:
    spec = tempered(_exp_base(), 2.0, scale="one_over_beta", name="soft_exp")
    assert spec.name == "soft_exp"
    assert spec.forward(0.3) == pytest.approx(spec.fastpath(0.3, 0))
    assert spec.derivative(0.3) == pytest.approx(spec.fastpath(0.3, 1))
    # integral scaled by beta**(p+1) = 2**2 = 4.
    assert spec.integral is not None
    assert spec.integral(0.3) == pytest.approx(math.exp(2.0 * 0.3) / 4.0)


def test_tempered_default_name_and_metadata() -> None:
    spec = tempered(_exp_base(), 1.5, limit_neg_inf=0.0, aliases=("t_exp",))
    assert spec.name == "tempered_exp"
    assert spec.aliases == ("t_exp",)
    assert spec.limit_neg_inf == 0.0
    assert spec.riccati_polynomial is None


def test_tempered_requires_base_fastpath() -> None:
    with pytest.raises(ValueError, match="fastpath"):
        tempered(ActivationSpec(name="id", forward=_identity), 2.0)


def test_tempered_rejects_unknown_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        tempered(_exp_base(), 2.0, scale="nope")


def test_tempered_fastpath_is_vectorized() -> None:
    np = pytest.importorskip("numpy")
    z = np.linspace(-1.0, 1.0, 5)
    fp = make_tempered_fastpath(lambda x, n: np.exp(x), 3.0, scale_power=1)
    got = fp(z, 2)
    assert np.allclose(got, 3.0 ** (2 - 1) * np.exp(3.0 * z))
