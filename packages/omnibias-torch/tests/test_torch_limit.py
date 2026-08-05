# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-based ``lim`` operator (torch): L'Hopital ratios + asymptote metadata."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from omnibias.core.spec import saturation_limit  # noqa: E402
from omnibias.torch.activations.registry import get_activation  # noqa: E402
from omnibias.torch.jet import (  # noqa: E402
    lhopital_ratio,
    limit_of_ratio,
    mlp_jet,
    removable_value,
)

_F64 = torch.float64


def _t(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=_F64)


def _deep_mlp(seed: int = 5, dims: tuple[int, ...] = (3, 5, 4, 2), act: str = "tanh"):
    """A small deep MLP ``(layers, x0, v)`` for the jet/forward helpers below."""
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        w = torch.tensor(rng.normal(scale=0.7, size=(dims[i + 1], dims[i])), dtype=_F64)
        b = torch.tensor(rng.normal(scale=0.5, size=(dims[i + 1],)), dtype=_F64)
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((w, b, spec))
    x0 = torch.tensor(rng.normal(size=(dims[0],)), dtype=_F64)
    v = torch.tensor(rng.normal(size=(dims[0],)), dtype=_F64)
    return layers, x0, v


def _mlp_forward(layers):  # type: ignore[no-untyped-def]
    def f(x):  # type: ignore[no-untyped-def]
        z = x
        for w, b, spec in layers:
            z = w @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z

    return f


_SIN = _t([0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0])
_X = _t([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_ONE_MINUS_COS = _t([0.0, 0.0, 0.5, 0.0, -1.0 / 24.0, 0.0])
_X2 = _t([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_lhopital_sinc() -> None:
    assert float(lhopital_ratio(_SIN, _X, order=1)) == pytest.approx(1.0, abs=1e-15)


def test_lhopital_one_minus_cos_over_x2() -> None:
    assert float(lhopital_ratio(_ONE_MINUS_COS, _X2, order=2)) == pytest.approx(0.5, abs=1e-15)


def test_limit_of_ratio_autodetects_order() -> None:
    assert float(limit_of_ratio(_SIN, _X)) == pytest.approx(1.0, abs=1e-15)
    assert float(limit_of_ratio(_ONE_MINUS_COS, _X2)) == pytest.approx(0.5, abs=1e-15)


def test_limit_of_ratio_numerator_vanishes_faster_is_zero() -> None:
    assert float(limit_of_ratio(_X2, _X)) == 0.0


def test_limit_of_ratio_pole_is_infinite() -> None:
    assert math.isinf(float(limit_of_ratio(_X, _X2)))


def test_limit_of_ratio_zero_denominator_raises() -> None:
    with pytest.raises(ValueError, match="vanishes to all"):
        limit_of_ratio(_SIN[:4], torch.zeros(4, dtype=_F64))


def test_lhopital_ratio_is_differentiable() -> None:
    a = torch.tensor(2.0, dtype=_F64, requires_grad=True)
    num = _t([0.0, 1.0, 0.0]) * a
    den = _t([0.0, 1.0, 0.0])
    limit = lhopital_ratio(num, den, order=1)
    limit.backward()
    assert float(a.grad) == pytest.approx(1.0, abs=1e-15)


def test_removable_value_is_zeroth_coefficient() -> None:
    jet = _t([3.5, 1.0, -2.0])
    assert float(removable_value(jet)) == 3.5


def test_limit_of_a_learned_field_along_a_ray() -> None:
    torch.manual_seed(0)
    W1 = torch.randn(4, 2, dtype=_F64)
    b1 = torch.zeros(4, dtype=_F64)
    W2 = torch.randn(1, 4, dtype=_F64)
    b2 = torch.zeros(1, dtype=_F64)
    x0 = _t([0.3, -0.7])
    v = _t([1.0, 0.5])
    layers = [(W1, b1, "tanh"), (W2, b2, None)]
    jet = mlp_jet(x0, v, layers, order=2)
    direct = W2 @ torch.tanh(W1 @ x0 + b1) + b2
    assert float(removable_value(jet)[0]) == pytest.approx(float(direct[0]), abs=1e-12)


def test_saturation_metadata_is_populated() -> None:
    assert get_activation("tanh").limit_pos_inf == 1.0
    assert get_activation("tanh").limit_neg_inf == -1.0
    assert get_activation("sigmoid").limit_neg_inf == 0.0
    assert get_activation("gaussian").limit_pos_inf == 0.0
    assert saturation_limit(get_activation("arctan"), +1.0) == pytest.approx(math.pi / 2.0)
    assert saturation_limit(get_activation("exp"), -1.0) == 0.0
    assert saturation_limit(get_activation("exp"), +1.0) is None


# ----- the defining equation: derivative == limit of the difference quotient ----


def test_difference_quotient_limit_equals_directional_derivative() -> None:
    r"""``f'(x0).v == lim_{t->0} (f(x0 + t v) - f(x0)) / t`` for a deep MLP.

    Wires the two primitives against the textbook definition rather than against
    each other:

    * the *limit* operator (:func:`lhopital_ratio`) is applied to the jet of the
      difference quotient ``(f(x0 + t v) - f(x0)) / t``;
    * the *derivative* is taken independently from forward-mode autodiff
      (:func:`torch.func.jvp`), which shares no code with the sigma-tower jet.

    The closed-form Taylor coefficient ``jet[1]`` (== directional derivative) is
    recovered *exactly* by the limit, and matches autodiff to float64 eps. The
    second-order limit ``2 lim (f(x0+tv) - f(x0) - f'.v t) / t^2`` likewise equals
    the directional second derivative, exercising a genuine ``0/0`` L'Hopital.
    """
    layers, x0, v = _deep_mlp(seed=5)
    f = _mlp_forward(layers)
    order = 3
    jet = mlp_jet(x0, v, layers, order)  # jet[k] = (1/k!) d^k/dt^k f(x0 + t v)

    # N(t) = f(x0 + t v) - f(x0): drop the constant term -> [0, a1, a2, a3].
    num = jet.clone()
    num[0] = 0.0
    t_jet = _t([0.0, 1.0, 0.0, 0.0])  # jet of D(t) = t
    limit1 = lhopital_ratio(num, t_jet, order=1)  # the lim operator

    _, jvp = torch.func.jvp(f, (x0,), (v,))  # independent autodiff derivative

    # lim of the difference quotient reproduces the derivative coefficient exactly
    assert torch.equal(limit1, jet[1])
    # ... and equals the autodiff directional derivative.
    assert torch.allclose(limit1, jvp, rtol=1e-12, atol=1e-12)

    # Second order: N2(t) = N(t) - (f'.v) t -> [0, 0, a2, a3]; D2(t) = t^2.
    num2 = num.clone()
    num2[1] = 0.0
    t2_jet = _t([0.0, 0.0, 1.0, 0.0])
    limit2 = lhopital_ratio(num2, t2_jet, order=2)
    d2 = torch.func.jvp(lambda x: torch.func.jvp(f, (x,), (v,))[1], (x0,), (v,))[1]
    assert torch.equal(limit2, jet[2])
    assert torch.allclose(2.0 * limit2, d2, rtol=1e-12, atol=1e-12)
