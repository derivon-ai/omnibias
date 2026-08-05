# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learnable-temperature blocks + a.e./tempered tower checks (Keras backend).

Bit-identical agreement with torch / jax is covered by
``tests/test_keras_parity.py``; here we pin the Keras layer surface
(:class:`TemperedActivation`, :class:`LearnablePReLU`): trainable / frozen
weights, config round-trip, the ``call`` forward, and the closed-form tower via
``keras.ops`` (beta-scaling identity, linear-piece ``n >= 2 -> 0``).
"""

from __future__ import annotations

import numpy as np
import pytest
from keras import ops
from omnibias.keras import LearnablePReLU, TemperedActivation
from omnibias.keras.activations import get_activation


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


_Z = np.linspace(-3.0, 3.0, 41)


def _z():
    return ops.convert_to_tensor(_Z)


# --- TemperedActivation ----------------------------------------------------


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 4.0])
def test_tempered_beta_scaling_identity(beta: float) -> None:
    """``TemperedActivation(softplus)^(n) == beta**(n-1) softplus^(n)(beta z)``."""
    mod = TemperedActivation("softplus", beta=beta, scale="one_over_beta")
    softplus = get_activation("softplus")
    z = _z()
    for n in range(0, 6):
        lhs = _np(mod.fastpath(z, n))
        rhs = _np((beta ** (n - 1)) * softplus.fastpath(beta * z, n))
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6, atol=1e-6, err_msg=f"n={n}")


def test_tempered_call_is_forward() -> None:
    mod = TemperedActivation("sigmoid", beta=1.5, scale="unit")
    z = _z()
    np.testing.assert_allclose(_np(mod(z)), _np(mod.fastpath(z, 0)), rtol=1e-6, atol=1e-6)


def test_tempered_learnable_beta_is_trainable() -> None:
    learn = TemperedActivation("softplus", beta=2.0, learnable_beta=True)
    assert len(learn.trainable_weights) == 1
    assert learn.trainable_weights[0].path.endswith("beta")


def test_tempered_frozen_beta_not_trainable() -> None:
    frozen = TemperedActivation("softplus", beta=2.0, learnable_beta=False)
    assert frozen.trainable_weights == []


def test_tempered_config_round_trip() -> None:
    mod = TemperedActivation("sigmoid", beta=2.5, scale="unit", learnable_beta=True)
    clone = TemperedActivation.from_config(mod.get_config())
    z = _z()
    np.testing.assert_allclose(_np(clone(z)), _np(mod(z)), rtol=1e-6, atol=1e-6)
    assert clone.scale == "unit"


def test_tempered_rejects_bad_scale() -> None:
    with pytest.raises(ValueError):
        TemperedActivation("softplus", beta=1.0, scale="bogus")


def test_tempered_converges_to_relu() -> None:
    relu = get_activation("relu")
    z = ops.convert_to_tensor(np.array([-3.0, -1.0, -0.3, 0.3, 1.0, 3.0]))
    err = float(np.max(np.abs(_np(TemperedActivation("softplus", beta=128.0)(z)) - _np(relu.forward(z)))))
    assert err < 1e-2


# --- LearnablePReLU --------------------------------------------------------


def test_learnable_prelu_forward_and_tower() -> None:
    lp = LearnablePReLU(0.25)
    zz = np.array([-2.0, -0.5, 0.5, 2.0])
    z = ops.convert_to_tensor(zz)
    np.testing.assert_allclose(_np(lp(z)), np.where(zz > 0, zz, 0.25 * zz))
    np.testing.assert_allclose(_np(lp.fastpath(z, 1)), np.where(zz > 0, 1.0, 0.25))
    assert np.all(_np(lp.fastpath(z, 2)) == 0.0)


def test_learnable_prelu_alpha_trainable() -> None:
    lp = LearnablePReLU(0.25, learnable=True)
    assert len(lp.trainable_weights) == 1
    assert lp.trainable_weights[0].path.endswith("alpha")
    frozen = LearnablePReLU(0.25, learnable=False)
    assert frozen.trainable_weights == []
