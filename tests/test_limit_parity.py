# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend parity for the jet ``lim`` operator and asymptote metadata.

The jax and torch limit primitives share the same elementwise algebra, so they
must agree to float64 precision; the ``ActivationSpec`` saturation metadata is
shared pure-Python data and must be byte-identical across backends.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
torch = pytest.importorskip("torch")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get  # noqa: E402
from omnibias.jax.jet import lhopital_ratio as jax_lhopital  # noqa: E402
from omnibias.jax.jet import limit_of_ratio as jax_limit  # noqa: E402
from omnibias.torch.activations.registry import get_activation as torch_get  # noqa: E402
from omnibias.torch.jet import lhopital_ratio as torch_lhopital  # noqa: E402
from omnibias.torch.jet import limit_of_ratio as torch_limit  # noqa: E402

_SATURATING = ("tanh", "sigmoid", "gaussian", "softplus", "exp", "arctan")


@pytest.mark.parametrize(
    ("num", "den", "order"),
    [
        ([0.0, 1.0, 0.0, -1.0 / 6.0], [0.0, 1.0, 0.0, 0.0], 1),
        ([0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0], 2),
        ([0.0, 2.5, 1.0, 0.0], [0.0, -0.5, 3.0, 0.0], 1),
    ],
)
def test_lhopital_ratio_parity(num: list[float], den: list[float], order: int) -> None:
    j = float(jax_lhopital(jnp.asarray(num), jnp.asarray(den), order))
    t = float(torch_lhopital(torch.tensor(num, dtype=torch.float64), torch.tensor(den, dtype=torch.float64), order))
    assert j == pytest.approx(t, rel=1e-12, abs=1e-12)


def test_limit_of_ratio_parity() -> None:
    num = [0.0, 1.0, 0.0, -1.0 / 6.0]
    den = [0.0, 1.0, 0.0, 0.0]
    j = float(jax_limit(jnp.asarray(num), jnp.asarray(den)))
    t = float(torch_limit(torch.tensor(num, dtype=torch.float64), torch.tensor(den, dtype=torch.float64)))
    assert j == pytest.approx(t, rel=1e-12, abs=1e-12)


def test_saturation_metadata_parity() -> None:
    for name in _SATURATING:
        js = jax_get(name)
        ts = torch_get(name)
        assert js.limit_pos_inf == ts.limit_pos_inf, name
        assert js.limit_neg_inf == ts.limit_neg_inf, name


def test_saturation_metadata_values() -> None:
    assert jax_get("tanh").limit_pos_inf == 1.0
    assert jax_get("tanh").limit_neg_inf == -1.0
    assert jax_get("sigmoid").limit_pos_inf == 1.0
    assert jax_get("sigmoid").limit_neg_inf == 0.0
    assert jax_get("gaussian").limit_pos_inf == 0.0
    assert jax_get("gaussian").limit_neg_inf == 0.0
    assert jax_get("arctan").limit_pos_inf == pytest.approx(math.pi / 2.0)
    assert np.isclose(jax_get("arctan").limit_neg_inf, -math.pi / 2.0)
    # Diverging activations record no finite right asymptote.
    assert jax_get("exp").limit_pos_inf is None
    assert jax_get("softplus").limit_pos_inf is None
