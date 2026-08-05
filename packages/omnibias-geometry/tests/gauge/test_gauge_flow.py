# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Yang-Mills gradient-flow / Langevin (stochastic-quantisation) drift.

Checks the continuum ``gauge_flow_rhs`` operator: (i) it vanishes on the BPST
instanton (an exact solution of the equations of motion, also in Lorenz gauge so
the DeTurck term vanishes too), (ii) the DeTurck-Zwanziger term is non-trivial on
a generic connection, and (iii) torch and jax are bit-identical twins, including
one Euler-Maruyama Langevin step driven by a shared noise array.
"""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from _gauge_helpers import instanton_arrays

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.geometry.gauge.jax.ops as JOPS  # noqa: E402
import omnibias.geometry.gauge.torch.ops as TOPS  # noqa: E402

torch.set_default_dtype(torch.float64)
SIG = (1, 1, 1, 1)


def _random_fields(seed: int = 5):
    rng = np.random.default_rng(seed)
    b, d, n = 6, 4, 3
    a = rng.normal(size=(b, d, n))
    da = rng.normal(size=(b, d, d, n))
    dda = rng.normal(size=(b, d, d, d, n))
    return a, da, dda


def test_flow_drift_vanishes_on_instanton() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(-2.0, 2.0, size=(32, 4))
    a, da, dda = instanton_arrays(pts, rho=1.0)
    su2 = la.su(2)
    for deturck in (0.0, 0.7):
        drift = TOPS.gauge_flow_rhs_from_arrays(
            torch.as_tensor(a),
            torch.as_tensor(da),
            torch.as_tensor(dda),
            algebra=su2,
            coupling=1.0,
            signature=SIG,
            deturck=deturck,
        )
        assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-9)


def test_deturck_term_is_nontrivial() -> None:
    a, da, dda = _random_fields()
    su2 = la.su(2)
    base = TOPS.gauge_flow_rhs_from_arrays(
        torch.as_tensor(a), torch.as_tensor(da), torch.as_tensor(dda),
        algebra=su2, coupling=0.8, signature=SIG, deturck=0.0,
    )
    fixed = TOPS.gauge_flow_rhs_from_arrays(
        torch.as_tensor(a), torch.as_tensor(da), torch.as_tensor(dda),
        algebra=su2, coupling=0.8, signature=SIG, deturck=0.5,
    )
    assert not torch.allclose(base, fixed, atol=1e-6)


@pytest.mark.parametrize("deturck", [0.0, 0.5])
def test_flow_rhs_torch_jax_parity(deturck: float) -> None:
    a, da, dda = _random_fields(seed=11)
    su2 = la.su(2)
    t = TOPS.gauge_flow_rhs_from_arrays(
        torch.as_tensor(a), torch.as_tensor(da), torch.as_tensor(dda),
        algebra=su2, coupling=0.8, signature=SIG, deturck=deturck,
    )
    j = JOPS.gauge_flow_rhs_from_arrays(
        jnp.asarray(a), jnp.asarray(da), jnp.asarray(dda),
        algebra=su2, coupling=0.8, signature=SIG, deturck=deturck,
    )
    np.testing.assert_allclose(t.detach().numpy(), np.asarray(j), rtol=1e-9, atol=1e-11)


def test_langevin_step_parity_with_shared_noise() -> None:
    a, da, dda = _random_fields(seed=13)
    su2 = la.su(2)
    rng = np.random.default_rng(0)
    noise = rng.normal(size=a.shape)
    t = TOPS.langevin_step(
        torch.as_tensor(a), torch.as_tensor(da), torch.as_tensor(dda),
        algebra=su2, coupling=0.8, signature=SIG, dt=0.02, deturck=0.3,
        temperature=1.0, noise=torch.as_tensor(noise),
    )
    j = JOPS.langevin_step(
        jnp.asarray(a), jnp.asarray(da), jnp.asarray(dda),
        algebra=su2, coupling=0.8, signature=SIG, dt=0.02, deturck=0.3,
        temperature=1.0, noise=jnp.asarray(noise),
    )
    np.testing.assert_allclose(t.detach().numpy(), np.asarray(j), rtol=1e-9, atol=1e-11)


def test_langevin_step_requires_noise_or_key() -> None:
    a, da, dda = _random_fields(seed=2)
    su2 = la.su(2)
    with pytest.raises(ValueError, match="noise"):
        JOPS.langevin_step(
            jnp.asarray(a), jnp.asarray(da), jnp.asarray(dda),
            algebra=su2, coupling=0.8, signature=SIG, dt=0.01,
        )
