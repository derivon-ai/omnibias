# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch <-> jax bit-identical parity for the gauge ops (float64, rtol 1e-9)."""

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


@pytest.fixture(scope="module")
def fields():
    rng = np.random.default_rng(99)
    pts = rng.uniform(-2.0, 2.0, size=(40, 4))
    return instanton_arrays(pts)


def _assert_parity(t, j, rtol=1e-9, atol=1e-11) -> None:
    np.testing.assert_allclose(
        np.asarray(t.detach().numpy()), np.asarray(j), rtol=rtol, atol=atol
    )


def test_field_strength_parity(fields) -> None:
    a, da, _ = fields
    su2 = la.su(2)
    t = TOPS.field_strength_from_arrays(torch.as_tensor(a), torch.as_tensor(da), algebra=su2, coupling=0.8)
    j = JOPS.field_strength_from_arrays(jnp.asarray(a), jnp.asarray(da), algebra=su2, coupling=0.8)
    _assert_parity(t, j)


def test_covariant_divergence_parity(fields) -> None:
    a, da, dda = fields
    su2 = la.su(2)
    t = TOPS.covariant_divergence_from_arrays(
        torch.as_tensor(a), torch.as_tensor(da), torch.as_tensor(dda),
        algebra=su2, coupling=0.8, signature=SIG,
    )
    j = JOPS.covariant_divergence_from_arrays(
        jnp.asarray(a), jnp.asarray(da), jnp.asarray(dda),
        algebra=su2, coupling=0.8, signature=SIG,
    )
    _assert_parity(t, j)


def test_bianchi_parity(fields) -> None:
    a, da, dda = fields
    su2 = la.su(2)
    t = TOPS.bianchi_from_arrays(
        torch.as_tensor(a), torch.as_tensor(da), torch.as_tensor(dda),
        algebra=su2, coupling=0.8, signature=SIG,
    )
    j = JOPS.bianchi_from_arrays(
        jnp.asarray(a), jnp.asarray(da), jnp.asarray(dda),
        algebra=su2, coupling=0.8, signature=SIG,
    )
    _assert_parity(t, j, atol=1e-9)


def test_action_and_charge_parity(fields) -> None:
    a, da, _ = fields
    su2 = la.su(2)
    Ft = TOPS.field_strength_from_arrays(torch.as_tensor(a), torch.as_tensor(da), algebra=su2, coupling=1.0)
    Fj = JOPS.field_strength_from_arrays(jnp.asarray(a), jnp.asarray(da), algebra=su2, coupling=1.0)
    _assert_parity(TOPS.action_density(Ft, signature=SIG), JOPS.action_density(Fj, signature=SIG))
    _assert_parity(TOPS.topological_charge_density(Ft), JOPS.topological_charge_density(Fj))
    _assert_parity(TOPS.dual_field_strength(Ft, signature=SIG), JOPS.dual_field_strength(Fj, signature=SIG))


def test_structure_constants_parity() -> None:
    for n in (2, 3, 4):
        su = la.su(n)
        _assert_parity(TOPS.structure_constants(su), JOPS.structure_constants(su), atol=1e-12)
