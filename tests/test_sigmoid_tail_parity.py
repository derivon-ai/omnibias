# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Documented torch / JAX sigmoid tail contract at extreme ``z``.

Both backends use their framework-native stable sigmoid
(``torch.sigmoid`` / ``jax.nn.sigmoid``). Shared polynomial coefficients make
the derivative tower bit-identical given the same ``s``, but the base
``s = sigma(z)`` itself is *not* guaranteed bit-identical in the extreme tails
because each vendor's ``exp`` / fused kernel can differ by a few ULPs once
``s`` is near 0 or 1. This test locks that documented contract rather than
forcing a common hand-rolled formula that would risk breaking existing
bit-identity fixtures. See:

* :mod:`omnibias.torch.fastpath.eulerian`
* :mod:`omnibias.jax._fastpath`
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402  (after jax_enable_x64)
from omnibias.jax._fastpath import jax_sigmoid
from omnibias.jax._fastpath import sigmoid_nth_derivative as jax_sigma_n
from omnibias.torch.fastpath.eulerian import sigmoid_nth_derivative as torch_sigma_n

# Extreme tails named in the module docs. float64 throughout.
_TAIL_Z = (-80.0, -40.0, 40.0, 80.0)

# A few ULPs of relative slack once ``s`` is denormalising toward 0/1 is the
# honest contract; absolute floor covers the underflow regime at |z|=80.
_REL_TOL = 1e-14
_ABS_TOL = 1e-300


def _ulps_between(a: float, b: float) -> int:
    """Integer ULP distance between two float64 values (same sign / finite)."""
    if a == b:
        return 0
    ai = np.float64(a).view(np.int64)
    bi = np.float64(b).view(np.int64)
    return int(abs(int(ai) - int(bi)))


@pytest.mark.parametrize("z", _TAIL_Z)
def test_native_sigmoid_tails_are_finite_and_close(z: float) -> None:
    """Framework-native sigmoids agree within a few ULPs at extreme tails."""
    t = float(torch.sigmoid(torch.tensor(z, dtype=torch.float64)))
    j = float(jax_sigmoid(jnp.asarray(z, dtype=jnp.float64)))
    assert math.isfinite(t) and math.isfinite(j)
    assert 0.0 <= t <= 1.0 and 0.0 <= j <= 1.0
    assert math.isclose(t, j, rel_tol=_REL_TOL, abs_tol=_ABS_TOL), (
        f"torch.sigmoid({z})={t!r} vs jax.nn.sigmoid({z})={j!r} "
        f"(ulp_distance={_ulps_between(t, j)})"
    )
    # Documented ceiling: a few ULPs, not bit-identity. Allow generous headroom
    # for vendor kernel drift without silently accepting large divergence.
    if t != 0.0 and j != 0.0 and t != 1.0 and j != 1.0:
        assert _ulps_between(t, j) <= 8, (
            f"tail ULP distance {_ulps_between(t, j)} exceeds documented few-ULP contract"
        )


def test_native_sigmoid_is_monotone_across_documented_tails() -> None:
    zs = sorted(_TAIL_Z)
    torch_vals = [
        float(torch.sigmoid(torch.tensor(z, dtype=torch.float64))) for z in zs
    ]
    jax_vals = [float(jax_sigmoid(jnp.asarray(z, dtype=jnp.float64))) for z in zs]
    assert torch_vals == sorted(torch_vals)
    assert jax_vals == sorted(jax_vals)


@pytest.mark.parametrize("z", _TAIL_Z)
@pytest.mark.parametrize("n", [0, 1, 2])
def test_closed_form_tower_tracks_native_sigmoid_at_tails(z: float, n: int) -> None:
    """Given each backend's own ``s``, the shared Eulerian tower stays coherent."""
    t = torch_sigma_n(torch.tensor(z, dtype=torch.float64), n)
    j = jax_sigma_n(jnp.asarray(z, dtype=jnp.float64), n)
    t_v = float(t)
    j_v = float(j)
    assert math.isfinite(t_v) and math.isfinite(j_v)
    # n=0 is exactly the native sigmoid; higher orders vanish at |z|->inf.
    assert math.isclose(t_v, j_v, rel_tol=_REL_TOL, abs_tol=max(_ABS_TOL, 1e-30))
