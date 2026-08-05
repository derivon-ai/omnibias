# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cross-backend invariant tests beyond the per-activation forward/derivative
parity that already lives in :mod:`tests.test_jax_parity`.

These tests pin three additional cross-backend contracts:

1. **Riccati identity**: for every activation that declares a
   ``riccati_polynomial``, both backends must produce a first
   derivative equal to ``P(sigma(z))`` to float64 ULP precision.

2. **Integral round-trip**: for every activation that declares both
   ``integral`` and ``derivative``, the analytic derivative of the
   integral must reproduce the forward (modulo a constant of
   integration). We check the *difference of derivatives at two
   points* to remove the constant.

3. **Negative-order parity**: both backends must raise the same
   error class (``ValueError``) for ``fastpath(z, -1)`` calls.

These tests run on the workspace stable surface only (no pinn /
qpinn / curvature) and require both torch + jax to be installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.jax.activations import get_activation as jax_get  # noqa: E402
from omnibias.jax.activations import list_activations as jax_list  # noqa: E402
from omnibias.torch.activations.registry import get_activation as torch_get  # noqa: E402


def _eval_poly(coeffs: tuple[float, ...], x):
    """Backend-agnostic Horner evaluation of ``sum_k c_k x^k``."""
    out = 0.0
    for c in reversed(coeffs):
        out = out * x + c
    return out


# ---------------------------------------------------------------------------
# 1. Riccati identity -- both backends.
# ---------------------------------------------------------------------------


_RICCATI_NAMES = sorted(
    {name for name in jax_list() if jax_get(name).riccati_polynomial is not None}
)


@pytest.mark.parametrize("name", _RICCATI_NAMES)
def test_riccati_identity_torch(name: str) -> None:
    """``derivative(z) - P(forward(z))`` must vanish in float64 torch."""
    spec = torch_get(name)
    poly = spec.riccati_polynomial
    assert poly is not None
    z = torch.linspace(-2.0, 2.0, 41, dtype=torch.float64)
    if name in ("tan", "cot", "coth"):
        # Avoid singularities; restrict to a safe band.
        if name == "tan":
            z = torch.linspace(-1.4, 1.4, 41, dtype=torch.float64)
        elif name == "cot":
            z = torch.linspace(0.1, math.pi - 0.1, 41, dtype=torch.float64)
        elif name == "coth":
            z = torch.linspace(0.5, 3.0, 41, dtype=torch.float64)
    sigma_p = spec.derivative(z).detach().numpy()
    s = spec.forward(z).detach().numpy()
    riccati = _eval_poly(poly, s)
    np.testing.assert_allclose(sigma_p, riccati, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("name", _RICCATI_NAMES)
def test_riccati_identity_jax(name: str) -> None:
    """``derivative(z) - P(forward(z))`` must vanish in float64 jax."""
    spec = jax_get(name)
    poly = spec.riccati_polynomial
    assert poly is not None
    z = jnp.linspace(-2.0, 2.0, 41)
    if name in ("tan", "cot", "coth"):
        if name == "tan":
            z = jnp.linspace(-1.4, 1.4, 41)
        elif name == "cot":
            z = jnp.linspace(0.1, math.pi - 0.1, 41)
        elif name == "coth":
            z = jnp.linspace(0.5, 3.0, 41)
    sigma_p = np.asarray(spec.derivative(z))
    s = np.asarray(spec.forward(z))
    riccati = _eval_poly(poly, s)
    np.testing.assert_allclose(sigma_p, riccati, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# 2. Integral round-trip.
# ---------------------------------------------------------------------------


_INTEGRAL_NAMES = sorted(
    {
        name
        for name in jax_list()
        if jax_get(name).integral is not None and jax_get(name).derivative is not None
    }
)


_INTEGRAL_TEST_RANGE = {
    "tan": (-1.4, 1.4),
    "cot": (0.2, math.pi - 0.2),
    "coth": (0.5, 3.0),
}


@pytest.mark.parametrize("name", _INTEGRAL_NAMES)
def test_integral_round_trip_torch(name: str) -> None:
    """``d/dz S(z) == sigma(z)`` (autograd of integral matches forward)."""
    spec = torch_get(name)
    lo, hi = _INTEGRAL_TEST_RANGE.get(name, (-1.5, 1.5))
    z = torch.linspace(lo, hi, 11, dtype=torch.float64, requires_grad=True)
    S = spec.integral(z).sum()
    (dS_dz,) = torch.autograd.grad(S, z)
    fwd = spec.forward(z).detach().numpy()
    np.testing.assert_allclose(
        dS_dz.detach().numpy(), fwd, rtol=1e-10, atol=1e-12,
        err_msg=f"integral round-trip failed for {name!r}",
    )


@pytest.mark.parametrize("name", _INTEGRAL_NAMES)
def test_integral_round_trip_jax(name: str) -> None:
    spec = jax_get(name)
    lo, hi = _INTEGRAL_TEST_RANGE.get(name, (-1.5, 1.5))
    z = jnp.linspace(lo, hi, 11)
    grad_int = jax.vmap(jax.grad(spec.integral))(z)
    fwd = spec.forward(z)
    np.testing.assert_allclose(
        np.asarray(grad_int), np.asarray(fwd), rtol=1e-10, atol=1e-12,
        err_msg=f"integral round-trip failed for {name!r}",
    )


# ---------------------------------------------------------------------------
# 3. Negative-order parity.
# ---------------------------------------------------------------------------


_FASTPATH_NAMES = sorted({name for name in jax_list() if jax_get(name).fastpath is not None})


@pytest.mark.parametrize("name", _FASTPATH_NAMES)
def test_negative_order_parity_torch(name: str) -> None:
    spec = torch_get(name)
    assert spec.fastpath is not None
    with pytest.raises(ValueError, match="order n must be"):
        spec.fastpath(torch.zeros(1), -1)


@pytest.mark.parametrize("name", _FASTPATH_NAMES)
def test_negative_order_parity_jax(name: str) -> None:
    spec = jax_get(name)
    assert spec.fastpath is not None
    with pytest.raises(ValueError, match="order n must be"):
        spec.fastpath(jnp.zeros(1), -1)


# ---------------------------------------------------------------------------
# 4. Fastpath value parity at the maximum supported order, with strict
#    float64 tolerances. Complements the looser per-activation parity in
#    ``tests/test_jax_parity.py``.
# ---------------------------------------------------------------------------


_TIGHT_RTOL = 1e-13


_TIGHT_PARITY = (
    # name, n: tested at this order using float64
    ("sigmoid", 3),
    ("tanh", 3),
    ("softplus", 3),
    ("gaussian", 3),
    ("exp", 6),
    ("sin", 4),
    ("cos", 4),
    ("sinh", 4),
    ("cosh", 4),
)


@pytest.mark.parametrize("name,n", _TIGHT_PARITY)
def test_fastpath_tight_parity(name: str, n: int) -> None:
    """High-tolerance parity for activations whose Horner schemes are
    bit-stable across the two backends in float64."""
    rng = np.random.default_rng(13 + n)
    z = rng.uniform(-1.5, 1.5, size=64).astype(np.float64)
    z_t = torch.from_numpy(z).double()
    z_j = jnp.asarray(z)

    t_out = np.asarray(torch_get(name).fastpath(z_t, n).detach().numpy(), dtype=np.float64)
    j_out = np.asarray(jax_get(name).fastpath(z_j, n), dtype=np.float64)

    np.testing.assert_allclose(j_out, t_out, rtol=_TIGHT_RTOL, atol=1e-14)
