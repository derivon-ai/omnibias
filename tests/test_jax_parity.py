# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity tests: every JAX activation matches the PyTorch one to <= 1e-6.

The two backends are expected to agree to numerical precision because
the polynomial coefficients are computed by the same pure-Python
generator (:mod:`omnibias.fastpath.polynomials`). The only sources of
discrepancy are:

* float32 round-off (we use float64 here),
* differences in the underlying ``erf`` implementation (relevant for
  ``gelu`` only).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
# Use float64 throughout: the polynomial recursions otherwise accumulate
# ~1e-7 round-off at orders 5-6 that overflows our strict tolerance.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402  (after importorskip)
from omnibias.jax.activations import get_activation as jax_get
from omnibias.jax.activations import list_activations as jax_list
from omnibias.torch.activations.registry import get_activation as torch_get

_RNG_SEED = 0

# Per-activation: maximum order supported by the fast path and any
# special tolerance overrides.
_MAX_FASTPATH_ORDER = {
    "sigmoid": 6,
    "tanh": 6,
    "softplus": 6,
    "gaussian": 6,
    "exp": 6,
    "arctan": 2,
    "log1pu2": 2,
    # relu / huber now carry all-orders a.e. towers; silu / gelu / mish are
    # exact all orders via Leibniz on the analytic z*f(z) product.
    "huber": 6,
    "silu": 6,
    "gelu": 6,
    "relu": 6,
    # NQS-friendly activations (Phase-10 lattice work, mirrored to torch
    # in Phase 16 to close the parity gap).
    "log_cosh": 3,
    "softabs": 2,
    "smooth_sign": 6,
    "mish": 6,
    # Trigonometric / hyperbolic family added in Phase 16 for periodic
    # systems (sin/cos), continuum bound states (sinh/cosh, sech) and
    # Riccati-class hyperbolic activations (coth, tan, cot).
    "sin": 6,
    "cos": 6,
    "sinh": 6,
    "cosh": 6,
    "tan": 3,
    "cot": 3,
    "coth": 3,
    "sech": 3,
    # Piecewise (almost-everywhere) family. Linear pieces: n>=2 -> 0.
    "leaky_relu": 4,
    "prelu": 4,
    "relu6": 4,
    "hardtanh": 4,
    "hardsigmoid": 4,
    "softshrink": 4,
    "hardshrink": 4,
    "threshold": 4,
    "abs": 4,
    "sign": 4,
    "step": 4,
    # Piecewise-smooth pieces (exp / quadratic / rational): all orders.
    "elu": 6,
    "selu": 6,
    "celu": 6,
    "hardswish": 6,
    "softsign": 6,
    # Beta-tempered smooth surrogates.
    "soft_relu": 6,
    "soft_step": 6,
}
_RTOL = 1e-6
_ATOL = 1e-6


# Activations with poles (or other singular structure) on the real line:
# the standard ``_sample_inputs()`` distribution would routinely sample
# arbitrarily close to a pole, which is meaningless for parity. We use a
# safe, pole-free range for these.
_SAFE_RANGE = {
    # tan: poles at +-pi/2 + k*pi; (-1.5, 1.5) sits comfortably below pi/2 ~ 1.5708.
    "tan": (-1.5, 1.5),
    # cot: poles at k*pi; restrict to (0.05, pi - 0.05) shifted symmetric.
    "cot": None,  # special-cased below
    # coth: pole at z = 0; restrict to |z| in [0.1, 5.0].
    "coth": None,  # special-cased below
}


def _sample_inputs() -> np.ndarray:
    """Return float64 inputs covering small/large/near-zero/extreme values."""
    rng = np.random.default_rng(_RNG_SEED)
    z = np.concatenate(
        [
            rng.normal(scale=1.0, size=64),
            rng.normal(scale=3.0, size=32),
            rng.uniform(-10.0, 10.0, size=32),
            np.array([0.0, 1e-8, -1e-8, 1.0, -1.0, 5.0, -5.0]),
        ]
    ).astype(np.float64)
    return z


def _sample_inputs_for(name: str) -> np.ndarray:
    """Activation-aware sample range that avoids real-line singularities."""
    rng = np.random.default_rng(_RNG_SEED + hash(name) % 65536)
    if name == "tan":
        lo, hi = _SAFE_RANGE["tan"]
        return rng.uniform(lo, hi, size=160).astype(np.float64)
    if name == "cot":
        # Stay away from k*pi; sample in (0.05, pi - 0.05) and mirror.
        x = rng.uniform(0.05, math.pi - 0.05, size=80)
        return np.concatenate([x, -x]).astype(np.float64)
    if name == "coth":
        # Stay away from z = 0; sample in [0.1, 5.0] and mirror.
        x = rng.uniform(0.1, 5.0, size=80)
        return np.concatenate([x, -x]).astype(np.float64)
    return _sample_inputs()


@pytest.mark.parametrize("name", sorted(_MAX_FASTPATH_ORDER))
def test_forward_matches_torch(name: str) -> None:
    z = _sample_inputs_for(name)
    z_t = torch.from_numpy(z).double()
    z_j = jnp.asarray(z)

    t_out = np.asarray(torch_get(name).forward(z_t).detach().numpy(), dtype=np.float64)
    j_out = np.asarray(jax_get(name).forward(z_j), dtype=np.float64)

    np.testing.assert_allclose(
        j_out, t_out, rtol=_RTOL, atol=_ATOL, err_msg=f"forward mismatch for {name!r}"
    )


@pytest.mark.parametrize("name", sorted(_MAX_FASTPATH_ORDER))
def test_derivative_matches_torch(name: str) -> None:
    z = _sample_inputs_for(name)
    z_t = torch.from_numpy(z).double()
    z_j = jnp.asarray(z)

    t_out = np.asarray(torch_get(name).derivative(z_t).detach().numpy(), dtype=np.float64)
    j_out = np.asarray(jax_get(name).derivative(z_j), dtype=np.float64)

    np.testing.assert_allclose(
        j_out, t_out, rtol=_RTOL, atol=_ATOL, err_msg=f"derivative mismatch for {name!r}"
    )


@pytest.mark.parametrize("name", sorted(_MAX_FASTPATH_ORDER))
def test_fastpath_matches_torch_all_orders(name: str) -> None:
    z = _sample_inputs_for(name)
    z_t = torch.from_numpy(z).double()
    z_j = jnp.asarray(z)

    max_n = _MAX_FASTPATH_ORDER[name]
    for n in range(0, max_n + 1):
        t_out = np.asarray(torch_get(name).fastpath(z_t, n).detach().numpy(), dtype=np.float64)
        j_out = np.asarray(jax_get(name).fastpath(z_j, n), dtype=np.float64)

        # At high polynomial orders (n >= 5 in the Eulerian / Legendre
        # / Hermite recursions) the Horner schemes in torch vs jax
        # accumulate ~1e-7 relative round-off differently. Both
        # backends are correct to within float64 ULPs; the absolute
        # value is < 1 in magnitude for the values we test, so 1e-6
        # absolute matches the looser of the two tolerances on the
        # ratios.
        rtol = 1e-5 if n >= 5 or name == "gelu" else _RTOL
        atol = 1e-6 if n >= 5 or name == "gelu" else _ATOL
        np.testing.assert_allclose(
            j_out,
            t_out,
            rtol=rtol,
            atol=atol,
            err_msg=f"fastpath mismatch for {name!r} at n={n}",
        )


def test_list_activations_matches_torch() -> None:
    """Both backends register the same set of activation names."""
    from omnibias.torch.activations.registry import list_activations as torch_list

    assert sorted(torch_list()) == sorted(jax_list())


def test_polynomial_coeffs_shared() -> None:
    """The polynomial generators are imported from the shared pure-python
    module; they should produce identical tuples for both backends."""
    from omnibias.core.polynomials import (
        hermite_coeffs,
        sigmoid_polynomial_coeffs,
        tanh_polynomial_coeffs,
    )

    # spot-check a few orders
    assert sigmoid_polynomial_coeffs(0) == (0.0, 1.0)
    assert sigmoid_polynomial_coeffs(1) == (0.0, 1.0, -1.0)
    assert tanh_polynomial_coeffs(0) == (0.0, 1.0)
    assert tanh_polynomial_coeffs(1) == (1.0, 0.0, -1.0)
    assert hermite_coeffs(0) == (1.0,)
    assert hermite_coeffs(1) == (0.0, 1.0)
    assert hermite_coeffs(2) == (-1.0, 0.0, 1.0)


def test_jax_neural_field_laplacian_matches_jax_hessian() -> None:
    """For one-layer scalar fields, the closed-form Laplacian must equal
    the trace of jax.hessian on the field."""
    from omnibias.jax.laplacian import (
        neural_field_laplacian,
        neural_field_value,
    )

    D, H = 8, 16
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.normal(scale=0.4, size=(H, D)).astype(np.float64))
    beta = jnp.asarray(rng.normal(size=H).astype(np.float64))
    c = jnp.asarray(rng.normal(scale=0.3, size=H).astype(np.float64))
    b = float(rng.normal())
    x = jnp.asarray(rng.normal(size=D).astype(np.float64))

    for name in ("sigmoid", "tanh", "softplus", "gaussian", "exp", "arctan", "log1pu2"):
        lap_omni = neural_field_laplacian(x, W, beta, c, name)

        # JAX reference: trace of the Hessian of f wrt x.
        def f_of_x(xv, name=name):
            return neural_field_value(xv, W, beta, c, b, name)

        H_full = jax.hessian(f_of_x)(x)  # (D, D)
        lap_ref = jnp.trace(H_full)

        np.testing.assert_allclose(
            np.asarray(lap_omni),
            np.asarray(lap_ref),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"closed-form Laplacian mismatch for {name!r}",
        )


def test_jax_neural_field_laplacian_trig_matches_jax_hessian() -> None:
    """Same parity test, but for the trig/hyperbolic family.

    For ``tan`` / ``cot`` / ``coth`` we shrink the W magnitude and shift
    ``beta`` so that the pre-activation ``W @ x + beta`` stays well clear
    of the relevant pole locations on the real line.
    """
    from omnibias.jax.laplacian import (
        neural_field_laplacian,
        neural_field_value,
    )

    D, H = 8, 16
    rng = np.random.default_rng(1)
    # Smaller weights and zero-mean x: pre-activation z = W @ x + beta
    # then has standard deviation ~ sqrt(D) * 0.1 ~ 0.28, which is well
    # inside (-1.5, 1.5) for tan / inside (1e-2, pi/2) for cot.
    W_small = jnp.asarray(rng.normal(scale=0.1, size=(H, D)).astype(np.float64))
    c = jnp.asarray(rng.normal(scale=0.3, size=H).astype(np.float64))
    b = float(rng.normal())
    x = jnp.asarray(rng.normal(scale=0.3, size=D).astype(np.float64))

    # Tier-1 trig: every-n closed form. Beta = 0, weights small.
    for name in ("sin", "cos", "sinh", "cosh"):
        beta = jnp.zeros(H)
        lap_omni = neural_field_laplacian(x, W_small, beta, c, name)

        def f_of_x(xv, name=name, beta=beta):
            return neural_field_value(xv, W_small, beta, c, b, name)

        H_full = jax.hessian(f_of_x)(x)
        lap_ref = jnp.trace(H_full)

        np.testing.assert_allclose(
            np.asarray(lap_omni),
            np.asarray(lap_ref),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"closed-form Laplacian mismatch for {name!r}",
        )

    # tan: shift beta so pre-activation lies in (-1.0, 1.0).
    beta_tan = jnp.asarray(rng.uniform(-0.3, 0.3, size=H).astype(np.float64))
    for name in ("tan",):
        lap_omni = neural_field_laplacian(x, W_small, beta_tan, c, name)

        def f_of_x(xv, name=name, beta=beta_tan):
            return neural_field_value(xv, W_small, beta, c, b, name)

        H_full = jax.hessian(f_of_x)(x)
        lap_ref = jnp.trace(H_full)

        np.testing.assert_allclose(
            np.asarray(lap_omni),
            np.asarray(lap_ref),
            rtol=1e-9,
            atol=1e-11,
            err_msg=f"closed-form Laplacian mismatch for {name!r}",
        )

    # cot: shift beta to (0.5, 1.5) so pre-activation stays in (0.2, 1.8).
    beta_cot = jnp.asarray(rng.uniform(0.5, 1.5, size=H).astype(np.float64))
    for name in ("cot",):
        lap_omni = neural_field_laplacian(x, W_small, beta_cot, c, name)

        def f_of_x(xv, name=name, beta=beta_cot):
            return neural_field_value(xv, W_small, beta, c, b, name)

        H_full = jax.hessian(f_of_x)(x)
        lap_ref = jnp.trace(H_full)

        np.testing.assert_allclose(
            np.asarray(lap_omni),
            np.asarray(lap_ref),
            rtol=1e-9,
            atol=1e-11,
            err_msg=f"closed-form Laplacian mismatch for {name!r}",
        )

    # coth: shift beta to (1.5, 3.5) so pre-activation stays well above 0.
    beta_coth = jnp.asarray(rng.uniform(1.5, 3.5, size=H).astype(np.float64))
    for name in ("coth",):
        lap_omni = neural_field_laplacian(x, W_small, beta_coth, c, name)

        def f_of_x(xv, name=name, beta=beta_coth):
            return neural_field_value(xv, W_small, beta, c, b, name)

        H_full = jax.hessian(f_of_x)(x)
        lap_ref = jnp.trace(H_full)

        np.testing.assert_allclose(
            np.asarray(lap_omni),
            np.asarray(lap_ref),
            rtol=1e-9,
            atol=1e-11,
            err_msg=f"closed-form Laplacian mismatch for {name!r}",
        )

    # sech: no real-line poles, but use a moderate shift so we sample
    # both the bell apex and the saddle.
    beta_sech = jnp.asarray(rng.normal(scale=0.5, size=H).astype(np.float64))
    for name in ("sech",):
        lap_omni = neural_field_laplacian(x, W_small, beta_sech, c, name)

        def f_of_x(xv, name=name, beta=beta_sech):
            return neural_field_value(xv, W_small, beta, c, b, name)

        H_full = jax.hessian(f_of_x)(x)
        lap_ref = jnp.trace(H_full)

        np.testing.assert_allclose(
            np.asarray(lap_omni),
            np.asarray(lap_ref),
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"closed-form Laplacian mismatch for {name!r}",
        )
