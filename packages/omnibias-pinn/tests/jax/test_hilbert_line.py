# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Whole-line numerical Hilbert vs exact H[Q]=-P (free-Ω path)."""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.equations.ccf_compactified import (  # noqa: E402
    alpha_from_lambda,
    hardy_even,
    hardy_odd,
    hilbert_transform_truncated_line,
)
from omnibias.pinn.jax.discovery.ccf_vorticity import (  # noqa: E402
    free_omega_vorticity_residual,
    hardy_omega_profile,
    vorticity_residual_samples,
    wang_residual,
)
from omnibias.pinn.jax.hilbert_line import (  # noqa: E402
    hilbert_gl_single_raw_tail,
    hilbert_gl_split_core,
    hilbert_tail_power_mapped,
    hilbert_wholeline_hp,
    integrate_velocity_from_hilbert,
)


LAM = 0.6057
ALPHA = float(alpha_from_lambda(LAM))


@pytest.mark.parametrize(
    ("a", "ymax", "n", "core_frac"),
    [
        (1.3, 40.0, 401, 0.9),
        (0.25, 20.0, 801, 0.9),
    ],
)
def test_wholeline_hp_beats_gl96_u48_on_planted_q(
    a: float, ymax: float, n: int, core_frac: float
) -> None:
    """Free-Ω quadrature vs exact H[Q]=-P. Stretch 1e-13 is not claimed."""
    y = jnp.linspace(-ymax, ymax, n, dtype=jnp.float64)
    omega_fn = lambda t, a=a: hardy_odd(t, a, ALPHA)
    values = omega_fn(y)
    h_exact = -hardy_even(y, a, ALPHA)
    h_old = hilbert_gl_single_raw_tail(
        y, values, omega_fn, y_trunc=ymax, n_gl=96, n_tail=48
    )
    h_new = hilbert_wholeline_hp(
        y, values, omega_fn, decay_power=ALPHA, y_trunc=ymax
    )
    core = jnp.abs(y) <= core_frac * ymax
    err_old = float(jnp.max(jnp.abs((h_old - h_exact)[core])))
    err_new = float(jnp.max(jnp.abs((h_new - h_exact)[core])))
    assert math.isfinite(err_new)
    assert err_new < err_old
    # Planted Q treated as a free omega_fn (no Hardy closed form in the
    # operator). Stretch 1e-13 is still unearned on a trained Wang net.
    if a >= 1.0:
        assert err_new < 1e-12
    else:
        assert err_new < 5e-13


def test_wholeline_hp_beats_truncated_fft_on_hardy_q_reproduce_grid() -> None:
    """Periodic truncated-line FFT is not the whole-line operator."""
    y = jnp.linspace(-40.0, 40.0, 257, dtype=jnp.float64)
    a = 1.3
    omega = hardy_odd(y, a, ALPHA)
    href = -hardy_even(y, a, ALPHA)
    core = jnp.abs(y) <= 36.0
    err_fft = float(
        jnp.max(jnp.abs((hilbert_transform_truncated_line(y, omega) - href)[core]))
    )
    h_hp = hilbert_wholeline_hp(
        y, omega, lambda t: hardy_odd(t, a, ALPHA), decay_power=ALPHA, y_trunc=40.0
    )
    err_hp = float(jnp.max(jnp.abs((h_hp - href)[core])))
    assert err_fft > 5e-2
    assert err_hp <= 1e-8


def test_wholeline_hp_is_even_for_odd_omega() -> None:
    y = jnp.linspace(-20.0, 20.0, 401, dtype=jnp.float64)
    omega_fn = lambda t: hardy_odd(t, 1.3, ALPHA)
    h = hilbert_wholeline_hp(y, omega_fn(y), omega_fn, decay_power=ALPHA)
    assert float(jnp.max(jnp.abs(h - h[::-1]))) < 1e-10


def test_split_core_plus_power_tail_finite() -> None:
    y = jnp.linspace(-8.0, 8.0, 201, dtype=jnp.float64)
    omega_fn = lambda t: t * jnp.exp(-t * t)
    values = omega_fn(y)
    h_core = hilbert_gl_split_core(
        y, values, omega_fn, y_trunc=8.0, y_near=2.0, n_near=64, n_far=32
    )
    h_tail = hilbert_tail_power_mapped(
        y, omega_fn, y_trunc=8.0, decay_power=ALPHA, n_tail=48
    )
    assert jnp.isfinite(h_core).all()
    assert jnp.isfinite(h_tail).all()


def test_integrate_velocity_matches_closed_form_hardy_u() -> None:
    """Trapezoid U from exact H[Q]=-P tracks closed-form U (not OperatorBlock)."""
    y = jnp.linspace(-20.0, 20.0, 4001, dtype=jnp.float64)
    om, omy, u_exact, uy = hardy_omega_profile(
        y, jnp.asarray([1.0]), jnp.asarray([1.3]), jnp.asarray([ALPHA])
    )
    u_trap = integrate_velocity_from_hilbert(y, uy)
    assert float(jnp.max(jnp.abs(u_trap - u_exact))) < 5e-6
    assert float(jnp.abs(jnp.interp(0.0, y, u_trap))) < 1e-14
    _ = om, omy


def test_free_omega_wang_residual_matches_hardy_samples() -> None:
    """Official free-Ω score path vs exact Hardy residual on planted Q."""
    y = jnp.linspace(-40.0, 40.0, 4001, dtype=jnp.float64)
    coeffs = jnp.asarray([1.0])
    scales = jnp.asarray([1.3])
    gammas = jnp.asarray([ALPHA])
    r_h, fields = vorticity_residual_samples(y, coeffs, scales, gammas, LAM)
    omega_fn = lambda t: hardy_omega_profile(t, coeffs, scales, gammas)[0]
    r_f, fields_f = free_omega_vorticity_residual(
        y,
        fields["omega"],
        fields["omega_y"],
        omega_fn,
        lam=LAM,
        y_trunc=40.0,
    )
    # hp tail is not exact on the truncation nodes; U is a trapezoid.
    core = jnp.abs(y) <= 20.0
    assert float(jnp.max(jnp.abs((r_f - r_h)[core]))) < 1e-5
    assert float(jnp.max(jnp.abs((fields_f["U_y"] - fields["U_y"])[core]))) < 1e-12
    r_direct = wang_residual(
        y, fields["omega"], fields["omega_y"], fields["U"], fields["U_y"], lam=LAM
    )
    assert float(jnp.max(jnp.abs(r_direct - r_h))) == 0.0


def test_official_free_omega_path_is_float64() -> None:
    """JAX defaults to float32; the stretch Hilbert must stay float64."""
    from omnibias.pinn.jax.hilbert_line import _gl_pm1

    y = jnp.linspace(-20.0, 20.0, 401, dtype=jnp.float64)
    coeffs = jnp.asarray([1.0], dtype=jnp.float64)
    scales = jnp.asarray([1.3], dtype=jnp.float64)
    gammas = jnp.asarray([ALPHA], dtype=jnp.float64)
    om, omy, _, _ = hardy_omega_profile(y, coeffs, scales, gammas)
    omega_fn = lambda t: hardy_omega_profile(t, coeffs, scales, gammas)[0]
    xi, w = _gl_pm1(16)
    assert xi.dtype == jnp.float64
    assert w.dtype == jnp.float64
    h = hilbert_wholeline_hp(y, om, omega_fn, decay_power=ALPHA, y_trunc=20.0)
    u = integrate_velocity_from_hilbert(y, h)
    r, fields = free_omega_vorticity_residual(
        y, om, omy, omega_fn, lam=LAM, y_trunc=20.0
    )
    assert h.dtype == jnp.float64
    assert u.dtype == jnp.float64
    assert r.dtype == jnp.float64
    assert fields["U"].dtype == jnp.float64
    assert fields["U_y"].dtype == jnp.float64
    assert fields["omega"].dtype == jnp.float64
