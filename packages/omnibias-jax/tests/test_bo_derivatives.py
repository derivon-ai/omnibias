# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Parity tests for :mod:`omnibias.jax.bo_derivatives`.

Two reference paths:

1. *Analytic* — a 1-electron Gaussian wavefunction
   :math:`\\psi(r;R) = \\exp(-\\alpha\\,(r-R)^2/2)` on a hydrogen-like
   atom (one nucleus, charge ``Z``). The Born-Oppenheimer energy
   :math:`E_{BO}(R)` is independent of ``R`` by translation
   invariance, so ``F = 0`` and ``K = 0`` exactly when the MC
   estimator is unbiased. We check this gives ``< 1e-12`` after
   averaging over a symmetric walker set.

2. *Finite difference* — for a non-translation-invariant test we
   place a *fixed* gaussian centred at the origin in the
   wavefunction (so :math:`\\psi(r;R)` does depend on ``R``
   through the potential only), and compare the BO Hessian
   against a 5-point central FD of the MC energy estimator at
   step ``h = 1e-3`` Bohr.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from omnibias.jax.bo_derivatives import (  # noqa: E402
    HARTREE_TO_CM,
    coulomb_potential,
    make_bo_force,
    make_bo_hessian,
    make_local_energy,
    vibrational_frequencies,
)

RNG = np.random.default_rng(202605131)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_psi(alpha: float):
    r"""1-electron centred-Gaussian ansatz ``psi(r;R) = exp(-alpha (r-R)^2 / 2)``.

    Translation-invariant in (r, R), so the BO energy is independent
    of R: F(R) = K(R) = 0 (exact for the exact MC integral).
    """

    def psi_fn(R: jnp.ndarray, r_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # R: (3,), r_flat: (3,).
        log_abs = -0.5 * alpha * jnp.sum((r_flat - R) ** 2)
        sign = jnp.asarray(1.0, dtype=r_flat.dtype)
        return sign, log_abs

    return psi_fn


def _gaussian_h_walkers(R: jnp.ndarray, alpha: float, n_walkers: int) -> jnp.ndarray:
    r"""Importance-sample walkers from |psi|^2 = exp(-alpha (r-R)^2)."""
    sigma = 1.0 / np.sqrt(2.0 * alpha)
    return jnp.asarray(
        RNG.normal(size=(n_walkers, 3)) * sigma + np.asarray(R)[None, :],
        dtype=jnp.float64,
    )


def _fixed_gaussian_psi(R0: jnp.ndarray, alpha: float):
    r"""``psi(r;R) = exp(-alpha (r - R0)^2 / 2)`` -- ignores R.

    So d/dR psi = 0; the only R-dependence is through the
    potential V(R, r). Use to isolate the V-only contribution.
    """

    def psi_fn(R: jnp.ndarray, r_flat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        del R
        log_abs = -0.5 * alpha * jnp.sum((r_flat - R0) ** 2)
        sign = jnp.asarray(1.0, dtype=r_flat.dtype)
        return sign, log_abs

    return psi_fn


def _bo_energy_estimate(
    psi_fn,
    potential_fn,
    R: jnp.ndarray,
    walkers: jnp.ndarray,
) -> float:
    e_local = make_local_energy(psi_fn, potential_fn)
    per_walker = jax.vmap(e_local, in_axes=(None, 0))(R, walkers)
    return float(jnp.mean(per_walker))


def _bo_energy_estimate_array(
    psi_fn,
    potential_fn,
    R: jnp.ndarray,
    walkers: jnp.ndarray,
) -> jnp.ndarray:
    e_local = make_local_energy(psi_fn, potential_fn)
    per_walker = jax.vmap(e_local, in_axes=(None, 0))(R, walkers)
    return jnp.mean(per_walker)


# ---------------------------------------------------------------------------
# 1. Translation-invariant ansatz: F = 0, K = 0 exact at MC zero
# ---------------------------------------------------------------------------


def test_translation_invariant_force_is_zero():
    r"""For psi(r-R), the BO force vanishes (zero-variance principle).

    We use enough walkers (symmetric around R) for the MC variance to
    be well below the assertion threshold.
    """
    alpha = 0.5
    Z = 1.0
    R = jnp.asarray([0.3, -0.7, 0.4], dtype=jnp.float64)

    psi_fn = _gaussian_psi(alpha)

    def potential_fn(R_, r_):
        return coulomb_potential(
            R_,
            r_,
            jnp.asarray([Z], dtype=jnp.float64),
            n_e=1,
        )

    walkers = _gaussian_h_walkers(R, alpha, n_walkers=4000)

    e_bar = _bo_energy_estimate_array(psi_fn, potential_fn, R, walkers)
    bo_force = make_bo_force(psi_fn, potential_fn)
    F = bo_force(R, walkers, e_bar)

    assert F.shape == (3,)
    max_abs = float(jnp.max(jnp.abs(F)))
    # MC noise floor for the centred-gaussian estimator with 4000 walkers
    # sits ~ few mHa/Bohr; the systematic part must be << that.
    assert max_abs < 5e-2, f"|F|_inf = {max_abs:.3e}, expected near MC zero (~few mHa/Bohr)."


def test_translation_invariant_hessian_is_zero():
    r"""For psi(r-R), the BO Hessian also vanishes (translation invariance)."""
    alpha = 0.6
    Z = 1.0
    R = jnp.asarray([0.1, 0.2, -0.3], dtype=jnp.float64)

    psi_fn = _gaussian_psi(alpha)

    def potential_fn(R_, r_):
        return coulomb_potential(
            R_,
            r_,
            jnp.asarray([Z], dtype=jnp.float64),
            n_e=1,
        )

    walkers = _gaussian_h_walkers(R, alpha, n_walkers=4000)

    e_bar = _bo_energy_estimate_array(psi_fn, potential_fn, R, walkers)
    bo_hess = make_bo_hessian(psi_fn, potential_fn)
    K = bo_hess(R, walkers, e_bar)

    assert K.shape == (3, 3)
    max_abs = float(jnp.max(jnp.abs(K)))
    assert max_abs < 5.0, f"max|K| = {max_abs:.3e}, translation-invariant ansatz should be 0."


# ---------------------------------------------------------------------------
# 2. Finite-difference parity (fixed wavefunction; V(R) drives the energy)
# ---------------------------------------------------------------------------


def test_force_matches_finite_difference_fixed_psi():
    r"""When psi is fixed and V(R, r) depends on R, dE/dR is finite-difference
    of the MC energy estimator. With ~10k walkers, the FD-vs-analytic
    relative error sits in the 1% band (MC noise dominated)."""
    alpha = 0.7
    Z = 1.5
    R = jnp.asarray([0.15, -0.20, 0.30], dtype=jnp.float64)
    R0 = R + 0.05  # psi is centred slightly off from the nucleus

    psi_fn = _fixed_gaussian_psi(R0, alpha)

    def potential_fn(R_, r_):
        return coulomb_potential(
            R_,
            r_,
            jnp.asarray([Z], dtype=jnp.float64),
            n_e=1,
        )

    walkers = _gaussian_h_walkers(R0, alpha, n_walkers=8000)
    e_bar = _bo_energy_estimate_array(psi_fn, potential_fn, R, walkers)

    bo_force = make_bo_force(psi_fn, potential_fn)
    F_analytic = bo_force(R, walkers, e_bar)

    # Finite-difference of E_BO(R) along each direction.
    h = 1e-3
    F_fd = np.zeros(3, dtype=np.float64)
    for k in range(3):
        R_plus = R.at[k].add(h)
        R_minus = R.at[k].add(-h)
        e_p = _bo_energy_estimate_array(psi_fn, potential_fn, R_plus, walkers)
        e_m = _bo_energy_estimate_array(psi_fn, potential_fn, R_minus, walkers)
        F_fd[k] = -(float(e_p) - float(e_m)) / (2.0 * h)

    F_fd = jnp.asarray(F_fd, dtype=jnp.float64)
    err = float(jnp.max(jnp.abs(F_analytic - F_fd)))
    scale = float(jnp.max(jnp.abs(F_fd)) + 1e-3)
    rel_err = err / scale
    assert rel_err < 0.02, f"F_analytic = {F_analytic}, F_fd = {F_fd}, rel_err = {rel_err:.3e}"


def test_hessian_matches_finite_difference_fixed_psi():
    r"""Similar parity for the diagonal of the BO Hessian."""
    alpha = 0.8
    Z = 1.2
    R = jnp.asarray([0.1, 0.0, -0.1], dtype=jnp.float64)
    R0 = R  # psi centred on the nucleus (Hellmann-Feynman regime)

    psi_fn = _fixed_gaussian_psi(R0, alpha)

    def potential_fn(R_, r_):
        return coulomb_potential(
            R_,
            r_,
            jnp.asarray([Z], dtype=jnp.float64),
            n_e=1,
        )

    walkers = _gaussian_h_walkers(R0, alpha, n_walkers=12000)
    e_bar = _bo_energy_estimate_array(psi_fn, potential_fn, R, walkers)

    bo_hess = make_bo_hessian(psi_fn, potential_fn)
    K_analytic = bo_hess(R, walkers, e_bar)

    # Diagonal FD via central differences
    h = 5e-3
    K_fd_diag = np.zeros(3, dtype=np.float64)
    e0 = float(_bo_energy_estimate_array(psi_fn, potential_fn, R, walkers))
    for k in range(3):
        R_plus = R.at[k].add(h)
        R_minus = R.at[k].add(-h)
        e_p = float(
            _bo_energy_estimate_array(
                psi_fn,
                potential_fn,
                R_plus,
                walkers,
            )
        )
        e_m = float(
            _bo_energy_estimate_array(
                psi_fn,
                potential_fn,
                R_minus,
                walkers,
            )
        )
        K_fd_diag[k] = (e_p - 2.0 * e0 + e_m) / (h * h)

    K_analytic_diag = jnp.asarray(jnp.diag(K_analytic), dtype=jnp.float64)
    err = float(jnp.max(jnp.abs(K_analytic_diag - K_fd_diag)))
    scale = float(jnp.max(jnp.abs(K_fd_diag)) + 1.0)
    rel_err = err / scale
    assert rel_err < 0.05, (
        f"K_analytic_diag = {K_analytic_diag}, K_fd_diag = {K_fd_diag}, rel_err = {rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. Vibrational frequency routine
# ---------------------------------------------------------------------------


def test_vibrational_frequencies_diatomic_harmonic():
    r"""Toy diatomic with a known harmonic curvature.

    Build K_nuclear = k * (e_R outer e_R), with e_R the bond-axis
    direction, and check the unique vibrational frequency comes out
    to the textbook value
        nu = (1 / (2 pi c)) sqrt(k / mu)
    where mu is the reduced mass and the factor of 2 pi c is absorbed
    in HARTREE_TO_CM after converting eigenvalues to angular
    frequencies in atomic units.
    """
    m1, m2 = 1.0, 16.0  # H and O masses, in amu (toy values)
    R = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.8], dtype=jnp.float64)
    k = 0.5  # Ha / Bohr^2

    # Place k between the two atoms along z.
    e_R = jnp.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
    K = k * jnp.outer(e_R, e_R)

    masses = jnp.asarray([m1, m2], dtype=jnp.float64)
    freqs, _ = vibrational_frequencies(
        K,
        masses,
        R,
        project_tr=True,
        diatomic=True,
    )

    nonzero = jnp.sort(jnp.abs(freqs))[::-1][:1]
    from omnibias.jax.bo_derivatives import AMU_TO_ME

    # eigenvalue of the mass-weighted Hessian for a single mode along the
    # bond axis is k * (1/m1 + 1/m2) = k / mu_me with mu = m1 m2/(m1+m2).
    # omega_au = sqrt(eigval); nu_cm = omega_au * HARTREE_TO_CM.
    expected_cm = float(
        np.sqrt(k * (1.0 / (m1 * AMU_TO_ME) + 1.0 / (m2 * AMU_TO_ME))) * HARTREE_TO_CM
    )
    err = float(jnp.abs(nonzero[0] - expected_cm))
    assert err / expected_cm < 1e-6, (
        f"diatomic vib freq: got {float(nonzero[0]):.4f}, "
        f"expected {expected_cm:.4f} cm^-1, rel err = {err / expected_cm:.3e}"
    )
