# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Parity tests for the Tier-2 restricted-FermiNet wavefunction.

We verify the four core claims of the Tier-2 restricted-FermiNet
derivation:

1. The closed-form per-electron column derivatives
   ``(phi(r_j), d phi / d r_j, d^2 phi / d r_j^2)``
   match ``jax.grad`` / ``jax.hessian`` of the same ``phi(r_j)`` at
   float64 ULP precision.
2. The closed-form Laplacian / |grad|^2 of ``log|det M|`` for the
   full multi-electron config matches ``jax.hessian`` / ``jax.grad``
   of :func:`tier2_log_abs_psi` to rel err :math:`\\le 10^{-10}` on
   random configurations.
3. The fused :func:`tier2_local_kinetic_energy` matches the
   autograd reference (sum of all per-electron Laplacian
   contributions plus ``-1/2 |grad log|psi||^2``) to the same
   tolerance.
4. The :func:`tier2_psi_fn` adapter is consumed cleanly by
   :func:`omnibias.jax.make_local_energy` -- i.e., the Tier-2
   wavefunction is a drop-in for the existing BO derivative
   pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.restricted import (  # noqa: E402
    Tier2Params,
    Tier2SymParams,
    _column_value_jac_hess,
    _features,
    _features_jac_and_hessian,
    _orbital_matrix,
    _orbital_matrix_sym,
    tier2_blocked_local_kinetic_energy,
    tier2_blocked_log_abs_psi,
    tier2_init_params,
    tier2_local_kinetic_energy,
    tier2_log_abs_psi,
    tier2_psi_fn,
    tier2_spin_blocked_init_params,
    tier2_value_grad_log_psi,
    tier2sym_blocked_local_kinetic_energy,
    tier2sym_blocked_log_abs_psi,
    tier2sym_init_params,
    tier2sym_local_kinetic_energy,
    tier2sym_log_abs_psi,
    tier2sym_psi_fn,
    tier2sym_spin_blocked_init_params,
    tier2sym_value_grad_log_psi,
)
from omnibias.jax import (  # noqa: E402
    coulomb_potential,
    make_local_energy,
)

RNG = np.random.default_rng(0)


def _rand_walker(n_e: int, atoms: jnp.ndarray, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    # walker = atoms[rand_atom] + 0.3 * randn(3)
    n_atoms = atoms.shape[0]
    out = []
    for _ in range(n_e):
        a = rng.integers(n_atoms)
        out.append(np.asarray(atoms[a]) + 0.3 * rng.normal(size=3))
    return jnp.asarray(np.concatenate(out), dtype=jnp.float64)


def test_features_jac_hess_parity_vs_jax():
    rng = np.random.default_rng(11)
    atoms = jnp.asarray(rng.normal(size=(3, 3)), dtype=jnp.float64)
    r_j = jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float64)
    f, J, H = _features_jac_and_hessian(r_j, atoms)

    def feat_fn(r):
        return _features(r, atoms)

    J_ref = jax.jacobian(feat_fn)(r_j)  # (n_feat, 3)
    H_ref = jax.hessian(feat_fn)(r_j)  # (n_feat, 3, 3)

    np.testing.assert_allclose(np.asarray(f), np.asarray(feat_fn(r_j)), atol=1e-14)
    np.testing.assert_allclose(np.asarray(J), np.asarray(J_ref), atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(H), np.asarray(H_ref), atol=1e-13, rtol=1e-13)


def test_column_value_grad_hess_parity_vs_jax():
    """Per-electron orbital column derivatives match jax.hessian."""
    params = tier2_init_params(n_atoms=3, n_orb=4, hidden=8, seed=2)

    rng = np.random.default_rng(33)
    r_j = jnp.asarray(rng.normal(size=(3,)), dtype=jnp.float64)
    phi, grad_phi, H_phi = _column_value_jac_hess(params, r_j)

    def phi_i(i, r):
        # i-th orbital evaluated as the i-th component of the column
        M_col = _orbital_matrix(
            params,
            jnp.concatenate([r, jnp.zeros(0)]),
            n_e=1,
        )  # (n_orb, 1)
        return M_col[i, 0]

    # phi via direct re-evaluation through _orbital_matrix
    M_ref = _orbital_matrix(params, r_j, n_e=1)  # (n_orb, 1)
    np.testing.assert_allclose(np.asarray(phi), np.asarray(M_ref[:, 0]), atol=1e-13, rtol=1e-13)

    for i in range(params.W_orb.shape[0]):
        g_ref = jax.grad(lambda r, i=i: phi_i(i, r))(r_j)
        H_ref = jax.hessian(lambda r, i=i: phi_i(i, r))(r_j)
        np.testing.assert_allclose(
            np.asarray(grad_phi[i]),
            np.asarray(g_ref),
            atol=1e-11,
            rtol=1e-11,
        )
        np.testing.assert_allclose(
            np.asarray(H_phi[i]),
            np.asarray(H_ref),
            atol=1e-10,
            rtol=1e-10,
        )


@pytest.mark.parametrize("n_e", [2, 3, 4])
def test_local_kinetic_parity_vs_jax_hessian(n_e: int):
    """Tier-2 closed-form T_loc matches jax.hessian on log|psi|."""
    n_atoms = 2
    params = tier2_init_params(
        n_atoms=n_atoms,
        n_orb=n_e,
        hidden=6,
        seed=4,
    )
    r_flat = _rand_walker(n_e, params.atoms, seed=5)

    T_closed = tier2_local_kinetic_energy(params, r_flat, n_e=n_e)

    def log_abs(r):
        return tier2_log_abs_psi(params, r, n_e=n_e)

    grad_log = jax.grad(log_abs)(r_flat)
    H_log = jax.hessian(log_abs)(r_flat)
    lap = jnp.trace(H_log)
    T_ref = -0.5 * (lap + jnp.sum(grad_log * grad_log))

    np.testing.assert_allclose(
        float(T_closed),
        float(T_ref),
        atol=5e-10,
        rtol=5e-10,
    )


def test_value_grad_log_psi_parity():
    """tier2_value_grad_log_psi matches jax.value_and_grad of the reference."""
    params = tier2_init_params(n_atoms=3, n_orb=4, hidden=8, seed=7)
    n_e = 4
    r_flat = _rand_walker(n_e, params.atoms, seed=12)

    log_closed, grad_closed = tier2_value_grad_log_psi(
        params,
        r_flat,
        n_e=n_e,
    )

    def log_abs(r):
        return tier2_log_abs_psi(params, r, n_e=n_e)

    log_ref, grad_ref = jax.value_and_grad(log_abs)(r_flat)

    np.testing.assert_allclose(
        float(log_closed),
        float(log_ref),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grad_closed),
        np.asarray(grad_ref),
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("n_alpha,n_beta", [(1, 1), (2, 2), (3, 2)])
def test_spin_blocked_local_kinetic_parity(n_alpha: int, n_beta: int):
    """Spin-blocked closed-form T_loc == jax.hessian on the same wavefunction."""
    n_atoms = 2
    params = tier2_spin_blocked_init_params(
        n_atoms=n_atoms,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=6,
        seed=15,
    )
    n_e = n_alpha + n_beta
    r_flat = _rand_walker(n_e, params.alpha.atoms, seed=16)

    T_closed = tier2_blocked_local_kinetic_energy(params, r_flat)

    def log_abs(r):
        return tier2_blocked_log_abs_psi(params, r)

    grad_log = jax.grad(log_abs)(r_flat)
    H_log = jax.hessian(log_abs)(r_flat)
    T_ref = -0.5 * (jnp.trace(H_log) + jnp.sum(grad_log * grad_log))

    np.testing.assert_allclose(
        float(T_closed),
        float(T_ref),
        atol=5e-10,
        rtol=5e-10,
    )


# ---------------------------------------------------------------------------
# Tier-2-full (symmetric-pool) parity tests
# ---------------------------------------------------------------------------


def _tier2sym_to_lite(params_sym: Tier2SymParams) -> Tier2Params:
    """Drop-in conversion when W1_b == 0 (used for the pool=0 parity test)."""
    return Tier2Params(
        W1=params_sym.W1_a,
        b1=params_sym.b1,
        W_orb=params_sym.W_orb,
        sigmas=params_sym.sigmas,
        alphas=params_sym.alphas,
        atoms=params_sym.atoms,
    )


@pytest.mark.parametrize("n_e", [2, 3, 4])
def test_tier2sym_local_kinetic_parity_vs_jax_hessian(n_e: int):
    """Tier-2-full closed-form T_loc matches jax.hessian on log|psi|."""
    n_atoms = 2
    params = tier2sym_init_params(
        n_atoms=n_atoms,
        n_orb=n_e,
        hidden=6,
        seed=21,
        pool_scale=1.0,
    )
    r_flat = _rand_walker(n_e, params.atoms, seed=22)

    T_closed = tier2sym_local_kinetic_energy(params, r_flat, n_e=n_e)

    def log_abs(r):
        return tier2sym_log_abs_psi(params, r, n_e=n_e)

    grad_log = jax.grad(log_abs)(r_flat)
    H_log = jax.hessian(log_abs)(r_flat)
    T_ref = -0.5 * (jnp.trace(H_log) + jnp.sum(grad_log * grad_log))

    np.testing.assert_allclose(
        float(T_closed),
        float(T_ref),
        atol=5e-10,
        rtol=5e-10,
    )


def _near_singular_walker(
    n_e: int, atoms: jnp.ndarray, seed: int, eps: float = 1e-5
) -> jnp.ndarray:
    """A walker with electrons 0 and 1 nearly coincident, so the orbital
    matrix has two near-equal columns -> near rank-deficient (ill-conditioned)
    ``M``. Exercises the linear-solve kinetic path against autodiff."""
    r = np.array(_rand_walker(n_e, atoms, seed), dtype=np.float64).reshape(n_e, 3)
    r[1] = r[0] + eps
    return jnp.asarray(r.reshape(-1), dtype=jnp.float64)


def test_tier2_local_kinetic_parity_on_near_singular_config() -> None:
    """Regression for ``inv -> solve``: the closed-form T_loc must still match
    autodiff when ``M`` is ill-conditioned (two electrons nearly coincident)."""
    n_e = 4
    params = tier2_init_params(n_atoms=2, n_orb=n_e, hidden=6, seed=4)
    r_flat = _near_singular_walker(n_e, params.atoms, seed=5)

    T_closed = tier2_local_kinetic_energy(params, r_flat, n_e=n_e)

    def log_abs(r):  # type: ignore[no-untyped-def]
        return tier2_log_abs_psi(params, r, n_e=n_e)

    grad_log = jax.grad(log_abs)(r_flat)
    H_log = jax.hessian(log_abs)(r_flat)
    T_ref = -0.5 * (jnp.trace(H_log) + jnp.sum(grad_log * grad_log))

    # Tolerance reflects the cond(M) ~ 1e5 floor shared by the closed form and
    # the slogdet-based autodiff reference; a broken solve would be off by O(1).
    np.testing.assert_allclose(float(T_closed), float(T_ref), rtol=1e-4, atol=1e-4)


def test_tier2sym_local_kinetic_parity_on_near_singular_config() -> None:
    """Same ill-conditioned guard for the Tier-2-full symmetric-pool path,
    whose Laplacian traces now go through ``solve(M, RHS)`` rather than an
    explicit ``M^{-1}``."""
    n_e = 4
    params = tier2sym_init_params(
        n_atoms=2, n_orb=n_e, hidden=6, seed=21, pool_scale=1.0
    )
    r_flat = _near_singular_walker(n_e, params.atoms, seed=22)

    T_closed = tier2sym_local_kinetic_energy(params, r_flat, n_e=n_e)

    def log_abs(r):  # type: ignore[no-untyped-def]
        return tier2sym_log_abs_psi(params, r, n_e=n_e)

    grad_log = jax.grad(log_abs)(r_flat)
    H_log = jax.hessian(log_abs)(r_flat)
    T_ref = -0.5 * (jnp.trace(H_log) + jnp.sum(grad_log * grad_log))

    # See note above: cond(M) ~ 1e5 floor; the guard catches O(1) regressions.
    np.testing.assert_allclose(float(T_closed), float(T_ref), rtol=1e-4, atol=1e-4)


def test_tier2sym_log_abs_psi_matches_orbital_matrix():
    """tier2sym_log_abs_psi == log|det(orbital matrix)|."""
    params = tier2sym_init_params(n_atoms=2, n_orb=3, hidden=5, seed=23)
    r_flat = _rand_walker(3, params.atoms, seed=24)
    M = _orbital_matrix_sym(params, r_flat, n_e=3)
    _sign, logdet = jnp.linalg.slogdet(M)
    np.testing.assert_allclose(
        float(tier2sym_log_abs_psi(params, r_flat, n_e=3)),
        float(logdet),
        atol=1e-13,
        rtol=1e-13,
    )


def test_tier2sym_reduces_to_tier2_lite_when_pool_zero():
    r"""``Tier-2-full`` with ``W1_b = 0`` matches Tier-2-lite exactly.

    This is a sanity check on the symmetric-pool closed-form
    derivation: at ``W1_b = 0`` the column-of-``M`` only depends
    on ``r_j``, so :func:`tier2sym_local_kinetic_energy` must
    bit-equal :func:`tier2_local_kinetic_energy`.
    """
    n_atoms, n_e = 3, 4
    params_sym = tier2sym_init_params(
        n_atoms=n_atoms,
        n_orb=n_e,
        hidden=6,
        seed=25,
        pool_scale=0.0,
    )
    assert jnp.allclose(params_sym.W1_b, 0.0)
    params_lite = _tier2sym_to_lite(params_sym)
    r_flat = _rand_walker(n_e, params_sym.atoms, seed=26)

    T_sym = tier2sym_local_kinetic_energy(params_sym, r_flat, n_e=n_e)
    T_lite = tier2_local_kinetic_energy(params_lite, r_flat, n_e=n_e)

    np.testing.assert_allclose(
        float(T_sym),
        float(T_lite),
        atol=1e-12,
        rtol=1e-12,
    )

    # log|psi| also matches at pool_scale=0
    L_sym = tier2sym_log_abs_psi(params_sym, r_flat, n_e=n_e)
    L_lite = tier2_log_abs_psi(params_lite, r_flat, n_e=n_e)
    np.testing.assert_allclose(
        float(L_sym),
        float(L_lite),
        atol=1e-13,
        rtol=1e-13,
    )


def test_tier2sym_value_grad_parity():
    """tier2sym_value_grad_log_psi matches jax.value_and_grad."""
    params = tier2sym_init_params(n_atoms=3, n_orb=4, hidden=8, seed=27)
    n_e = 4
    r_flat = _rand_walker(n_e, params.atoms, seed=28)
    log_closed, grad_closed = tier2sym_value_grad_log_psi(
        params,
        r_flat,
        n_e=n_e,
    )
    log_ref, grad_ref = jax.value_and_grad(
        lambda r: tier2sym_log_abs_psi(params, r, n_e=n_e),
    )(r_flat)
    np.testing.assert_allclose(
        float(log_closed),
        float(log_ref),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(grad_closed),
        np.asarray(grad_ref),
        atol=1e-11,
        rtol=1e-11,
    )


@pytest.mark.parametrize("n_alpha,n_beta", [(1, 1), (2, 2), (3, 2)])
def test_tier2sym_spin_blocked_local_kinetic_parity(
    n_alpha: int,
    n_beta: int,
):
    """Spin-blocked Tier-2-full T_loc matches jax.hessian."""
    n_atoms = 2
    params = tier2sym_spin_blocked_init_params(
        n_atoms=n_atoms,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=6,
        seed=29,
        pool_scale=1.0,
    )
    n_e = n_alpha + n_beta
    r_flat = _rand_walker(n_e, params.alpha.atoms, seed=30)

    T_closed = tier2sym_blocked_local_kinetic_energy(params, r_flat)

    def log_abs(r):
        return tier2sym_blocked_log_abs_psi(params, r)

    grad_log = jax.grad(log_abs)(r_flat)
    H_log = jax.hessian(log_abs)(r_flat)
    T_ref = -0.5 * (jnp.trace(H_log) + jnp.sum(grad_log * grad_log))

    np.testing.assert_allclose(
        float(T_closed),
        float(T_ref),
        atol=5e-10,
        rtol=5e-10,
    )


def test_tier2sym_psi_fn_drops_into_local_energy_pipeline():
    """The Tier-2-full wavefunction is a drop-in for make_local_energy."""
    n_e = 3
    n_atoms = 2
    params = tier2sym_init_params(
        n_atoms=n_atoms,
        n_orb=n_e,
        hidden=6,
        seed=31,
    )
    psi_fn = tier2sym_psi_fn(params, n_e)
    charges = jnp.array([1.0, 1.0], dtype=jnp.float64)

    def pot_fn(R, r):
        return coulomb_potential(R, r, charges, n_e=n_e)

    e_local = make_local_energy(psi_fn, pot_fn)

    R_flat = params.atoms.reshape(-1)
    r_flat = _rand_walker(n_e, params.atoms, seed=32)

    V = coulomb_potential(R_flat, r_flat, charges, n_e=n_e)
    T = tier2sym_local_kinetic_energy(params, r_flat, n_e=n_e)
    E_closed = T + V

    E_pipeline = e_local(R_flat, r_flat)
    np.testing.assert_allclose(
        float(E_closed),
        float(E_pipeline),
        atol=5e-9,
        rtol=5e-9,
    )


def test_tier2_psi_fn_drops_into_local_energy_pipeline():
    """The Tier-2 wavefunction is a drop-in for omnibias.jax.make_local_energy."""
    n_e = 3
    n_atoms = 2
    params = tier2_init_params(
        n_atoms=n_atoms,
        n_orb=n_e,
        hidden=6,
        seed=13,
    )
    psi_fn = tier2_psi_fn(params, n_e)
    charges = jnp.array([1.0, 1.0], dtype=jnp.float64)

    def pot_fn(R, r):
        return coulomb_potential(R, r, charges, n_e=n_e)

    e_local = make_local_energy(psi_fn, pot_fn)

    R_flat = params.atoms.reshape(-1)
    r_flat = _rand_walker(n_e, params.atoms, seed=14)

    V = coulomb_potential(R_flat, r_flat, charges, n_e=n_e)
    T = tier2_local_kinetic_energy(params, r_flat, n_e=n_e)
    E_closed = T + V

    E_pipeline = e_local(R_flat, r_flat)
    np.testing.assert_allclose(
        float(E_closed),
        float(E_pipeline),
        atol=5e-9,
        rtol=5e-9,
    )
