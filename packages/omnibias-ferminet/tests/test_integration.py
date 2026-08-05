# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Parity tests for :mod:`omnibias.jax.ferminet_integration`.

These tests stand on the same JAX float64 substrate as
``tests/test_jax_folx_compat.py`` and verify the four claims of
the FermiNet envelope-bridge derivation:

1. The closed-form envelope :func:`envelope_value_grad_hessian`
   matches ``jax.grad`` / ``jax.hessian`` at relative error
   :math:`\\le 2 \\times 10^{-15}` across random
   ``(orbital, electron, atom)`` configurations.

2. With ``BackflowParams(a=0, gamma_pre=...)``,
   :func:`apply_optional_backflow` reproduces the identity
   transform ``q = r, J = I, lap_q = 0`` bit-for-bit (the
   warm-start contract).

3. With non-zero ``a``, :func:`apply_optional_backflow` matches
   ``jax.jacobian`` and the per-component Laplacian of
   ``q(r)`` to floating-point precision.

4. The mock FermiNet local kinetic energy
   :func:`mock_local_kinetic_omnibias_envelope` matches the
   reference ``-0.5 (∇² + ||∇||²) log|psi|`` evaluated by
   ``jax.grad`` / ``jax.hessian`` on
   :func:`mock_ferminet_log_abs_psi`.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.integration import (  # noqa: E402
    BackflowParams,
    MockFermiNetParams,
    apply_optional_backflow,
    envelope_value_grad_hessian,
    make_omnibias_envelope_local_kinetic_energy,
    make_omnibias_tier2_local_kinetic_energy,
    mock_ferminet_log_abs_psi,
    mock_local_kinetic_omnibias_envelope,
)
from omnibias.ferminet.restricted import (  # noqa: E402
    tier2sym_blocked_local_kinetic_energy,
    tier2sym_blocked_log_abs_psi,
    tier2sym_init_params,
    tier2sym_local_kinetic_energy,
    tier2sym_log_abs_psi,
    tier2sym_spin_blocked_init_params,
)

RNG = np.random.default_rng(0)


def _to_jax(*arrays):
    return tuple(jnp.asarray(a, dtype=jnp.float64) for a in arrays)


def _random_envelope_problem(n_orb: int, n_e: int, n_atoms: int):
    r = RNG.normal(size=(n_e, 3)) * 1.5
    atoms = RNG.normal(size=(n_atoms, 3)) * 0.5
    sigmas = RNG.normal(size=(n_orb, n_atoms)) * 0.6
    alphas = 0.3 + RNG.uniform(size=(n_orb, n_atoms)) * 1.4
    return _to_jax(r, atoms, sigmas, alphas)


# ---------------------------------------------------------------------------
# 1. envelope_value_grad_hessian vs jax.grad / jax.hessian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_orb,n_e,n_atoms", [(3, 3, 4), (5, 4, 2)])
def test_envelope_value_grad_hessian_parity_vs_jax(n_orb, n_e, n_atoms):
    r, atoms, sigmas, alphas = _random_envelope_problem(n_orb, n_e, n_atoms)
    env, grad, H = envelope_value_grad_hessian(r, atoms, sigmas, alphas)

    def env_scalar(r_j, sigma_i, alpha_i):
        delta = r_j[None, :] - atoms
        d = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
        return jnp.sum(sigma_i * jnp.exp(-alpha_i * d))

    grad_ref_fn = jax.grad(env_scalar, argnums=0)
    hess_ref_fn = jax.hessian(env_scalar, argnums=0)

    for i in range(n_orb):
        for j in range(n_e):
            v_ref = env_scalar(r[j], sigmas[i], alphas[i])
            g_ref = grad_ref_fn(r[j], sigmas[i], alphas[i])
            H_ref = hess_ref_fn(r[j], sigmas[i], alphas[i])

            scale_v = max(1.0, float(abs(v_ref)))
            scale_g = max(1.0, float(jnp.linalg.norm(g_ref)))
            scale_H = max(1.0, float(jnp.linalg.norm(H_ref)))

            assert abs(float(env[i, j]) - float(v_ref)) / scale_v < 2e-15
            assert float(jnp.linalg.norm(grad[i, j] - g_ref)) / scale_g < 2e-15
            assert float(jnp.linalg.norm(H[i, j] - H_ref)) / scale_H < 2e-15


# ---------------------------------------------------------------------------
# 2. apply_optional_backflow: identity at a = 0
# ---------------------------------------------------------------------------


def test_backflow_identity_at_a_zero():
    n_e = 4
    n_atoms = 3
    r = jnp.asarray(RNG.normal(size=(n_e, 3)), dtype=jnp.float64)
    atoms = jnp.asarray(RNG.normal(size=(n_atoms, 3)), dtype=jnp.float64)
    bf = BackflowParams(
        a=jnp.zeros((n_atoms,), dtype=jnp.float64),
        gamma_pre=jnp.zeros((n_atoms,), dtype=jnp.float64),
    )
    q, J, lap_q = apply_optional_backflow(r, atoms, bf)
    # Exact identity (no autograd) so the warm-start contract holds.
    assert float(jnp.max(jnp.abs(q - r))) == 0.0
    eye_batch = jnp.broadcast_to(jnp.eye(3, dtype=r.dtype), (n_e, 3, 3))
    assert float(jnp.max(jnp.abs(J - eye_batch))) == 0.0
    assert float(jnp.max(jnp.abs(lap_q))) == 0.0


def test_backflow_none_equals_identity_call():
    n_e = 4
    n_atoms = 3
    r = jnp.asarray(RNG.normal(size=(n_e, 3)), dtype=jnp.float64)
    atoms = jnp.asarray(RNG.normal(size=(n_atoms, 3)), dtype=jnp.float64)
    q_none, J_none, lap_none = apply_optional_backflow(r, atoms, None)
    assert float(jnp.max(jnp.abs(q_none - r))) == 0.0
    eye_batch = jnp.broadcast_to(jnp.eye(3, dtype=r.dtype), (n_e, 3, 3))
    assert float(jnp.max(jnp.abs(J_none - eye_batch))) == 0.0
    assert float(jnp.max(jnp.abs(lap_none))) == 0.0


# ---------------------------------------------------------------------------
# 3. apply_optional_backflow: closed-form (J, lap_q) vs jax.jacobian / hessian
# ---------------------------------------------------------------------------


def test_backflow_chain_rule_parity_vs_jax():
    n_e = 3
    n_atoms = 4
    r = jnp.asarray(RNG.normal(size=(n_e, 3)) * 1.2, dtype=jnp.float64)
    atoms = jnp.asarray(RNG.normal(size=(n_atoms, 3)) * 0.5, dtype=jnp.float64)
    bf = BackflowParams(
        a=jnp.asarray(RNG.normal(size=(n_atoms,)) * 0.1, dtype=jnp.float64),
        gamma_pre=jnp.asarray(
            RNG.normal(size=(n_atoms,)) * 0.5,
            dtype=jnp.float64,
        ),
    )

    def per_electron_q(r_j, atoms_, bf_):
        gamma = jax.nn.softplus(bf_.gamma_pre)
        delta = r_j[None, :] - atoms_
        d = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
        e = jnp.exp(-gamma * d)
        return r_j + jnp.sum((bf_.a * e)[:, None] * delta, axis=0)

    q, J, lap_q = apply_optional_backflow(r, atoms, bf)
    for j in range(n_e):
        q_ref = per_electron_q(r[j], atoms, bf)
        J_ref = jax.jacobian(per_electron_q, argnums=0)(r[j], atoms, bf)
        H_ref = jax.hessian(per_electron_q, argnums=0)(r[j], atoms, bf)
        lap_ref = jnp.trace(H_ref, axis1=1, axis2=2)  # per component

        scale_q = max(1.0, float(jnp.linalg.norm(q_ref)))
        scale_J = max(1.0, float(jnp.linalg.norm(J_ref)))
        scale_lap = max(1.0, float(jnp.linalg.norm(lap_ref)))

        assert float(jnp.linalg.norm(q[j] - q_ref)) / scale_q < 5e-15
        assert float(jnp.linalg.norm(J[j] - J_ref)) / scale_J < 5e-15
        assert float(jnp.linalg.norm(lap_q[j] - lap_ref)) / scale_lap < 5e-15


# ---------------------------------------------------------------------------
# 4. mock FermiNet local kinetic energy vs jax.hessian reference
# ---------------------------------------------------------------------------


def _random_mock_params(n_orb: int, n_e: int, n_atoms: int):
    sigmas = jnp.asarray(
        RNG.normal(size=(n_orb, n_atoms)) * 0.4 + 1.5,
        dtype=jnp.float64,
    )
    alphas = jnp.asarray(
        0.4 + RNG.uniform(size=(n_orb, n_atoms)) * 0.8,
        dtype=jnp.float64,
    )
    g_weight = jnp.asarray(
        RNG.normal(size=(n_orb, n_e, 3)) * 0.2,
        dtype=jnp.float64,
    )
    g_bias = jnp.asarray(
        RNG.normal(size=(n_orb, n_e)) * 0.05 + 1.0,
        dtype=jnp.float64,
    )
    return MockFermiNetParams(
        sigmas=sigmas,
        alphas=alphas,
        g_weight=g_weight,
        g_bias=g_bias,
    )


@pytest.mark.parametrize("n_e", [3, 4])
def test_mock_local_kinetic_energy_parity_vs_jax(n_e):
    n_orb = n_e
    n_atoms = 2
    params = _random_mock_params(n_orb, n_e, n_atoms)
    r = jnp.asarray(
        RNG.normal(size=(n_e, 3)) * 0.7,
        dtype=jnp.float64,
    )
    atoms = jnp.asarray(
        RNG.normal(size=(n_atoms, 3)) * 0.3,
        dtype=jnp.float64,
    )

    T_closed = mock_local_kinetic_omnibias_envelope(params, r, atoms)

    def log_psi_flat(r_flat):
        r_reshaped = r_flat.reshape((n_e, 3))
        return mock_ferminet_log_abs_psi(params, r_reshaped, atoms)

    r_flat = r.reshape((-1,))
    grad_ref = jax.grad(log_psi_flat)(r_flat)
    H_ref = jax.hessian(log_psi_flat)(r_flat)
    lap_ref = jnp.trace(H_ref)
    T_ref = -0.5 * (lap_ref + jnp.sum(grad_ref * grad_ref))

    rel_err = float(abs(T_closed - T_ref)) / max(1.0, float(abs(T_ref)))
    assert rel_err < 1e-12, (
        f"T_closed = {float(T_closed):.12e}  T_ref = {float(T_ref):.12e}  rel_err = {rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# 5. Production factory plumbs end-to-end (autograd-equivalent Tier-1 path)
# ---------------------------------------------------------------------------


def test_production_factory_kinetic_energy_matches_jax_reference():
    r"""``make_omnibias_envelope_local_kinetic_energy`` produces the same
    ``T_loc = -0.5 (\nabla^2 + ||\nabla||^2) \log|\psi|`` as a hand-rolled
    ``jax.grad`` + ``jax.hessian`` reference, on a FermiNet-shaped scalar
    wavefunction. This is the Tier-1 plumbing contract documented in
    ``omnibias.jax.ferminet_integration.make_omnibias_envelope_local_kinetic_energy``.
    """
    n_e = 4
    n_atoms = 2
    n_orb = n_e
    params = MockFermiNetParams(
        sigmas=jnp.asarray(
            RNG.normal(size=(n_orb, n_atoms)) * 0.3 + 1.3,
            dtype=jnp.float64,
        ),
        alphas=jnp.asarray(
            0.4 + RNG.uniform(size=(n_orb, n_atoms)) * 0.6,
            dtype=jnp.float64,
        ),
        g_weight=jnp.asarray(
            RNG.normal(size=(n_orb, n_e, 3)) * 0.2,
            dtype=jnp.float64,
        ),
        g_bias=jnp.asarray(
            RNG.normal(size=(n_orb, n_e)) * 0.05 + 1.0,
            dtype=jnp.float64,
        ),
    )
    atoms = jnp.asarray(RNG.normal(size=(n_atoms, 3)) * 0.3, dtype=jnp.float64)

    # FermiNet-shaped network: takes (params, positions_flat, spins, atoms, charges)
    # and returns (sign, log_abs).
    def fermi_like(p, positions_flat, spins, atoms_, charges):
        del spins, charges
        r = positions_flat.reshape((n_e, 3))
        env, _, _ = envelope_value_grad_hessian(r, atoms_, p.sigmas, p.alphas)
        g = jnp.einsum("ijk,jk->ij", p.g_weight, r) + p.g_bias
        A = env * g
        sign, logdet = jnp.linalg.slogdet(A)
        return sign, logdet

    # Mock FermiNetData-like object holding what local_kinetic_energy needs.
    from types import SimpleNamespace

    data = SimpleNamespace(
        positions=jnp.asarray(
            RNG.normal(size=(n_e * 3,)) * 0.5,
            dtype=jnp.float64,
        ),
        spins=jnp.asarray([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64),
        atoms=atoms,
        charges=jnp.asarray([1.0, 1.0], dtype=jnp.float64),
    )

    kinetic = make_omnibias_envelope_local_kinetic_energy(
        fermi_like,
        complex_output=False,
    )
    T_omnibias = kinetic(params, data)

    # Reference via jax.grad / jax.hessian on log|psi|.
    def log_abs(positions_flat):
        return fermi_like(
            params,
            positions_flat,
            data.spins,
            data.atoms,
            data.charges,
        )[1]

    grad_ref = jax.grad(log_abs)(data.positions)
    H_ref = jax.hessian(log_abs)(data.positions)
    lap_ref = jnp.trace(H_ref)
    T_ref = -0.5 * (lap_ref + jnp.sum(grad_ref * grad_ref))

    rel_err = float(abs(T_omnibias - T_ref)) / max(1.0, float(abs(T_ref)))
    assert rel_err < 1e-12, (
        f"T_omnibias = {float(T_omnibias):.12e}  "
        f"T_ref = {float(T_ref):.12e}  rel_err = {rel_err:.3e}"
    )


# ---------------------------------------------------------------------------
# 6. Tier-2 production factory: closed-form path
# ---------------------------------------------------------------------------


def _fake_data(positions_flat, atoms, charges, n_e):
    from types import SimpleNamespace

    return SimpleNamespace(
        positions=positions_flat,
        spins=jnp.asarray([1.0] * n_e, dtype=jnp.float64),
        atoms=atoms,
        charges=charges,
    )


def test_tier2_factory_closed_form_matches_omnibias_kernel():
    r"""``make_omnibias_tier2_local_kinetic_energy`` invokes the
    closed-form ``local_kinetic_fn`` when provided.

    Sanity check: with ``local_kinetic_fn`` set to
    :func:`tier2sym_local_kinetic_energy` (closure over ``n_e``),
    the factory must return exactly the same scalar as a direct
    call to the kernel. This pins the adapter as a thin wrapper
    around the omnibias closed-form path -- the upstream PR can
    therefore land the ``elif`` plumbing without worrying about
    extra autograd in the kinetic-energy hot loop.
    """
    n_e = 3
    n_atoms = 2
    params = tier2sym_init_params(
        n_atoms=n_atoms,
        n_orb=n_e,
        hidden=6,
        seed=101,
        pool_scale=1.0,
    )
    positions = jnp.asarray(
        RNG.normal(size=(n_e * 3,)) * 0.5,
        dtype=jnp.float64,
    )
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    data = _fake_data(positions, params.atoms, charges, n_e)

    def fermi_like(p, positions_flat, spins, atoms_, charges_):
        del spins, charges_
        p_R = p._replace(atoms=atoms_)
        return jnp.array(1.0), tier2sym_log_abs_psi(p_R, positions_flat, n_e)

    def local_kinetic_fn(p, pos):
        return tier2sym_local_kinetic_energy(p, pos, n_e=n_e)

    kinetic = make_omnibias_tier2_local_kinetic_energy(
        fermi_like,
        complex_output=False,
        local_kinetic_fn=local_kinetic_fn,
    )

    T_factory = kinetic(params, data)
    T_direct = tier2sym_local_kinetic_energy(params, positions, n_e=n_e)
    np.testing.assert_allclose(
        float(T_factory),
        float(T_direct),
        atol=1e-13,
        rtol=1e-13,
    )

    # And the factory matches jax.hessian on log|psi| (closed-form parity).
    def log_abs(positions_flat):
        return tier2sym_log_abs_psi(params, positions_flat, n_e)

    grad_ref = jax.grad(log_abs)(positions)
    H_ref = jax.hessian(log_abs)(positions)
    T_ref = -0.5 * (jnp.trace(H_ref) + jnp.sum(grad_ref * grad_ref))
    rel_err = float(abs(T_factory - T_ref)) / max(1.0, float(abs(T_ref)))
    assert rel_err < 5e-10


def test_tier2_factory_spin_blocked_closed_form():
    r"""Production factory with the spin-blocked Tier-2-full kinetic energy."""
    n_alpha, n_beta = 2, 2
    n_e = n_alpha + n_beta
    n_atoms = 1
    params = tier2sym_spin_blocked_init_params(
        n_atoms=n_atoms,
        n_alpha=n_alpha,
        n_beta=n_beta,
        hidden=6,
        seed=103,
        pool_scale=0.7,
    )
    positions = jnp.asarray(
        RNG.normal(size=(n_e * 3,)) * 0.6,
        dtype=jnp.float64,
    )
    charges = jnp.asarray([4.0], dtype=jnp.float64)
    data = _fake_data(positions, params.alpha.atoms, charges, n_e)

    def fermi_like(p, positions_flat, spins, atoms_, charges_):
        del spins, charges_
        p_R = p._replace(
            alpha=p.alpha._replace(atoms=atoms_),
            beta=p.beta._replace(atoms=atoms_),
        )
        return jnp.array(1.0), tier2sym_blocked_log_abs_psi(p_R, positions_flat)

    kinetic = make_omnibias_tier2_local_kinetic_energy(
        fermi_like,
        complex_output=False,
        local_kinetic_fn=tier2sym_blocked_local_kinetic_energy,
    )
    T_factory = kinetic(params, data)
    T_direct = tier2sym_blocked_local_kinetic_energy(params, positions)
    np.testing.assert_allclose(
        float(T_factory),
        float(T_direct),
        atol=1e-13,
        rtol=1e-13,
    )

    def log_abs(positions_flat):
        return tier2sym_blocked_log_abs_psi(params, positions_flat)

    grad_ref = jax.grad(log_abs)(positions)
    H_ref = jax.hessian(log_abs)(positions)
    T_ref = -0.5 * (jnp.trace(H_ref) + jnp.sum(grad_ref * grad_ref))
    rel_err = float(abs(T_factory - T_ref)) / max(1.0, float(abs(T_ref)))
    assert rel_err < 5e-10


def test_tier2_factory_autograd_fallback_matches_envelope_factory():
    r"""``make_omnibias_tier2_local_kinetic_energy`` with
    ``local_kinetic_fn=None`` is bit-stable to
    :func:`make_omnibias_envelope_local_kinetic_energy` -- both
    route through the same forward-Laplacian autograd path. This
    is the contract that lets the upstream ``elif`` branch land
    immediately even on FermiNet networks that are NOT yet
    Tier-2-shaped.
    """
    n_e = 3
    n_atoms = 2
    n_orb = n_e
    params = MockFermiNetParams(
        sigmas=jnp.asarray(
            RNG.normal(size=(n_orb, n_atoms)) * 0.3 + 1.3,
            dtype=jnp.float64,
        ),
        alphas=jnp.asarray(
            0.4 + RNG.uniform(size=(n_orb, n_atoms)) * 0.6,
            dtype=jnp.float64,
        ),
        g_weight=jnp.asarray(
            RNG.normal(size=(n_orb, n_e, 3)) * 0.2,
            dtype=jnp.float64,
        ),
        g_bias=jnp.asarray(
            RNG.normal(size=(n_orb, n_e)) * 0.05 + 1.0,
            dtype=jnp.float64,
        ),
    )
    atoms = jnp.asarray(RNG.normal(size=(n_atoms, 3)) * 0.3, dtype=jnp.float64)
    positions = jnp.asarray(
        RNG.normal(size=(n_e * 3,)) * 0.5,
        dtype=jnp.float64,
    )
    charges = jnp.asarray([1.0, 1.0], dtype=jnp.float64)
    data = _fake_data(positions, atoms, charges, n_e)

    def fermi_like(p, positions_flat, spins, atoms_, charges_):
        del spins, charges_
        r = positions_flat.reshape((n_e, 3))
        env, _, _ = envelope_value_grad_hessian(r, atoms_, p.sigmas, p.alphas)
        g = jnp.einsum("ijk,jk->ij", p.g_weight, r) + p.g_bias
        A = env * g
        sign, logdet = jnp.linalg.slogdet(A)
        return sign, logdet

    kin_tier1 = make_omnibias_envelope_local_kinetic_energy(
        fermi_like,
        complex_output=False,
    )
    kin_tier2_fallback = make_omnibias_tier2_local_kinetic_energy(
        fermi_like,
        complex_output=False,
        local_kinetic_fn=None,
    )

    T1 = kin_tier1(params, data)
    T2 = kin_tier2_fallback(params, data)
    np.testing.assert_allclose(
        float(T1),
        float(T2),
        atol=1e-13,
        rtol=1e-13,
    )
