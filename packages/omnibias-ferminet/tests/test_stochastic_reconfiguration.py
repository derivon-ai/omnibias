# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Tests for the stochastic-reconfiguration (SR) VMC optimizer step.

Covers, in order: (1) the ``S`` / ``g`` Monte-Carlo estimators against a
hand-computable fixed sample set (plain numpy, cross-checked against the
JAX implementation), (2) shape/validation guards, (3) that
:func:`stochastic_reconfiguration_step` calls
:mod:`omnibias.curvature.natural_gradient`'s ``damped_solve`` /
``natural_gradient_step`` correctly rather than reimplementing them, (4) the
toy harmonic-oscillator SR convergence demonstration (a single trainable
variational-width parameter, deterministic, fixed seed/steps/tolerance), (5)
correct handling of a multi-leaf parameter pytree, and (6) ``jax.jit``
safety for the driver.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.curvature.natural_gradient import (  # noqa: E402
    damped_solve,
    natural_gradient_step,
)
from omnibias.ferminet.sampling import (  # noqa: E402
    autodiff_grad_and_laplacian,
    kinetic_energy_from_grad_lap,
    metropolis_sample,
)
from omnibias.ferminet.stochastic_reconfiguration import (  # noqa: E402
    StochasticReconfigurationResult,
    sr_energy_gradient,
    sr_overlap_matrix,
    stochastic_reconfiguration_step,
)


# ---------------------------------------------------------------------------
# 1. Hand-computable S / g estimators
# ---------------------------------------------------------------------------


class TestOverlapMatrixAndEnergyGradient:
    """Cross-check :func:`sr_overlap_matrix` / :func:`sr_energy_gradient`
    against a plain-numpy computation on a tiny fixed sample set."""

    _O = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 2.0],
        ]
    )
    _E_L = np.array([10.0, 20.0, 5.0, 15.0])

    def test_overlap_matrix_matches_numpy_population_covariance(self):
        # bias=True is numpy's population (ddof=0) covariance, matching this
        # module's documented convention exactly.
        expected = np.cov(self._O.T, bias=True)
        actual = np.asarray(sr_overlap_matrix(jnp.asarray(self._O)))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_overlap_matrix_matches_hand_expansion(self):
        # S_ij = <O_i O_j> - <O_i><O_j>, computed by hand term-by-term.
        mean = self._O.mean(axis=0)
        n = self._O.shape[0]
        expected = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                expected[i, j] = np.mean(self._O[:, i] * self._O[:, j]) - mean[i] * mean[j]
        actual = np.asarray(sr_overlap_matrix(jnp.asarray(self._O)))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert n == 4  # sanity: population covariance divides by N, not N-1.

    def test_energy_gradient_matches_hand_expansion(self):
        mean_O = self._O.mean(axis=0)
        mean_E = self._E_L.mean()
        expected = 2.0 * (np.mean(self._O * self._E_L[:, None], axis=0) - mean_O * mean_E)
        actual = np.asarray(sr_energy_gradient(jnp.asarray(self._O), jnp.asarray(self._E_L)))
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_overlap_matrix_is_symmetric_and_psd(self):
        rng = np.random.default_rng(0)
        O = rng.normal(size=(50, 4))
        S = np.asarray(sr_overlap_matrix(jnp.asarray(O)))
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals >= -1e-10)

    def test_zero_variance_log_derivative_gives_zero_overlap_row(self):
        """A parameter with a constant (sample-independent) log-derivative
        contributes exactly zero to S and to g (zero covariance with anything)."""
        rng = np.random.default_rng(1)
        n = 30
        varying = rng.normal(size=n)
        constant = np.full(n, 3.14)
        O = np.stack([varying, constant], axis=1)
        E_L = rng.normal(size=n)
        S = np.asarray(sr_overlap_matrix(jnp.asarray(O)))
        g = np.asarray(sr_energy_gradient(jnp.asarray(O), jnp.asarray(E_L)))
        np.testing.assert_allclose(S[1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(S[:, 1], 0.0, atol=1e-10)
        np.testing.assert_allclose(g[1], 0.0, atol=1e-10)

    def test_output_shapes(self):
        O = jnp.zeros((10, 5), dtype=jnp.float64)
        E_L = jnp.zeros((10,), dtype=jnp.float64)
        assert sr_overlap_matrix(O).shape == (5, 5)
        assert sr_energy_gradient(O, E_L).shape == (5,)


class TestValidationGuards:
    def test_overlap_matrix_rejects_non_2d(self):
        with pytest.raises(ValueError, match="n_samples, n_params"):
            sr_overlap_matrix(jnp.zeros((5,)))

    def test_energy_gradient_rejects_non_2d_log_derivatives(self):
        with pytest.raises(ValueError, match="n_samples, n_params"):
            sr_energy_gradient(jnp.zeros((5,)), jnp.zeros((5,)))

    def test_energy_gradient_rejects_mismatched_sample_count(self):
        with pytest.raises(ValueError, match="local_energies must be"):
            sr_energy_gradient(jnp.zeros((5, 3)), jnp.zeros((4,)))


# ---------------------------------------------------------------------------
# Shared toy ansatz: 1-D quantum harmonic oscillator, one variational
# width parameter alpha. psi_alpha(x) = exp(-alpha x^2 / 2); alpha=1 is the
# exact ground state (E_0 = 1/2). E(alpha) = alpha/4 + 1/(4 alpha) has its
# unique minimum at alpha=1 -- an elementary, hand-verifiable variational
# problem (see the module docstring's derivation for why this energy
# functional is exactly what stochastic reconfiguration measures).
# ---------------------------------------------------------------------------


def _log_abs_psi_variational_ho(alpha, r):
    return -0.5 * alpha * jnp.sum(r * r)


def _kinetic_fn(alpha, r):
    grad, lap = autodiff_grad_and_laplacian(_log_abs_psi_variational_ho, alpha, r)
    return kinetic_energy_from_grad_lap(grad, lap)


def _potential_fn(r):
    return 0.5 * jnp.sum(r * r)


_TRUE_GROUND_STATE_ENERGY = 0.5
_TRUE_OPTIMAL_ALPHA = 1.0


def _make_sample_fn(n_steps: int, step_size: float):
    return functools.partial(metropolis_sample, n_steps=n_steps, step_size=step_size)


# ---------------------------------------------------------------------------
# 3. damped_solve / natural_gradient_step are called correctly (not reimplemented)
# ---------------------------------------------------------------------------


class TestNaturalGradientCrossCheck:
    def test_sr_step_output_matches_direct_natural_gradient_step_call(self):
        alpha0 = jnp.asarray(1.7, dtype=jnp.float64)
        positions = jnp.zeros((32, 1), dtype=jnp.float64)
        sample_fn = _make_sample_fn(n_steps=50, step_size=1.0)
        damping, lr = 0.05, 0.4

        result = stochastic_reconfiguration_step(
            _log_abs_psi_variational_ho,
            alpha0,
            positions,
            jax.random.PRNGKey(7),
            kinetic_fn=_kinetic_fn,
            potential_fn=_potential_fn,
            sample_fn=sample_fn,
            damping=damping,
            learning_rate=lr,
        )

        # Cross-check against calling natural_gradient_step directly with the
        # exact (S, g) the SR step estimated and reported.
        expected_flat = natural_gradient_step(
            jnp.atleast_1d(alpha0),
            result.energy_gradient,
            result.overlap_matrix,
            learning_rate=lr,
            damping=damping,
        )
        np.testing.assert_allclose(
            np.asarray(jnp.atleast_1d(result.params)), np.asarray(expected_flat), atol=1e-12
        )

        # And against damped_solve + the update formula written out by hand.
        delta = damped_solve(result.overlap_matrix, result.energy_gradient, damping=damping)
        expected_flat_2 = jnp.atleast_1d(alpha0) - lr * delta
        np.testing.assert_allclose(
            np.asarray(jnp.atleast_1d(result.params)), np.asarray(expected_flat_2), atol=1e-12
        )

    def test_cg_iterations_is_none_direct_solve_only(self):
        """This module reuses damped_solve's exact direct solve; no CG path
        is implemented, so cg_iterations is always None (see module docstring)."""
        alpha0 = jnp.asarray(1.5, dtype=jnp.float64)
        positions = jnp.zeros((16, 1), dtype=jnp.float64)
        sample_fn = _make_sample_fn(n_steps=30, step_size=1.0)
        result = stochastic_reconfiguration_step(
            _log_abs_psi_variational_ho,
            alpha0,
            positions,
            jax.random.PRNGKey(3),
            kinetic_fn=_kinetic_fn,
            potential_fn=_potential_fn,
            sample_fn=sample_fn,
            damping=0.1,
            learning_rate=0.3,
        )
        assert result.cg_iterations is None
        assert isinstance(result, StochasticReconfigurationResult)


# ---------------------------------------------------------------------------
# 4. Toy harmonic-oscillator SR convergence demonstration
# ---------------------------------------------------------------------------


class TestHarmonicOscillatorConvergence:
    """Deterministic (fixed seed, fixed step count, fixed tolerance) end-to-end
    demonstration that sample -> accumulate -> solve -> update is wired
    correctly and genuinely improves the energy estimate."""

    def test_sr_converges_toward_true_ground_state_from_off_optimal_start(self):
        alpha0 = 1.8  # deliberately off the true optimum, alpha = 1.0
        n_walkers = 128
        n_sr_steps = 12
        sample_fn = _make_sample_fn(n_steps=150, step_size=1.0)
        damping, learning_rate = 0.1, 0.3

        jitted_step = jax.jit(
            stochastic_reconfiguration_step,
            static_argnames=(
                "log_abs_psi_fn",
                "kinetic_fn",
                "potential_fn",
                "sample_fn",
                "damping",
            ),
        )

        alpha = jnp.asarray(alpha0, dtype=jnp.float64)
        positions = jnp.zeros((n_walkers, 1), dtype=jnp.float64)
        key = jax.random.PRNGKey(0)

        energy_trajectory = []
        alpha_trajectory = [float(alpha)]
        for _ in range(n_sr_steps):
            key, subkey = jax.random.split(key)
            result = jitted_step(
                _log_abs_psi_variational_ho,
                alpha,
                positions,
                subkey,
                kinetic_fn=_kinetic_fn,
                potential_fn=_potential_fn,
                sample_fn=sample_fn,
                damping=damping,
                learning_rate=learning_rate,
            )
            alpha = result.params
            positions = result.positions
            energy_trajectory.append(float(result.energy_mean))
            alpha_trajectory.append(float(alpha))

        initial_error = abs(energy_trajectory[0] - _TRUE_GROUND_STATE_ENERGY)
        final_error = abs(energy_trajectory[-1] - _TRUE_GROUND_STATE_ENERGY)

        # Genuine, substantial improvement (not just noise): the final energy
        # estimate is at least 10x closer to the true ground-state energy
        # than the first step's, and alpha lands near its true optimum 1.0.
        assert final_error < initial_error / 10.0
        assert final_error < 0.02
        assert abs(alpha_trajectory[-1] - _TRUE_OPTIMAL_ALPHA) < 0.02

        # Reproducibility: the same fixed seed must reproduce this exactly.
        alpha_repeat = jnp.asarray(alpha0, dtype=jnp.float64)
        positions_repeat = jnp.zeros((n_walkers, 1), dtype=jnp.float64)
        key_repeat = jax.random.PRNGKey(0)
        for _ in range(n_sr_steps):
            key_repeat, subkey_repeat = jax.random.split(key_repeat)
            result_repeat = jitted_step(
                _log_abs_psi_variational_ho,
                alpha_repeat,
                positions_repeat,
                subkey_repeat,
                kinetic_fn=_kinetic_fn,
                potential_fn=_potential_fn,
                sample_fn=sample_fn,
                damping=damping,
                learning_rate=learning_rate,
            )
            alpha_repeat = result_repeat.params
            positions_repeat = result_repeat.positions
        np.testing.assert_array_equal(np.asarray(alpha_repeat), np.asarray(alpha))


# ---------------------------------------------------------------------------
# 5. Multi-leaf parameter pytree wiring
# ---------------------------------------------------------------------------


class TestPytreeParams:
    def test_updated_params_preserve_pytree_structure_and_shapes(self):
        def log_abs_psi(params, r):
            w, b = params
            return jnp.sum(jnp.tanh(w * r)) + b * jnp.sum(r)

        params = (
            jnp.array([0.5, -0.3, 0.2], dtype=jnp.float64),
            jnp.asarray(0.1, dtype=jnp.float64),
        )

        def kinetic_fn(p, r):
            grad, lap = autodiff_grad_and_laplacian(log_abs_psi, p, r)
            return kinetic_energy_from_grad_lap(grad, lap)

        positions = jax.random.normal(jax.random.PRNGKey(11), (24, 3), dtype=jnp.float64)
        sample_fn = _make_sample_fn(n_steps=20, step_size=0.5)

        result = stochastic_reconfiguration_step(
            log_abs_psi,
            params,
            positions,
            jax.random.PRNGKey(12),
            kinetic_fn=kinetic_fn,
            potential_fn=_potential_fn,
            sample_fn=sample_fn,
            damping=0.5,
            learning_rate=0.1,
        )

        new_w, new_b = result.params
        assert new_w.shape == params[0].shape
        assert new_b.shape == params[1].shape
        assert result.overlap_matrix.shape == (4, 4)  # 3 (w) + 1 (b)
        assert result.energy_gradient.shape == (4,)
        assert not bool(jnp.any(jnp.isnan(new_w)))
        assert not bool(jnp.isnan(new_b))


# ---------------------------------------------------------------------------
# 6. jax.jit safety
# ---------------------------------------------------------------------------


class TestJitSafety:
    def test_stochastic_reconfiguration_step_under_jit(self):
        jitted = jax.jit(
            stochastic_reconfiguration_step,
            static_argnames=(
                "log_abs_psi_fn",
                "kinetic_fn",
                "potential_fn",
                "sample_fn",
                "damping",
            ),
        )
        alpha0 = jnp.asarray(1.3, dtype=jnp.float64)
        # n_walkers=32: sampling.py's Sokal auto-window integrated-autocorrelation
        # estimator (`standard_error_of_mean`) is a pre-existing, unmodified
        # primitive that -- like any short-chain autocorrelation-time estimator --
        # is only guaranteed non-degenerate with "enough" samples; at very small
        # n (e.g. 16) it can occasionally self-select a window where the noisy
        # windowed sum estimates a negative integrated time, propagating a NaN
        # standard error even though energy_mean itself stays finite. That is a
        # property of the already-shipped estimator (out of scope to change
        # here; see the module docstring's honesty section), not of this jit
        # wiring, so this smoke test simply uses a large-enough n_walkers to stay
        # away from that small-n edge case.
        positions = jnp.zeros((32, 1), dtype=jnp.float64)
        sample_fn = _make_sample_fn(n_steps=25, step_size=1.0)

        result = jitted(
            _log_abs_psi_variational_ho,
            alpha0,
            positions,
            jax.random.PRNGKey(5),
            kinetic_fn=_kinetic_fn,
            potential_fn=_potential_fn,
            sample_fn=sample_fn,
            damping=0.1,
            learning_rate=0.3,
        )
        assert result.positions.shape == (32, 1)
        assert result.overlap_matrix.shape == (1, 1)
        assert result.energy_gradient.shape == (1,)
        assert result.cg_iterations is None
        assert not bool(jnp.isnan(result.energy_mean))
        assert not bool(jnp.isnan(result.energy_standard_error))

    def test_damping_must_be_static_under_jit(self):
        """damped_solve branches on damping's concrete value; a non-static
        damping under jit must fail with JAX's usual concretization error."""
        jitted = jax.jit(
            stochastic_reconfiguration_step,
            static_argnames=("log_abs_psi_fn", "kinetic_fn", "potential_fn", "sample_fn"),
        )
        alpha0 = jnp.asarray(1.3, dtype=jnp.float64)
        positions = jnp.zeros((8, 1), dtype=jnp.float64)
        sample_fn = _make_sample_fn(n_steps=10, step_size=1.0)
        with pytest.raises(jax.errors.TracerBoolConversionError):
            jitted(
                _log_abs_psi_variational_ho,
                alpha0,
                positions,
                jax.random.PRNGKey(6),
                kinetic_fn=_kinetic_fn,
                potential_fn=_potential_fn,
                sample_fn=sample_fn,
                damping=0.1,
                learning_rate=0.3,
            )
