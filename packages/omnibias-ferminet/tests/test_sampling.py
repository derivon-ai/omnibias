# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Tests for the reproducible quantum Monte-Carlo sampling substrate.

Covers, in order: (1) bit-identical reproducibility of both walkers from a
fixed PRNG key, (2) correctness on the exactly-solvable standard-normal toy
target (``|psi|^2`` a unit Gaussian) for both the Metropolis and Langevin
(MALA) walkers, (3) the harmonic-oscillator ground state's *exactly* constant
local energy -- the strong regression check for :func:`local_energy` and
:func:`kinetic_energy_from_grad_lap`, (4) the autocorrelation / IAT / ESS /
SEM estimators on synthetic AR(1) and i.i.d. series with known ground truth,
(5) the walker-level log-derivative accumulator cross-checked against a plain
central finite difference, and (6) ``jax.jit`` safety for every public
function (with the callable arguments marked ``static_argnames``, per the
module docstring).
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.ferminet.sampling import (  # noqa: E402
    autocorrelation_function,
    autodiff_grad_and_laplacian,
    effective_sample_size,
    energy_chain_diagnostics,
    integrated_autocorrelation_time,
    kinetic_energy_from_grad_lap,
    langevin_sample,
    langevin_step,
    local_energy,
    log_derivative_accumulator,
    metropolis_sample,
    metropolis_step,
    standard_error_of_mean,
)

#: ``|psi|^2 = exp(2 * log_abs_psi) = exp(-r^2)`` up to normalization, i.e.
#: ``exp(-r^2 / (2 * 0.5))``: the target is ``N(0, 0.5)`` per axis (variance
#: ``1/2``, *not* ``1``), the exactly-solvable toy target used below.
_TARGET_VARIANCE = 0.5


def _log_abs_psi_std_normal(params: None, r: jnp.ndarray) -> jnp.ndarray:
    return -0.5 * jnp.sum(r * r)


def _log_abs_psi_ho_ground_state(params: None, r: jnp.ndarray) -> jnp.ndarray:
    """Exact 1-D quantum harmonic-oscillator ground state, ``m = omega = hbar = 1``."""
    return -0.5 * jnp.sum(r * r)


def _ho_potential(r: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * jnp.sum(r * r)


class TestMetropolisReproducibility:
    def test_same_key_gives_bit_identical_trajectory(self):
        key = jax.random.PRNGKey(0)
        init = jnp.zeros((8, 2), dtype=jnp.float64)
        r1 = metropolis_sample(
            _log_abs_psi_std_normal, None, init, key, n_steps=200, step_size=1.0
        )
        r2 = metropolis_sample(
            _log_abs_psi_std_normal, None, init, key, n_steps=200, step_size=1.0
        )
        np.testing.assert_array_equal(np.asarray(r1.trajectory), np.asarray(r2.trajectory))
        np.testing.assert_array_equal(np.asarray(r1.positions), np.asarray(r2.positions))
        np.testing.assert_array_equal(np.asarray(r1.n_accepted), np.asarray(r2.n_accepted))

    def test_different_key_gives_different_trajectory(self):
        init = jnp.zeros((8, 2), dtype=jnp.float64)
        r1 = metropolis_sample(
            _log_abs_psi_std_normal, None, init, jax.random.PRNGKey(0), n_steps=200, step_size=1.0
        )
        r2 = metropolis_sample(
            _log_abs_psi_std_normal, None, init, jax.random.PRNGKey(1), n_steps=200, step_size=1.0
        )
        assert not bool(jnp.all(r1.trajectory == r2.trajectory))

    def test_single_step_reproducible(self):
        key = jax.random.PRNGKey(3)
        positions = jax.random.normal(jax.random.PRNGKey(9), (5, 2), dtype=jnp.float64)
        log_abs0 = jax.vmap(_log_abs_psi_std_normal, in_axes=(None, 0))(None, positions)
        a = metropolis_step(_log_abs_psi_std_normal, None, positions, log_abs0, key, 0.5)
        b = metropolis_step(_log_abs_psi_std_normal, None, positions, log_abs0, key, 0.5)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


class TestLangevinReproducibility:
    @pytest.mark.parametrize("adjusted", [True, False])
    def test_same_key_gives_bit_identical_trajectory(self, adjusted):
        key = jax.random.PRNGKey(0)
        init = jnp.zeros((8, 2), dtype=jnp.float64)
        r1 = langevin_sample(
            _log_abs_psi_std_normal,
            None,
            init,
            key,
            n_steps=200,
            step_size=0.3,
            adjusted=adjusted,
        )
        r2 = langevin_sample(
            _log_abs_psi_std_normal,
            None,
            init,
            key,
            n_steps=200,
            step_size=0.3,
            adjusted=adjusted,
        )
        np.testing.assert_array_equal(np.asarray(r1.trajectory), np.asarray(r2.trajectory))
        np.testing.assert_array_equal(np.asarray(r1.positions), np.asarray(r2.positions))

    def test_single_step_reproducible(self):
        key = jax.random.PRNGKey(4)
        positions = jax.random.normal(jax.random.PRNGKey(10), (5, 2), dtype=jnp.float64)
        log_abs0 = jax.vmap(_log_abs_psi_std_normal, in_axes=(None, 0))(None, positions)
        a = langevin_step(_log_abs_psi_std_normal, None, positions, log_abs0, key, 0.2)
        b = langevin_step(_log_abs_psi_std_normal, None, positions, log_abs0, key, 0.2)
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


class TestToyTargetCorrectness:
    """Both walkers must recover the exact ``N(0, 1/2)`` moments of the toy target."""

    def test_metropolis_recovers_standard_normal_moments(self):
        n_walkers, n_steps, burn_in = 64, 3000, 1000
        init = jnp.zeros((n_walkers, 1), dtype=jnp.float64)
        result = metropolis_sample(
            _log_abs_psi_std_normal,
            None,
            init,
            jax.random.PRNGKey(42),
            n_steps=n_steps,
            step_size=1.2,
        )
        samples = np.asarray(result.trajectory[burn_in:])  # (n_steps - burn_in, n_walkers, 1)
        mean = float(np.mean(samples))
        var = float(np.var(samples))
        acc = np.asarray(result.acceptance_rate)
        assert abs(mean) < 0.1
        assert abs(var - _TARGET_VARIANCE) < 0.1
        assert np.all((acc > 0.05) & (acc < 0.98))

    def test_langevin_mala_recovers_standard_normal_moments(self):
        n_walkers, n_steps, burn_in = 64, 3000, 1000
        init = jnp.zeros((n_walkers, 1), dtype=jnp.float64)
        result = langevin_sample(
            _log_abs_psi_std_normal,
            None,
            init,
            jax.random.PRNGKey(43),
            n_steps=n_steps,
            step_size=0.7,
            adjusted=True,
        )
        samples = np.asarray(result.trajectory[burn_in:])
        mean = float(np.mean(samples))
        var = float(np.var(samples))
        assert abs(mean) < 0.1
        assert abs(var - _TARGET_VARIANCE) < 0.1


class TestLocalEnergyHarmonicOscillator:
    """The strong exact regression check: a true eigenstate has constant E_L."""

    def test_local_energy_is_exactly_constant(self):
        def kinetic_fn(params, r):
            g, lap = autodiff_grad_and_laplacian(_log_abs_psi_ho_ground_state, params, r)
            return kinetic_energy_from_grad_lap(g, lap)

        positions = jax.random.normal(jax.random.PRNGKey(1), (25, 1), dtype=jnp.float64)
        energies = local_energy(kinetic_fn, _ho_potential, None, positions)
        expected_e0 = 0.5  # 1-D ground state, m = omega = hbar = 1: E_0 = 1/2.
        np.testing.assert_allclose(
            np.asarray(energies), np.full((25,), expected_e0), atol=1e-10, rtol=0.0
        )

    def test_local_energy_multi_dimensional_ground_state(self):
        """A D-dimensional isotropic oscillator ground state has E_0 = D / 2."""
        d = 3

        def log_abs_psi(params, r):
            return -0.5 * jnp.sum(r * r)

        def kinetic_fn(params, r):
            g, lap = autodiff_grad_and_laplacian(log_abs_psi, params, r)
            return kinetic_energy_from_grad_lap(g, lap)

        positions = jax.random.normal(jax.random.PRNGKey(2), (10, d), dtype=jnp.float64)
        energies = local_energy(kinetic_fn, _ho_potential, None, positions)
        np.testing.assert_allclose(
            np.asarray(energies), np.full((10,), d / 2.0), atol=1e-10, rtol=0.0
        )

    def test_kinetic_energy_from_grad_lap_matches_definition(self):
        grad = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float64)
        lap = jnp.asarray(3.0, dtype=jnp.float64)
        expected = -0.5 * (3.0 + (1.0 + 4.0 + 0.25))
        np.testing.assert_allclose(
            float(kinetic_energy_from_grad_lap(grad, lap)), expected, atol=1e-14
        )


class TestAutocorrelationDiagnostics:
    def test_iid_noise_gives_tau_near_one(self):
        x = jax.random.normal(jax.random.PRNGKey(2), (4000,), dtype=jnp.float64)
        tau = float(integrated_autocorrelation_time(x))
        assert abs(tau - 1.0) < 0.3

    def test_ar1_process_recovers_known_tau(self):
        """AR(1) ``x_t = phi x_{t-1} + eps_t`` has exact ``tau = (1+phi)/(1-phi)``."""
        phi = 0.8
        n = 20000
        eps = jax.random.normal(jax.random.PRNGKey(5), (n,), dtype=jnp.float64)

        def body(carry, e):
            new = phi * carry + e
            return new, new

        _, chain = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), eps)
        tau_est = float(integrated_autocorrelation_time(chain))
        tau_true = (1.0 + phi) / (1.0 - phi)
        assert abs(tau_est - tau_true) / tau_true < 0.15

    def test_autocorrelation_function_rho_zero_is_one(self):
        x = jax.random.normal(jax.random.PRNGKey(6), (500,), dtype=jnp.float64)
        rho = autocorrelation_function(x, max_lag=10)
        assert float(rho[0]) == pytest.approx(1.0, abs=1e-12)

    def test_autocorrelation_rejects_out_of_range_max_lag(self):
        x = jnp.zeros((10,), dtype=jnp.float64)
        with pytest.raises(ValueError):
            autocorrelation_function(x, max_lag=10)
        with pytest.raises(ValueError):
            autocorrelation_function(x, max_lag=-1)

    def test_constant_chain_has_zero_error_no_nan(self):
        """A zero-variance chain (a true eigenstate's local energy) must not yield NaN."""
        x = jnp.full((16,), 0.5, dtype=jnp.float64)
        diag = energy_chain_diagnostics(x)
        assert not bool(jnp.isnan(diag.standard_error))
        assert not bool(jnp.isnan(diag.integrated_autocorr_time))
        assert not bool(jnp.isnan(diag.effective_sample_size))
        assert float(diag.mean) == pytest.approx(0.5)
        assert float(diag.standard_error) == pytest.approx(0.0, abs=1e-14)
        assert float(diag.effective_sample_size) == pytest.approx(16.0)

    def test_effective_sample_size_le_n_for_correlated_chain(self):
        phi = 0.9
        n = 5000
        eps = jax.random.normal(jax.random.PRNGKey(7), (n,), dtype=jnp.float64)

        def body(carry, e):
            new = phi * carry + e
            return new, new

        _, chain = jax.lax.scan(body, jnp.asarray(0.0, dtype=jnp.float64), eps)
        ess = float(effective_sample_size(chain))
        assert 0.0 < ess < n

    def test_standard_error_of_mean_matches_naive_for_iid(self):
        x = jax.random.normal(jax.random.PRNGKey(8), (4000,), dtype=jnp.float64)
        sem = float(standard_error_of_mean(x))
        naive_sem = float(jnp.std(x, ddof=1) / jnp.sqrt(x.shape[0]))
        assert abs(sem - naive_sem) / naive_sem < 0.3


class TestLogDerivativeAccumulator:
    def test_matches_finite_difference(self):
        def log_abs_psi(params, r):
            w, b = params
            return jnp.sum(jnp.tanh(w * r)) + b * jnp.sum(r)

        params = (
            jnp.array([0.5, -0.3, 0.2], dtype=jnp.float64),
            jnp.asarray(0.1, dtype=jnp.float64),
        )
        positions = jax.random.normal(jax.random.PRNGKey(11), (6, 3), dtype=jnp.float64)

        analytic = np.asarray(log_derivative_accumulator(log_abs_psi, params, positions))

        flat_w, flat_b = params
        h = 1e-6
        finite_diff = np.zeros_like(analytic)
        for j in range(flat_w.shape[0]):
            plus = (flat_w.at[j].add(h), flat_b)
            minus = (flat_w.at[j].add(-h), flat_b)
            for i in range(positions.shape[0]):
                f_plus = float(log_abs_psi(plus, positions[i]))
                f_minus = float(log_abs_psi(minus, positions[i]))
                finite_diff[i, j] = (f_plus - f_minus) / (2 * h)
        # last flattened parameter is the scalar b.
        for i in range(positions.shape[0]):
            plus_b = (flat_w, flat_b + h)
            minus_b = (flat_w, flat_b - h)
            f_plus = float(log_abs_psi(plus_b, positions[i]))
            f_minus = float(log_abs_psi(minus_b, positions[i]))
            finite_diff[i, -1] = (f_plus - f_minus) / (2 * h)

        np.testing.assert_allclose(analytic, finite_diff, atol=1e-5, rtol=1e-5)

    def test_output_shape_is_n_samples_by_n_params(self):
        def log_abs_psi(params, r):
            w, b = params
            return jnp.sum(w * r) + b

        params = (
            jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )
        positions = jnp.zeros((7, 4), dtype=jnp.float64)
        out = log_derivative_accumulator(log_abs_psi, params, positions)
        assert out.shape == (7, 5)  # n_samples=7, n_params = 4 (w) + 1 (b) = 5


class TestJitSafety:
    """Every public sampler/estimator must run under ``jax.jit`` with the
    documented ``static_argnames`` for its callable argument(s)."""

    def test_metropolis_sample_under_jit(self):
        jitted = jax.jit(metropolis_sample, static_argnames=("log_abs_psi_fn", "n_steps"))
        init = jnp.zeros((4, 2), dtype=jnp.float64)
        result = jitted(
            _log_abs_psi_std_normal, None, init, jax.random.PRNGKey(0), n_steps=25, step_size=1.0
        )
        assert result.positions.shape == (4, 2)

    @pytest.mark.parametrize("adjusted", [True, False])
    def test_langevin_sample_under_jit(self, adjusted):
        jitted = jax.jit(
            langevin_sample, static_argnames=("log_abs_psi_fn", "n_steps", "adjusted")
        )
        init = jnp.zeros((4, 2), dtype=jnp.float64)
        result = jitted(
            _log_abs_psi_std_normal,
            None,
            init,
            jax.random.PRNGKey(0),
            n_steps=25,
            step_size=0.3,
            adjusted=adjusted,
        )
        assert result.positions.shape == (4, 2)

    def test_local_energy_under_jit(self):
        def kinetic_fn(params, r):
            g, lap = autodiff_grad_and_laplacian(_log_abs_psi_ho_ground_state, params, r)
            return kinetic_energy_from_grad_lap(g, lap)

        jitted = jax.jit(local_energy, static_argnames=("kinetic_fn", "potential_fn"))
        positions = jax.random.normal(jax.random.PRNGKey(1), (5, 1), dtype=jnp.float64)
        energies = jitted(kinetic_fn, _ho_potential, None, positions)
        np.testing.assert_allclose(np.asarray(energies), np.full((5,), 0.5), atol=1e-10)

    def test_log_derivative_accumulator_under_jit(self):
        def log_abs_psi(params, r):
            w, b = params
            return jnp.sum(w * r) + b

        jitted = jax.jit(log_derivative_accumulator, static_argnames=("log_abs_psi_fn",))
        params = (jnp.array([0.5, -0.3], dtype=jnp.float64), jnp.asarray(0.1, dtype=jnp.float64))
        positions = jax.random.normal(jax.random.PRNGKey(3), (5, 2), dtype=jnp.float64)
        out = jitted(log_abs_psi, params, positions)
        assert out.shape == (5, 3)

    def test_autocorrelation_estimators_under_jit(self):
        jitted_tau = jax.jit(integrated_autocorrelation_time)
        x = jax.random.normal(jax.random.PRNGKey(2), (1000,), dtype=jnp.float64)
        tau = jitted_tau(x)
        assert float(tau) > 0.0

        jitted_diag = jax.jit(energy_chain_diagnostics)
        diag = jitted_diag(jnp.full((10,), 0.5, dtype=jnp.float64))
        assert float(diag.standard_error) == pytest.approx(0.0, abs=1e-14)


class TestAcceptanceRateDiagnostic:
    def test_step_size_extremes_bracket_acceptance_rate(self):
        """A tiny step size should accept almost always; a huge one, almost never."""
        init = jnp.zeros((16, 1), dtype=jnp.float64)
        tiny = metropolis_sample(
            _log_abs_psi_std_normal,
            None,
            init,
            jax.random.PRNGKey(20),
            n_steps=300,
            step_size=1e-3,
        )
        huge = metropolis_sample(
            _log_abs_psi_std_normal,
            None,
            init,
            jax.random.PRNGKey(21),
            n_steps=300,
            step_size=50.0,
        )
        assert float(jnp.mean(tiny.acceptance_rate)) > 0.9
        assert float(jnp.mean(huge.acceptance_rate)) < 0.1
        assert jnp.all((tiny.acceptance_rate >= 0.0) & (tiny.acceptance_rate <= 1.0))
        assert jnp.all((huge.acceptance_rate >= 0.0) & (huge.acceptance_rate <= 1.0))
