# Quantum Monte-Carlo sampling substrate

`omnibias.ferminet.sampling` is the **sampling** foundation for a
variational Monte Carlo (VMC) trainer (see
[`omnibias.ferminet.stochastic_reconfiguration`](stochastic_reconfiguration.md)
for the natural-gradient optimizer step built on top of it): reproducible
Metropolis / Langevin
walkers over electron (or any) positions, a local-energy estimator,
autocorrelation-aware error diagnostics, and a walker-level log-derivative
accumulator. It is deliberately **ansatz-agnostic** -- every function takes a
caller-supplied `log_abs_psi_fn(params, position) -> log|psi(position)|`
callable for a *single* walker, so it plugs into
[`omnibias.ferminet.restricted`](ferminet.md)'s Tier-2 wavefunction (bind its
extra `n_e` argument with `functools.partial`) or any future ansatz without
modification. The target density every walker samples is `|psi|² =
exp(2 * log_abs_psi_fn(...))`.

Every piece is labeled honestly by register:

| Register | What | Functions |
| --- | --- | --- |
| **Numerical / statistical** (standard MCMC, not closed-form, not autodiff) | The Metropolis-Hastings and Langevin accept/reject mechanics; the autocorrelation / IAT / ESS / SEM estimators | `metropolis_step`, `metropolis_sample`, `langevin_step`, `langevin_sample`, `autocorrelation_function`, `integrated_autocorrelation_time`, `effective_sample_size`, `standard_error_of_mean`, `energy_chain_diagnostics` |
| **Plain JAX autodiff** (ordinary `jax.grad`/`jax.hessian`, *not* the closed-form derivative tower) | The Langevin position-gradient drift; the general-ansatz Laplacian fallback; the parameter log-derivative accumulator | `langevin_step`'s drift, `autodiff_grad_and_laplacian`, `log_derivative_accumulator` |
| **Closed-form (consumed, not computed here)** | `local_energy`'s `kinetic_fn` may route through `omnibias.ferminet.restricted`'s closed-form Laplacian | `local_energy` (via a caller-supplied `kinetic_fn`) |

## 30-second tour: the harmonic-oscillator constant-local-energy oracle

A true eigenstate has **exactly constant** local energy `E_L(r) = H psi(r) /
psi(r)` at every point -- the strongest available regression oracle, since it
needs no statistics at all. For the 1-D quantum harmonic oscillator
(`m = omega = hbar = 1`), the ground state is `psi(x) = exp(-x^2 / 2)`, so
`log|psi| = -x^2/2`, and the exact ground-state energy is `E_0 = 1/2`:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.ferminet.sampling import (
    autodiff_grad_and_laplacian,
    kinetic_energy_from_grad_lap,
    local_energy,
)


def log_abs_psi_ho(params, r):
    return -0.5 * jnp.sum(r * r)


def potential(r):
    return 0.5 * jnp.sum(r * r)


def kinetic_fn(params, r):
    grad, lap = autodiff_grad_and_laplacian(log_abs_psi_ho, params, r)
    return kinetic_energy_from_grad_lap(grad, lap)


positions = jax.random.normal(jax.random.PRNGKey(0), (8, 1), dtype=jnp.float64)
energies = local_energy(kinetic_fn, potential, None, positions)
assert jnp.allclose(energies, 0.5, atol=1e-10)  # E_0 = 1/2 at *every* sample.
```

`kinetic_energy_from_grad_lap` implements the numerically stable log-domain
identity `lap(psi)/psi = lap(log|psi|) + |grad(log|psi|)|^2`, so
`T_loc = -1/2 * lap(psi)/psi` never divides by `psi` -- safe even arbitrarily
close to a nodal surface.

## Sampling the target density

The same `log_abs_psi_fn` drives a reproducible random-walk Metropolis
sampler (explicit PRNG key, `jax.vmap` over independent walkers, `jax.lax.scan`
over steps -- calling it twice with the same key gives a bit-identical
trajectory):

```python
key = jax.random.PRNGKey(1)
init_positions = jnp.zeros((4, 1), dtype=jnp.float64)

from omnibias.ferminet.sampling import metropolis_sample

result = metropolis_sample(
    log_abs_psi_ho, None, init_positions, key, n_steps=200, step_size=1.0
)
print(result.positions.shape, result.acceptance_rate.shape)  # (4, 1) (4,)
```

`omnibias.ferminet.sampling.langevin_sample` offers the same reproducibility
guarantee for an (unadjusted or Metropolis-adjusted) Langevin walker driven by
the *position* gradient of `log|psi|` -- ordinary autodiff, not a closed-form
claim.

## Error diagnostics and the log-derivative accumulator

`energy_chain_diagnostics` bundles the mean, the autocorrelation-corrected
standard error of the mean, the integrated autocorrelation time, and the
effective sample size for a scalar local-energy chain (a zero-variance chain,
as in the oracle above, correctly reports a zero standard error rather than
`nan`):

```python
from omnibias.ferminet.sampling import energy_chain_diagnostics

diagnostics = energy_chain_diagnostics(energies)
assert float(diagnostics.standard_error) < 1e-9
```

`log_derivative_accumulator` computes the per-sample parameter gradient
`O_p(r) = d(log|psi(params, r)|) / d(theta_p)` via
`jax.vmap(jax.grad(...))`, flattening `params` with
`jax.flatten_util.ravel_pytree`, and returns an `(n_samples, n_params)` array
-- exactly the shape
[`omnibias.ferminet.stochastic_reconfiguration`](stochastic_reconfiguration.md)
uses for the Fisher-style overlap matrix `S_ij = <O_i* O_j> - <O_i><O_j>`
(matching [`omnibias.curvature.natural_gradient`](curvature.md)'s `(P, P)` /
`(P,)` convention):

```python
from omnibias.ferminet.sampling import log_derivative_accumulator


def log_abs_psi_linear(params, r):
    w, b = params
    return jnp.sum(w * r) + b


params = (jnp.array([0.5, -0.3]), jnp.asarray(0.1))
positions = jax.random.normal(jax.random.PRNGKey(2), (5, 2))
log_derivatives = log_derivative_accumulator(log_abs_psi_linear, params, positions)
assert log_derivatives.shape == (5, 3)  # n_samples=5, n_params=2+1
```

!!! warning "Sampling substrate, now with a natural-gradient trainer on top"
    This module is the sampling substrate for a VMC trainer.
    [`omnibias.ferminet.stochastic_reconfiguration`](stochastic_reconfiguration.md)
    builds the natural-gradient (stochastic-reconfiguration) optimizer step on
    top of it, reusing `omnibias.curvature.natural_gradient`'s generic
    `(P, P)` / `(P,)` `damped_solve` / `natural_gradient_step` unmodified. This
    module itself still does **not** implement lattice Hamiltonians,
    antisymmetric-ansatz changes, or ground-state certificates. See
    [scope & guarantees](../scope-and-guarantees.md) §6 for the full
    closed-form / autodiff / numerical breakdown of `omnibias-ferminet` and
    `omnibias-qpinn`.

::: omnibias.ferminet.sampling
    options:
      show_root_heading: false
      heading_level: 3
