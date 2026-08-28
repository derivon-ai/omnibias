# Stochastic reconfiguration (SR)

`omnibias.ferminet.stochastic_reconfiguration` is the natural-gradient VMC
optimizer step built **on top of** the
[QMC sampling substrate](qmc_sampling.md). It is explicitly a
**composition**, not a new optimizer: the only new math here is the two
Monte-Carlo estimators below (`sr_overlap_matrix`, `sr_energy_gradient`); the
linear solve that turns them into a parameter update is delegated, unmodified,
to [`omnibias.curvature.natural_gradient`](curvature.md)'s `damped_solve` /
`natural_gradient_step` -- functions written with no knowledge of quantum
Monte Carlo at all. Requires the optional `curvature` extra
(`pip install "omnibias-ferminet[curvature]"`).

Every piece is labeled honestly by register:

| Register | What | Functions |
| --- | --- | --- |
| **Monte-Carlo statistical estimator** (ordinary sampling noise, exact only as `n_samples -> infinity`) | The overlap (quantum geometric tensor) matrix `S` and the VMC energy-gradient estimator `g` | `sr_overlap_matrix`, `sr_energy_gradient` |
| **Exact linear algebra** (deterministic given `(S, g)`; no additional randomness) | The damped solve `(S + damping*I)^{-1} g` and the parameter update, reused unmodified from `omnibias-curvature` | `omnibias.curvature.natural_gradient.damped_solve` / `natural_gradient_step` |
| **Plain JAX autodiff** (already labeled as such in `sampling.py`) | The per-sample log-derivative accumulator and the autodiff Laplacian fallback that feeds the local energy | `omnibias.ferminet.sampling.log_derivative_accumulator`, `local_energy` |

## The estimators, in isolation

`sr_overlap_matrix` is the *population* (`ddof=0`) sample covariance of a
`(n_samples, n_params)` log-derivative matrix `O`; `sr_energy_gradient` is the
standard VMC "log-derivative trick" energy-gradient estimator
`g_p = 2 * (<O_p E_L> - <O_p><E_L>)`. Both take plain arrays, so they are
testable against hand-computed values with no sampling involved at all:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.ferminet.stochastic_reconfiguration import (
    sr_energy_gradient,
    sr_overlap_matrix,
)

# 4 samples, 2 parameters.
O = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 2.0]])
E_L = jnp.array([10.0, 20.0, 5.0, 15.0])

S = sr_overlap_matrix(O)
g = sr_energy_gradient(O, E_L)
assert S.shape == (2, 2)
assert g.shape == (2,)
assert jnp.allclose(S, S.T)  # S is always symmetric.
```

See the module docstring
([`omnibias.ferminet.stochastic_reconfiguration`][omnibias.ferminet.stochastic_reconfiguration])
for the full derivation of `g` (the score-function / log-derivative identity)
and the exact sign/factor convention used.

## The full sample -> accumulate -> solve -> update step

`stochastic_reconfiguration_step` composes one Metropolis (or Langevin) walk,
the log-derivative / local-energy accumulators, `sr_overlap_matrix` /
`sr_energy_gradient`, and `natural_gradient_step` into a single call. The toy
example below is the 1-D quantum harmonic oscillator with a single
variational-width parameter `alpha` (`psi_alpha(x) = exp(-alpha x^2 / 2)`,
exact ground state at `alpha = 1`, `E_0 = 1/2`) -- the same oracle
[`qmc_sampling.md`](qmc_sampling.md) uses, now actually *optimized*:

```python
import functools

from omnibias.curvature.natural_gradient import damped_solve
from omnibias.ferminet.sampling import (
    autodiff_grad_and_laplacian,
    kinetic_energy_from_grad_lap,
    metropolis_sample,
)
from omnibias.ferminet.stochastic_reconfiguration import stochastic_reconfiguration_step


def log_abs_psi(alpha, r):
    return -0.5 * alpha * jnp.sum(r * r)


def kinetic_fn(alpha, r):
    grad, lap = autodiff_grad_and_laplacian(log_abs_psi, alpha, r)
    return kinetic_energy_from_grad_lap(grad, lap)


def potential_fn(r):
    return 0.5 * jnp.sum(r * r)


alpha = jnp.asarray(1.8, dtype=jnp.float64)  # deliberately off the optimum (1.0).
positions = jnp.zeros((64, 1), dtype=jnp.float64)
sample_fn = functools.partial(metropolis_sample, n_steps=100, step_size=1.0)

result = stochastic_reconfiguration_step(
    log_abs_psi,
    alpha,
    positions,
    jax.random.PRNGKey(0),
    kinetic_fn=kinetic_fn,
    potential_fn=potential_fn,
    sample_fn=sample_fn,
    damping=0.1,
    learning_rate=0.3,
)

# One step already moves alpha toward 1.0 and the energy toward 0.5.
assert abs(float(result.params) - 1.8) > 0.0
print(result.energy_mean, result.params, result.overlap_matrix.shape)
```

`result` is a [`StochasticReconfigurationResult`][omnibias.ferminet.stochastic_reconfiguration.StochasticReconfigurationResult]:
the updated `params`, the walker `positions` (feed these back in as the next
call's `init_positions` to warm-start the chain), the energy estimate and its
`omnibias.ferminet.sampling.standard_error_of_mean`-based standard error, the
`(S, g)` this step estimated (so a caller can independently cross-check the
update with `damped_solve(result.overlap_matrix, result.energy_gradient,
damping=...)`), the damping used, and a reserved (always `None` today)
`cg_iterations` slot.

Running several such steps in a loop, feeding `result.params` /
`result.positions` back in as the next step's inputs, converges `alpha`
toward `1.0` and the energy estimate toward `0.5` -- see
`packages/omnibias-ferminet/tests/test_stochastic_reconfiguration.py`'s
`TestHarmonicOscillatorConvergence` for the full, deterministic (fixed-seed)
12-step demonstration.

!!! warning "`damping` (and every callable argument) must be static under `jax.jit`"
    `damped_solve` performs concrete Python control flow on `damping`'s value,
    so wrapping `stochastic_reconfiguration_step` in `jax.jit` requires listing
    it -- and `log_abs_psi_fn` / `kinetic_fn` / `potential_fn` / `sample_fn`,
    all plain Python callables -- in `static_argnames`:

    ```python
    jax.jit(
        stochastic_reconfiguration_step,
        static_argnames=("log_abs_psi_fn", "kinetic_fn", "potential_fn", "sample_fn", "damping"),
    )
    ```

    `learning_rate` has no such restriction and may safely be a traced array.

!!! note "Why no conjugate-gradient solver"
    `damped_solve` currently does a dense `jnp.linalg.solve`, and the only
    conjugate-gradient implementation in the repository
    (`omnibias.torch.optim.conjugate_gradient`) is PyTorch-only -- reusing it
    here would mean a from-scratch JAX port, not reuse. Every test in this
    module uses `n_params` on the order of 1-6, for which the dense solve is
    fast and robust, so `StochasticReconfigurationResult.cg_iterations` stays
    `None`; the field exists so a future CG-based solve path can be wired in
    without changing this result type's shape.

::: omnibias.ferminet.stochastic_reconfiguration
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]
