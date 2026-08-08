# omnibias.pinn.train (causal marching + diagnostics)

Training drivers that close the loop
[`TimeMarcher`](../api/pinn.md) deliberately left open: per-window
optimisation, Wang–Perdikaris causal weights, warm-start handoff, and
diagnostics for temporal discordance and trivial-solution collapse.

Maturity: **alpha** submodule of Beta `omnibias-pinn`.

## Honest method labels

- The **causality index** is a *measurement* (Kendall-tau discordance of
  per-bin residuals), not a proof of temporal consistency.
- **`march_solve`** uses Adam + soft (or caller-provided hard) IC penalties;
  residual operators stay on whatever closed-form / autodiff path the field
  already exposes.
- **`SpectralBandScheduler`** grows Fourier / Mscale bands from the residual
  spectrum — a numerical curriculum, not a certificate against spectral bias.

## Core schemas

::: omnibias.pinn.train
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - CausalityReport
        - TrivialSolutionVerdict
        - SpectralBandScheduler
        - causality_index
        - unlocked_fraction
        - report_causality
        - trivial_solution_guard

## Torch driver

::: omnibias.pinn.train.torch
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - MarchResult
        - WindowResult
        - march_solve

## JAX twin

`omnibias.pinn.train.jax.march_solve` is the functional twin (optax Adam,
pytree params). Window geometry and diagnostics are shared pure numpy.

## Example

See [`docs/examples/pinn_causal_marching.py`](../examples/pinn_causal_marching.py).
