# omnibias.pinn.train (causal marching + diagnostics)

Training drivers that close the loop
[`TimeMarcher`](../api/pinn.md) deliberately left open: per-window
optimisation, Wang–Perdikaris causal weights, warm-start handoff, and
diagnostics for temporal discordance and trivial-solution collapse.

Maturity: **alpha** submodule of Beta `omnibias-pinn`.

## Guarantee level

| Piece | Level | Acceptance domain |
| --- | --- | --- |
| Window geometry / handoff / advance gate | by construction | shared numpy `TimeMarcher` |
| Causality index | empirical measurement | inversion fraction of per-bin residual energies (`tau = 1 - 2 * index`); not a proof of temporal consistency |
| `march_solve` | empirical (multi-seed) | Adam (torch) / hand-rolled Adam (JAX); soft IC or caller `hard_ic_factory`; refuses silent advance under `advance_policy="gate"` |
| `SpectralBandScheduler` | numerical curriculum | applies measured residual bands at deterministic steps; not a certificate against spectral bias |

Same-time triviality guards use variance / energy / residual modes — do **not**
compare a late-time decaying field against the global `t=0` IC amplitude.

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

`omnibias.pinn.train.jax.march_solve` is the functional twin (hand-rolled Adam
over pytree params; no `optax` dependency). Window geometry and diagnostics are
shared pure numpy.

## Example

See [`docs/examples/pinn_causal_marching.py`](../examples/pinn_causal_marching.py).
