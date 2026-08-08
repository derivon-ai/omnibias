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

## Regime note (heat vs reaction)

Whether marching beats whole-interval is **regime-dependent**, and the
acceptance script measures both families under an equal advertised step budget
([`benchmarks/causal_marching.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/causal_marching.py)):

- **Heat** (`u_t = u_xx`, smooth manufactured decay): whole-interval is expected
  to win; the gate requires every arm `skill_score > 0` and the best-arm median
  rel-L2 under a named threshold.
- **Reaction** (Krishnapriyan `u_t = ρ u(1−u)` at large `ρ`): whole-interval is
  the classical causality failure; the gate requires the best marching arm to
  beat `whole_interval` on median rel-L2.

Capability matrix:
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

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
Acceptance artifact + matrix:
[`docs/benchmarks/causal_marching.json`](../benchmarks/causal_marching.json),
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).
