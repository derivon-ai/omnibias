# Certified fluid-dynamics demo

A proof-carrying residual loop for **incompressible periodic flow**, the
nonlinear companion to [`../proof_carrying_pde`](../proof_carrying_pde):

1. generate an exact analytic flow on a periodic torus (no external data),
2. evaluate the momentum, continuity and pressure-Poisson residuals by periodic
   spectral differentiation,
3. seal a residual-only Navier-Stokes certificate (with energy / enstrophy
   diagnostics and a regenerable fixture descriptor),
4. adjudicate it through the proof machine with schema, **independent numpy
   replay** and honesty gates, and
5. re-certify a short Taylor-Green "rollout" to show the residual stays at
   machine zero with no numerical drift.

Two flows are certified:

- **Taylor-Green vortex** — the exact 2-D decaying solution, used as the laminar
  correctness baseline.
- **Kolmogorov forced shear** — the exact steady base state `u = (A sin ky, 0)`
  balanced by a monochromatic body force; the entry point into the forced /
  chaotic-facing regime.

Run from the repository root:

```bash
python -m examples.certified_fluid_dynamics.run_demo
python -m examples.certified_fluid_dynamics.run_demo --n 128 --viscosity 0.05
```

## Claim boundary

These are finite-grid, finite-time residual certificates for **known analytic
model flows**. The residuals are computed by FFT sampling, not interval
enclosure, so the certificates set `interval_verified=False`. They explicitly do
**not** claim perfect weather, a continuum Navier-Stokes regularity theorem,
high-Reynolds turbulence, or pointwise long-horizon chaos tracking
(`unproven_claim`, `continuum_navier_stokes_claim`, `chaotic_tracking_claim`,
`perfect_weather_claim` and `turbulence_closure_claim` are all `False`). The open
obligation toward rigour is an interval / Taylor-model enclosure of the residual
between grid nodes.

No data is downloaded. The optional `--scratch-dir` (or `$OMNIBIAS_SCRATCH`) is a
runtime cache for generated arrays only.

## Extended studies (3D + fractional)

Beyond the residual baseline above, this directory hosts a three-track study of
the 3D Navier-Stokes equation, tied together in the cookbook page *Navier-Stokes
tracks (numerical / certified / fractional)*:

| Track | Driver | Write-up |
|---|---|---|
| A -- validated 3D PINN vs exact ABC flow | `run_abc_3d_pinn.py` | [`RESULTS_abc_3d_pinn.md`](RESULTS_abc_3d_pinn.md) |
| B -- replayable certificates | `run_abc_3d_certified.py` | [`RESULTS_abc_3d_certified.md`](RESULTS_abc_3d_certified.md) |
| C -- fractional / hyperdissipative + criticality ladder | `run_fractional_ns.py` | [`RESULTS_fractional_ns.md`](RESULTS_fractional_ns.md) |
| C -- learnable-alpha inverse-problem PINN | `run_fractional_pinn.py` | [`RESULTS_fractional_pinn.md`](RESULTS_fractional_pinn.md) |
| C -- learnable-beta at Tao's log-supercritical edge | `run_log_supercritical_pinn.py` | [`RESULTS_log_supercritical.md`](RESULTS_log_supercritical.md) |
| C -- GPU-scale 3D fractional PINN vs exact Beltrami shell | `run_fractional_abc_3d_pinn.py` | [`RESULTS_fractional_abc_3d.md`](RESULTS_fractional_abc_3d.md) |

Shared, importable math (spectral fractional operators, exact solutions --
including the alpha-dependent **Beltrami-shell** flow -- the `alpha_c = 5/4`
ladder, Tao's log-supercritical diagnostic + differentiable symbol, learnable
order) is in `fractional_ns_theory.py`, tested by `tests/test_fractional_ns.py`.
Figures are regenerated deterministically by `make_figures.py`.

The whole study is assembled into a self-contained flagship-style preprint in the
separate [omnibias-papers](https://github.com/derivon-ai/omnibias-papers/tree/main/papers/navier-stokes)
project (`main.tex` + executed companion notebook + `build.sh`), whose figures are
the *same* committed computations.

All of this is *validated numerics on finite models*, never a global-regularity claim:
`unproven_claim = False` is hard-wired everywhere and the honesty gate is shown
blocking a forged claim.
