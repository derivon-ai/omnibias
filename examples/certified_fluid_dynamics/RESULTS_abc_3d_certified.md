# Track B results: rigorous certified-evidence bridge for 3D Navier-Stokes

Track A produced a *trained numerical* 3D velocity field. Track B turns that
field into **replayable, schema-validated residual certificates**, adjudicates
an exact 3D fixture through the prove/disprove machine, and exercises the
genuine 3D-reduced **interval-enclosure** pipeline.

Nothing here is a global-regularity statement. `unproven_claim = False`
everywhere, and the honesty gate is shown to *block* a forged claim. See
[`../../docs/scope-and-guarantees.md`](../../docs/scope-and-guarantees.md) and
[`../../docs/cookbook/navier-stokes-certified.md`](../../docs/cookbook/navier-stokes-certified.md).

Driver: [`run_abc_3d_certified.py`](run_abc_3d_certified.py). Builds on the
Track A checkpoint ([`RESULTS_abc_3d_pinn.md`](RESULTS_abc_3d_pinn.md)).

## Environment

CPU, `python` (torch 2.9.0, float64). No GPU needed:
the residual is recomputed with a numpy/FFT spectral twin, and the trained field
is only *sampled* (closed-form) on a periodic grid. Full run: **~78 s**,
`n = 32` grid, time slices `t in {0, 0.25, 0.5, 0.75, 1.0}`.

## Stage A -- exact manufactured ABC baseline

`manufactured_abc_flow` (steady Beltrami ABC) -> `build_ns_cap_bundle` -> schema
check -> independent numpy replay (`verify_ns_cap_bundle`). Two independent
spectral implementations agree at machine precision.

| Quantity | Value |
|---|---|
| RMS momentum residual | 3.5e-15 |
| max \|momentum residual\| | 2.2e-14 |
| max \|divergence\| | 0.0 |
| max \|pressure-Poisson\| | 4.1e-13 |
| replay momentum max-abs-diff | 0.0 (bit-for-bit) |
| `residual_samples_match` / `schema_ok` | `True` / `True` |

## Stage B -- bridge the trained field into CAP bundles

Load the Track A `VectorPotentialField`, sample `u = curl(A)`, `p`, and the
**closed-form** time derivative `u_t` (`ops.derivative(..., axis=t)`) on a
periodic grid at each time slice, seal a CAP bundle, validate the schema, and
independently replay it. `rel-L2 vs exact ABC` is a *fully independent* numpy
re-check of Track A's torch-side accuracy.

| t | rel-L2 vs exact ABC | RMS momentum residual | max \|div u\| | replay match |
|---|---|---|---|---|
| 0.00 | 0.0014 | 0.0616 | 3.8e-14 | yes |
| 0.25 | 0.0096 | 0.0284 | 3.2e-14 | yes |
| 0.50 | 0.0069 | 0.0746 | 5.9e-14 | yes |
| 0.75 | 0.0149 | 0.0556 | 8.5e-14 | yes |
| 1.00 | 0.0110 | 0.1560 | 9.4e-14 | yes |
| **agg** | **mean 0.0088 / max 0.0149** | **mean 0.075** | **<= 9.4e-14** | **all yes** |

- **Accuracy replay:** mean 0.9% velocity error vs the exact decaying ABC,
  recomputed in numpy -- an independent confirmation of Track A's 0.0097.
- **Incompressibility:** `div u` stays at machine zero (the `curl(A)` cage is
  structural, and the spectral twin sees it too).
- **Residual:** the momentum residual (mean RMS 0.075) matches Track A's 0.069;
  it is larger near `t = 1` where the field is least accurate -- reported
  honestly, not hidden.
- Every slice: `schema_ok = True`, `residual_samples_match = True`,
  `honesty.unproven_claim = False`, `honesty.interval_verified = False`.

## Stage C -- proof-machine verdict + honesty gate

Adjudicate the shipped 3D `beltrami_abc_flow` fixture through
`build_default_machine()` (`navier_stokes_periodic_residual` kind).

| Conjecture | status | schema | replay | honesty | note |
|---|---|---|---|---|---|
| honest ABC residual | **PROVED** | ok | ok | ok | `exact_solution_claim=True`, `unproven_claim=False` |
| forged `unproven_claim=True` | **BLOCKED** | ok | ok | **fail** | *"an asserted claim is not supported by the certificate's honesty flags"* |

The forged claim is rejected purely by the honesty gate -- the certificate
cannot be made to assert a global-regularity result.

## Stage D -- axisymmetric-swirl interval + blow-up closure (3D-reduced rigor)

The genuine rigorous frontier: a finite-dimensional axisymmetric-swirl candidate
with outward-rounded **interval** certificates, each cross-checked by its numpy
replay twin.

| Stage | result |
|---|---|
| refined candidate (`residual_descended`, replay) | `True` / match |
| interval report (`interval_verified`) | **`True`** |
| interval replay (`interval_report_match`, `stage`) | match / `interval_obligation_ready` |
| tail / axis / continuum certified | `True` / `True` / `True` |
| blow-up closure replay (`closure_report_match`) | match |

Closure obligations (honest):

| obligation | met |
|---|---|
| `axis_smoothness` | yes |
| `finite_energy_initial_data` | yes |
| `linearized_invertibility` | yes |
| `norm_divergence` | yes |
| `radii_polynomial_closure` | **no (open)** |
| `operator_theoretic_invertibility` | **no (open)** |

The interval bounds are certified and replay-checked (`interval_obligation_ready`),
but the two *continuum* obligations remain open -- so no theorem-grade upgrade is
claimed. `unproven_claim = False` throughout. This is the honest boundary between
what the interval machinery certifies today and what a continuum proof would
require.

## Artifacts

Large CAP bundles (~24 MB each: full velocity / pressure / residual samples) and
the axisymmetric reports are persisted under
`artifacts/omnibias_runs/abc3d_certified/` (gitignored), with a compact
`certified_summary.json`. Bundles are JSON and re-verifiable offline via
`omnibias.symbolic.verify_ns_cap_bundle`.

## Reproduce

```bash
# smoke (CPU, ~10 s)
python -m examples.certified_fluid_dynamics.run_abc_3d_certified --smoke

# full run against the Track A v2 checkpoint
python -m examples.certified_fluid_dynamics.run_abc_3d_certified \
  --ckpt-dir "artifacts/omnibias_runs/abc3d_gpu_v2" \
  --out-dir  "artifacts/omnibias_runs/abc3d_certified"
```
