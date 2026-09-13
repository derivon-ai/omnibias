# Neuromanifold collision-coordinate benchmark

Coalescing sigmoid neurons made ordinary amplitude/bias coordinates poorly
conditioned → centered moments keep a finite, trainable derivative atom at
the collision → `omnibias.{torch,jax}.confluence` and the opt-in
`ConfluentPackBank`. This benchmark measures that specific change in
coordinates on the [operator surface](../operator-surface.md).

The recorded experiment uses CPU float64, twenty fixed seeds, four trainable
parameters in each chart, 129 points on `[-2,2]`, and 200 Adam steps with
learning rate `0.01`. Both charts start from the same real-arithmetic pair
function. The objective is the sampled mean squared error against `sigmoid'`.
All reported fitting errors use this finite grid; they are not a held-out
generalization result or a uniform domain bound.

## Reproduce

```bash
uv run python benchmarks/neuromanifold_collisions.py
uv run python benchmarks/neuromanifold_collisions.py --full20
```

The default smoke run has two seeds, 33 points, and 30 steps per chart.
`--full20` writes `$OMNIBIAS_SCRATCH/neuromanifold_collisions/full20.json`, with
`OMNIBIAS_SCRATCH` defaulting to `artifacts/`. `--output` selects another file.
The [recorded full report](neuromanifold-collisions.json) includes every trial,
the conditioning scan, call counts, allocation measurements, and the boundary
training trace. The script alternates chart execution order across seeds and
warms the optimizer before timing.

## The two coordinate systems

The ordinary chart is
`a_plus sigmoid(x+b_plus) + a_minus sigmoid(x+b_minus)`.
Write `b_plus=mean+h`, `b_minus=mean-h`, `rho=h^2`,
`m0=a_plus+a_minus`, and `m1=h(a_plus-a_minus)`. The same finite pair is
`m0 A + m1 B`, where

\[
A=\frac{\sigma(z+h)+\sigma(z-h)}2,\qquad
B=\frac{\sigma(z+h)-\sigma(z-h)}{2h},\qquad z=x+\mathrm{mean}.
\]

At the attained boundary `rho=0`, the surviving operators are
`A=sigmoid(z)` and `B=sigmoid'(z)`. The implementation uses stable finite
identities away from collision and the shared derivative tower near it; it
does not compute a subtractive numerical difference quotient to define the
boundary derivative. Projection of `rho` onto `[0,infinity)` occurs between
optimizer steps and retains the right derivative at zero.

This is the bias-collapse axis. No temperature annealing appears in this
experiment. A small floating residual is also not enclosure collapse or proof.

## Recorded results

Medians over the twenty paired seeds:

| Quantity | Ordinary amplitude/bias chart | Confluent moment chart |
| --- | ---: | ---: |
| Final sampled MSE | `1.15097e-7` | `8.05340e-12` |
| Final sampled maximum absolute error | `6.47489e-4` | `7.82280e-6` |
| Instrumented wall time per run | `0.0752 s` | `0.6182 s` |
| Native sigmoid calls per run | `404` | `808` |
| Explicit parameter, gradient, and optimizer bytes | `132` | `132` |
| Peak Python allocation bytes | `4684.5` | `21890` |

The confluent chart has lower final MSE in 18 of the 20 seeds; its median MSE
is about 14,292 times lower on this workload. It takes about 8.22 times as long
and makes twice as many native sigmoid calls. This is a conditioning and
representation experiment, and it makes no claim of a forward-pass speed win.
Equal optimizer-step budgets do not mean equal wall-clock budgets.

Python allocation peaks exclude the backend tensor allocator. The explicit
tensor-byte count covers parameters, gradients, and optimizer state; it
excludes intermediate computation graphs. Wall times include the same call
instrumentation in both charts and vary with the host and load.

Fourteen confluent runs touch `rho=0` during joint training. None of those
twenty final iterates remains exactly on the boundary: later updates of the
other coordinates can move the spread back into the interior. A separate
live training fixture keeps `(m0,m1,mean)=(0,1,0)` and optimizes only `rho`.
It reaches zero on its first projected Adam step, stays there for the
remaining three steps, and returns the same float64 derivative-atom values
as the existing sigmoid derivative kernel.

## Conditioning and cancellation

The observation Jacobian uses the same 129-point grid. Representative
numerical SVD results are:

| Half-spread `h` | Ordinary condition estimate | Confluent condition estimate |
| --- | ---: | ---: |
| `1e-1` | `1.49817e5` | `164.469` |
| `1e-2` | `1.33532e9` | `163.939` |
| `1e-3` | `1.33371e13` | `163.933` |
| `1e-4` | `1.33249e17` | `163.933` |
| `1e-8` | `1.84284e23` | `163.933` |

The report also records numerical rank and the smallest singular value.
Condition estimates beyond machine resolution are diagnostics, not exact
rank certificates. At `h=1e-8`, numerical rank is three in the ordinary
coordinates and four in the confluent coordinates.

An independent 80-digit Decimal evaluation of the finite centered formula,
rounded once to float64, provides a numerical accuracy reference. At
`h=1e-8`, maximum sampled error is `1.20830e-8` for the ordinary evaluation
and `9.71445e-17` for the confluent evaluation. The Decimal reference is not
an outward-rounded enclosure and earns no proof tier.

## Numerical, rigorous, and derivative scope

The finite centered identities are exact over real arithmetic. The near
branch evaluates six even-series terms when `abs(h*eta)<=0.01`; its finite
spread truncation must be included in a rigorous error claim. Bank snapshots
bind both settings, the activation, capacity, maximum order, and transition
version. The shared rigorous confluence routines supply remainder bounds;
the benchmark itself reports numerical measurements. Neither this report
nor a successful training run sets `theorem_prover_verified` or
`mathlib_verified`.

Each allocated bank slot now evaluates one registered tower and shares it
across the ordinary, moment, and pair-series branches. The stable pair's
finite branch separately evaluates three sigmoids. This reuse preserves
the established accumulation order and per-backend polynomial evaluation.
Native transcendental kernels can still differ across backends by a ULP;
the shared coefficients do not prove that independently implemented native
exponentials are identical on every device.

Joint input/weight jets and JVPs of the described dense six-role realization
use the closed-form tower. Its VJP uses first-order backend reverse-mode
autodiff of the live evaluation. The observation-Jacobian calculation in this
benchmark uses forward-mode autodiff as an independent numerical diagnostic;
it is not labeled a closed-form weight adjoint.

The older `pinn.operator.{torch,jax}.parameter_jets.mixed_jet` manufactured
heat entrypoint preserves its shared Python-scalar result when `field=None`
and the method is `closed_form`. Select `live=True` to differentiate or JIT
that manufactured backend formula. Supplied field providers and callable
autodiff methods remain live automatically. The new generic realization APIs
always keep their input and weight graphs live.

Birth, cluster, pair, and derivative transitions are host transactions with
versioned proposals and explicit acceptance. Changed slots reset their
optimizer moments. Exact permutations instead transport all moment rows;
ordinary tanh sign relabeling changes first-moment signs and preserves second
moments. Unknown coupled optimizer states require an explicit adapter.
Legacy `sync_from` is rejected on the opt-in confluent bank because it cannot
preserve these transition and optimizer-state obligations.

The acceptance tests exercise fourth-order joint jets, finite-spread and
boundary behavior, shared tower call counts, checkpoint round trips, stale
proposal rejection, rollback, zero-output births, and moment transport. The
scientific continuation and PDE experiments have their own
[observation and certificate contracts](../api/neuromanifold-science.md).
