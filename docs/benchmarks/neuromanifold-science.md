# Neuromanifold science acceptance ledger

The [science APIs](../api/neuromanifold-science.md) are exercised by
`benchmarks/neuromanifold_science.py`. This benchmark checks numerical proposals,
rigorous controls, equal-budget synthetic observation design, and matrix-free
quantum geometry. It makes no new scientific discovery claim.

```bash
python benchmarks/neuromanifold_science.py
python benchmarks/neuromanifold_science.py --full
```

Outputs default to
`$OMNIBIAS_SCRATCH/neuromanifold_science/{smoke,full}.json`, with
`OMNIBIAS_SCRATCH=artifacts` when unset. `--output` selects an explicit JSON path.
Smoke uses three fixed seeds; full uses twenty consecutive seeds, 1729–1748.
The full run below was actually executed on 2026-09-13. The generated artifact
is `artifacts/neuromanifold_science/full.json`; rerunning regenerates it.

| Workload | Acceptance or control | Measured full run |
| --- | --- | --- |
| Bratu analytic solution family | Pseudo-arclength branch, numerical fold, interval augmented-root and normal-form conditions | 71 points; maximum residual `9.52e-11`; fold certified |
| Bratu fold parameter | Sound enclosure of the selected analytic family's fold | `lambda` in `[3.5138307191191203, 3.5138307191312026]` |
| Information geometry | Full-rank interval Jacobian passes; duplicate-column control is inconclusive | Both controls pass |
| Reduction | Value and first/second derivative error coverage; nonuniform singular-corner control refuses | Both controls pass |
| Exact finite QGT | Rational centered covariance and complete null-space replay | Pass |
| Observation design | Four observations per method from ten candidates; twenty noise seeds | Mean parameter error: designed `0.04566`, uniform `0.10793`, random `0.08596` |
| Small QGT/SR | Same 16 samples and energies for dense and matrix-free updates; eight parameters; twenty seeds | Maximum update discrepancy `8.97e-15`; maximum relative solve residual `6.33e-13` |
| Large QGT action | 10,000 parameters, 256 observations, chunk size 32; finite output without forming dense covariance | Pass; estimated dense covariance storage would be 800,000,000 bytes |

The design experiment is a manufactured linear inverse problem with independent
Gaussian noise. The selected set uses the existing greedy log-determinant
algorithm with an SPD prior and a cardinality budget. The benchmark also computes
the brute-force D-optimal oracle. The same candidate pool, observation budget,
per-seed noise realization, and unregularized least-squares estimator are used
for all three methods. The prior enters the greedy design objective and its
brute-force oracle, not the estimator.
The measured skill over uniform sampling is `1 - designed_error/uniform_error =
0.5770`. This is evidence for this finite workload, not an accuracy guarantee
for nonlinear inverse PDEs or an empirical universal advantage of D-design.
The `all_passed` flag concerns structural controls and numerical equivalence;
the positive empirical-recovery gate is recorded separately and requires full mode.

The large QGT action took about 0.71 seconds in this run. That measurement includes
the reported action's execution/compilation behavior and has no dense timing
comparison. The process maximum resident size was about 1,542 MiB across the full
Python process, not a per-operation memory measurement. The 800 MB dense estimate
is a shape-and-dtype calculation, not a measured allocation saved by a profiler.
These timings and process memory are environment-dependent.

The Bratu certificate uses an explicit analytic solution family rather than a
trained discretized PDE. General PDE continuation still needs discretization
error, solution-space coverage, and any relevant continuum existence theorem.
Generic Hopf certification consumes supplied sound normal-form enclosures; it
does not generate interval Lyapunov coefficients for every vector field.
The reduction certificate consumes explicit interval error callbacks; it is not
an asymptotic theorem generator. Finite QGT algebra does not establish sampling
accuracy or a wavefunction's physical validity. The current acceptance workload
does not test a new many-body phase, experimental data, or ground-state discovery.

Regression tests also cover numerical positive/negative/degenerate Hopf normal
forms, simple equilibrium branch switching (both pitchfork sides, transcritical,
coupled states, and changed coordinates), segment endpoint uniqueness, correlated observation covariance, redundant
nuisance projection, torch/JAX design gradients, singular reduction paths,
complex log amplitudes, failed-solve nonupdates, and operand-replay tampering.
The numerical producers retain their previous dense and scalar consumer APIs.
