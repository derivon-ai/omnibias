# PINN four-gap capability matrix

Acceptance-gated status after the four-gap closure program. Scripts under
`benchmarks/` support `--smoke` (CI) and `--full` (multi-seed). Compact JSON
summaries are committed under `docs/benchmarks/`; heavier full-run copies also
land under `$OMNIBIAS_SCRATCH` (default `artifacts/`).

| Gap | Implemented capability | Empirical result (smoke) | Structural / by-construction | Certified scope |
| --- | --- | --- | --- | --- |
| Causality | Gated `march_solve` (torch+JAX), required IC, seam + same-time triviality diagnostics | Equal-budget 1-D heat arms in [`causal_marching.json`](causal_marching.json) | Window geometry / handoff / refuse unconverged advance under `advance_policy="gate"` | A-posteriori residual evidence only; not a solution bound |
| Geometry | Negative-inside R-CSG, SDF solver sampling, Dirichlet/Neumann/Robin cages + jets | Disk/annulus/CSG/nonconvex vs soft-penalty in [`geometry_sdf_smoke.json`](geometry_sdf_smoke.json) | Dirichlet on `φ=0`; junction failure for missing normals | Linear PDE certificates reusable when a stability constant exists |
| Operators | Multi-head DeepONet conditioning, geometry hard-BC wrap, parametric slabs | Held-out vs unconditioned vs PINN retrain in [`operator_zero_shot.json`](operator_zero_shot.json) | Per-instance hard BC when wrapped; function-only API preserved | Residual enclosure over coefficient boxes (operator verified path) — not solution error |
| Spectral | Multilevel FBPINN, mutation-free NTK + Lanczos, band scheduler controller | Equal-param arms + NTK alignment in [`spectral_bias_fbpinn.json`](spectral_bias_fbpinn.json) | POU blend weights ≥0 / sum-to-one when partition combine is used | NTK spectra are measurements, not training certificates |

**Not claimed:** universal elimination of non-convex optimization failure,
finite approximation capacity, identifiability, or PDE-dependent stability.
