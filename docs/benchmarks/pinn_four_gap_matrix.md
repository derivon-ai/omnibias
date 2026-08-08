# PINN four-gap capability matrix

Acceptance-gated status after the four-gap closure program. Scripts under
`benchmarks/` support smoke (default, CI; writes `*_smoke.json` where
applicable) and `--full` (multi-seed acceptance artifact). Compact JSON
summaries are committed under `docs/benchmarks/`; heavier full-run copies also
land under `$OMNIBIAS_SCRATCH` (default `artifacts/`). Every artifact emits a
`gates` block via `benchmarks/_gates.py`.

| Gap | Constructive capability | Absolute gate | Empirical result (`--full`, 5 seeds, CPU float64) | Structural / by-construction |
| --- | --- | --- | --- | --- |
| Causality | Hard IC/BC cage + closed-form residual + gated `march_solve` (torch+JAX); soft-IC reaction family for the classical failure | Every arm `skill_score > 0`; heat best-arm median rel-L2 clears named threshold; reaction marching beats whole-interval; `advance_policy="gate"`; IC via `ic_fn` | **PASS** — **heat**: `whole_interval` median rel-L2 ≈ 9.4e-3, skill ≈ 0.9999 (seam ~ machine zero with `ic_fn`); **reaction** (`rho=12`): `causal_marching` ≈ 8.4e-2 beats `whole_interval` ≈ 0.99 ([`causal_marching.json`](causal_marching.json)) | Window geometry / handoff / refuse unconverged advance; hard IC/BC exact on heat; soft IC weight cliff on reaction |
| Geometry | Negative-inside R-CSG, SDF cages; manufactured interior Poisson | Boundary identity by construction; hard skill > 0 and hard boundary ≪ soft; nonconvex reports boundary-focused gate | **PASS** — disk hard skill ≈ 0.91, boundary max\|u−g\| ~1e-16; nonconvex L keeps boundary identity (~1e-12) while interior remains open ([`geometry_sdf.json`](geometry_sdf.json)) | Dirichlet on `φ=0`; junction failure for missing normals |
| Operators | Multi-head DeepONet + ETDRK4 / exact mode-1 heat reference; residual-PINN retrain | Maximum principle on slabs; conditioned+retrain skill > 0; conditioned median rel-L2 beats unconditioned **and** per-instance residual PINN | **PASS** — median rel-L2 conditioned 1.48e-2 < unconditioned 1.86e-2 and residual PINN 1.85e-2 ([`operator_zero_shot.json`](operator_zero_shot.json)) | Scalar parameter head must not LayerNorm-collapse; ETDRK4 / analytic Fourier exact linear advance |
| Spectral | Multilevel FBPINN + **one-shot least-squares** (no GD dynamics) + NTK diagnostics; instrumented wall-clock + `lstsq_matched` | `lstsq` rel-L2 < 5e-6 through f=16; capacity falsification at f=64 by raising feature count | **PASS** — median `lstsq` rel-L2 ≈ 5.2e-9 (~20× wall vs Adam at this scale; see `median_wall_seconds`); `lstsq_matched` weaker but beats plain GD; f=64 restored 0.165 → 1.3e-7; memory is structural `O(N H)` ([`spectral_bias_fbpinn.json`](spectral_bias_fbpinn.json)) | POU blend weights ≥0 / sum-to-one; spectral bias is GD dynamics, dissolved by one-shot collocation |

## Open frontier

Routes still open for attack (not structural blockers):

| Frontier | Constructive route |
| --- | --- |
| Nonconvex CSG interior accuracy (hard cage skill can lag soft) | Better Φ / sampling near re-entrant corners; exact-curvature optimizers; richer manufactured solutions |
| Non-convex optimization failure on nonlinear stiff PDEs | Exact-curvature optimizers (`omnibias.torch.optim`) + hard cages + residual certificates |
| Finite approximation capacity at extreme frequency | Raise feature bandwidth / multilevel FBPINN depth; capacity falsification already gates this |
| Identifiability of inverse coefficients | `solve_inverse` with cubic Gauss–Newton; seal coefficient enclosures |
| PDE-dependent stability constants for solution bounds | Reuse a-posteriori linear certificates when the constant exists; otherwise seal residual evidence and keep improving |

Claim a row solved when its absolute gate passes. Prefer that over pre-emptive
hedging.
