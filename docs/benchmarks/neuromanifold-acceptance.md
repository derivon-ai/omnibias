# Neuromanifold: acceptance map for the sixteen gaps

This ledger maps the sixteen research gaps to implemented, scoped capabilities
and independent checks. The [operator surface](../operator-surface.md) is the
capability contract; the [geometry API](../api/neuromanifold.md),
[exact algebra API](../api/realization-algebra.md), and
[science API](../api/neuromanifold-science.md) give executable usage.
Coverage of a row means the stated finite capability is available. The accepted
implementation scope does not require benchmark superiority or a new discovery.

Bias collapse produces the activation derivative/moment coordinates. Temperature
collapse supplies a smooth choice over a fixed finite architecture catalogue.
Enclosure collapse concerns a sound enclosure becoming decisive. These three
limits remain distinct. Joint jets support all six `OperatorBlock` roles:
`identity`, `grad`, `laplacian`, `derivative`, `band`, and `integral`. The last
is the activation antiderivative window `S(z+b_hi)-S(z+b_lo)`, with `S'=sigma`;
domain and measure quadrature retain separate observation semantics.

## Capabilities and independent acceptance

API names below are public within the stated modules. Test labels resolve to
repository-relative paths in the following section.

| Gap: blocked object → usable capability | Implemented home and APIs | Independent acceptance and explicit boundary |
| --- | --- | --- |
| 1. Unspecified function image → declared realization and observation scope | `core.realization`: `ParameterLayout`, `RealizationSpec`, `ObservationSpec`; `{torch,jax}.realization.observe` | **Schema, Jets, Legacy inverse:** layouts and fingerprints, supplied-provider use, derivatives versus quadrature, and integral-kind distinctions. Legacy location-only APIs reject unsupported supplied models rather than substituting their manufactured example. Finite observations never establish global function equality. A complete-coefficient label needs an actual determining representation. |
| 2. Whole-network polynomial geometry → exact coefficient maps and real witnesses | `core.realization.polynomial.compile_coefficient_map`; `.rank.certify_generic_rank`; `.membership.classify_linear`, `classify_scalar_quadratic`, `classify_binary_quadratic`; `verify.neuroalgebra.bounded_realization_search` | **Algebra, Search:** independent rational dense forward evaluation, symbolic minors, signed quadratic congruence, algebraic substitution, and the `(x²,xy)` closure-only control. Complete membership covers arbitrary-depth homogeneous linear maps, scalar homogeneous quadratics, and binary homogeneous quadratics with two shared square units and arbitrary outputs. General compiled architectures get bounded search, which can be inconclusive. |
| 3. Hidden weight dependence → live joint input/parameter jets | `{torch,jax}.realization.parameter_jet`, `operator_parameter_jet`; `jet.mlp_joint_jet`, `jet_mv.mlp_joint_jet_mv`, `jet_mv.parameter_jet`; Torch `operator_block_jet` | **Jets, Joint entry points:** independent nested forward AD for every mixed coefficient through total order four, all six roles, live outer gradients, public wrapper graphs, JIT, JVP/VJP duality, and coefficient-budget refusal. Jets use the shared tower; VJP is first-order backend reverse AD. Full parameter tensors still have combinatorial coefficient counts. |
| 4. Diverging collision coordinates → bounded centered pairs and cluster moments | `{torch,jax}.confluence.centered_pair`, `moment_atom`; `core.confluence.initialize_moments`, `cluster_remainder`; `verify.neuromanifold.certify_transition` | **Pairs, Transactions, Rigorous, Confluence replay:** polynomial and independent AD controls at zero/tiny/finite spread, fourth derivatives, tails, exact stored-input moment conversion, and source-bound error budgets. Includes affine collision directions. Finite cluster truncation and both runtime series errors enter the budget; derivative domains crossing a numerical branch switch are refused. This covers declared collision families, not every neural boundary. |
| 5. Equivalent parameters → explicit symmetries and regular charts | `geometry.neuromanifold`: `hidden_permutation`, `hidden_sign`, `homogeneous_scaling`, `homogeneous_anchor_chart`, `duplicate_fibers`, `affine_quotient` | **Geometry, Transactions:** independent function evaluation after tanh parity/sigmoid complement/permutation/monomial actions; exact affine factorization, complete kernel and right inverse; state transport checks. Arbitrary monomial scaling is not assigned to sigmoid/tanh. Nonlinear global fibers are not inferred from a sampled nullspace. |
| 6. Unexplained small singular values → scoped rank and stratum evidence | `geometry.neuromanifold`: `rank_report`, `exact_rank_report`, `diagnose_stratum`, `monomial_curve_stratum`, `higher_order_visibility` | **Geometry, Algebra:** `t²`, `t³`, `(t²,t³)`, undersampling, a nonlinear map sharing an affine Jacobian, and nearly dependent stored dyadic matrices. Exact rank needs a nonzero source minor plus a source factorization; symbolic next-order minors can close the generic-rank bound. General image-singularity classification remains unresolved. |
| 7. Coordinate-dependent distances → observation metrics and reduced steps | `geometry.neuromanifold.ObservationMetric`, `Realization.metric_action`, `quotient_step`; existing `torch.optim.NaturalGradient`; `geometry.ChartSpec` derivative provider | **Geometry, Coordinate covariance, Chart provider, Jets:** matrix-free/dense actions, exact affine retraction, an independent regular least-squares solve, live derivatives, and invariant represented steps under a nonorthogonal coordinate change. The covariance control uses zero damping: adding the same `lambda I` in both charts is not a covariant regularizer. Observation provenance remains explicit; population information and quadrature accuracy are not inferred. |
| 8. Ignored realization curvature → regular extrinsic geometry and escape proposals | `geometry.neuromanifold.extrinsic_geometry`, `normal_acceleration`, `least_squares_hessian`, `escape_candidates` | **Geometry:** the circle's analytic normal curvature and least-squares Hessian, tangent/normal orthogonality, singular-chart refusal, and higher-order directional visibility controls. Escape candidates are finite-order proposals; complete singular critical-point enumeration and general tangent cones are unsupported. |
| 9. Flat symmetry families → transverse minimum certificates | `verify.neuromanifold.IntervalObjective`, `certify_slice_minimum`, `certify_quotient_minimum`, `formalize_minimum` | **Rigorous, Minimum replay:** independent quadratic roots, saddle/degenerate refusals, whole-box Hessians, actual stationary-point enclosures, affine quotient/Morse–Bott families, and source/rank/root-box/dimension tampering. A quotient claim defines the full loss as `ell(R theta)` for a verified affine chart. The critical-family scope is the preimage of the certified reduced box. |
| 10. Unsafe architecture edits → versioned proposals and certified commits | `{torch,jax}.confluent_bank.ConfluentPackBank`, `commit_transition`, `detect_transitions`; `verify.neuromanifold.selection` and `decode_architecture` | **Transactions, Rigorous:** pair/cluster/derivative/birth changes, unchanged parameter identities, checkpoint migration, stale rejection, rollback, moment reset/transport and unknown-optimizer refusal. Value and requested spatial/PDE/integral budgets cover the accepted domain. The temperature gap applies to the fixed finite catalogue, not retraining or global architecture optimality. |
| 11. Confounded scientific parameters → observation and nuisance geometry | `pinn.inverse.observation.ObservationModel`, `LikelihoodScores`, `observation_information`, `profile_information`, `identifiability_report`, `observation_visibility`; `verify.neuromanifold.scientific.certify_identifiability` | **Observations, Likelihood information, Scientific controls:** correlated covariance, redundant nuisance columns, Gaussian mean/log-scale Fisher from independent Gaussian quadrature, and actual second/third-order visibility where the first derivative vanishes. Parameter-dependent covariance requires explicit likelihood scores; nuisance profiling uses `info.information_factor`. Fixed-covariance interval full-rank checks can remain inconclusive for redundant nuisances. Neither local rank nor finite jets prove global mechanism uniqueness. |
| 12. Repeated uninformative measurements → finite observation design | `pinn.inverse.design.design_score`, `design_gradient`, `differentiable_design_score`, `select_design`; `submodular.design.information_design_problem` | **Observations, E-design, Finite selection, Design:** D/A/E derivatives against independent formulas and both AD backends; greedy/exhaustive selection against complete subset oracles; deterministic ties, overflow guards and preflight budget refusal. E is smooth only at a simple smallest eigenvalue. Numerical `select_design` earns no approximation factor. The separate cardinality `1-1/e` guarantee belongs to eligible normalized additive-PSD D-design with an SPD prior. |
| 13. Missed folds and branches → continuation, simple equilibrium switching and supplied event certificates | `geometry.continuation.ImplicitFamily`, `continue_branch`, `locate_fold`, `locate_hopf`; `geometry.continuation_directional.directional_residual_family`; `geometry.continuation_switch.switch_equilibrium_branch`; `dynamics.continuation.certify_event`, `certify_segment`, `certify_join` | **Continuation, Directional continuation, Branch switching, Validated continuation:** fold crossing, phase-normalized Hopf controls, Bratu's analytic family, pitchfork/transcritical switching, coupled nonlinear correction, uniform segment roots and endpoint uniqueness. Directional callbacks match independent dense derivatives under explicit allocation budgets. Switching requires a known parent tangent, rank `DF=n-1` and a nondegenerate mixed quadratic crossing; failed correctors remain inconclusive. Periodic-branch construction, higher codimension and general PDE tail control are outside scope. |
| 14. An apparent boundary without a law → explicit reduction grammar and uniform errors | `geometry.continuation.follow_model_boundary`; `symbolic.reduction.ParameterReduction`, `reduction_candidate`, `evaluate_reduction`; `verify.neuromanifold.scientific.certify_reduction` | **Reduction, Continuation, Scientific controls:** exponential coalescence, saturating-kinetic scaling with an explicit limit, analytic geodesic speed, value/first/second-derivative error coverage, and a nonuniform singular-corner refusal. Singular reconstruction needs a supplied reduced model; numerical path endpoints do not prove a limiting law. |
| 15. Dense or redundant quantum geometry → matrix-free QGT and SR | `ferminet.operator_sr.qgt_operator`, `matrixfree_sr_step`; `curvature.operators.pcg_solve`; `verify.neuromanifold.scientific.certify_quantum_geometry`, `replay_quantum_geometry` | **Quantum, Linear operators, Scientific controls:** independent dense complex QGT/SR, constant amplitude/phase null directions, failed-solve nonupdates, implicit solve gradients and exact rational covariance/kernel replay. The large benchmark forms no parameter-square covariance. Sampling uncertainty, ansatz-growth discovery and improved ground-state energy/infidelity are not established by these tests. |
| 16. Geometry claims without witnesses → operand-bound finite replay | `core.proof.realization_replay`, `core.proof.realization_algebra_replay`; `core.proof.lean_check.check_certificate`; `formal.mathlib_check.check_certificate` | **Finite replay, Algebra replay, Mathlib replay, Confluence replay, Minimum replay, Build lock:** actual positive builds in both projects, re-sealed false source operands rejected, unavailable tools leave flags false, and concurrent generated-file operations serialize and restore. Exact finite arithmetic is the earned formal scope described below. |

## Test paths and reproducible checks

All paths are relative to the repository root. Several rows share a test because
it checks an integration boundary; these are not sixteen independent benchmark
campaigns.

| Label | Test file |
| --- | --- |
| Schema | `packages/omnibias-core/tests/test_realization_schema.py` |
| Legacy inverse | `packages/omnibias-pinn/tests/test_inverse_api.py` |
| Algebra | `packages/omnibias-core/tests/test_realization_algebra.py` |
| Search | `packages/omnibias-verify/tests/test_neuroalgebra.py` |
| Jets | `tests/test_realization_jets.py` |
| Joint entry points | `tests/test_joint_jet_entrypoints.py` |
| Pairs | `tests/test_confluence_parity.py` |
| Transactions | `tests/test_confluent_bank_parity.py` |
| Collision benchmark | `tests/test_neuromanifold_collision_benchmark.py` |
| Geometry | `packages/omnibias-geometry/tests/test_neuromanifold.py` |
| Coordinate covariance | `packages/omnibias-geometry/tests/test_neuromanifold_coordinate_covariance.py` |
| Chart provider | `packages/omnibias-geometry/tests/test_neuromanifold_chart_provider.py` |
| Rigorous | `packages/omnibias-verify/tests/test_neuromanifold.py` |
| Observations | `packages/omnibias-pinn/tests/test_observation_geometry.py` |
| Likelihood information | `packages/omnibias-pinn/tests/test_likelihood_observation_information.py` |
| E-design | `packages/omnibias-pinn/tests/test_e_optimal_design.py` |
| Finite selection | `packages/omnibias-pinn/tests/test_design_selection.py` |
| Design | `packages/omnibias-submodular/tests/test_information_design.py` |
| Continuation | `packages/omnibias-geometry/tests/test_continuation.py` |
| Directional continuation | `packages/omnibias-geometry/tests/test_continuation_directional.py` |
| Branch switching | `packages/omnibias-geometry/tests/test_continuation_switch.py` |
| Validated continuation | `packages/omnibias-dynamics/tests/test_continuation_certificates.py` |
| Reduction | `packages/omnibias-symbolic/tests/test_model_reduction.py` |
| Quantum | `packages/omnibias-ferminet/tests/test_operator_sr.py` |
| Linear operators | `packages/omnibias-curvature/tests/test_matrixfree_operators.py` |
| Scientific controls | `packages/omnibias-verify/tests/test_neuromanifold_scientific.py` |
| Finite replay | `packages/omnibias-core/tests/proof/test_realization_replay.py` |
| Algebra replay | `packages/omnibias-core/tests/proof/test_realization_algebra_replay.py` |
| Mathlib replay | `packages/omnibias-formal/tests/test_realization_replay.py` |
| Confluence replay | `packages/omnibias-verify/tests/test_confluence_formal_coordinates.py` |
| Minimum replay | `packages/omnibias-verify/tests/test_neuromanifold_formal_minima.py` |
| Build lock | `packages/omnibias-core/tests/proof/test_lean_lock.py` |

Run a row with `python -m pytest <test-file> -q`. Execute both real Lean projects
with `lake build` from `formal/omnibias-verified-kernel` and
`formal/omnibias-analytic`. The Python bridges serialize writes and builds of
their shared generated modules. Missing toolchains skip applicable build tests
and cannot earn a formal flag. This ledger does not assert that the entire
repository test suite passes.

### Integration snapshot

The following checks completed on 2026-09-13. These selections overlap; their
counts must not be added into a purported total of distinct tests. Subsequent
extension additions have their own targeted checks.

| Check | Recorded result |
| --- | --- |
| Full `packages/omnibias-core/tests` | 3,402 passed; 207.87 seconds |
| Full `packages/omnibias-verify/tests` | 366 passed; 172.65 seconds |
| Default-device parity selection | 695 passed; 265.94 seconds |
| Executable documentation selection | 276 passed, 8 deselected; nine current science-page Python blocks additionally passed |
| Stable-workspace strict typing | 328 modules passed |
| Curated strict typing | 121 modules passed |
| Repository guard selection | 26 passed |
| Final finite-selector/E-design/submodular-oracle checks | 23 passed after the selector and overflow guards landed |
| Ruff over packages and tests; strict MkDocs | Passed |
| Both Lean projects and applicable positive/negative wrapper tests | Actual builds completed; false obligations rejected |

The default-device parity result does not resolve the separately reproduced CPU
exact-parity mismatch below. These are validation results for their named
selections, not an assertion that every test in every workspace distribution was
executed or that the scientific research programs are complete.

## Measured workloads

```bash
python benchmarks/neuromanifold_collisions.py --full20
python benchmarks/neuromanifold_science.py --full
```

The [collision report](neuromanifold-collisions.md) and
[recorded JSON](neuromanifold-collisions.json) describe CPU float64 runs with
twenty fixed seeds, 200 Adam steps per chart and 129 observations. Median sampled
MSE is `1.15097e-7` in ordinary coordinates and `8.05340e-12` in confluent
coordinates, with lower confluent MSE in 18/20 seeds. Median measured times are
`0.0752 s` and `0.6182 s`. This earns a conditioning/representation result at an
equal step budget, not a speed win or equal-compute advantage. The original
artifact is `artifacts/neuromanifold_collisions/full20.json`. The independent
collision-benchmark test checks budgets, Decimal-reference cancellation error,
and a separate live training fixture that attains `rho=0` and returns the
existing derivative atom. Attainment in that fixture does not mean every joint
training run ends on the boundary.

The [science report](neuromanifold-science.md) records the twenty-seed run in
`artifacts/neuromanifold_science/full.json`: Bratu fold parameter enclosure
`[3.5138307191191203, 3.5138307191312026]`; mean synthetic parameter error
`0.04566` for designed observations, `0.10793` for uniform and `0.08596` for random;
maximum dense-versus-matrix-free SR update difference `8.97e-15`; and a finite
10,000-parameter, 256-observation QGT action. Dense covariance storage would be
800 MB by shape and dtype; no dense timing comparison was measured. Bratu uses
an explicit analytic solution family, not an arbitrary trained PDE solution.

The scripts default to `$OMNIBIAS_SCRATCH`, or repository-relative `artifacts/`
when unset. Their JSON `all_passed`/empirical gates concern their declared
workloads, not universal performance or new scientific discovery.

### Separate CPU native-kernel parity issue

`artifacts/neuromanifold_collisions/native_parity_reproduction.json` compares the
pre-change `git HEAD` MultiPack modules with shared-tower evaluation on the same
CPU float64 fixture, `tests/test_multipack_parity.py::test_g3_torch_jax_bit_identical`.
Torch is bit-identical before/after; JAX is bit-identical before/after. Both
versions have the same cross-backend maximum absolute difference,
`1.1102230246251565e-16`. The stored differing values are
`0x1.ed070da2e0b6cp-2` and `0x1.ed070da2e0b6ap-2`.

This is a reproduced baseline native-transcendental mismatch. Shared integer
polynomials and preserved backend accumulation do not make distinct native
exponential kernels bit-identical on every device. Approximate parity and
within-backend preservation tests do not turn this failed exact-parity fixture
into a pass; it remains separate from new-feature acceptance.

## What the formal flags earn

Both Lean projects have actually built the supported finite obligations. Their
bridges bind the original operand lists, rather than accepting an unrelated
positive pivot or a claimed root box. The replay families cover rational
polynomial evaluation/inequalities, rank factorization plus a nonzero source
minor, interval LDL arithmetic, full Krawczyk matrix products and strict box
inclusion, error budgets, Sturm-chain arithmetic, algebraic substitution modulo
a defining polynomial, Laurent cancellation/tails, cluster moments, and complete
finite subdivision trees. Tampering remains a failure even after re-sealing the
JSON when the changed arithmetic is false.

The earned theorem is the emitted finite statement. General Sturm root-count
soundness, compiler semantics, matrix-factorization implications, Taylor
remainder theorems, Krawczyk existence/uniqueness, and analytic/geometric family
interpretation remain explicit dependencies where the Lean family does not prove
them. An interval callback also has to enclose the intended smooth quantity;
checking its returned rational endpoints cannot establish that correspondence.
Reducible algebraic representations whose equality holds only at the selected
root need a refined primitive before the current divisibility replay accepts
them. Budget exhaustion and unsupported elimination cases remain inconclusive
or explicit errors.

Core, backend and geometry producers keep their permissive licensing. Verification
and Mathlib consumers keep their existing copyleft/commercial tier; numerical
producers do not acquire imports of those consumers.

## Research claims beyond the accepted scope

Benchmark superiority, new scientific discoveries, and solutions of famous open
problems are not acceptance requirements for this implementation. The measurements
above describe their specific workloads. General nonlinear quotient certification,
complete neural-image singularity stratification, universal real feasibility,
and continuum PDE tail coverage require additional machinery and independent
evidence. These limitations do not turn a supported finite result into an
unresolved result; equally, a supported result must retain its declared scope.
