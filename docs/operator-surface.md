# omnibias operator surface (canonical capability matrix)

This page is the **single source of truth** for "what operators / integrals /
derivatives does omnibias actually expose". It exists to stop a common mistake:
under-stating the surface (for example, forgetting that omnibias has a
**closed-form integral operator**, not only closed-form derivatives).

Ground capability claims here (or in the cited code), never in memory. The code
of record is
[`omnibias.torch.blocks.operator`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/src/omnibias/torch/blocks/operator.py)
and [`omnibias.core.spec`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-core/src/omnibias/core/spec.py).

## `OperatorBlock` dispatch (all six roles)

`OperatorBlock(op=...)` selects the multi-bias arity `K` and the forward path.
There are **six** roles, not four:

| `op`          | K     | Forward path                                                    | Kind                         |
|---------------|-------|-----------------------------------------------------------------|------------------------------|
| `identity`    | 1     | `sigma(z + b)`                                                  | literal (Lemma identity)     |
| `grad`        | 2     | `sigma'(z + b_mean)`                                            | closed form (fast path)      |
| `laplacian`   | 3     | `sigma''(z + b_mean)`                                           | closed form (fast path)      |
| `derivative`  | n+1   | `sigma^(n)(z + b_mean)` for arbitrary order `n`                 | closed form (fast path)      |
| `band`        | 2     | `sigma(z + b_hi) - sigma(z + b_lo)`                             | literal window difference    |
| `integral`    | 2     | `S(z + b_hi) - S(z + b_lo)`, with `S' = sigma`                  | **closed-form antiderivative** |

The `grad` / `laplacian` / `derivative` paths require a base activation with a
closed-form derivative kernel (`ActivationSpec.fastpath`); the `integral` path
requires a closed-form antiderivative kernel (`ActivationSpec.integral`).
`OperatorBlock` raises a clear `TypeError` at construction when the base lacks
the required kernel.

`grad` and `laplacian` are just fixed-order aliases of `derivative` (orders 1
and 2); `derivative` takes an explicit `derivative_order`.

### One geometry behind all six roles

The roles are not six unrelated features. With `z = w . x`, each bias `b_k`
places a transition on the hyperplane `w . x + b_k = 0`, so an OMBU's `K` terms
are `K` **parallel hyperplanes**. Every role is a choice about the gap between
them:

| Gap | Roles | What you get |
|---|---|---|
| one plane (`K = 1`) | `identity` | the boundary itself, `sigma(z + b)` |
| gap `-> 0` (planes coalesce) | `grad`, `laplacian`, `derivative` | the transverse derivative tower `sigma^(n)` on the single surviving plane |
| gap held finite | `band`, `integral` | the slab between two planes: its response (`band`) or its accumulated mass (`integral`) |

That is why `integral` is closed form for the same reason the derivatives are:
it is the same parallel-hyperplane family, read in the antiderivative direction
instead of the derivative direction. The window `S(z+b_hi)-S(z+b_lo)` is
bias-geometry held finite, **not** Enclosure Collapse (`width -> 0` of a
sound enclosure; output is a point plus a proof; theory 01-14). See
[`theory.md` sec 4a](theory.md#4a-the-geometric-statement-what-collapses-geometrically).

The gated Wave-1 `BiasScan` ([scan.md](api/scan.md)) templates reuse these
same six roles; it is not a seventh `OperatorBlock` role. Equivariance is an
interior shift along `w` only. Gated Wave-3 `ScanNet`
([scannet.md](api/scannet.md)) stacks those templates; equivariance stays
**per-layer, per-direction, on-lattice**, not the translation group of
`R^D`. The convolution-class family is `scan(role)` over those six roles;
the catalog (role × scan, non-scan operators, inventable pointers, and
rejects) is theory spec 01-13 (**shipped**). A named
`BiasScan(op="integral")` layer is the first spend of that family
(then 09-14, **shipped**), not a seventh role.

## The antiderivative kernel `S` (why `integral` is closed form)

An [`ActivationSpec`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-core/src/omnibias/core/spec.py) may
carry an `integral` field: a closed-form antiderivative `S` with `S'(z) =
sigma(z)`. Definite bias-window integrals are then evaluated exactly as
`S(z + b_hi) - S(z + b_lo)` (fundamental theorem of calculus), with no
quadrature. This is the FTC twin of the bias-collapse derivative tower: the
same OMBU machinery, run in the antiderivative direction.

Canonical example: for `sigmoid`, `S(z) = softplus(z)` (since
`d/dz softplus(z) = sigmoid(z)`). The set of activations that ship a stable
antiderivative kernel is defined by the backend registry; the primitive
contract and its tests live in
[`omnibias.torch.fastpath.dispatch`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-torch/src/omnibias/torch/fastpath/dispatch.py)
and `packages/omnibias-torch/tests/test_integral_primitives.py`.

## Three distinct senses of "operator" (do not conflate)

"Operator" names three *different* objects in omnibias. Qualify which one you
mean:

1. **`OperatorBlock` / OMBU** -- the activation-level multi-bias unit
   (`omnibias.torch.blocks.operator`, the six roles above). Acts on scalar
   channels; closed-form derivative / antiderivative tower. This is what
   *this page* is about.
2. **Field operators** -- `grad` / `div` / `curl` / `laplacian` / `hessian` /
   `jacobian` / ... applied to a typed `FieldState`
   (`omnibias.fields`, `omnibias.pinn`). Closed-form when the field declares a
   dispatch tag (`one_layer`, `jet_mlp`, `spectral`, ...).
3. **Neural operator learning** -- a map between function spaces,
   `G: u(.) ↦ v(.)` (`omnibias.pinn.operator`: DeepONet / FNO). The DeepONet
   trunk is a jet field, so *query-coordinate* derivatives of `G(u)` are
   closed form; FNO derivatives stay FFT-based and periodic-grid-bound. This
   is *not* an `OperatorBlock` and is *not* a field operator.

Senses 1-3 are all real capabilities. Do not cite this page for sense 3
claims: the code of record for neural operators is
[`omnibias.pinn.operator`](api/pinn-operator.md).

## Three distinct senses of "integral" (do not conflate)

"Integral" names three *different* objects in omnibias. Qualify which one you
mean:

1. **Activation antiderivative window** -- `S(z + b_hi) - S(z + b_lo)`
   (`OperatorBlock(op="integral")`, `OMBU.analytic_integral`). Closed form, 1-D,
   per-channel, autograd-differentiable. This is *not* a domain integral.
2. **Domain quadrature** -- `sum_q w_q u(x_q)` over a spatial domain
   (`omnibias.fields` `integrate` / `inner_product` / `l2_norm` /
   `sobolev_norm`, and `omnibias.variational` / `omnibias.geometry`
   integrators). Numerical (quadrature), multidimensional, autograd through the
   field values and weights.
3. **Measure integral** -- `integral f dmu` against an abstract measure
   (`omnibias.measure`: measure-weighted quadrature, layer-cake /
   superlevel-set, importance sampling, simple-function approximation).
   Numerical/quadrature with a `Measure` abstraction, autograd; a rigorous
   certified variant (`certified_domain_integral`, `certified_lp_norm`,
   `certified_sobolev_norm`) lives in `omnibias.verify` / `omnibias.core.verified`.

Senses 1-3 are all real capabilities. The founding **bias collapse** is the
`delta -> 0` limit producing the *derivative* tower (senses `grad` / `laplacian`
/ `derivative` above); it is a different limit from **temperature collapse**, the
`beta -> inf` feasibility penalty in `omnibias.convex` / `-control` / `-routing`. See [`theory.md`](theory.md) sections 3-4 and the "two senses of
collapse" note.

## Capability matrix (honest labels)

`closed form` = the sigma tower / antiderivative (exact to machine precision);
`autodiff-exact` = autodiff of an analytic expression; `numerical` = grid /
quadrature; `certified` = a sound outward-rounded enclosure.

| Capability | How | Label | Where |
|---|---|---|---|
| `sigma^(n)(z)`, arbitrary `n` | closed-form tower | closed form | `omnibias.{torch,jax}`, `OperatorBlock(op="derivative")` |
| Gradient / Laplacian of a field | closed-form tower | closed form | `omnibias.fields`, `OperatorBlock(op="grad"/"laplacian")` |
| Directional / multivariate jets | Bell / Faa di Bruno | closed form | `omnibias.{torch,jax}.jet`, `.jet_mv` |
| Joint input / parameter jets, including six operator roles | explicit direction basis; affine coefficient convolution; live activation towers | closed form; budgeted coefficient count | `omnibias.{torch,jax}.realization` |
| Finite polynomial realization image | sparse rational coefficient compilation; algebraic witnesses and complete solvers for declared quadratic / linear families | exact rational / algebraic; general searches incomplete | `omnibias.core.realization`, `omnibias.verify.neuroalgebra` |
| Colliding neurons in attained zero-spread coordinates | centered pairs in `rho=h^2`; shared tower and bounded even-series evaluation | trainable; analytic and series errors enclosed separately from backend rounding | `omnibias.{torch,jax}.confluence`, `.confluent_bank`, `omnibias.verify.neuromanifold` |
| Observation geometry, fibers and regular affine quotients | live JVP / VJP, `J^T W J`, exact source factorizations, weighted normal acceleration | closed-form parameter jets; numerical geometry; exact declared affine relations | `omnibias.geometry.neuromanifold` |
| Certified collision commits and quotient minima | source-bound proposals; whole-box hyperdual Hessians; strict Krawczyk inclusion; exact affine loss factorization | certified on declared domains; finite Lean replay plus explicit analytic dependencies | `omnibias.verify.neuromanifold` |
| Residual-family continuation and simple fold / Hopf events | directional callbacks; budgeted dense correctors; adaptive pseudo-arclength, simple equilibrium branch switching and augmented systems; interval segment / event validation | numerical producer; finite-system certificate; analytic Bratu fold reference | `omnibias.geometry.continuation`, `.continuation_directional`, `.continuation_switch`, `omnibias.dynamics.continuation` |
| Identifiability and experiment design | whitened vector-observation jets; explicit likelihood-score providers for parameter-dependent noise; nuisance projection; D-/A-/E-objectives | numerical; E differentiable on simple spectrum; approximation guarantee only for fixed additive PSD information with SPD prior | `omnibias.pinn.inverse.observation`, `.design`, `omnibias.submodular.design` |
| Effective-law boundary reduction | finite fixing / coalescence / integer-power grammar; explicit reconstruction and domain error callbacks | numerical proposals; conditional interval value / derivative certificates | `omnibias.symbolic.reduction`, `omnibias.verify.neuromanifold` |
| Matrix-free quantum geometry and stochastic reconfiguration | chunked centered JVP / VJP actions; projected PCG | numerical; exact finite score-table replay is separate from Monte Carlo estimates | `omnibias.ferminet.operator_sr`, `omnibias.curvature.operators` |
| Antiderivative window of `sigma` | `S(z+b_hi)-S(z+b_lo)` | closed form | `OperatorBlock(op="integral")`, `OMBU.analytic_integral` |
| Definite domain integral `integral_Omega u dx` | quadrature | numerical | `omnibias.fields.{torch,jax}.integrate`, `omnibias.geometry`, `omnibias.variational` |
| L2 / Sobolev field norms | quadrature | numerical | `omnibias.fields.{torch,jax}.l2_norm` / `sobolev_norm` |
| Measure integral `integral f dmu` | measure-weighted quadrature / layer-cake / IS | numerical | `omnibias.measure` |
| Rigorous 1-D integral | interval quadrature + remainder | certified | `omnibias.core.verified.quadrature`, `omnibias.verify.certified_integral` |
| Rigorous multivariate integral / NN `L^p` / `H^k` norm | TM/interval + branch-and-bound | certified | `omnibias.verify.certified_domain_integral` / `certified_lp_norm` / `certified_sobolev_norm` |
| Fractional derivative (analytic class) | Gamma-ratio jet series | closed form | `omnibias.fractional...analytic` |
| Fractional derivative (general sampled `f`) | Grunwald-Letnikov / spectral | numerical | `omnibias.fractional...fractional` |
| Neural operator `G(u)(y)` query derivatives (DeepONet) | trunk jet × branch coeffs | closed form | `omnibias.pinn.operator` |
| Neural operator 4th-order residual (KS on DeepONet) | one order-4 trunk jet × live coeffs | closed form | `omnibias.pinn.operator` + shipped `KuramotoSivashinsky` |
| PirateNet α-skip (`α=0` identity) | gated residual apply | numerical | `omnibias.{jax,torch}.architectures.piratenet` (not ImageNet / not CCF stretch) |
| Boussinesq `(q, β)` envelopes | closed-form chart / product rule | autodiff-exact hats; closed-form envelope | `omnibias.pinn.{jax,torch}.equations.boussinesq_compactified` |
| Neural operator spectral-conv (FNO 1-D / 2-D) | FFT multiply | numerical | `omnibias.pinn.operator` |
| Operator multi-head conditioning (params / BC / geometry) | LayerNorm head encoders + fusion MLP; **width-1 parameter heads skip LayerNorm** (`nn.Identity`) so a scalar diffusivity is not collapsed to 0 | numerical | `omnibias.pinn.operator.ConditioningSpec` |
| Causal time-marching PINN training | Wang–Perdikaris weights + gated window ladder | numerical | `omnibias.pinn.train` |
| Depth-causal residual (08-05) | PDE residual of a per-layer decode + local GN | closed-form jet; numerical GN | `omnibias.pinn.train.{torch,jax}.depth_residual` (not time marching; not CCF Hilbert) |
| Causality / trivial-solution diagnostics | inversion fraction / same-time variance | measurement | `omnibias.pinn.train` |
| Curved-boundary hard Dirichlet (`u = g + φ·NN`) | SDF / ADF multiplicative cage | exact on `φ=0`; `φ` autodiff-exact | `omnibias.pinn.domain` |
| Curved Neumann / Robin (smooth primitives) | normalized-distance factor modes | by construction where normals exist | `omnibias.pinn.domain.torch.DistanceConstrainedField` |
| Negative-inside R-function CSG | Rvachev ops via `r_intersect_sdf` / `r_union_sdf` | algebraic zero-set | `omnibias.pinn.domain` |
| Multilevel FBPINN spectral-bias mitigation | hierarchy + partition combine / POU | numerical | `omnibias.pinn.{torch,jax}.fields.FBPINNField` |
| NTK eigenspectrum / spectral-bias index | empirical Jacobian + Lanczos / mode LRs | measurement | `omnibias.pinn.{torch,jax}.losses.ntk` |
| Keller n=3 Jacobian identity (Alpöge / Gallagher) | exact `Q` 3×3 Jacobian + witness eval | exact rational | `omnibias.holonomic.keller` (not a Jacobian-conjecture proof; `n=2` open) |
| Jacobian n=2 finite box `C_box(d,h,G)` | identical `det JF` + rational grid collision or Gabber inverse failure | exact rational | `omnibias.holonomic.jacobian_n2` (miss is not the parent; `jacobian_n2_claim` only on a violator) |
| Jacobian n=2 Case A `(b11,b21,b31)` leftover | exact `Q` identities; origin-only variety | exact rational | `omnibias.holonomic.jacobian_n2_case_a` (local seal; parent stays open; not JC) |
| Jacobian n=2 Case A `(b02,b03,b04)` leftover | exact `Q` identities; origin-only; Case A chart then empty | exact rational | `omnibias.holonomic.jacobian_n2_case_a_b02` (local seal; parent stays open; not JC) |
| Blind deg-2 tangent-sweep Keller search | side conditions + constant-Jac 3-to-1 fiber | exact rational | `omnibias.holonomic.keller_search` |
| Deg-3 tangent-sweep + finite discovery loop | `run_discovery` + score-guided walk; exact `Q` checker | exact rational | `omnibias.core.proof.discovery` / `keller_search` (family witness, not a parent proof) |
| Discovery catalog / characterization | statement → family → proposer → exact check; box-scoped uniqueness | exact rational or honest `BLOCKED` | `omnibias.core.proof.catalog` / `discovery` (not a parent proof) |
| Condition language / kind meta-family | `ConditionHypothesis` in a finite grammar; snap-to-`Q` accept gate; incomplete miss is `search_incomplete` | exact rational or honest `BLOCKED` | `omnibias.core.proof.condition` / `lift` (not “no condition exists”) |
| Shared observation / class loop | one `Observation`; binders return `Family \| None`; `select_class` ranks certified hits; gate proposes only | exact rational or honest `BLOCKED` | `omnibias.core.proof.observe` / `select_class` (never “no condition exists”) |
| Autonomy stack (ingest / grow / bilevel) | tag-optional packers; grow-on-miss; STLSQ / NeuralJet / optional field+PINN; `discover_observation` | exact rational or honest `BLOCKED` | `omnibias.symbolic.ingest` / `classloop` (PINN off by default; never “no condition exists”) |
| Exact activation identity (tanh Riccati) | closed-form `T_n` at rational `t`; nullity-one span | exact rational | `omnibias.symbolic.families.ActivationIdentityFamily` |
| P-recurrence span | exact `Q` null space in an `(order, degree)` box | exact rational | `omnibias.symbolic.families.RecurrenceSpanFamily` |
| Blind `K_5` colouring search | 1-flip walk; triangle-free predicate | exact finite | `omnibias.combinatorics.ramsey_search` (not Erdős 183) |
| DGG / Rybin unsplittable cost separation | exact `Fraction` H* enumerator | exact rational | `omnibias.combinatorics.unsplittable` (congestion theorem stays true) |
| 3-terminal DAG ≤6 (capped) | same cost-separation predicate; miss is `BLOCKED` | exact rational | `omnibias.combinatorics.unsplittable_dags` |
| Finite triangle-free colouring / `IsSaturated` | triple enumeration; matrix predicate | exact finite | `omnibias.combinatorics.ramsey` (not Erdős 183) |
| Compactness / 2-degenerate templates | adjacency-list predicates | exact finite | `omnibias.combinatorics.extremal` (not Erdős 146 / 180) |
| Banded Fourier radii (manufactured nearest-neighbour) | `BandedLinearPart` + radii polynomial | certified | `omnibias.core.verified.radii_spectral` (diagonal `laplacian_symbol` path unchanged) |
| Laguerre function basis on `[0, inf)` | exact `Q` coeffs + interval Horner | certified | `omnibias.core.verified.laguerre_basis` (not Fourier self-dual) |
| Cone-field hyperbolicity (finite orbit) | expanding cone of interval Jacobians | certified | `omnibias.dynamics._core.cone` (not Anosov / continuum chaos) |
| Poincare compactification / finite Dulac model | exact-Q chart remap; confluent-exponential derivation chain; interval exponent cover | exact rational plus certified intervals; model only | `omnibias.dynamics.{compactify,dulac,graphic}` (no physical return membership, remainder theorem, graphic cyclicity, or DRR closure) |
| Exact-Q Groebner basis / ideal membership | Buchberger (both criteria) + reduced basis; ideal/radical membership with a cofactor witness | exact rational, budget-refusing | `omnibias.holonomic._core.groebner` (doubly exponential worst case; `GroebnerBudgetExceeded` is an engineering limit) |
| Poincare-Lyapunov focal values / Bautin ideal | homological-equation solve degree by degree; Groebner basis of `(V_1..V_N)` | exact rational (or `Q[params]`); `bautin_ideal_stabilization_proved` always false | `omnibias.dynamics.focal`, `omnibias.dynamics.bautin` (first nonzero focal value is gauge-invariant; later ones are not) |
| Hilbert-16 finite Bautin-jet barrier | exact `V4 in (V1,V2,V3)` witness plus a formal next coefficient with nonzero Gröbner remainder | exact-Q finite stabilization and exact jet-only counterexample | `omnibias.dynamics.bautin_stabilization_barrier` (no all-orders recurrence, arbitrary singular return map, G2, or Hilbert XVI) |
| Hilbert-16 Songling lower-bound audit | exact quadratic source plus binary64 cancellation test against published 2048-bit four-cycle data | exact-Q source and sound negative backend-readiness result | `omnibias.dynamics.songling_lower_bound` (published `H(2)>=4` is not replayed; no H(2) upper bound or Hilbert XVI) |
| Hilbert-16 Part-A polygon/SOS audit | exact 22-annulus target layout; finite Route-2 inequalities; certified toy Putinar emptiness witness; Gram-size accounting | exact-Q geometry plus certified finite SOS sanity check | `omnibias.geometry.part_a_obstruction` (no all-realizations semialgebraic reduction, octic obstruction, 22-oval realization, or Hilbert XVI) |
| Resonant Poincare-Dulac normal form / derived corner map | per-degree diagonal operator at a rational hyperbolic saddle; first-order log-correction perturbation | exact rational normal form; first-order-derived corner coefficients | `omnibias.dynamics.saddle_normal_form` (rational eigenvalues only; not a physical Dulac-map-membership proof) |
| Collar-membership agreement / unique-cycle proof | interval overlap of a declared vs. a derived Dulac model on `x in [delta, delta0]`; optional `interval_newton` | sound, bounded away from the corner | `omnibias.dynamics.membership` (`corner_window_external` always true; `physical_return_membership_proved` always false) |
| Hilbert-16 tracked entry-exit product | slow-line partial fractions; `sep^2 * (h_1/(eps^3 mu sep^2))^(C eps)` | exact rational / real-log identities | `omnibias.dynamics.entry_exit_leading` (not a C2 remainder, G1, or Hilbert XVI) |
| Hilbert-16 fold I-map | implicit `I(r-delta)=kappa`; `dx/dkappa = delta^2/(r-delta)`; leading C2 of `log D'` | exact rational identities | `omnibias.dynamics.fold_leading` (not a physical remainder, G1, or Hilbert XVI) |
| Hilbert-16 shrinking-root I-map | two-root `dx/dkappa = (x-r1)(x-r2)/x`; `r1 -> 0` remainder | exact rational identities | `omnibias.dynamics.shrinking_root_leading` (not outgoing first-hit, G1, or Hilbert XVI) |
| Hilbert-16 canonical zeta | algebraic `r=-1` slow-line `zeta` at `lambda=0`; Cauchy majorant for `Z` | exact rational plus rectangular complex enclosure | `omnibias.dynamics.canonical_zeta` (fold compact is `fold_zeta`; not G1 or Hilbert XVI) |
| Hilbert-16 fold-compact zeta | Picard `k` plus Cauchy majorant for `Z` on `r in [1.4, 1.6]`, `L=r^2`, `lambda1=-2 r` | exact rational plus rectangular complex enclosure | `omnibias.dynamics.fold_zeta` (not physical C2, G1, or Hilbert XVI) |
| Hilbert-16 frozen-Z C2 | first-log-derivative and C2 remainder identities versus the lifted fold | exact rational identities | `omnibias.dynamics.physical_c2` (not `Z_x`, `sep>0`, G1, or Hilbert XVI) |
| Hilbert-16 unfrozen-Z `Z_x` gap | first-log-derivative identities including `Z_x` versus the lifted fold | exact rational identities | `omnibias.dynamics.z_x_gap` (not a bound on `Z_x`, `sep>0`, G1, or Hilbert XVI) |
| Hilbert-16 holomorphic `Z_v` bound | identities plus Interval `|Z_v|<1/4` on the cancelled-N kill compact, excluding 0 | exact rational plus Interval enclosure | `omnibias.dynamics.z_v_bound` (not fold `Z_x`, `sep>0`, G1, or Hilbert XVI) |
| Hilbert-16 slow-line `Z_V` chain | identities plus Interval `|Z_V|<1/4` on the kill compact and holomorphic `|Z_v|<1/4` on the fold wall | exact rational plus Interval enclosure | `omnibias.dynamics.z_slow_v` (not fold I-map `Z_x`, `sep>0`, G1, or Hilbert XVI) |
| Hilbert-16 fold I-map `Z_x` | matching-chart `Z_x=Z_v eps/ell` plus Interval `|Z_x|<1/100` on `r in [1.4, 1.6]`, `eps in [0, 0.02]` | exact rational plus Interval enclosure | `omnibias.dynamics.fold_z_x` (not `sep>0`, G1, or Hilbert XVI) |
| Hilbert-16 Stage-B inflation | kill-line Picard `|Delta x|<1/3` on `sep in (0, 1]`, `eps in [0, 1/16]` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_b` (not Stage A/C, C2, G1, or Hilbert XVI) |
| Hilbert-16 Stage-A wall | kill-line `B_-(a)=theta(1+theta)sep^2` at `theta=1/8`; Interval `a>1/4` and `Psi_pre` factor `<1/4` on `sep in [0, 1]` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_a` (not `dx_e/dkappa`, Stage C, G1, or Hilbert XVI) |
| Hilbert-16 `chi_b` threshold | kill-line `sep*S_pre<3` and `chi_b<9` on `sep in [1/2^16, 1]`; decay `c=1/16` | exact rational plus Interval enclosure | `omnibias.dynamics.chi_b` (not `dx_e/dkappa`, Stage C, G1, or Hilbert XVI) |
| Hilbert-16 `dx_e` leading | kill-line prefactor `<1/2` and threshold net exponent `>1/8` after the `y0` log remainder | exact rational plus Interval enclosure | `omnibias.dynamics.dx_e_leading` (not uniform-in-`chi`, Stage C, G1, or Hilbert XVI) |
| Hilbert-16 uniform-in-`chi` `dx_e` | kill-line `C<2` and extra `>1/16` on the `chi_b` compact; extra coefficient `3/32` | exact rational plus Interval enclosure | `omnibias.dynamics.dx_e_unif` (not Stage C, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C `a_min` | kill-line `end_lo>1/4` and `1/x<8` on the Stage-B end box; written `a_min=1/2` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c` (not an outgoing orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C exit energy | kill-line `T_e/eps^2 in (1/16, 1)` and `T_e > h_e` on the Stage-B end box | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_exit` (not an outgoing orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C leading `T_h` | kill-line `T_h>1/2` on the Stage-B end box at `y_1=1`; worst leading `3/4` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_th` (not an outgoing orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C start gap | kill-line `(T_e-h_1)/eps^2>1/32` at `y_1=1`; exact wall `T-h=1/4096` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_gap` (not an outgoing orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C C=0 envelope | kill-line `(1/32)(eps^2+h)<=T(h)<=eps^2+h`; declared `c=1/32` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_env` (not a C≠0 orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C C=2 integrating factor | kill-line `(h/h_1)^{C eps}<32`; exponent `<=3`; edge `9/4` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_if` (not `T(h)` after the remaining integral, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C C=2 `T(h)` majorant | kill-line `T(h)<=64(eps^2+h)`; slope `16/7` at the compact edge | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_int` (not first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C C=2 lower `T(h)` envelope | kill-line `T(h)>=(1/32)(eps^2+h)`; start remainder `> 1/16` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_lo` (not first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C C=2 tight `T(h)` ratio | kill-line `T(h)<=6(eps^2+h)`; edge factor `< 3` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_k` (not `T-h=O(eps)`, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C C=2 `T-h` bootstrap | kill-line `T-h<1` at `K=6`; linear `15 eps` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_boot` (not `O(eps)`, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C continuation rectangle | kill-line `V in [-2, -1/64]` on `h in [h_1, 1]`; left wall `sqrt(4)=2` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_rect` (not first-hit, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C height first-hit | kill-line comparison first-hit of `h=1`; time `<= 192 ln(16)` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_hit` (not Lohner, signed-label section, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C `E_out` first-hit | kill-line comparison first-hit of `E_out` from Stage-C start; `h_hit<1/8` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_sec` (not Lohner, chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C matching-chart Lohner | unique transverse first-hit of `x=4` from `(x,y)=(1/4,1)` on `sep in {0, 3/5, 1}` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_oneshot` (not every `eps`, chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C shrinking-eps Lohner | unique transverse first-hit of `x=n/4` at `n in {16, 20, 25}` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_oneshot_eps` (not every `eps`, chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C parametric-eps Lohner | unique transverse first-hit of `4 eps x=1` on four `1/800` slabs covering `[23/400, 1/16]` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_eps_span` (not every `eps`, chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C chart-O Lohner pack | unique transverse first-hit of `x=4` from `(x,y)=(1/4,1)` on `sep in {3/2, 7/4, 2}` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_origin` (not every `r1`, complete first-hit on chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C parametric-sep Lohner | unique transverse first-hit of `x=4` on eight `1/16` slabs covering `[3/2, 2]` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_origin_span` (not every `r1`, complete first-hit on chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C nearer-interface Lohner | unique transverse first-hit of `x=4` from `(x,y)=(1/8,1)` on eight `1/32` slabs covering `[7/4, 2]` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_origin_iface` (not every `r1`, complete first-hit on chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C `x=1/16` Lohner | unique transverse first-hit of `x=4` from `(x,y)=(1/16,1)` on eight `1/64` slabs covering `[15/8, 2]` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_origin_near` (not every `r1`, complete first-hit on chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C `x=1/32` Lohner | unique transverse first-hit of `x=4` from `(x,y)=(1/32,1)` on eight `1/128` slabs covering `[31/16, 2]` | exact rational plus QR-Lohner | `omnibias.dynamics.stage_c_origin_x32` (not every `r1`, complete first-hit on chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C uniform-`r1` comparison | phase-wise speed bound from `(x,y)=(1/4,1)` reaches `x=8` for every `r1` in `[0,1]` and every `eps` in `[1/32, 1/16]` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_compare` (not every `eps`, Lohner, complete first-hit on chart O, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C uniform comparison | `dx/dσ >= eps/2` from `(x,y)=(1/4,1)` for every `r1` in `[0,1]` and every `eps` in `(0, 1/16]`, so `x=(1/4)/eps` is hit | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_uniform` (not Lohner, `eps>1/16`, the shrinking interface, G1, or Hilbert XVI) |
| Hilbert-16 Stage-C interface comparison | `dx/dσ >= eps/5` from every start in `(0, 1/2]` for every `r1` in `[0,1]` and every `eps` in `(0, 1/16]` | exact rational plus Interval enclosure | `omnibias.dynamics.stage_c_interface` (not Lohner, the height-section flag, G1, or Hilbert XVI) |
| Hilbert-16 kill-line `sep * S_pre` | `sep * S_pre < 11/5` and `chi_b <= 8` for every `sep` in `(0, 1]`; net exponent `> 1/8` | exact rational plus Interval enclosure | `omnibias.dynamics.sep_spre` (not `dx_e` off the kill line, G1, or Hilbert XVI) |
| Hilbert-16 `dx_e` on `lambda1` in `[-4, -2]` | `h(sep/r1) < 11/5`; net exponent `> 1/8`; `chi_b <= 21/5`; `C < 2` | exact rational plus Interval enclosure | `omnibias.dynamics.dx_e_off` (not `lambda1 < -4`, not `lambda1` in `(-2, 0)`, G1, or Hilbert XVI) |
| Hilbert-16 `dx_e` for `lambda1 <= -2` | net exponent `> 1/8` under `X <= 2 rstar`; `chi_b <= 21/5`; `C < 2` | exact rational plus Interval enclosure | `omnibias.dynamics.dx_e_ray` (not `lambda1` in `(-2, 0)`, G1, or Hilbert XVI) |
| Hilbert-16 `dx_e` on `lambda1` in `[-3/2, -2)` | net exponent `> 1/8`; `chi_b <= 26/5`; `C < 2` | exact rational plus Interval enclosure | `omnibias.dynamics.dx_e_near` (not `lambda1` in `(-3/2, 0)`, G1, or Hilbert XVI) |
| Hilbert-16 `dx_e` on `lambda1` in `(-3/2, 0)` | net exponent `> 1/8` under an `eps` cap; `C < (1/5)/a` | exact rational plus Interval enclosure | `omnibias.dynamics.dx_e_open` (not a fixed `eps=1/16` slab, G1, or Hilbert XVI) |
| Hilbert-16 weighted-section assessment | finite weighted hit-time jets but ordinary `q=sep^2` derivatives grow as `q^-1`, `q^-2`; chart-O event speed vanishes with `r1` | exact rational negative assessment | `omnibias.dynamics.weighted_section` (falsifies H1 as a G1 discharge; not every closing map or Hilbert XVI) |
| Hilbert-16 quasi-homogeneous scale dichotomy | every rational monomial scale excluded against a frozen `eps^N` section; moving `eps^3 sep^2` section is a scalar counterexample to the all-atlas claim | exact rational negative assessment | `omnibias.dynamics.quasihomogeneous_dichotomy` (frozen-section one-scale no-go only; not physical C2, G1, or Hilbert XVI) |
| Hilbert-16 direct LN-format barrier | finite two-function `tau`/log-W chains with fixed degree and coefficients; analytic outer radius and chain norm grow linearly | exact-Q LN-chain replay plus rational growth identities | `omnibias.dynamics.ln_format_barrier` (direct representation only; normalized actual-return membership, G3, and Hilbert XVI remain open) |
| Hilbert-16 Abelian-to-return transfer | exact Picard--Fuchs syzygy plus a conditional rational epsilon threshold preserving supplied simple-zero margins under supplied remainder bounds | exact-Q holonomic and threshold certificates | `omnibias.dynamics.abelian_return_transfer` (regular Hamiltonian oval only; no open-DRR graphic reduction, physical remainder, endpoint capture, or Hilbert XVI) |
| Hilbert-16 outgoing x-corridor | cleared two-root I-map; `r1 log r1` majorant `2 sqrt(r1)-2 r1` | exact rational plus interval majorant | `omnibias.dynamics.outgoing_corridor` (not height-section first-hit, G1, or Hilbert XVI) |
| Hilbert-16 post-corridor margin | `T_*=Theta(eps^2)` and `(h/h_e)^{C eps}->1` independently of `r1` | exact rational plus interval majorant | `omnibias.dynamics.post_corridor` (not height-section first-hit, G1, or Hilbert XVI) |
| Hilbert-16 height envelope | `C=0` `T-h` conservation; uniform `|q|` ratio as `r1->0` | exact rational plus interval majorant | `omnibias.dynamics.height_envelope` (not height-section first-hit, G1, or Hilbert XVI) |
| Hilbert-16 C=2 `|q|` ratio | leading `|q|` ratio `<2` for every `x` on `lambda1=-2` | exact rational identities | `omnibias.dynamics.q_ratio_c2` (not `k=1+O(eps)`, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 `k`/cubic remainder | `C=0` normal `k=1+O(nu)` with exact `O(nu^2)` remainder; cubic `eps^4 x^3/3` | exact rational identities | `omnibias.dynamics.k_zeta_remainder` (not a `Z` bound, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 kill-compact `Z` | Cauchy majorant on `lambda1=-2`, `L in [0,1]` including `L=0` | Picard plus rectangular Cauchy | `omnibias.dynamics.kill_zeta` (not small enough for `C=2+delta`, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 cancelled-N `Z` | Holomorphic `Z` after cubic cancellation; `2 eps |V| |Z| < 1` on a declared slow-line compact | exact rational plus real Interval enclosure | `omnibias.dynamics.cancelled_n` (not `T-h` along the orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 `C!=0` height mix | `ell`/`V` mixing; `|ell_h|=|C| nu^2` on a declared compact with `ell>0` | exact rational plus Interval enclosure | `omnibias.dynamics.height_mix` (not `T-h` along the orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 `T_h` gap | Actual-versus-comparison `T_h` equals `k-1` at flux touching | exact rational identities | `omnibias.dynamics.orbit_th` (not an integrated `T-h` orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 `T-h` integral | Comparison-bootstrap `(T-h)_h` split; majorant `< 9 eps` on a declared compact | exact rational plus Interval sqrt majorant | `omnibias.dynamics.th_integral` (not a Lohner orbit, first-hit, G1, or Hilbert XVI) |
| Hilbert-16 cubic `(V,h)` orbit | QR-Lohner prefix; certified first-hit of `V=-1/4` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.vh_orbit` (not GRAZING `E_sigma`, G1, or Hilbert XVI) |
| Hilbert-16 matching-chart `E_out` | Certified first-hit of `E_out=V+rho+nu rho h+C nu^2 rho h^2` on `L in {9/25, 1/16, 0}`; GRAZING `E_sigma` excluded | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_out_section` (not uniform `eps->0`, G1, or Hilbert XVI) |
| Hilbert-16 shrinking-`eps` `E_out` pack | Certified first-hit on `n in {16,20,25}` at `L=0` inside `T=n^2/8` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_out_eps` (not a uniform-in-`eps` theorem, G1, or Hilbert XVI) |
| Hilbert-16 kill-line comparison speed | `F_V=eps phi`, `phi(-eps)>0`, `g<0`; `T<=(rho-eps)/(3 eps^3)` | exact rational plus Interval enclosure | `omnibias.dynamics.e_out_speed` (not Lohner for every `eps`, G1, or Hilbert XVI) |
| Hilbert-16 incoming GRAZING comparison speed | `F(0)=-4 eps^3(1+eps)`, `phi(0)<0`; `T<=1/(4 eps^3(1+eps))` | exact rational plus Interval enclosure | `omnibias.dynamics.e_sigma_speed` (not certified `E_sigma` first-hit, G1, or Hilbert XVI) |
| Hilbert-16 incoming `V=1/4` wall | Certified first-hit of `V=1/4` on the reverse cubic from `V=0`, `h=4 eps^3`, `L in {9/25, 1/16, 0}` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_in` (not certified `E_sigma` from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 declared-point `E_sigma` | Certified first-hit of GRAZING `E_sigma` from `(V,h)=(3/4,1/4)` on `L in {9/25, 1/16, 0}` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_hit` (not the GRAZING band from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 comparison `E_sigma` from `V=0` | Unique increasing `E_sigma` zero on the reverse-cubic comparison tube from GRAZING `V=0` | exact rational plus Interval enclosure | `omnibias.dynamics.e_sigma_from0` (not a Lohner event from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 uniform comparison `E_sigma` | `E_sigma>0` on eight Interval slabs covering `eps in [0, 1/8]` | exact rational plus Interval enclosure | `omnibias.dynamics.e_sigma_unif` (not a Lohner event for every `eps`, G1, or Hilbert XVI) |
| Hilbert-16 orbit-aligned wall `E_sigma` | Certified first-hit of GRAZING `E_sigma` from `(1/4,1/40)` inside the `V=1/4` GRAZING-from-`V=0` box | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_wall` (not a single Lohner run from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 wall-box `h`-interval `E_sigma` | Certified first-hit of GRAZING `E_sigma` on twelve `h`-slabs covering `[1/50, 4/125]` at `V=1/4` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_box` (not the whole wall `h`-interval, not a single Lohner run from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 L=0 whole-wall `h`-span `E_sigma` | Certified first-hit of GRAZING `E_sigma` on twenty-one `h`-slabs covering `[19/1000, 1/25]` at `V=1/4` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_span` (not the `L in {9/25, 1/16}` walls, not a single Lohner run from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 L-pack wall-span `E_sigma` | Certified first-hit of GRAZING `E_sigma` on eighteen `h`-slabs covering `[17/1000, 7/200]` at `V=1/4` on `L in {9/25, 1/16}` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_pack` (not a single Lohner run from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 shrinking-eps aligned `E_sigma` | Certified first-hit of GRAZING `E_sigma` from `(1/4,1/40)` at `eps=1/n` for `n in {16, 20, 25}` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_eps` (not a wall-span cover at every `n`, not a single Lohner run from `V=0`, G1, or Hilbert XVI) |
| Hilbert-16 one-shot Lohner `E_sigma` | Certified first-hit of GRAZING `E_sigma` from `V=0` in one `certify_stopped_event` at `eps=1/16` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_oneshot` (not uniform in `eps`, not `Z_x` C2, G1, or Hilbert XVI) |
| Hilbert-16 shrinking-eps one-shot `E_sigma` | Certified first-hit of GRAZING `E_sigma` from `V=0` at `eps=1/n` for `n in {16, 20, 25}` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_oneshot_eps` (not uniform in `eps`, not `Z_x` C2, G1, or Hilbert XVI) |
| Hilbert-16 compact aligned parametric-eps `E_sigma` | Certified first-hit of GRAZING `E_sigma` on three `eps`-slabs covering `[1/25, 1/16]` from `(1/4,1/40)` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_eps_span` (not every `eps`, not Lohner from `V=0` on that compact, not `Z_x` C2, G1, or Hilbert XVI) |
| Hilbert-16 lower aligned parametric-eps `E_sigma` | Certified first-hit of GRAZING `E_sigma` on six `eps`-slabs covering `[1/64, 1/16]` from `(1/4,1/40)` | exact rational plus Interval Taylor / Lohner | `omnibias.dynamics.e_sigma_eps_lo` (not every `eps`, not Lohner from `V=0` on that compact, not `Z_x` C2, G1, or Hilbert XVI) |
| Hilbert-16 obligation ledger | per-gate / per-DRR-case declared status; parent flags re-derived from entries | derived, unforgeable | `omnibias.dynamics.hilbert16_ledger` (`full_hilbert16_solved` false on every shipped ledger) |
| Two-site Hubbard GS sandwich | Ritz + Temple + blocked `LDL^T` | certified | `omnibias.core.verified.lattice_ground_state` (finite lattice; not thermo limit) |
| NPA linear-Hamiltonian lower bound | interval `LDL^T` moment matrix | certified | `omnibias.sos.npa` (lower bounds only; not full diagonalization) |
| Cohn-Elkies 1-D Hermite packing bound | exact Fourier + grid signs | certified | `omnibias.core.verified.cohn_elkies` (loose vs density `1`; not `d -> inf`) |
| IPM banded Fourier toy CAP | residual + tail + radii | certified toy; full IPM unearned | `omnibias.pinn.certified.ipm` (`navier_stokes_proof_claim=False`) |
| Volume-uniform strong-coupling family | polymer glueball on growing finite `d` | certified finite family | `omnibias.geometry.gauge.transfer.strong_coupling` (`yang_mills_claim=False`) |
| Named `Lambda <= 0.2` attempt | `Phi` / `H_t` enclosure + cited far-field | unearned (`certified=False`) | `omnibias.core.verified.debruijn_newman` (not RH; far-field stays external) |
| Combinatorial SOS (clique / 3-XOR) | degree-indexed strictly-PD residual | certified residual; oracle on graph | `omnibias.sos.combinatorial` (not P vs NP; not `SosDegreeFamily`) |
| Gaussian Slater `log|det M|` Laplacian | matrix identity on sampling contract | closed form | `omnibias.ferminet.antisymmetric` |
| Bloch / twist mixed partials | multivariate jet on a real affine twist | closed form | `omnibias.ferminet.bloch` (not a complex Bloch phase) |
| L-infinity minimax step | linearized epigraph + monotone accept | numerical | `omnibias.jax.optim.linf_minimax_step` (toy residual; not CCF champion) |
| Rationalize-and-certify discovery | exact `Fraction` residual | exact rational | `omnibias.symbolic.certify` (never `theorem_prover_verified`) |

## Where NOT to look for a capability

- omnibias has **no** measure-theoretic Lebesgue integral *beyond* the numerical
  forms in sense 3 above -- the abstract Lebesgue integral of an arbitrary
  measurable function is not a computable primitive (true for Riemann too).
- The certified integral / norm capability is **post-hoc and rigorous** (not
  autograd-trainable); the differentiable measure layer in `omnibias.measure`
  is the trainable counterpart.

If a capability is not on this page and not in the cited source, treat it as
absent and say so, rather than guessing.
