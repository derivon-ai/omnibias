# omnibias theory primer

This document gives a 2-page operator-framing summary of the multi-bias
activation primitive that omnibias is built around. For full proofs and
empirical validation, see the multi-bias activation paper.

## 1. The primitive

A K-bias multi-bias unit applied per channel is

```
f_K(z; b, s)  =  sum_{k=1}^{K}  s_k * sigma(z + b_k)
```

with K learnable biases `b = (b_1, ..., b_K)`, K signs `s = (s_1, ..., s_K)`,
and a fixed base activation `sigma`. We call this an
*OperatorMultiBiasUnit* (OMBU). For `K = 1` this is the standard
single-bias activation `sigma(z + b)`.

## 2. Lemma identity (identity nesting)

**Statement.** For any `sigma` and any `s` with `sum_k s_k = 1`, tying
`b_1 = ... = b_K = b` gives

```
f_K(z; (b, ..., b), s)  =  sigma(z + b)
```

bit-identically on any IEEE-754 implementation.

**Consequence.** A freshly-instantiated OMBU with this initialisation is
a drop-in replacement for `sigma`, no matter what `K` is. omnibias's
`identity_init_signs` (alternating `+1, -1, ...` for odd K; doubled
first sign `+2, -1, +1, ...` for even K) and `identity_init_biases`
(tied at `init_bias`) realise this.

## 3. Lemma collapse (derivative tower)

**Statement.** Place the biases on the central-difference stencil

```
b_k  =  b  +  (k - (K+1)/2) * delta,    k = 1..K,
```

and choose the rescaled signs

```
s_k  =  (-1)^(K-k) * binom(K-1, k-1) / delta^(K-1).
```

Then for `sigma in C^K`,

```
f_K(z; b, s)  ->  sigma^(K-1)(z + b)    as delta -> 0+,
```

with truncation error `O(delta^2)` (central) or `O(delta)` (forward).

**Consequence.** The K-bias unit is a finite-difference approximation
to the (K-1)-th derivative of `sigma`, evaluated at the bias mean `b`.
For `sigma` whose derivative tower has a closed form (sigmoid, tanh,
softplus, gaussian, exp -- the "Riccati class" plus exp), the limit is
computable exactly and cheaply, without going through the rescaled
finite difference. This is the *fast path*.

## 4. The fast path

For sigmoid, with `s = sigma(z)`,

```
sigma^(n)(z)  =  P_n(s),    P_0(s) = s,    P_{n+1}(s) = s (1 - s) * P_n'(s).
```

`P_n` is a polynomial of degree `n+1` in `s`. omnibias precomputes its
coefficients lazily (the *Eulerian* recursion) and evaluates by Horner.
**One** `torch.sigmoid` call per OMBU, regardless of `K`.

Analogous closed forms for the other smooth activations:

| Base       | Polynomial family               | One-call kernel     |
|------------|---------------------------------|---------------------|
| `sigmoid`  | Eulerian (in `sigma`)           | `eulerian.py`       |
| `tanh`     | Legendre-style (in `tanh`)      | `legendre.py`       |
| `softplus` | shift of sigmoid family         | `eulerian.py`       |
| `gaussian` | probabilist's Hermite (in `z`)  | `hermite.py`        |
| `exp`      | trivial (`exp^(n) = exp`)       | `classical.py`      |

The naive forward `sum_k s_k * sigma(z + b_k)` rescaled by
`1/delta^(K-1)` loses about `log10(1/delta^(K-1))` digits to
catastrophic cancellation. The fast path has no such cancellation: the
output is a polynomial in **one** sigmoid call, with no division by
`delta`.

## 4a. The geometric statement (what collapses, geometrically)

Sections 3-4 are the analytic statement. Here is the same thing said in the
input space, which is often the faster way to see why the primitive is one idea
rather than several.

Write the pre-activation as `z = w . x` for a weight row `w`. Then the single
term `sigma(z + b_k)` has its transition centred on the **hyperplane**

```
H_k  =  { x : w . x + b_k = 0 }.
```

Changing `b_k` slides that hyperplane along `w` without rotating it, so the `K`
terms of an OMBU are `K` **parallel** hyperplanes, offset from one another by
the bias spread. On the central-difference stencil of sec 3 the offsets are
`(k - (K+1)/2) * delta`, so the family is `K` parallel copies at spacing
`delta`, straddling the mean plane `w . x + b = 0`.

**Bias collapse is those `K` parallel hyperplanes coalescing into one.** As
`delta -> 0` the copies merge onto the single mean hyperplane, and what survives
the merge is not the plane itself -- one term already gives that -- but the
`(K-1)`-th derivative *transverse to it*. The unit stops reporting "which side
of the boundary am I on" and starts reporting "how sharply does the field turn
as I cross the boundary". That is the whole primitive: **one decision boundary,
carrying its derivative tower.**

Two consequences worth stating explicitly.

- The tower is **transverse and one-dimensional**. Every derivative `sigma^(n)`
  is taken along `w`, across the one surviving hyperplane. Multidimensional
  structure comes from composing units (`omnibias.{torch,jax}.jet_mv`), never
  from a single unit's collapse.
- **Order costs biases, not evaluations.** Reaching order `n` needs `K = n + 1`
  hyperplanes before the merge, and the closed forms of sec 4 mean the merged
  unit is still one `sigma` call. This is why the cost is `O(1)` in `n`.

### The `integral` role is the same geometry, uncollapsed

Keep exactly two of those parallel hyperplanes and *do not* shrink the gap. The
slab between them is a finite window, and integrating `sigma` across it is the
`integral` role of `OperatorBlock`:

```
integral_{z + b_lo}^{z + b_hi} sigma(t) dt  =  S(z + b_hi) - S(z + b_lo),   S' = sigma.
```

So the derivative roles and the `integral` role are the two directions of one
construction on the same parallel-hyperplane family:

| Gap between the planes | Operation | Output |
|---|---|---|
| `delta -> 0` (planes coalesce) | bias collapse | `sigma^(K-1)`, a derivative transverse to the single plane |
| gap held finite | antiderivative window | `S(z + b_hi) - S(z + b_lo)`, the mass of `sigma` in the slab |

Both are closed form, both cost one kernel evaluation, and both are per-channel
and one-dimensional. The `band` role is the finite-gap case read as a *response*
(a bump supported on the slab) rather than as an accumulated integral. See
[`operator-surface.md`](operator-surface.md) for the full role table and for the
three distinct things "integral" can mean.

### Wave-1 primitives

Three extensions of this geometry have code and CI smoke.

- **Order / packs** ([01-01](api/multipack.md)): `MultiPackUnit` evaluates a
  heterogeneous Birkhoff sample `sum_g c_g sigma^(n_g)(z + mu_g)` along one
  `w`. G1–G5 earned. Status is **shipped**.
- **Position / scan** ([01-02](api/scan.md)): `BiasScan` shares one pack
  template across a bank of offsets. Equivariance is an **interior lattice
  shift** along `w` (`R(z+Delta)[..., :-1]` vs `R(z)[..., 1:]`), not a
  circular wrap of `tanh'`. Soft-argmax `gamma` is a softmax readout;
  `gamma -> inf` would be temperature collapse, not `delta -> 0`. G1–G4
  CI-gated; G4 is a warmed-up voxelize-then-`cmbConv1d` pipeline win.
  Status is **shipped**.
- **Irregular stencils** ([difference API](api/difference.md)): exact-`Q`
  Birkhoff weights for arbitrary nodes and per-node orders. Order is
  asymptotic in the node scale `h`. G1–G4 earned. Status is **shipped**.
- **Enclosure Collapse** ([01-14](api/enclosure_collapse.md)): `width -> 0`
  of a sound enclosure (a point plus a proof); Width Law plus six
  `squeeze_*` wrappers. Not a derivative and not a 0/1 step. G1–G6
  earned. Status is **shipped**.

### Wave-3 gated architectures (not shipped)

Algebra plus stacked consumers of Wave-1, still **gated**. Cost / wall-time
gates are earned on smoke, not in CI `all_passed`.

- **Mollifier** ([01-05](api/mollifier.md)): `MollifierSpec` / `tail_bound`.
  Analytic bases have certified exponential tails, not compact support;
  higher-order kernels take **negative** values. G1–G4 CI-gated.
  Status is **shipped**.
- **Spectral design** ([01-07](api/spectral_design.md)): `BandPlan` /
  `peak_frequency`. Pack order is a **band selector**, not a Littlewood-Paley
  completeness claim. G1–G2/G4 CI-gated; G3 leftover-recorded, not in CI
  `all_passed`. Status is **shipped**.
- **OMBU frames** ([01-06](api/frames.md)): `FrameSpec` /
  `admissibility_constant`. `sigma'` is **not** admissible; frames are not
  orthonormal and not compactly supported. G1–G3 CI-gated; G4 denoising
  leftover-recorded, not in CI `all_passed`. Status is **shipped**.
- **Scan-Net** ([02-01](api/scannet.md)): stacked `BiasScan` banks. Equivariance
  is per-layer, per-direction, on-lattice; `gamma` is not `delta -> 0`.
  G1/G2/G3/G5 CI-gated; G4 k-NN leftover-recorded. Status is **shipped**.
- **Jet-KAN** ([02-03](api/jetkan.md)): univariate multi-pack edges. Exactness
  is of the **model jet**, not the target; the KA theorem does **not**
  justify the architecture. G2 leftover-recorded (leftover #41), not
  in CI `all_passed`. Status is **shipped**.
- **Weak-form VPINN** ([02-04](api/weak.md)): exact integrals only for
  polynomial coeffs on boxes; path recorded; boundary bound **on by default**.
  Status is **shipped**.
- **Transmission PINN** ([02-05](api/interface.md)): parallel interfaces;
  `alpha -> inf` is **interface sharpening**, neither collapse. G3 vs
  PartitionedField / FBPINN leftover-recorded. G4 hard vs penalized
  leftover-recorded. Import `Interface` from
  `omnibias.pinn.interface`, not the XPINN glue in
  `omnibias.pinn._core.interface`. Status is **shipped**.
- **Arrangement geometry** ([01-03](api/arrangement.md)): cells / tope graph /
  `soft_membership`. `beta -> inf` is temperature collapse, not founding
  `delta -> 0`. Sampling is a subgraph. Sound gap, not P vs NP. Cost vs
  `n`/`D` leftover-recorded. Status is **shipped**.
- **Tropical homotopy** ([01-08](api/tropical.md)): log / max-plus path;
  reuses `logsumexp_gap_bound`. G4 path-following earned
  (`path_follow` wired to `AnnealSchedule`). Cost vs `n`/`D`
  leftover-recorded. Status is **shipped**.
- **Equality locus** ([01-09](api/locus.md) / [02-12](api/locus.md)):
  constraint manifold, not a general PDE solver. Always returns
  `branch` / `condition` / `converged`. 01-09 status is **shipped**.
  02-12 G4 Burgers RH leftover-recorded. 02-12 status is **shipped**.
- **Jet vocabulary** ([01-10](api/jets.md)): dictionary and contact test,
  not a discovery and not a package. Status is **shipped**.
- **Rational stencil obligations** ([01-11](api/rational_stencil.md)):
  `C_j` / poisedness as finite rational Lean obligations. Algebra
  only, not the collapse. Status is **shipped**.
- **Conjugate Hilbert** ([01-12](api/conjugate.md)): line Hilbert only;
  G5 leftover-recorded (leftover #11); projection defect, not a
  stretch-gate clearing. Status is **shipped**.
- **Face-Net** ([02-02](api/facenet.md)): sampled tope subgraph; temperature
  collapse; sound gap, not P vs NP. G3 vs k-NN leftover-recorded. Cost
  vs `n`/`D` leftover-recorded. Status is **shipped**.
- **BEM-Net** ([02-06](api/bem.md)): PDE exact off-surface; BC approximated;
  linear constant-coeff homogeneous only. G2 disc-accuracy earned
  (`circle_dirichlet_density`). Single-layer cost leftover-recorded
  (leftover #42). G3 exterior win leftover-recorded. Status is
  **shipped**.
- **Pack tree** ([02-07](api/hierarchy.md)): 1-D offsets; `eta=0` bit-identical
  to dense; far-field is a truncation with a bound. G3 complexity
  earned (cached `O(p)` multipole, dense crossover). Status is **shipped**.
- **Equivariant scan** ([02-08](api/equivariant_scan.md)): gaussian-family
  steering only; discrete `C_L`, not SO(2)/SO(3). G5 anisotropic-interface
  leftover-recorded. Orbit cost leftover-recorded (leftover #43).
  Status is **shipped**.
- **Soliton tanh-method** ([02-09](api/travelling.md)): tanh algebra, not a
  collapse; multi-kink is not the n-soliton formula. G4 PINN init-win
  leftover-recorded. Algebraic cost leftover-recorded (leftover #44).
  Status is **shipped**.
- **Hermite ladder** ([02-10](api/ladder.md)): raw tower is not the QHO
  eigenbasis; Rodrigues reweight required. G4 FermiNet many-body
  leftover-recorded. Exact-vs-FD leftover-recorded (leftover #45).
  Anharmonic G5 leftover-recorded (may lose). Status is **shipped**.
- **Layered transfer** ([02-11](api/layered.md)): 1-D only;
  `continuum_claim=False`; distinct from `geometry.gauge.transfer`.
  G4 inverse-design leftover-recorded. Stack cost leftover-recorded
  (leftover #46). G5 conservation leftover-recorded. Status is
  **shipped**.
- **Linearizing transforms** ([02-13](api/transforms_pde.md)): named Cole-Hopf
  / Miura / Bäcklund / Darboux; exactness to jet order N; G3 Burgers
  leftover-recorded. 03-11 search stays designed. Status is **shipped**.
- **Holonomy band** ([02-14](api/holonomy_band.md)): closed form only abelian
  + transverse-constant; open lines gauge-dependent; no Yang-Mills / mass
  gap / continuum claim. G3 Magnus leftover-recorded. G4 gauge
  covariance earned (`random_u1_gauge`). Status is **shipped**.
- **Soft-population evolution** ([03-01](api/soft_evolution.md)):
  `log(P)/beta` gap; temperature collapse, not founding bias collapse.
  Not P vs NP. Status is **shipped**.
- **Arrangement LP** ([03-02](api/arrangement_lp.md)): learned-facet
  front end; not a new LP algorithm. Temperature collapse, not
  founding bias collapse. Status is **shipped**.
- **CSP collapse** ([03-03](api/csp.md)): finite-domain CSP; not a
  complete solver; not P vs NP. Temperature collapse, not founding
  bias collapse. Status is **shipped**.
- **Sliced OT** ([03-04](api/sliced_ot.md)): exact 1-D `W_1` per slice;
  founding bias collapse, not temperature collapse. Not Wasserstein.
  Status is **shipped**.
- **Morphology** ([03-05](api/morphology.md)): soft dilation / erosion
  via `logsumexp_beta`. Temperature collapse, not founding bias
  collapse. Status is **shipped**.
- **Neural quadrature** ([03-06](api/neural_quadrature.md)):
  moment-solved nodes plus Peano enclosure. Founding bias collapse,
  not temperature collapse. Status is **shipped**.
- **Scale flow** ([03-07](api/scale_flow.md)): `alpha` is a tempering
  scale, not a collapse. Nonlinear flow is a recorded truncation.
  Status is **shipped**.
- **Certified localization** ([03-08](api/certified_localization.md)):
  Krawczyk unique-peak enclosure. `Inconclusive` is first-class.
  Status is **shipped**.
- **Differentiable topology** ([03-09](api/differentiable_topology.md)):
  no differentiable Betti number. Temperature collapse, not founding
  bias collapse. Status is **shipped**.
- **Singularity tracking** ([03-10](api/singularity_tracking.md)):
  diagnostic, not a blow-up proof. Founding bias collapse, not
  temperature collapse. Status is **shipped**.
- **Lie symmetry discovery** ([03-11](api/symmetry_discovery.md)):
  in-ansatz only, not a classification. Founding bias collapse, not
  temperature collapse. Status is **shipped**.
- **Exact jet line search** ([03-12](api/line_search.md)): G4/G5
  leftover-recorded (leftovers #47 / #48). Founding bias collapse.
  Status is **shipped**.
- **Adaptive pack refinement** ([03-13](api/refine.md)): birth/growth
  bit-identical; death reports a bound. G4 earned. Status is
  **shipped**.
- **Pack Fisher** ([04-01](api/pack_fisher.md)): pack-parameter metric,
  not scalar `A''(theta)`. G1–G5 earned. Founding bias collapse.
  Status is **shipped**.
- **Conformal slabs** ([04-02](api/conformal_slabs.md)): three
  guarantee kinds stay apart; conformal is not sealable. G1–G6 CI.
  Status is **shipped**.
- **Inverse imaging** ([05-01](api/pinn_inverse.md)): G1–G7 earned.
  Locally-seeded `sd ~ alpha^(n-5/2)`; global search for `n=3` only.
  Status is **shipped**.
- **Beyond-PDE applications** ([05-02](api/tab.md)): G3b leftover
  #49 and G4 leftover #50 leftover-recorded. Temperature collapse
  on tabular; founding bias collapse on the scan. Status is **shipped**.
- **Acceptance gates** (06-01): shared protocol in
  `benchmarks/_gates.py`. Design record. Status is **shipped**.
- **Honesty and claim boundaries** ([06-02](honesty.md)): claim
  ladder + forbidden-claims register. Design record. Status is
  **shipped**.
- **Packaging and rollout** (06-03): G1–G5 earned (94/94 homes).
  G4/G5 vacuous, not promoted. Status is **shipped**.
- **Public primitive** (06-05): citation path; extract / paper /
  external stay later. Status is **shipped**.
- **Sub-obligation ledger** ([07-01](frontier-ledger.md)): RH is a
  non-entry. Design record. Status is **shipped**.
- **NS-adjacent weak form** ([07-02](api/ns_weak_form.md)): finite
  box / horizon / test space. Not a continuum regularity claim.
  Status is **shipped**.
- **Yang-Mills adjacent** ([07-04](api/geometry-gauge.md)): finite
  transfer gap; G1 factor measured. Mass gap / continuum stay
  external. Status is **shipped**.
- **Spectral floors** ([07-05](api/trial_spaces.md)): multi-pack
  trial spaces + arrangement SOS. Not a continuum gap or YM mass
  gap. Status is **shipped**.
- **Validated dynamics** ([07-06](api/validated_dynamics.md)):
  finite horizon; not a continuum existence theorem. Status is
  **shipped**.
- **Domain programs** ([07-07](api/domain_programs.md)): tooling,
  not a discovery. Status is **shipped**.
- **Training-idea ledger** (08-01): recommended stack; trainers do
  not clear Hilbert stretch. Design record. Status is **shipped**.
- **Composed curvature** ([08-02](api/composed_curvature.md)): slice
  escape, not a global min. Status is **shipped**.
- **Depth-causal local jet** ([08-03](api/local_jet.md)): greedy
  warm start, not ImageNet. Status is **shipped**.
- **Kantorovich Newton** ([08-04](api/kantorovich_newton.md)): empty
  ball is a reject, not a continuum PDE. Status is **shipped**.
- **Depth-causal residual** ([08-05](api/depth_residual.md)): marches
  in network depth, not time. Status is **shipped**.
- **Sharpness schedule** ([08-06](api/sharpness_schedule.md)):
  step-size signal, not a generalization claim. Status is
  **shipped**.
- **Block exact search** ([08-07](api/block_exact_search.md)):
  coordinate sweep, not a global solver. Status is **shipped**.
- **Implicit DEQ Newton** ([08-08](api/implicit.md)): IFT, not
  unrolled BPTT. Status is **shipped**.
- **Certified step** ([08-09](api/certified_step.md)): empty is a
  reject, not robustness. Status is **shipped**.
- **Invention ledger** (09-01): first-bet ranking; inventions do
  not clear Hilbert stretch. Design record. Status is **shipped**.
- **Jet-token transformer** ([09-02](api/jet_token.md)): model jet,
  not ImageNet. Status is **shipped**.
- **FTC-Net** ([09-03](api/ftc_net.md)): integral cell, not a
  VPINN. Status is **shipped**.
- **Frame-UNet** ([09-04](api/frame_unet.md)): band skip is not a
  collapse head. Status is **shipped**.
- **Taylor-model neuron** ([09-05](api/tm_neuron.md)): remainder
  sound, not a deep-net certificate. Status is **shipped**.
- **Coupling jet-flow** ([09-06](api/coupling_jet_flow.md)): finite
  couplings, not `integrate_cnf`. Status is **shipped**.
- **Pack-MoE** ([09-07](api/pack_moe.md)): slab-mass router, not
  softmax. Status is **shipped**.
- **Characteristic-Net** ([09-08](api/characteristic_net.md)):
  transport along learned `v`, not 02-13. Status is **shipped**.
- **Sheaf-atlas net** ([09-09](api/sheaf_atlas_net.md)): jet
  cocycle on partition charts, not a sheaf theorem. Status is
  **shipped**.

## Three senses of "collapse" (do not conflate)

"Collapse" names three *different* limits in this codebase. Only the first --
the one defined in sec 3 above -- is **the** bias collapse.

| Sense | What moves | Limit | Output | Where |
|-------|------------|-------|--------|-------|
| **Bias collapse** (founding, this document) | the `K` biases coalesce, spread `delta -> 0` | finite difference -> derivative | a smooth `sigma^(K-1)(z + b_mean)` | `omnibias.torch.unit`, `omnibias.torch.stencil`; sec 3-4 above |
| **Temperature collapse** (downstream) | one gate sharpened, `beta -> inf` | soft threshold -> hard step | a 0/1 feasibility indicator (a step, *not* a derivative) | `omnibias-convex` / `-control` / `-routing` |
| **Enclosure Collapse** (verified register) | enclosure width, `w = hi-lo -> 0` of a *sound enclosure* | certificates contract | a point **plus a proof**, or `Inconclusive` | `omnibias.core.verified.enclosure_collapse`, `omnibias.verify.enclosure_collapse`; [01-14](api/enclosure_collapse.md) |

The first two put a threshold inside `sigma` and take a limit, which is why they
were once both called "collapse" -- but they are not the same operation, and
each now has its own name. **Bias collapse** takes **many** biases to **one**
and yields a derivative; **temperature collapse** takes **one** soft gate and
hardens it into a constraint indicator. As `beta -> inf` a sigmoid saturates to
0 or 1 (a step), so if you find yourself describing "bias collapse" as a hard
step or a constraint, you mean temperature collapse. The `K=2` "collapse output"
column in sec 5 below is the *founding* sense (`sigma'`).

**Enclosure Collapse** is not a derivative and not a 0/1 step. `lo` and `hi`
are not two biases. The integral window `S(z+b_hi)-S(z+b_lo)` is bias-geometry
held finite, not this limit. Forcing `lo = hi` by clamping is unsound.

## 5. Operator dictionary

The choice of `sigma` is not arbitrary: it picks both an inductive bias
and a classical statistical / proximal-operator role. The following
table is the omnibias *activation dictionary*, indexed by the K=2
bias-collapse output `sigma'(z)`:

| Base activation | K=2 collapse output       | Operator role                                |
|-----------------|---------------------------|----------------------------------------------|
| `sigmoid`       | `s (1 - s)`               | Bernoulli variance / IRLS for logistic       |
| `tanh`          | `1 - tanh^2`              | symmetric IRLS bell                          |
| `softplus`      | `sigmoid`                 | Bernoulli mean / log-link Newton step        |
| `gaussian`      | `-z * exp(-z^2/2)`        | Hermite spectral basis, RBF kernel           |
| `huber`         | `clip(z, -tau, tau)`      | LASSO ISTA soft-shrink (proximal of L1)      |
| `arctan`        | `1 / (1 + z^2)`           | Cauchy IRLS weight (heavy-tailed regression) |
| `log1pu2`       | `2 z / (1 + z^2)`         | redescending M-estimator (Black-Anandan)     |
| `exp`           | `exp(z)`                  | Poisson-regression Newton step               |
| `relu`          | Heaviside step            | equality-constraint indicator / pseudoinverse|

The first three are log-partition functions of standard exponential
families, so a softplus-K=2 stack realises **logistic-regression IRLS
in network form**, and an exp-K=2 stack realises **Poisson-regression
Newton steps**. The Huber line is the operator-side equivalent of the
ISTA soft-thresholding step that powers LISTA and friends.

## 6. From OMBU to operator-typed layers

`OperatorBlock(op=...)` ties the operator role explicitly into the
forward dispatch. There are **six** roles (see the canonical
[operator-surface](operator-surface.md) page for the full capability matrix):

| `op`         | K   | Forward path                                                  |
|--------------|-----|---------------------------------------------------------------|
| `identity`   | 1   | `sigma(z + b)` (literal, Lemma identity)                      |
| `grad`       | 2   | `sigma'(z + b_mean)` (closed form, fast path)                 |
| `laplacian`  | 3   | `sigma''(z + b_mean)` (closed form, fast path)                |
| `derivative` | n+1 | `sigma^(n)(z + b_mean)` for arbitrary `n` (closed form)       |
| `band`       | 2   | `sigma(z + b_hi) - sigma(z + b_lo)` (literal window difference)|
| `integral`   | 2   | `S(z + b_hi) - S(z + b_lo)` with `S' = sigma` (**closed-form antiderivative**) |

`grad` / `laplacian` are fixed-order aliases of `derivative`; the `band` and
`integral` roles are the literal window and the closed-form antiderivative
window respectively -- the `integral` role is the fundamental-theorem twin of
the bias-collapse derivative tower (`S` is the antiderivative kernel
`ActivationSpec.integral`), **not** a difference of `sigma` values.

`cmbLinear`, `cmbConv1d`, `cmbConv2d` are the operator-typed analogues
of the standard `nn.Linear` / `nn.Conv*d` layers.

## 7. Three reference architectures

- `PINNHeat` (`omnibias.torch.architectures.pinn`): single-hidden-layer PINN
  for the 1D heat equation. Spatial and temporal derivatives come from
  closed-form `sigma'` and `sigma''` evaluations; **no
  `torch.autograd.grad` in the inner loop.**
- `CmbNet` (`omnibias.torch.architectures.cmbnet`): operator-typed CNN. The
  three convolutions carry explicit gradient / Laplacian / integral
  roles, recovering Sobel / LoG / DoG kernels by training.
- `CvxLasso` and `CvxLogistic` (`omnibias.torch.architectures.cvxlayer`):
  unrolled differentiable embedded convex solvers. Each unrolled layer
  is one ISTA / Newton step realised by a K=2 multi-bias collapse on
  Huber / softplus.

## 8. Where omnibias intentionally stops

- **Higher-order proximal kernels** (Huber for `n >= 2`, ReLU for
  `n >= 2`) raise a clear `NotImplementedError` rather than silently
  returning a distributional limit.
- **Mixed-order operators** (e.g. `op="biharmonic"` requiring K=5) are
  not pre-baked; build them by composing two `op="laplacian"` blocks or
  by writing a custom `OperatorBlock` subclass.
- **Auto-dispatch from literal to fast-path during training of free OMBU**
  is intentionally not done. The OMBU primitive's forward is always the
  literal sum, because for a free-form OMBU the analytic `sigma^(K-1)`
  is only equal to the literal in the stencil-rescaled-signs regime.
  `OperatorBlock` controls when the analytic path is the right thing.

## 9. Structural constraints (omnibias-pinn cages)

`omnibias-pinn` lifts the OMBU calculus to *constrained* PDE
solutions. A "cage" wraps an underlying field `phi` and exposes a
transformed component view such that a physical invariant holds *by
construction*.

### Streamfunction (2D incompressible)

Given a base field carrying a single scalar component `psi`, define

```
u = ∂_y psi,        v = -∂_x psi.
```

Then `∂_x u + ∂_y v = ∂_x ∂_y psi - ∂_y ∂_x psi ≡ 0`, so the
incompressibility constraint `div u = 0` holds for every input
sample, every parameter setting, and every numerical precision -- to
floating-point round-off.

Importantly, all *higher-order* derivatives of `(u, v)` reduce to
mixed partials of `psi`, which the spectral / one-layer / chebyshev
fields compute closed-form. Cage layering preserves the omnibias
fastpath end-to-end.

### Vector potential (3D incompressible)

Given three scalar components `(A1, A2, A3)`, set
`u = curl(A) = (∂_y A3 - ∂_z A2, ∂_z A1 - ∂_x A3, ∂_x A2 - ∂_y A1)`.
Then `div u = ∂_x(∂_y A3 - ∂_z A2) + ∂_y(∂_z A1 - ∂_x A3) + ∂_z(∂_x A2 - ∂_y A1) ≡ 0`
by the Schwarz / Clairaut symmetry of mixed partials.

Two corollaries:

1. **Gauge ambiguity**: `A` is determined only up to the gradient
   of an arbitrary scalar `chi`. Adding `∇chi` to `A` leaves `u`
   invariant, so a Coulomb-gauge constraint
   `div A = 0` may optionally be imposed as a *soft* term
   (`coulomb_gauge_loss`) to fix this redundancy.
2. **High-order spatial derivatives** of `u` reduce to mixed partials
   of `A`. For a Fourier basis these are diagonal multipliers in
   coefficient space, so the cage costs only one extra rearrangement
   per residual evaluation.

### Helmholtz projection (soft Hodge decomposition)

For applications that need a learned pressure (e.g. compressible-leaning
incompressible solvers), define `u = u_pred - ∇phi` with the
constraint `Δphi = div u_pred` enforced by the Poisson loss
`helmholtz_gauge_loss`. Hard incompressibility is then traded for a
training objective that converges quadratically once the Poisson
loss is small.

### Skew-symmetric advection (energy / enstrophy conservation)

The *naive* advection `(u . ∇) v` is energy-conserving only when
`div u = 0`. The skew-symmetric form

```
(u . ∇) v + ½ (div u) v
```

conserves `½ ∫ |v|^2 dx` exactly even when `div u != 0`, and the
combined form

```
½ [(u . ∇) v + ∇ . (u v)]
```

(the canonical "skew-symmetric" Navier-Stokes splitting) is bit-stable
for any predicted velocity field. `EnergyConserving.advection` and
`EnstrophyConserving.advection` implement these two flavours via the
ops surface, reusing the closed-form derivative path.

### Why this matters for PINNs

A standard PINN imposes invariants like `div u = 0` as soft penalties,
which produce competing loss terms with hand-tuned weights. Structural
cages eliminate the soft term entirely, removing one source of
ill-conditioning from the optimisation. Empirically, the
`VectorPotentialField` cage reduces 3D NS PINN training time by ~3x
compared to the soft-incompressibility baseline at the same final
forecast horizon (see
[`benchmarks.md`](benchmarks.md)).
