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
instead of the derivative direction. See
[`theory.md` sec 4a](theory.md#4a-the-geometric-statement-what-collapses-geometrically).

The gated Wave-1 `BiasScan` ([scan.md](api/scan.md)) templates reuse these
same six roles; it is not a seventh `OperatorBlock` role. Equivariance is an
interior shift along `w` only. Gated Wave-3 `ScanNet`
([scannet.md](api/scannet.md)) stacks those templates; equivariance stays
**per-layer, per-direction, on-lattice**, not the translation group of
`R^D`.

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
| Neural operator spectral-conv (FNO 1-D / 2-D) | FFT multiply | numerical | `omnibias.pinn.operator` |
| Operator multi-head conditioning (params / BC / geometry) | LayerNorm head encoders + fusion MLP; **width-1 parameter heads skip LayerNorm** (`nn.Identity`) so a scalar diffusivity is not collapsed to 0 | numerical | `omnibias.pinn.operator.ConditioningSpec` |
| Causal time-marching PINN training | Wang–Perdikaris weights + gated window ladder | numerical | `omnibias.pinn.train` |
| Causality / trivial-solution diagnostics | inversion fraction / same-time variance | measurement | `omnibias.pinn.train` |
| Curved-boundary hard Dirichlet (`u = g + φ·NN`) | SDF / ADF multiplicative cage | exact on `φ=0`; `φ` autodiff-exact | `omnibias.pinn.domain` |
| Curved Neumann / Robin (smooth primitives) | normalized-distance factor modes | by construction where normals exist | `omnibias.pinn.domain.torch.DistanceConstrainedField` |
| Negative-inside R-function CSG | Rvachev ops via `r_intersect_sdf` / `r_union_sdf` | algebraic zero-set | `omnibias.pinn.domain` |
| Multilevel FBPINN spectral-bias mitigation | hierarchy + partition combine / POU | numerical | `omnibias.pinn.{torch,jax}.fields.FBPINNField` |
| NTK eigenspectrum / spectral-bias index | empirical Jacobian + Lanczos / mode LRs | measurement | `omnibias.pinn.{torch,jax}.losses.ntk` |

## Where NOT to look for a capability

- omnibias has **no** measure-theoretic Lebesgue integral *beyond* the numerical
  forms in sense 3 above -- the abstract Lebesgue integral of an arbitrary
  measurable function is not a computable primitive (true for Riemann too).
- The certified integral / norm capability is **post-hoc and rigorous** (not
  autograd-trainable); the differentiable measure layer in `omnibias.measure`
  is the trainable counterpart.

If a capability is not on this page and not in the cited source, treat it as
absent and say so, rather than guessing.
