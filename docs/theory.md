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

## Two senses of "collapse" (do not conflate)

"Collapse" names two *different* limits in this codebase. Only the first --
the one defined in sec 3 above -- is **the** bias collapse.

| Sense | What moves | Limit | Output | Where |
|-------|------------|-------|--------|-------|
| **Bias collapse** (founding, this document) | the `K` biases coalesce, spread `delta -> 0` | finite difference -> derivative | a smooth `sigma^(K-1)(z + b_mean)` | `omnibias.torch.unit`, `omnibias.torch.stencil`; sec 3-4 above |
| **Temperature collapse** (downstream) | one gate sharpened, `beta -> inf` | soft threshold -> hard step | a 0/1 feasibility indicator (a step, *not* a derivative) | `omnibias-convex` / `-control` / `-routing` |

Both put a threshold inside `sigma` and take a limit, which is why they were once
both called "collapse" -- but they are not the same operation, and each now has
its own name. **Bias collapse** takes **many** biases to **one** and yields a
derivative; **temperature collapse** takes **one** soft gate and hardens it into
a constraint indicator. As `beta -> inf` a sigmoid saturates to 0 or 1 (a step),
so if you find yourself describing "bias collapse" as a hard step or a
constraint, you mean temperature collapse. The `K=2` "collapse output"
column in sec 5 below is the *founding* sense (`sigma'`).

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
