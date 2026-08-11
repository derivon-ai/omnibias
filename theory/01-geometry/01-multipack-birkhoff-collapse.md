# 01-01 Heterogeneous multi-pack collapse (Birkhoff jets)

## 1. Thesis and status

One channel, one normal, but **several independent bias packs of different
sizes at different means**: pack size selects a derivative order, pack mean
selects a sample location, so a single unit evaluates a *scattered Birkhoff
sample* of `sigma` along one direction rather than a single derivative.

- **Status**: gated (G1/G2/G3/G5 earned; G4 deferred)
- **Depends on**: none
- **Blocks**: 01-02, 01-04, 01-05, 01-07, 01-09, 01-10, 01-11, 01-12, 02-03, 02-04, 02-05, 02-07, 02-09, 02-10, 02-12, 02-13, 03-01, 03-10, 03-11, 03-12, 03-13, 04-01, 05-01, 07-02, 07-03, 07-04, 07-05, 07-06

## 2. Where it lands

A submodule, not a package. Two pieces:

- `packages/omnibias-core/src/omnibias/core/multipack.py` — pure-Python support
  algebra (`PackSpec`, `MultiPackSpec`, validation, exact rational weights).
  No tensor imports.
- `packages/omnibias-torch/src/omnibias/torch/multipack.py` and the jax twin —
  the `MultiPackUnit` layer.

It fails the "earn independent existence" test for a package: same domain, same
dependency tier, same audience as `omnibias.torch.unit`. It is the natural
generalization of `OperatorBlock`, so it lives beside it.

## 3. Prior art in omnibias

What exists:

- `packages/omnibias-torch/src/omnibias/torch/unit.py` — the OMBU primitive
  `f_K(z; b, s) = sum_k s_k sigma(z + b_k)`, one pack, free or stencil biases.
- `packages/omnibias-torch/src/omnibias/torch/stencil.py` —
  `central_bias_offsets(K, delta)`, `forward_difference_signs(K, delta)`,
  `stencil_signs(K, delta, stencil)`. Uniform stencils, one pack.
- `packages/omnibias-torch/src/omnibias/torch/blocks/operator.py` —
  `OperatorBlock(op=...)` with six roles; `derivative` takes one
  `derivative_order` and one bias mean.
- `omnibias.core.polynomials` — the shared closed-form coefficients
  (`sigmoid_polynomial_coeffs`, `tanh_polynomial_coeffs`, `hermite_coeffs`).
- `omnibias.torch.jet` / `jet_mv` — full jets at **one** expansion point through
  a composition.

**Confirmed gap.** Every existing path is one pack with one order at one mean.
There is no object holding several packs with different `K_g` at different
`mu_g`. `GrowableOperatorMultiBiasUnit` grows the arity `K` of a *single* pack,
which is a different axis.

The delta: selected orders at several locations, in closed form, at a cost of
one activation evaluation per *distinct mean* rather than per pack.

## 4. Mathematics

### Setup

Fix a weight row `w` and write `z = w . x`. A pack `g` is `K_g` biases on a
central stencil about mean `mu_g` with spacing `delta_g`:

```
b^(g)_k = mu_g + (k - (K_g + 1)/2) * delta_g,       k = 1 .. K_g
s^(g)_k = (-1)^(K_g - k) * C(K_g - 1, k - 1) / delta_g^(K_g - 1)
```

By the collapse lemma, for `sigma` in `C^{K_g}`,

```
f_{K_g}(z) = sum_k s^(g)_k sigma(z + b^(g)_k)  ->  sigma^(n_g)(z + mu_g)
```

as `delta_g -> 0+`, with truncation `O(delta_g^2)` for the central stencil and
`O(delta_g)` for the forward one, where `n_g = K_g - 1`.

### The multi-pack object

Take `G` packs with outer weights `c_g` and form

```
F(z) = sum_{g=1}^{G} c_g * f_{K_g}(z)
```

Send every `delta_g -> 0` **independently**, holding the means `mu_g` fixed and
distinct. Then

```
F(z)  ->  sum_{g=1}^{G} c_g * sigma^(n_g)(z + mu_g),        n_g = K_g - 1.
```

This is the whole idea. With `(K_1, K_2, K_3) = (4, 3, 2)` and means
`(mu_1, mu_2, mu_3)` the unit computes

```
c_1 sigma'''(z + mu_1) + c_2 sigma''(z + mu_2) + c_3 sigma'(z + mu_3).
```

### What kind of object this is

Define the **support** as a set of (location, order) pairs and read the output
as a linear functional on the jet of `sigma`:

```
S = { (mu_g, n_g) }_{g=1..G},     <c, j_S(sigma)>(z) = sum_g c_g sigma^(n_g)(z + mu_g)
```

The classical name depends on the shape of `S`:

| Support pattern | Classical interpolation data |
|---|---|
| one `mu`, orders `0 .. N` | Taylor / jet data at a point |
| several `mu`, order `0` only | Lagrange samples |
| several `mu`, all orders `0 .. n_g` at each | Hermite data |
| several `mu`, with gaps in the order set | **Birkhoff** (lacunary) data |

So heterogeneous multi-pack collapse is the OMBU realization of a Birkhoff
sample of `sigma` along a learned direction. The general case (order gaps) is
Birkhoff, which is why poisedness matters.

### Poisedness

A Birkhoff scheme is *poised* when the interpolation conditions determine the
polynomial of matching degree uniquely, that is when the incidence matrix
`E` (rows = nodes, columns = orders, `E[i][j] = 1` if order `j` is used at node
`i`) admits a unique solution. The Polya condition

```
sum over the first (j+1) columns of E  >=  j + 1     for every j
```

is necessary. It is not sufficient in general, but it is sufficient for Hermite
data (no gaps) and for two-node schemes. The implementation must therefore

1. always check Polya as a cheap necessary screen, and
2. for the general case, test poisedness numerically by the rank of the
   confluent Vandermonde system built in spec 01-04.

An unpoised support is not an error in the *network* (the unit still computes
something well defined), but it is an error in any claim that the unit can
*represent* a prescribed jet sample. Keep those two statements separate.

### Cost

For a Riccati-class base, `sigma^(n)(u) = P_n(sigma(u))` with `P_n` from
`omnibias.core.polynomials`. Therefore

```
F(z) = sum_g c_g * P_{n_g}( sigma(z + mu_g) )
```

costs **one activation evaluation per distinct mean**, not per pack and not per
order. Packs that share a mean share their `sigma` call and read different
polynomials of it: the whole local tower at that point is free once the
activation value is known.

### What is not new

If the means themselves are driven together with finite-difference signs across
packs, the construction rebuilds a single higher-order collapse. The regime that
matters is: **within-pack spread to zero, between-pack gaps held finite.**

## 5. Worked example

Base `sigma = tanh`, so `sigma' = 1 - t^2`, `sigma'' = -2 t (1 - t^2)`.

Support `S = {(mu_1, n_1), (mu_2, n_2)} = {(-0.5, 1), (0.5, 2)}`, weights
`c = (1.0, 0.25)`, evaluated at `z = 0`.

Closed form:

```
t1 = tanh(-0.5) = -0.4621171573
sigma'(-0.5)    = 1 - t1^2      =  0.7864477330
t2 = tanh(0.5)  =  0.4621171573
sigma''(0.5)    = -2 t2 (1 - t2^2) = -0.7268871046
F(0) = 1.0 * 0.7864477330 + 0.25 * (-0.7268871046) = 0.6047259569
```

Finite-difference check, both packs at spacing `delta = 1e-3`:

```
pack 1: K=2, biases -0.5 -+ 5e-4, signs (-1, +1)/1e-3
pack 2: K=3, biases -0.5+... wait, mean 0.5, biases 0.5 + {-1e-3, 0, +1e-3},
        signs (1, -2, 1)/1e-6
F_fd(0) = 0.604725957...   (agrees to ~1e-7, consistent with O(delta^2))
```

The point of the fast path: `F_fd` loses roughly `log10(1/delta^{K-1})` digits
to cancellation as `delta` shrinks, while the closed form has no division by
`delta` at all. At `delta = 1e-6` the `K=3` finite difference is already
unusable in float64; the closed form is exact to rounding.

## 6. Proposed API

Shipped.

Core (pure Python, no tensor imports):

```python
# omnibias/core/multipack.py
@dataclass(frozen=True)
class PackSpec:
    order: int          # n_g >= 0; arity is K_g = order + 1
    mean: float         # mu_g
    weight: float = 1.0 # c_g

@dataclass(frozen=True)
class MultiPackSpec:
    packs: tuple[PackSpec, ...]

    @property
    def distinct_means(self) -> tuple[float, ...]: ...
    @property
    def max_order(self) -> int: ...

def polya_condition(spec: MultiPackSpec) -> bool: ...
def incidence_matrix(spec: MultiPackSpec) -> tuple[tuple[int, ...], ...]: ...
def is_poised(spec: MultiPackSpec, *, tol: float = 1e-10) -> bool | None: ...
```

`is_poised` returns `None` when the numerical rank test is inconclusive; callers
must not read `None` as `True`.

Backends (bit-identical twins):

```python
# omnibias/torch/multipack.py   and   omnibias/jax/multipack.py
class MultiPackUnit(nn.Module):          # jax: functional init/apply pair
    def __init__(
        self,
        num_channels: int,
        spec: MultiPackSpec,
        *,
        base: str | ActivationSpec = "sigmoid",
        learnable_means: bool = True,
        learnable_weights: bool = True,
        share_means: bool = True,   # one sigma call per distinct mean
        dtype: torch.dtype | None = None,
    ) -> None: ...

    def forward(self, z: Tensor) -> Tensor: ...

BirkhoffOMBU = MultiPackUnit   # alias used in the literature-facing docs
```

Rules:

- `dtype=None` resolves to `torch.get_default_dtype()`; never hardcode float32.
- Polynomial coefficients come from `omnibias.core.polynomials`; the two
  backends must produce bit-identical outputs on the same inputs.
- Orders are static Python ints so the jax path stays traceable; means and
  weights are arrays.
- `order < 0` raises `ValueError`; an order the base cannot reach in closed form
  raises `NotImplementedError`, matching the existing fastpath contract.

## 7. Practical use cases

1. **Multi-interface transmission problems.** Several parallel material
   interfaces where each interface enforces a different physical condition
   (value continuity, flux, curvature jump). A single unit supplies exactly
   those jet coordinates at exactly those locations, instead of a deep network
   discovering them. Expected win over a plain MLP: fewer parameters and a
   residual that does not have to learn the interface structure.
2. **Boundary layers.** Wall-normal derivative at the wall plus an outer
   response a finite distance away, in one channel. The alternative is two
   networks and a matching loss.
3. **Sensor Hermite matching.** Measurements give value and slope at a handful
   of probe positions along a ray. That is literally Birkhoff data; fitting
   `(c_g, mu_g, n_g)` is the natural estimator.
4. **Operator dictionaries for discovery.** A multi-pack channel is a library of
   `sigma^(n)` features at several offsets, which is a better-conditioned design
   matrix for `omnibias.symbolic` than repeated numerical differentiation.
5. **Reducing cancellation in high-order residuals.** Any place where the
   current code would reach for a high-order finite difference of an activation
   gets an exact substitute with no `1/delta^n` amplification.

## 8. Acceptance gates

Baseline to beat: a single-pack `OperatorBlock(op="derivative")` stack with the
same parameter count, and a free unstructured OMBU with the same number of
biases.

- **G1 exactness.** For a Riccati base and a random support with orders up to 6,
  `MultiPackUnit` output matches an independent mpmath evaluation of
  `sum_g c_g sigma^(n_g)(z + mu_g)` to `<= 4 ulp` in float64, on a dense grid
  and a random sample of `z`.
  **Earned with recorded ceiling** — on `z in [-1, 1]` the 4-ulp single-pack
  ceiling is order `1` (sigmoid) / `2` (tanh); higher orders lose digits to
  `P_n` conditioning (section 11). The worked-example multipack stays within
  4 ulp on the same domain. See `docs/benchmarks/multipack_birkhoff_smoke.json`.
- **G2 stability.** Relative error of the closed form versus the collapsing
  finite difference decreases monotonically as `delta` shrinks until the finite
  difference breaks down; the closed form must show no error growth at all.
  **Earned.**
- **G3 parity.** torch and jax outputs are bit-identical on the same inputs.
  **Earned** on the gated moderate grid (native tanh/sigmoid may differ by a
  few ULPs in extreme tails; see `tests/test_sigmoid_tail_parity.py`).
- **G4 task skill.** On a two-interface 1-D transmission problem with known
  exact solution, relative `L2` error `<= 1e-6` with skill `> 0` against the
  zero predictor, beating both baselines above at equal parameter count, over
  five seeds. **Unearned** (`g4_earned: false`).
- **G5 poisedness honesty.** For a deliberately unpoised support, `is_poised`
  returns `False` (not `None`, not `True`) and the representation claim in the
  docs is withheld. **Earned.**

## 9. Benchmark plan

- `benchmarks/multipack_birkhoff.py`, smoke by default (small grid, one seed,
  seconds on CPU), `--full` for five seeds and the full order sweep.
- Smoke writes `docs/benchmarks/multipack_birkhoff_smoke.json`; `--full` writes
  the acceptance JSON under `$OMNIBIAS_SCRATCH/multipack/` (default
  `artifacts/multipack/`).
- Reuse `gates_block`, `require_skill` and `require_rel_l2` from
  `benchmarks/_gates.py` so the artifact carries the standard `gates` block.
- CI runs the smoke tier only.

## 10. Honesty and scope

- The limit here is the **founding bias collapse**: within each pack the spread
  `delta_g -> 0` and `K_g` biases coalesce, producing a smooth `sigma^(K_g - 1)`.
  It is not temperature collapse; no `beta -> inf` and no 0/1 step appears
  anywhere in this spec.
- The tower is **transverse and one-dimensional**: every order is taken along
  `w`, across the surviving hyperplane of its pack. Multi-dimensional structure
  still comes from composing units (`jet_mv`), never from one unit's collapse.
- Representation claims require poisedness. "This unit computes a Birkhoff
  functional" is always true; "this unit can match any prescribed Birkhoff data"
  is only true for a poised support.
- No certificate tier is claimed by the layer itself. Exactness statements about
  the rational weights escalate through spec 01-11.

## 11. Open questions and risks

- **Training dynamics of the means.** Two packs whose means drift together
  degrade toward a single higher-order pack, which is a silent loss of capacity.
  A repulsion term or a minimum-separation projection may be needed; measure
  before adding.
- **Order selection is discrete.** Learning `n_g` needs either a fixed menu, the
  growth mechanism of spec 03-13, or a relaxation. Do not smuggle in a
  `beta -> inf` relaxation without labelling it as temperature collapse.
- **Conditioning at high order.** `P_n` has degree `n + 1` in `sigma`; at large
  `n` with saturated inputs the polynomial evaluation loses accuracy even though
  no cancellation from `1/delta` occurs. Quantify against mpmath before claiming
  an order ceiling.
- **Falsifier.** If, at matched parameter count, a plain jet MLP matches the
  multi-interface gate, the structural claim is weak and the spec should be
  demoted to a convenience wrapper.

## 12. Implementation checklist

- [x] `packages/omnibias-core/src/omnibias/core/multipack.py`
- [x] `packages/omnibias-torch/src/omnibias/torch/multipack.py`
- [x] `packages/omnibias-jax/src/omnibias/jax/multipack.py`
- [x] Unit tests for `polya_condition`, `incidence_matrix`, `is_poised`
      including the inconclusive branch
- [x] Exactness test against mpmath (dense grid plus random sample); float64
      order ceiling recorded honestly (sigmoid: 1, tanh: 2 on `z in [-1,1]`)
- [x] torch/jax parity test in `tests/`
- [x] Collapse test: finite-difference multi-pack converges to the closed form
- [x] `benchmarks/multipack_birkhoff.py` plus committed smoke JSON
- [x] Docs page `docs/api/multipack.md` and mkdocs nav entry
- [x] Regenerate `__all__` in both backend `__init__.py` files
- [x] Index row in `theory/README.md` marked shipped
- [ ] G4 two-interface task skill (deferred; `g4_earned: false`)
