# 02-03 Jet-KAN: univariate edge bases with exact derivative towers

## 1. Thesis and status

Kolmogorov-Arnold networks put a learned univariate function on every edge; make
that function an OMBU multi-pack and every edge gains an **exact derivative
tower to arbitrary order at no extra activation cost**, which removes the main
practical objection to KANs in scientific settings: their derivatives are only
as good as their spline basis.

- **Status**: gated (G1/G3/G5 earned; G2 cost smoke-earned, not CI `all_passed`; KA theorem does not justify)
- **Depends on**: 01-01, 03-13
- **Blocks**: 02-05, 03-11

## 2. Where it lands

`packages/omnibias-torch/src/omnibias/torch/architectures/jetkan.py` plus the
jax twin. An architecture, not a package.

## 3. Prior art in omnibias

- `packages/omnibias-torch/src/omnibias/torch/growable.py` —
  `GrowableOperatorMultiBiasUnit`, which grows the bias arity `K` of a single
  pack during training. This is the refinement mechanism a KAN needs, already
  present on the correct axis for order.
- `omnibias.torch.unit` / `omnibias.torch.blocks.operator` — the univariate
  building block.
- `omnibias.{torch,jax}.jet` and `jet_mv` — exact composition of towers through
  depth, which is what makes a *deep* KAN differentiable exactly rather than
  layer by layer.
- Spec 01-01's `MultiPackUnit` is the proposed edge function.

**Confirmed gap.** There is no KAN-style architecture in the repo. The
ingredients (univariate units, growth, exact composition) all exist and have
never been assembled this way.

## 4. Mathematics

### The architecture

A KAN layer maps `R^{n_in} -> R^{n_out}` by

```
y_q = sum_{p=1}^{n_in} phi_{q,p}( x_p )
```

with each `phi_{q,p}` a learned univariate function. Classical KANs use B-splines
on a grid. Here:

```
phi_{q,p}(u) = sum_{g} c_{q,p,g} sigma^(n_g)( alpha_{q,p,g} u + mu_{q,p,g} )
```

a multi-pack with per-edge orders, scales and offsets.

### Why the derivative tower matters here

For a spline of degree `d`, the `(d+1)`-st derivative is identically zero and the
`d`-th is piecewise constant. So a cubic-spline KAN cannot represent a smooth
fourth derivative at all, and its third derivative is discontinuous. In any
application that reads high derivatives from the model — PDE residuals, symbolic
discovery, symmetry search — that is a hard ceiling.

With OMBU edges:

```
phi^(k)(u) = sum_g c_g alpha_g^k sigma^(n_g + k)( alpha_g u + mu_g )
```

exact for every `k`, analytic, and computable from the same `sigma(alpha_g u +
mu_g)` values the forward pass produced. **Every order costs one polynomial
evaluation, not one extra activation.**

### Depth

Composition of layers is handled by `compose_jet` / `mlp_jet`: the jet of a
composition is a Bell-polynomial combination of the jets of the parts. So a
depth-`L` Jet-KAN has exact derivatives of every order up to the jet truncation
`N`, propagated in a single forward pass, without an autodiff graph of depth
`L * k`.

This is the quantitative claim to test: computing the `k`-th derivative of a
depth-`L` KAN by repeated autodiff costs roughly `O(L * 2^k)` in graph size,
while the jet path costs `O(L * N^2)` for all orders up to `N` at once.

### Refinement instead of grid extension

Classical KANs refine by increasing spline grid resolution. Here refinement has
two axes:

1. **Order** (`K` within a pack): `GrowableOperatorMultiBiasUnit` already does
   this, and it raises the local approximation order.
2. **Location** (new packs at new offsets): spec 03-13's pack birth, which
   raises resolution where the residual demands it.

Both are refinements of the same object, both preserve everything learned so far
(a new pack starts at zero outer weight), and neither requires re-gridding.

### Approximation

The Kolmogorov-Arnold representation theorem guarantees that a two-layer
composition of univariate functions can represent any continuous multivariate
function, but the inner functions may be highly irregular, so the theorem does
not by itself justify the architecture. State this. The practical justification
is empirical: on many scientific functions, low-order smooth univariate edges
suffice, and then the exact tower is a real advantage.

## 5. Worked example

Target: `f(x_1, x_2) = exp(sin(pi x_1) + x_2^2)`, the standard KAN toy problem,
on `[-1, 1]^2`.

Structure `[2, 2, 1]`: 4 edges in layer 1, 2 in layer 2, 6 univariate functions
total.

Edge budget: 3 packs each, orders `(1, 2, 3)`, so 6 parameters per edge
(3 weights, 3 offsets) plus 3 scales: 9 per edge, 54 total, plus biases.
A cubic-spline KAN with grid size 5 uses `5 + 3 = 8` coefficients per edge, 48
total — comparable, which is what makes the comparison fair.

The property to check is not the fit but the derivatives. Take the exact
second derivative of the target in `x_1`:

```
f_{11} = f * ( pi^2 (cos^2(pi x_1) - sin(pi x_1)) )    ... times pi^2 terms
```

Concretely at `(x_1, x_2) = (0.25, 0.5)`:

```
s = sin(pi * 0.25) = 0.7071068,  c = cos(pi * 0.25) = 0.7071068
f = exp(0.7071068 + 0.25) = exp(0.9571068) = 2.6041460
f_1   = f * pi c            = 2.6041460 * 2.2214415 =  5.7849570
f_11  = f * pi^2 (c^2 - s)  = 2.6041460 * 9.8696044 * (-0.2071068)
      = -5.3230340
```

A Jet-KAN reads `f_11` from the composed jet exactly (to the accuracy of the fit
itself). A cubic-spline KAN reads a piecewise-linear `f_11` with visible
kinks at grid points, and its `f_1111` is identically zero regardless of the
target. The benchmark measures exactly that: relative error of `f`, `f_11`, and
`f_1111` separately, at matched parameters.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/torch/architectures/jetkan.py  (and jax twin)
@dataclass(frozen=True)
class JetKANConfig:
    widths: tuple[int, ...]            # e.g. (2, 2, 1)
    packs_per_edge: int = 3
    orders: tuple[int, ...] = (1, 2, 3)
    base: str = "tanh"
    learnable_scales: bool = True
    growable: bool = True

class JetKAN(nn.Module):
    def __init__(self, config: JetKANConfig, *, dtype=None) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def jet(self, x: Tensor, order: int, direction: Tensor) -> Tensor:
        """Directional jet to `order` in one pass, via `compose_jet`."""
    def jet_mv(self, x: Tensor, total_order: int) -> Tensor:
        """All mixed partials to total order, via `mlp_jet_mv`."""
    def refine(self, criterion: RefineCriterion) -> None:
        """Order growth (GrowableOMBU) and/or pack birth (spec 03-13)."""

def edge_functions(model: JetKAN) -> tuple[MultiPackSpec, ...]:
    """Inspect the learned univariate functions; the interpretability story."""
```

## 7. Practical use cases

1. **PDE residuals of high order.** Fourth-order problems (biharmonic,
   Cahn-Hilliard, thin plates) need `f_xxxx`; a cubic-spline KAN cannot supply
   it, and a Jet-KAN can, exactly.
2. **Symbolic discovery.** `omnibias.symbolic` reads jets off a fitted field;
   edge functions that are already sums of `sigma^(n)` are far easier to read
   symbolically than spline coefficients.
3. **Symmetry search** (spec 03-11) needs prolongations, which are jets. Exact
   towers make the determining equations exactly linear rather than
   approximately so.
4. **Interpretable scientific fitting.** Each edge is a small, plottable,
   univariate function, and its order tells you the local smoothness the model
   chose.
5. **Adaptive scientific models.** Refinement by order and location, driven by
   the residual, without re-gridding or re-initializing.

## 8. Acceptance gates

Baseline: a cubic-spline KAN at matched parameter count, and a plain MLP with
`tanh`.

- **G1 derivative accuracy.** On a suite of analytic targets, relative error of
  the `k`-th derivative for `k = 0, 2, 4` is measured; Jet-KAN must beat the
  spline KAN by at least `10x` at `k = 2` and be the only method with finite
  error at `k = 4`.
- **G2 jet cost.** Computing all derivatives to order `N = 6` in one pass is at
  least `5x` faster than repeated autodiff at depth `L = 3`, measured.
- **G3 fit parity.** Function-value accuracy is within `1.5x` of the spline KAN,
  so the derivative win is not bought with a worse fit.
- **G4 refinement.** Order growth and pack birth each reduce the residual
  monotonically on a held-out set, and neither degrades a previously learned
  fit (a new pack starts at zero outer weight, so this is checkable exactly).
- **G5 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/jetkan.py`: derivative-accuracy table by order, jet-versus-autodiff
  timing, fit parity, refinement curves.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/jetkan/`.

## 10. Honesty and scope

- Edge functions are built by the founding bias collapse (`delta -> 0`). No
  temperature collapse appears.
- **The Kolmogorov-Arnold theorem does not justify this architecture.** The
  theorem's inner functions can be non-smooth, so it guarantees representability
  in a class this model does not contain. The justification is empirical and
  must be presented that way.
- Exactness is of the *derivative of the model*, not of the target. A perfectly
  differentiated bad fit is still a bad fit; G3 exists to keep that honest.
- No certificate tier. Interval enclosures of a Jet-KAN's output are possible
  through `omnibias.verify`, but that is a separate exercise.

## 11. Open questions and risks

- **Parameter efficiency.** KANs are often less parameter-efficient than MLPs;
  if that holds here, the honest claim narrows to "use it when you need
  derivatives".
- **Optimization.** Learnable scales inside an activation argument are prone to
  saturation. A scale parameterization through `exp` or a bounded map is likely
  necessary; measure.
- **Order selection.** Which orders to put on an edge is discrete; refinement
  handles growth but not pruning. Spec 03-13 covers death, and the interaction
  needs testing.
- **Falsifier.** If a plain MLP differentiated with `jet_mv` matches Jet-KAN on
  derivative accuracy at matched cost, the edge-wise structure adds nothing and
  the right answer is to use the MLP.

## 12. Implementation checklist

- [ ] `packages/omnibias-torch/src/omnibias/torch/architectures/jetkan.py`
- [ ] `packages/omnibias-jax/src/omnibias/jax/architectures/jetkan.py`
- [ ] Wire `jet` and `jet_mv` methods through `compose_jet` / `mlp_jet_mv`
- [ ] Reuse `GrowableOperatorMultiBiasUnit` for order growth; do not reimplement
- [ ] Derivative-accuracy test versus analytic references at `k = 0, 2, 4`
- [ ] Zero-weight-birth test: refinement never degrades the current fit
- [ ] torch/jax parity test
- [ ] `benchmarks/jetkan.py` plus smoke JSON
- [ ] Register in the architectures `__init__.py`, regenerate `__all__`
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
