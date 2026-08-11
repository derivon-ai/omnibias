# 03-05 Differentiable mathematical morphology

## 1. Thesis and status

Dilation and erosion are max-plus convolutions, so the `logsumexp_beta`
homotopy of spec 01-08 makes the whole morphological algebra — opening, closing,
top-hat, skeletons, distance transforms — differentiable with exact derivatives
and a `log(N)/beta` gap to the hard operator.

- **Status**: designed
- **Depends on**: 01-02, 01-06, 01-08
- **Blocks**: 03-09, 05-01

## 2. Where it lands

`packages/omnibias-shape/src/omnibias/shape/morphology/` with torch and jax
twins. The shape package already owns occupancy fields and soft unions; this is
the operator algebra on them.

## 3. Prior art in omnibias

- `packages/omnibias-shape/` — differentiable soft shape and occupancy fields,
  soft-coverage (soft-OR / log-sum-exp union) operators with a closed-form
  derivative tower.
- `omnibias.struct` — the semiring driver with `MaxPlusSemiring`, `LogSemiring`
  and `CountingSemiring`, `logsumexp_beta`, and `logsumexp_gap_bound`.
- `omnibias.pinn.domain` — SDF machinery and R-functions (`r_union_sdf`,
  `r_intersect_sdf`), which are a different (Rvachev) route to the same
  set-theoretic operations.
- `omnibias.torch.blocks.conv` — `cmbConv1d`, `cmbConv2d`, the grid convolution
  path a morphological operator mirrors.

**Confirmed gap.** Soft-OR unions exist, and the max-plus semiring exists in the
dynamic-programming context. Nobody has connected them into a morphological
operator algebra: no dilation, erosion, opening, closing or distance transform.

## 4. Mathematics

### The max-plus formulation

Grayscale dilation of an image `f` by a structuring element `b`:

```
( f (+) b )(x) = max_y ( f(x - y) + b(y) )
```

and erosion:

```
( f (-) b )(x) = min_y ( f(x + y) - b(y) )
```

These are exactly convolution in the max-plus (tropical) semiring, with `max`
for addition and `+` for multiplication. Ordinary convolution is the same
expression in the sum-product semiring. **Morphology and convolution are the same
algorithm in two semirings**, which is the structural observation that makes the
`omnibias.struct` driver applicable.

### The homotopy

Replace `max` by `logsumexp_beta`:

```
( f (+)_beta b )(x) = (1/beta) log sum_y exp( beta ( f(x - y) + b(y) ) )
```

Then:

- `beta -> inf` recovers exact dilation (temperature collapse).
- `beta -> 0+` approaches the mean, a linear blur.
- Every finite `beta` is smooth with closed-form derivatives, since
  `d/du logsumexp = softmax` and the whole tower follows.

The gap is the standard one: for `N` elements in the structuring element,

```
0 <= ( f (+)_beta b )(x) - ( f (+) b )(x) <= log(N) / beta
```

so the soft dilation is an **upper bound** on the hard one with a computable
slack. That one-sided property is useful: soft dilation is always conservative
in the "covers at least as much" direction, which matters for safety-style
applications.

Erosion is the same with a sign flip, and is correspondingly a lower bound.

### The algebra that follows

| Operation | Definition | Soft version |
|---|---|---|
| opening | `(f (-) b) (+) b` | compose the soft versions |
| closing | `(f (+) b) (-) b` | likewise |
| top-hat | `f - opening(f)` | difference |
| gradient | `dilation - erosion` | difference, bounded by `2 log(N)/beta` |
| distance transform | erosion by a cone / iterated | see below |

Composition sums the gaps, so an opening's bound is `2 log(N)/beta`. Track this
explicitly rather than reporting a single-operator bound for a composite.

### Structuring elements from packs

A structuring element is a small function on a neighbourhood. Building it from a
pack means:

- its shape is parameterized by order, scale and offset, so it is *learnable*;
- its derivatives with respect to those parameters are closed form, so the
  structuring element trains by gradient descent;
- a bias-scan bank (spec 01-02) gives a family of structuring elements at
  multiple scales in one pass, which is a morphological scale space.

Learnable structuring elements are the practical selling point: classical
morphology requires hand-designing them.

### Distance transforms

The Euclidean distance transform is an erosion by a paraboloid in the max-plus
algebra, and its exact algorithm is a lower-envelope computation. The soft
version replaces the envelope with a `logsumexp`, which is differentiable and
converges to the exact transform as `beta -> inf` with the same `log(N)/beta`
bound. This connects to the SDF machinery: a soft distance transform of an
occupancy field is a differentiable approximate SDF with a *bounded* error,
which is stronger than the usual heuristic SDF regularizers.

## 5. Worked example

**1-D soft dilation, numbers.**

Signal `f = (0, 1, 3, 1, 0)` on integer positions, flat structuring element
`b = (0, 0, 0)` over a 3-neighbourhood.

Hard dilation at each position (max over the 3-window, with `-inf` padding):

```
position 0: max(0, 1)       = 1
position 1: max(0, 1, 3)    = 3
position 2: max(1, 3, 1)    = 3
position 3: max(3, 1, 0)    = 3
position 4: max(1, 0)       = 1
hard dilation = (1, 3, 3, 3, 1)
```

Soft dilation at position 2 with `beta = 2`, window values `(1, 3, 1)`:

```
exp(2*1) = 7.389056,  exp(2*3) = 403.428793,  exp(2*1) = 7.389056
sum = 418.206905
(1/2) log(418.206905) = (1/2)(6.035977) = 3.017988
```

so the soft value is `3.017988` versus the hard `3`, an excess of `0.017988`.
The bound is `log(3)/2 = 0.549306`, which contains it with room to spare — the
bound is worst-case (all values equal) and this window is far from that.

At `beta = 10`: window `exp(10), exp(30), exp(10)`, and

```
(1/10) log( e^30 (1 + 2 e^-20) ) = 3 + (1/10) log(1 + 2 e^-20) = 3 + 4.1e-10
```

essentially exact, with bound `log(3)/10 = 0.1099`. The bound is loose because
the window has a clear winner; the bound is tight only when all window values
coincide, which is the honest characterization.

**Learnable structuring element.** Replacing the flat `b = (0,0,0)` with a pack
`b(y) = c sigma'(alpha y)` makes `b` a smooth bump whose width `1/alpha` and
height `c` are learnable, with

```
d ( f (+)_beta b )(x) / d alpha = sum_y softmax_y * d b(y) / d alpha
```

and `db/dalpha = c y sigma''(alpha y)`, closed form from the tower. So the
structuring element's shape trains by gradient descent with exact gradients —
which classical morphology cannot do at all.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/shape/morphology/_core.py
@dataclass(frozen=True)
class StructuringElement:
    offsets: FloatArray               # neighbourhood positions
    values: FloatArray | PackSpec     # flat array, or a learnable pack
    @property
    def size(self) -> int: ...

def morphology_gap_bound(*, size: int, beta: float, compositions: int = 1) -> float:
    """compositions * log(size) / beta. Composite operators must pass their
    actual composition count, never 1."""
```

```python
# omnibias/shape/morphology/torch.py  (and jax twin)
def dilate(f, se: StructuringElement, *, beta: float) -> Tensor: ...
def erode(f, se, *, beta: float) -> Tensor: ...
def opening(f, se, *, beta: float) -> Tensor: ...
def closing(f, se, *, beta: float) -> Tensor: ...
def top_hat(f, se, *, beta: float, dual: bool = False) -> Tensor: ...
def morphological_gradient(f, se, *, beta: float) -> Tensor: ...
def soft_distance_transform(occupancy, *, beta: float, metric="euclidean") -> Tensor: ...

@dataclass
class MorphResult:
    value: Tensor
    gap_bound: float          # always reported, composition-aware
    direction: Literal["upper", "lower", "two_sided"]
```

Carrying `direction` matters: dilation is a one-sided upper bound, erosion a
one-sided lower bound, and the gradient two-sided. Losing that distinction
throws away the conservative-safety property.

## 7. Practical use cases

1. **Learnable morphological layers** for segmentation and denoising, where the
   structuring element is trained rather than designed.
2. **Differentiable SDF construction** from occupancy with a bounded error,
   feeding `omnibias.pinn.domain`.
3. **Shape regularization in inverse problems** (spec 05-01): opening and
   closing penalties encode "no thin spikes" and "no small holes" in a
   differentiable way.
4. **Certified coverage.** Soft dilation over-covers by at most `log(N)/beta`,
   so a coverage guarantee survives the relaxation.
5. **Topological feature extraction** (spec 03-09): morphological granulometry
   is a classical route to size distributions.

## 8. Acceptance gates

Baselines: exact morphology (`max`/`min` filters) and a learned CNN of matched
parameter count.

- **G1 hard-limit convergence.** As `beta` doubles, the deviation from exact
  morphology halves, over at least four doublings, matching the predicted rate.
- **G2 bound soundness and direction.** The gap bound holds on a dense grid
  **and** a random sample with zero violations, and the one-sidedness
  (`dilation >= hard`, `erosion <= hard`) holds exactly at every point.
- **G3 composition accounting.** For openings and closings, the reported bound
  uses the correct composition count; a test asserts that a single-operator
  bound would be violated, so the accounting cannot silently regress.
- **G4 learnable-element win.** On a segmentation task with structured noise, a
  learned structuring element beats a hand-designed one by at least `20` percent
  relative improvement in the task metric, with skill `> 0`, over five seeds.
- **G5 distance-transform accuracy.** The soft distance transform is within the
  stated bound of the exact Euclidean distance transform everywhere.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/morphology.py`: convergence rates, bound soundness and tightness,
  segmentation task against both baselines, distance-transform accuracy.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/morphology/`.

## 10. Honesty and scope

- `beta -> inf` here is **temperature collapse**, the feasibility sense: a soft
  max hardening into a hard max. It is **not** the founding bias collapse
  (`delta -> 0`), which is what builds the pack-based structuring elements. The
  module carries the cross-reference note and a `PENALTY_FILES` entry.
- Mathematical morphology, its max-plus formulation, and softmax relaxations of
  it are all established. The contributions are the exact derivative tower on
  the structuring element, the composition-aware certified gap, and the
  integration with the occupancy and SDF machinery.
- The gap bound is **worst-case** and is tight only when all window values
  coincide. Report both the bound and the observed deviation, so the looseness
  is visible.
- No certificate tier beyond the sound gap bound.

## 11. Open questions and risks

- **Numerical overflow.** `exp(beta * value)` overflows for large `beta`; a
  shifted (max-subtracted) `logsumexp` is mandatory, and its interaction with
  gradient stability must be tested.
- **Large structuring elements** make the softmax over the window expensive; the
  separable decomposition classical morphology uses may not survive the
  relaxation exactly, and the loss must be quantified.
- **Gradient vanishing at high `beta`.** The softmax becomes a hard selection and
  gradients flow only to the argmax, which is the known pathology of annealed
  methods. Schedule design matters.
- **Falsifier.** If a plain CNN at matched parameters matches the learnable
  morphological layer on every task, the structured operator is not earning its
  place.

## 12. Implementation checklist

- [ ] `packages/omnibias-shape/src/omnibias/shape/morphology/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse `logsumexp_beta` and `logsumexp_gap_bound`; fork neither
- [ ] Shifted `logsumexp` with an overflow test at high `beta`
- [ ] One-sidedness test (exact inequality, not approximate)
- [ ] Composition-accounting test that would fail with a single-operator bound
- [ ] Distance-transform accuracy test against the exact transform
- [ ] Terminology cross-reference note plus `PENALTY_FILES` registration
- [ ] `benchmarks/morphology.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
