# 02-01 Scan-Net: a grid-free convolutional network

## 1. Thesis and status

Stack bias-scan banks into a depth architecture and you get the structural
benefits of a convolutional network — weight sharing, translation equivariance,
a multiscale hierarchy — on inputs that have **no grid at all**: point clouds,
collocation sets, scattered sensors, implicit fields.

- **Status**: gated (G1/G2/G5 earned; G3 cost and G4 k-NN recorded, not CI `all_passed`)
- **Depends on**: 01-02, 01-06, 01-07
- **Blocks**: 02-07, 05-01, 05-02

## 2. Where it lands

`packages/omnibias-torch/src/omnibias/torch/architectures/scannet.py` plus the
jax twin, beside the existing reference architectures. Reference architectures
already live there; this is one more, not a package.

## 3. Prior art in omnibias

- `packages/omnibias-torch/src/omnibias/torch/architectures/cmbnet.py` —
  `CmbNet`, the reference operator-typed CNN with edge, blob and scale layers.
  It is the honest baseline: same typing idea, but on a grid.
- `packages/omnibias-torch/src/omnibias/torch/blocks/conv.py` — `cmbConv1d`,
  `cmbConv2d`.
- `packages/omnibias-torch/src/omnibias/torch/architectures/__init__.py` — the
  registry these are exported from.
- Spec 01-02's `BiasScan` is the layer this architecture stacks.

**Confirmed gap.** Every convolutional path in the repo requires a regular
array. There is no architecture whose translation equivariance comes from the
bias axis rather than from array indexing.

## 4. Mathematics

### One layer

A Scan-Net layer maps a feature vector `u in R^C` at a point `x in R^D` to a new
feature vector:

```
z_c      = w_c . [x ; u]  + b_c                  (C_out learned directions)
r_{c,j}  = F_{T_c}( z_c + tau_j )                (bank of M offsets, shared template)
u'_c     = sum_j a_{c,j} r_{c,j}                 (learned taps over the bank)
```

Weight sharing is in `T_c` and `a_{c,:}`: the same template and the same taps are
applied at every bank position, exactly as a convolution kernel is applied at
every pixel. The bank index `j` plays the role of the kernel index.

### What equivariance survives

For a shift of the *pre-activation* by `t` (which a translation of `x` along
`w_c` induces), the response vector shifts along the bank. So each layer is
equivariant to translation along its own learned direction, on the bank lattice.

Two honest limitations, both of which must be stated in any comparison with a
CNN:

1. Directions are per-channel and learned, so the group being represented is a
   product of one-parameter translation groups, not the full translation group
   of `R^D`.
2. Depth composes these directions nonlinearly. There is no clean statement that
   a deep Scan-Net is equivariant to anything; the honest claim is that each
   layer has a structured, interpretable equivariance, and that the inductive
   bias helps.

Compare: a CNN's equivariance is exact and global for the full translation group
on the grid. Scan-Net trades that for the ability to operate without a grid. Say
the trade, do not hide it.

### Receptive field and scale

Bank width `M * spacing` in pre-activation units maps to a spatial extent
`M * spacing / |w_c|` along `w_c`. So `|w_c|` is a learned scale, and the
tempered scale axis (spec 01-02) gives a second one. The multiscale hierarchy of
a CNN (pooling doubling the receptive field) is reproduced by geometrically
growing the bank spacing with depth, which spec 01-07 lets you compute rather
than tune.

### Pooling and downsampling

There is no grid to downsample. The analogue is **bank coarsening**: reduce `M`
and increase spacing between layers, keeping the extent fixed while lowering
resolution. Feature aggregation over neighbourhoods (what pooling does
spatially) is done by the integral role over a window, which is closed form.

### Parameter count

Per layer: `C_out (D + C_in + 1)` for the directions and biases, plus
`C_out * M` taps, plus the template parameters. Because the template is shared,
this is smaller than `C_out * M` independent units by the template size, and the
comparison with `CmbNet` must be at matched total parameters.

## 5. Worked example

A 1-D problem: locate two interfaces in a scattered sample of a field.

Data: 200 points `x_i` drawn uniformly on `[-1, 1]`, values
`u(x) = tanh(20(x + 0.4)) - tanh(20(x - 0.3))`, that is a plateau between
`-0.4` and `0.3` with sharp edges. No grid: the `x_i` are unordered and
irregularly spaced.

Layer 1: `C_out = 4`, template = order-1 collapse (`sigma'`, a bump), bank of
`M = 9` offsets uniform on `[-2, 2]`, input `[x ; u]`.

At a point `x = -0.4` (an edge), the channel whose direction is dominated by `x`
produces a bank response peaked at `tau = 0.4 |w|`, and the taps `a` learn to
read "peak position" rather than "peak value". The layer output is therefore an
edge-proximity feature, and it was computed with 9 activation evaluations and no
neighbourhood search.

Contrast the alternatives at this task:

| Method | Needs a grid | Needs neighbour search | Cost per point |
|---|---|---|---|
| 1-D CNN | yes (bin the points) | no | `O(M)` after binning |
| Point-cloud network | no | yes (`k`-NN) | `O(k log N)` |
| Scan-Net | no | no | `O(M)` |

The middle column is the reason to care. A `k`-NN graph is the usual price of
grid-free convolution, and the scan does not pay it, because the "neighbourhood"
is a bank of biases rather than a set of nearby points.

The caveat, stated plainly: the scan's neighbourhood is along the *learned
direction in feature space*, not a metric ball in input space. Those coincide
only when the direction is informative. On tasks where genuine spatial
neighbourhoods matter, a `k`-NN method should win, and the benchmark must
include such a task so the boundary is visible.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/torch/architectures/scannet.py  (and jax twin)
@dataclass(frozen=True)
class ScanNetConfig:
    dim_in: int
    channels: tuple[int, ...]          # per layer
    bank_sizes: tuple[int, ...]        # M per layer
    bank_extents: tuple[float, ...]    # in pre-activation units
    template: MultiPackSpec | OpName = "grad"
    base: str = "tanh"
    readout: Literal["pooled", "response", "argmax"] = "pooled"

class ScanNet(nn.Module):
    def __init__(self, config: ScanNetConfig, *, dtype=None) -> None: ...
    def forward(self, x: Tensor, u: Tensor | None = None) -> Tensor: ...

def scannet_from_band_plan(plan: BandPlan, **kw) -> ScanNetConfig:
    """Use spec 01-07 to pick bank extents from a target spectrum."""
```

## 7. Practical use cases

1. **Meshless PDE collocation.** Collocation points are scattered by design;
   Scan-Net gives them convolutional structure without meshing.
2. **Sensor networks.** Irregularly placed sensors with no interpolation step.
3. **Implicit shape processing.** Features of an SDF along its own gradient
   direction, sampled wherever you like.
4. **Very high dimension.** Grid convolution is impossible for `D > 3`; a bank
   along a learned direction costs the same in any `D`.
5. **Operator learning on irregular discretizations**, where a fixed grid is
   exactly what one wants to avoid.

## 8. Acceptance gates

Baselines, all at matched parameter count: `CmbNet` on a binned grid, a
`k`-NN point-cloud network, and a plain MLP.

- **G1 layer equivariance.** Each layer's on-lattice equivariance holds to
  `<= 4 ulp` (inherited from spec 01-02, retested in the stack).
- **G2 grid-free win.** On a scattered-sample interface task, Scan-Net beats the
  binned `CmbNet` in mean absolute interface position error, with skill `> 0`
  against the domain-midpoint predictor, over five seeds.
- **G3 no-neighbour-search cost.** Wall time per point is independent of `N`
  (measured across `N` spanning two decades), unlike the `k`-NN baseline.
- **G4 honest boundary.** On a task where spatial neighbourhoods genuinely
  matter (local density estimation), the `k`-NN baseline is *allowed* to win,
  and the result is reported rather than omitted.
- **G5 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/scannet.py` with three arms (Scan-Net, binned `CmbNet`, `k`-NN)
  and two tasks (interface localization, local density), smoke and `--full`.
- Record `wall_seconds` per arm, as the existing benchmarks do.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/scannet/`.

## 10. Honesty and scope

- Templates collapse by the founding `delta -> 0` bias collapse. A soft-argmax
  readout, if used, is a softmax sharpness knob; driving it to infinity would be
  temperature collapse and is a different limit, to be labelled as such.
- **Equivariance is per-layer, per-direction, and on-lattice.** Scan-Net is not
  equivariant to the translation group of the input space, and any text
  suggesting otherwise is wrong.
- The claim is about grid-free settings. Where a grid exists and is cheap, a CNN
  is likely better, and G4 exists to keep that visible.
- No certificate tier. This is an empirical architecture.

## 11. Open questions and risks

- **Direction collapse.** Nothing stops several channels from learning the same
  direction. Measure the spread of `w_c` and consider a diversity term.
- **Depth.** Composing bank responses may be harder to optimize than composing
  pointwise nonlinearities; if depth does not help, the architecture is really a
  single-layer feature extractor and should be presented as one.
- **Falsifier.** If a plain MLP on `[x ; u]` matches Scan-Net at matched
  parameters on every task, the structure is decorative.

## 12. Implementation checklist

- [ ] `packages/omnibias-torch/src/omnibias/torch/architectures/scannet.py`
- [ ] `packages/omnibias-jax/src/omnibias/jax/architectures/scannet.py`
- [ ] Register in the architectures `__init__.py` with regenerated `__all__`
- [ ] Per-layer equivariance test inside the stack
- [ ] torch/jax parity test
- [ ] `benchmarks/scannet.py` with all three arms plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
