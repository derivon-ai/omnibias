# 02-05 Multi-interface transmission PINN

## 1. Thesis and status

Stratified and multi-material problems have several parallel interfaces, each
carrying a **different order** of transmission condition (value, flux,
curvature); a heterogeneous multi-pack unit supplies exactly those jet
coordinates at exactly those locations, so the interface structure is built into
the basis instead of being learned.

- **Status**: gated (`alpha -> inf` is sharpening, neither collapse; parallel interfaces only)
- **Depends on**: 01-01, 01-07, 02-03, 02-04
- **Blocks**: 02-11, 05-01, 07-02

## 2. Where it lands

`packages/omnibias-pinn/src/omnibias/pinn/interface/` with torch and jax twins.
A submodule of the PINN package: same audience, same tier.

## 3. Prior art in omnibias

- `packages/omnibias-partition/` — `PartitionedField` and `RegionModels`: a
  region-wise model whose regions come from a depth-`d` oblique gate tree, with
  `certify_partition_gap` for the soft-to-hard gap. This is the baseline.
- `omnibias.pinn.partition` — the discontinuity-PINN bridge onto that machinery.
- `FBPINNField` — finite-basis PINN domain decomposition with overlapping
  windows.
- `omnibias.pinn.domain` — SDF and R-function machinery, `HardConstraintField`,
  `DistanceConstrainedField`, `ConstrainedExpressionField` for hard boundary
  conditions.
- `omnibias.pinn._core.constrained` — the switching algebra and the relative
  `periodic()` condition.

**Confirmed gap.** Existing decompositions give each region its own subnetwork
and glue them with windows or gates. Nothing puts the *transmission conditions*
into the basis, and nothing distinguishes the order of the condition at
different interfaces.

## 4. Mathematics

### The problem class

On `Omega` split by parallel interfaces `Gamma_g = { w . x + mu_g = 0 }` into
layers, solve

```
- div( a(x) grad u ) = f      in each layer, with a piecewise constant
```

subject to transmission conditions at each interface. The standard set is

```
[ u ]_{Gamma}            = 0        (continuity of value)
[ a du/dn ]_{Gamma}      = 0        (continuity of flux)
```

but real problems vary: imperfect contact gives `[u] = R a du/dn` (a jump
proportional to flux), a thin interphase layer gives a curvature condition, and a
crack gives a free `[u]` with zero flux. **Different interfaces in the same
problem can carry different conditions of different orders.**

### The basis

Put a multi-pack on the shared normal `w`, with one pack per interface:

```
u_theta(x) = MLP(x)  +  sum_g c_g(x_tangential) sigma^(n_g)( w . x + mu_g )
```

where `n_g` is chosen to match the *order* of the condition at interface `g`:

| Condition at `Gamma_g` | Behaviour needed | Pack order |
|---|---|---|
| value jump `[u] != 0` | a step across `Gamma_g` | `n_g = 0` (the base `sigma` itself) |
| kink, flux jump `[a u_n] != 0` | a slope change | `n_g = 1` |
| curvature jump | a second-derivative jump | `n_g = 2` |

The reason this works is the founding structure: `sigma^(n)` is a smoothed
`n`-th derivative of a step, so it carries a jump in exactly the `n`-th
transverse derivative in the sharp limit, and a controlled smooth version at
finite scale.

The tangential dependence `c_g(x_t)` lets the jump magnitude vary along the
interface, which is what makes this more than a 1-D construction.

### Sharpness is a scale, not a limit

The interface width is `1 / alpha_g` from the tempered scale. Making it small
sharpens the transition; it does **not** collapse anything. This is worth stating
because `alpha -> inf` looks superficially like a collapse limit and is neither
of the two the repo names:

- `delta -> 0` (bias collapse) coalesces `K` biases into `sigma^(K-1)`.
- `beta -> inf` (temperature collapse) hardens a gate into a 0/1 step.
- `alpha -> inf` here sharpens an interface profile toward a distributional jump.

Three different limits. The third is a modelling choice with a resolution
trade-off, and calling it a collapse would be wrong.

### Exact conditions instead of penalties

Because the jump carried by each pack is known in closed form, the coefficients
`c_g` can be *solved* to satisfy the transmission conditions rather than
penalized toward them. For the sharp limit, the jump in the `n`-th transverse
derivative across pack `g` is `c_g` times a known constant, so a linear solve
gives the `c_g` that realize prescribed jumps. That is a hard-constraint
construction in the spirit of `ConstrainedExpressionField`, applied to interfaces
instead of boundaries.

At finite `alpha` the conditions hold up to a computable smoothing error, so the
honest statement is "conditions satisfied to a stated tolerance set by `alpha`",
with the tolerance reported.

## 5. Worked example

Two-layer 1-D conductivity problem on `[-1, 1]`:

```
a(x) = 1   for x < 0,     a(x) = 4   for x > 0
-(a u')' = 0,   u(-1) = 0,   u(1) = 1
```

Exact solution: `u` is piecewise linear with continuous flux `a u' = const = q`.
Then `u' = q` on the left and `q/4` on the right, so

```
u(x) = q (x + 1)              for x <= 0
u(x) = q + (q/4) x            for x >= 0
u(1) = q + q/4 = 1   =>   q = 4/5 = 0.8
```

so `u(0) = 0.8`, left slope `0.8`, right slope `0.2`. The solution has a **kink**
at `x = 0`: a jump in `u'` of `0.2 - 0.8 = -0.6`, with no jump in `u`.

Basis: one pack of order `n = 1` at `mu = 0`, since a flux condition with
piecewise constant `a` produces a slope jump. Take
`v(x) = sigma^(1)(alpha x) / alpha` with `sigma = tanh`, so
`v' = sigma''(alpha x)`... more usefully, use the antiderivative form: the
function whose derivative jumps is `S(alpha x) / alpha` with `S' = sigma`, that
is a smoothed ramp. Concretely take

```
u_theta(x) = A + B x + C * ( log cosh(alpha x) ) / alpha
```

since `d/dx [ log cosh(alpha x) / alpha ] = tanh(alpha x)`, which goes from `-1`
to `+1`: a smoothed unit slope jump of size `2`. So `C = -0.3` gives the required
slope jump of `-0.6`.

Matching: `u'(x) = B + C tanh(alpha x)`, so far from the interface
`u'(-1) = B - C = B + 0.3` and `u'(1) = B + C = B - 0.3`. Setting these to the
exact slopes `0.8` and `0.2` gives `B = 0.5` in both equations — consistent,
which confirms the ansatz can represent the solution exactly in the sharp limit.

Then `A` is fixed by a boundary condition. At `alpha = 50`:

```
log cosh(50) / 50 = (50 - log 2 + tiny) / 50 = 0.9861
u(-1) = A - 0.5 - 0.3 * 0.9861 = A - 0.7958 = 0   =>   A = 0.7958
u(1)  = 0.7958 + 0.5 - 0.3 * 0.9861 = 1.0000
u(0)  = 0.7958 + 0 - 0.3 * (log cosh 0)/50 = 0.7958
```

The exact `u(0)` is `0.8`, so the interface error is `4.159e-3`. It is not
mysterious: `log cosh(t) = |t| - log 2 + O(e^{-2|t|})`, so the error is exactly

```
|C| log 2 / alpha  =  0.3 * 0.693147 / 50  =  4.1589e-3
```

Doubling `alpha` to `100` gives `u(0) = 0.797921`, error `2.079e-3`: **first
order in `1/alpha` with a known constant**. Three parameters `(A, B, C)`
reproduce a two-material solution to `4e-3` with no training at all, and the
residual is a formula rather than a surprise.

A plain MLP needs far more capacity to represent the kink, and it will smooth it
by an amount it does not report.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/pinn/interface/_core.py
@dataclass(frozen=True)
class Interface:
    normal: tuple[float, ...]
    offset: float                     # mu_g
    condition: Literal["value", "flux", "curvature", "imperfect", "free"]
    order: int                        # n_g, derived from `condition` by default
    sharpness: float = 50.0           # alpha_g
    parameter: float | None = None    # e.g. contact resistance R

def order_for_condition(condition: str) -> int: ...
def smoothing_error_bound(iface: Interface, *, deriv_bound: float) -> Interval:
    """Certified bound on the transmission-condition residual at finite alpha."""
```

```python
# omnibias/pinn/interface/torch.py  (and jax twin)
class MultiInterfaceField(nn.Module):
    def __init__(
        self, base_field, interfaces: Sequence[Interface], *,
        tangential_rank: int = 0,   # 0 = constant jump; >0 = learned along Gamma
        hard: bool = True,          # solve for c_g instead of penalizing
        dtype=None,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    def interface_residuals(self, x: Tensor) -> dict[int, Tensor]:
        """Measured transmission-condition residual per interface, always
        available so the smoothing error is never invisible."""
```

## 7. Practical use cases

1. **Layered composites and laminates.** Many parallel interfaces, each with its
   own contact model. The basis matches the geometry exactly.
2. **Geophysical layered media.** Wave propagation through strata, where the
   interface depths are the unknowns of interest (spec 05-01 inverts for them).
3. **Battery and electrochemical models.** Electrode-electrolyte interfaces with
   flux conditions and contact resistance.
4. **Thermal barrier coatings.** Thin interphase layers modelled by a curvature
   condition rather than by resolving the layer.
5. **Cracks and delamination.** A free interface (`[u]` unconstrained, zero
   flux) is just another pack with a different condition tag.

## 8. Acceptance gates

Baselines: `PartitionedField` with a matched number of regions, `FBPINNField`
with matched windows, and a plain MLP, all at matched parameter count.

- **G1 exactness in the sharp limit.** For piecewise-linear and piecewise-
  quadratic reference solutions, the ansatz reproduces them to `<= 1e-10`
  relative as `alpha` grows, confirming the basis is complete for the class.
- **G2 smoothing-error rate.** The measured interface residual decays at the
  predicted rate in `1 / alpha` over at least three doublings, and
  `smoothing_error_bound` upper-bounds it with zero violations.
- **G3 accuracy.** On a three-layer problem with mixed conditions (value, flux,
  curvature) and a manufactured exact solution, relative `L2 <= 1e-6` with skill
  `> 0`, beating all three baselines over five seeds.
- **G4 hard versus penalized.** With `hard=True`, interface residuals are at the
  smoothing-error floor without any interface term in the loss; with
  `hard=False` they are strictly worse at equal training budget.
- **G5 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/multi_interface_pinn.py`: four arms, three problems (two-layer
  flux, three-layer mixed, imperfect contact), with the interface residual
  reported per interface.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/interface/`.

## 10. Honesty and scope

- The basis functions come from the founding bias collapse (`delta -> 0`,
  `K` biases coalescing into `sigma^(K-1)`).
- **`alpha -> inf` is neither collapse.** It is an interface-sharpening scale
  with a resolution trade-off. Do not call it a collapse; the repo reserves that
  word for the `delta -> 0` bias limit and the `beta -> inf` temperature limit.
- Transmission conditions hold **to a stated tolerance** at finite `alpha`, not
  exactly. `interface_residuals` is always available so the tolerance is visible.
- The interfaces are **parallel** (a shared normal) in the basic construction.
  Non-parallel interfaces need one normal per pack and lose the single-`z`
  efficiency; say so rather than implying generality.
- Certificate tier: sound enclosure for the smoothing-error bound; the rest is
  empirical.

## 11. Open questions and risks

- **Curved interfaces.** Replacing `w . x + mu` with an SDF makes the geometry
  general but breaks the closed-form transverse tower, since the SDF's own
  derivatives enter. Quantify the loss before claiming curved support.
- **Many interfaces.** Cost is linear in the number of packs, but conditioning
  degrades when interfaces are closer than `1 / alpha`. Report the minimum
  separation constraint.
- **Interface position as an unknown.** Learning `mu_g` is the inverse problem
  of spec 05-01, and it is much harder than learning jump magnitudes; keep the
  two claims separate.
- **Falsifier.** If `PartitionedField` at matched parameters reaches the same
  accuracy on the mixed-condition problem, the built-in condition orders are not
  earning their complexity.

## 12. Implementation checklist

- [ ] `packages/omnibias-pinn/src/omnibias/pinn/interface/_core.py`
- [ ] torch and jax twins with a parity test
- [ ] Reuse `ConstrainedExpressionField` patterns for the hard path
- [ ] Sharp-limit exactness test on piecewise polynomial references
- [ ] Rate test for the smoothing error, plus bound soundness
- [ ] `interface_residuals` always-on test
- [ ] `benchmarks/multi_interface_pinn.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Regenerate `__all__` in `omnibias/pinn/__init__.py`
- [ ] Index row in `theory/README.md`
