# 07-07 Nobel-adjacent domain programs

## 1. Thesis and status

Three domains where omnibias can contribute a measurably better **tool** —
quantum many-body wavefunctions, magnetized-plasma residuals, and layered
materials design — framed honestly as tooling with named baselines, because a
better tool is a real contribution and a discovery claim from a tool is not.

- **Status**: concept
- **Depends on**: 02-10, 02-11, 07-01
- **Blocks**: none

## 2. Where it lands

Three existing homes: `omnibias.ferminet` (quantum many-body),
`omnibias.fields` plus `omnibias.pinn` (plasma / MHD), and
`omnibias.pinn` with `omnibias.geometry` (layered materials). None of the three
earns a package; each is an application of an existing surface.

## 3. Prior art in omnibias

- `omnibias.ferminet` — the FermiNet bridge: `jastrow`, `multiblock`,
  `multiblock_integration`, `restricted`, `integration`, `folx_compat`. A
  **stable-tier** package, one of the four in the strict typing gate.
- `omnibias.fields.{torch,jax}.ops.mhd` — `induction_residual`
  (`d_t B - curl(u x B) - eta lap B`, with the ideal limit) and
  `ideal_mhd_momentum_residual` (incompressible momentum plus `-J x B`),
  bit-identical twins.
- `omnibias.fields` — the full field-operator surface: grad, div, curl,
  laplacian, hessian, jacobian, integration, norms, tensor divergence,
  finite strain, Wirtinger.
- `omnibias.pinn.{solver,operator,domain,train,certified}`.
- `omnibias.geometry` — metric, curvature, pullback metric of a learned chart.
- Spec 02-10 (Hermite ladder oscillator networks) and spec 02-11
  (transfer-matrix layered media) supply the architectures.

**Confirmed gap.** The MHD residuals exist as operators but no plasma
application is built on them. The FermiNet bridge exists but no Hermite-ladder
basis feeds it. There is no materials-design surface at all.

## 4. Mathematics

### (a) Quantum many-body: an exact ladder inside a neural ansatz

Spec 02-10 establishes that the gaussian base's derivative tower is the Hermite
function family in Rodrigues normalization, and that the quantum harmonic
oscillator eigenfunctions are recovered by an explicit Gaussian reweighting —
the two are related precisely, not identical, and the API requires the
normalization to be named.

For a neural wavefunction ansatz, the contribution is a basis in which the
oscillator's raising and lowering operators act **exactly**:

```
a   = ( x + d/dx ) / sqrt(2)
a^+ = ( x - d/dx ) / sqrt(2)
```

Both are first-order differential operators the tower supplies in closed form,
so `a psi` and `a^+ psi` are evaluated without autodiff and without error. For a
trapped system — cold atoms, quantum dots, vibrational modes — the ansatz then
carries exact algebra where a generic network carries an approximation.

The honest framing: this improves the **basis and the derivative evaluation**
inside a variational Monte Carlo calculation. It does not change what VMC can
converge to, which is set by the ansatz's expressiveness and the optimizer.

### (b) Plasma: residuals that already exist, applied

`induction_residual` and `ideal_mhd_momentum_residual` are shipped field ops
with closed-form derivatives. What is missing is the application: an MHD
equilibrium or stability calculation built on them.

The specific technical fit is that magnetized-plasma problems have **thin
current sheets and boundary layers** — exactly the structure spec 02-05's
multi-interface transmission bases and spec 01-01's multi-packs are designed
for. A resistive layer of thickness `d` in a domain of size `L` needs
`O(L/d)` uniform elements and `O(1)` multi-pack elements placed at the layer.

For fusion-relevant geometry, `omnibias.geometry`'s pullback metric handles
curved flux coordinates, which is where general-purpose PDE tooling usually
struggles.

### (c) Layered materials: transfer matrices with exact derivatives

Spec 02-11 makes a stack of parallel interfaces a product of `2 x 2` transfer
matrices, with energy conservation and reciprocity as algebraic identities
rather than learned approximations. For photonic and phononic stack design, the
design problem is:

```
minimize  || T(theta) - T_target ||   over layer thicknesses and contrasts theta
```

The derivative `dT/dtheta` is exact through the matrix product, so gradient-based
design converges where finite-difference design stalls — and the physical
identities hold at every iterate, so an intermediate design is always physically
realizable rather than only the converged one.

### What "Nobel-adjacent" means here, precisely

It means these domains are where prizes have been awarded, and it means nothing
more. A better wavefunction basis is not a discovery; a better MHD residual
solver is not a fusion result; a better stack optimizer is not a new material.
The word appears in the spec title because the plan uses it, and section 13
exists to keep it from doing any work.

## 5. Worked example

**Layer design, with a baseline that is hard to beat.**

Design a `10`-layer dielectric mirror for normal incidence at wavelength
`lambda_0 = 550 nm`, target reflectance `R >= 0.999`, using indices
`n_H = 2.35` and `n_L = 1.46`.

*The classical answer is already optimal.* A quarter-wave stack — every layer
of optical thickness `lambda_0 / 4` — has effective admittance, for `N` pairs of
`HL` on a substrate of index `n_s = 1.52` in air,

```
Y = ( n_H / n_L )^(2N) * n_s        R = ( (1 - Y) / (1 + Y) )^2
```

With `n_H / n_L = 1.6096` and `N = 5` pairs (the `10` layers specified):

```
(1.6096)^10 = 116.7        Y = 116.7 * 1.52 = 177.4
R = ( (1 - 177.4) / (1 + 177.4) )^2 = ( -176.4 / 178.4 )^2 = 0.9777
```

`97.8%`, short of the target. Reaching `R >= 0.999` needs `Y >= 3999`, hence
`(1.6096)^(2N) >= 2631`, hence `N >= 8.3` — so `9` pairs, `18` layers. That is
the whole design, in closed form, and no optimizer improves on it.

**The honest framing this forces.** For the *standard* problem — normal
incidence, one wavelength, two materials — a century of thin-film optics has
closed-form optimal designs. Proposing gradient-based design here would be
proposing to rediscover a known answer slowly.

The problems where an exact-derivative optimizer earns its place are the ones
without closed-form answers:

- **broadband** targets over a wavelength range,
- **off-normal incidence** with polarization constraints,
- **many materials** with manufacturing constraints on thickness,
- **inverse design** against a specified spectral shape.

In those settings the baseline is not a closed form but the needle optimization
and gradient methods already standard in thin-film design software, and the
claim to test is narrow: **exact `dT/dtheta` through the matrix product versus
the finite-difference or adjoint gradients those tools use.** For a `20`-layer
stack, finite differences need `20` extra forward evaluations per gradient;
the exact derivative needs one backward pass. That is a `20x` gradient-cost
reduction, and whether it translates into better final designs depends on
whether gradient cost was the constraint — which the benchmark must measure
rather than assume.

**The lesson generalizes to all three domains in this spec.** Each has decades
of strong domain-specific tooling. The contribution is a specific, measurable
improvement against that tooling, and the first job of each benchmark is to
establish that the domain baseline was implemented competently. A win against a
strawman is worse than no result, because it will be believed for a while.

## 6. Proposed API

```python
# omnibias/ferminet/hermite.py
def hermite_ladder_basis(
    n_modes: int, *, normalization: Literal["rodrigues", "oscillator"],
) -> LadderBasis:
    """Explicit normalization, required. Spec 02-10 documents why silently
    choosing one conflates two different function families."""

def apply_ladder(psi, op: Literal["raise", "lower"]) -> Array:
    """Exact first-order operator from the closed-form tower."""

# omnibias/pinn/plasma.py
def resistive_layer_basis(location, thickness, *, orders) -> MultiPackBasis: ...
def mhd_equilibrium_residual(state, *, geometry) -> Array:
    """Composes the existing induction_residual and
    ideal_mhd_momentum_residual field ops."""

# omnibias/pinn/stack.py
def transfer_stack(layers, *, wavelength) -> TransferStack:
    """Spec 02-11's product, with exact dT/dtheta."""
def design_stack(target, *, n_layers, materials, constraints) -> StackDesign: ...
```

torch and jax twins for anything evaluated on tensors, sharing coefficients from
`omnibias.core.polynomials`.

## 7. Practical use cases

1. **Trapped-system VMC** where the oscillator basis is physically natural and
   the ladder algebra is used explicitly.
2. **Resistive-layer MHD** where a thin current sheet is the whole physics and
   uniform discretization is wasteful.
3. **Broadband and off-normal stack design**, where closed-form answers do not
   exist and gradient cost is a real constraint.
4. **Curved-geometry plasma equilibria** using the pullback metric.
5. **Teaching and reproduction**, since all three have well-known reference
   solutions to validate against.

## 8. Acceptance gates

Baselines must be **the domain's standard tools, implemented competently**, and
establishing that is itself gated.

- **G0 baseline validity.** Each baseline reproduces a published reference
  result for its domain to within the reference's stated accuracy, before any
  comparison is run. A comparison against an unvalidated baseline is not
  reported. **This gate comes first and is the most important one in the spec.**
- **G1 quantum: ladder exactness.** `a` and `a^+` applied to basis states
  reproduce the analytic `sqrt(n)` and `sqrt(n+1)` coefficients to `1e-14`.
- **G2 quantum: VMC energy.** On a trapped few-body system with a known
  reference energy, the Hermite-ladder ansatz reaches the reference within
  chemical accuracy in at most half the optimization steps of the standard
  FermiNet ansatz, over five seeds — or the result is reported as no
  improvement.
- **G3 plasma: layer scaling.** For a resistive layer of thickness `d`, the
  multi-pack basis achieves fixed residual accuracy with a basis size
  independent of `d` over `d in [1e-4, 1e-1]`, while the uniform baseline grows
  at least as `1/d`.
- **G4 plasma: reference equilibrium.** Reproduces a published MHD equilibrium
  to within `1%` in the relevant diagnostic.
- **G5 materials: gradient cost.** Exact `dT/dtheta` reduces gradient cost by at
  least `10x` versus finite differences at `20` layers, and final design quality
  is at least as good as a standard needle-optimization baseline on a broadband
  target.
- **G6 physical identities.** Energy conservation and reciprocity hold to
  `1e-14` at **every** optimization iterate, not only at convergence.

## 9. Benchmark plan

- `benchmarks/quantum_hermite_vmc.py`, `benchmarks/plasma_resistive_layer.py`,
  `benchmarks/materials_stack_design.py`, each with its G0 baseline-validation
  stage run and recorded first.
- Smoke JSON committed; full runs under `$OMNIBIAS_SCRATCH/domain_programs/`.
  Heavy runs route through the optional GPU submit wrapper.

## 10. Honesty and scope

- **These are tooling contributions.** Better basis, better derivatives, better
  conditioning. No result here is a physical discovery, and none should be
  described as one.
- "Nobel-adjacent" describes the **domains**, not the results. The phrase is
  inherited from the plan and is deliberately given no work to do.
- Each domain has decades of specialized tooling that is very good. G0 exists
  because the realistic failure mode is beating a weak reimplementation of a
  strong method.
- The layered-media example shows the discipline: for the standard problem the
  classical answer is already optimal, and the honest move is to narrow the
  claim to the settings where no closed form exists.
- Spec 02-10's normalization distinction is load-bearing: the tower's Hermite
  functions and the oscillator eigenfunctions are related by an explicit
  reweighting, and the API requires naming which one is meant.
- The founding `delta -> 0` bias collapse supplies the ladder operators and the
  multi-pack bases. No temperature collapse appears.
- Certificate tier: **empirical gates** throughout. Nothing here is a sound
  enclosure, and no Lean flag is involved. Where a certified statement is wanted
  (a certified ground-state lower bound, say) the path is spec 07-05, not this
  one.

## 11. Open questions and risks

- **G0 is expensive.** Implementing a domain baseline well enough to validate
  against a published result is often more work than the contribution itself.
  That is the correct cost of an honest comparison, and underestimating it is
  the main scheduling risk.
- **VMC improvements are hard to attribute.** Energy differences depend on
  ansatz, optimizer, sampler and seed; attributing a win to the basis requires
  ablations that the gate does not currently specify in enough detail.
- **Plasma geometry is a large hidden cost.** Realistic fusion geometry is far
  from a periodic box, and the gap between a resistive-layer demonstration and
  anything of interest to a plasma physicist is wide.
- **Materials design may be gradient-cost-insensitive.** If the standard tools
  already converge quickly, a `10x` cheaper gradient changes nothing, and G5's
  second clause exists to detect that.
- **Three domains is probably two too many** for one spec. If effort is limited,
  the layered-materials track has the clearest baseline and the cheapest
  validation, and should go first.
- **Falsifier.** If G0 cannot be passed in a domain — the baseline cannot be
  validated against a published reference — then no comparison in that domain
  should be published, and the track should be dropped rather than reported with
  a caveat.

## 12. Implementation checklist

- [ ] `packages/omnibias-ferminet/src/omnibias/ferminet/hermite.py` with
      required explicit normalization
- [ ] `packages/omnibias-pinn/src/omnibias/pinn/plasma.py` composing the
      existing `induction_residual` and `ideal_mhd_momentum_residual`
- [ ] `packages/omnibias-pinn/src/omnibias/pinn/stack.py` with exact
      `dT/dtheta`
- [ ] G0 baseline validation implemented and recorded **before** any comparison
- [ ] Ladder-coefficient exactness test to `1e-14`
- [ ] Physical-identity check at every optimization iterate, not only at
      convergence
- [ ] torch / jax parity tests for every tensor-evaluated path
- [ ] Three benchmark scripts plus smoke JSON
- [ ] Heavy runs under `$OMNIBIAS_SCRATCH`, only summaries committed
- [ ] Docs pages and nav entries
- [ ] Index rows in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parents: the open scientific problems these domains sit inside** — a general
solution of the quantum many-body problem, controlled thermonuclear fusion
energy gain, and the inverse design of materials with arbitrary prescribed
properties.

Each stays external for the same reason, and it is not a subtlety: **this spec
produces better numerical tools, and the parents are not tool problems.**

- The many-body problem's difficulty is the exponential scaling of the Hilbert
  space and the sign problem; an exact oscillator basis inside a variational
  ansatz addresses neither. A better ansatz improves an approximation; it does
  not change the complexity class of the problem.
- Fusion energy gain is a problem of confinement physics, materials under
  neutron flux, and engineering at scale. A residual solver for an MHD model on
  a model geometry contributes to one small part of the modelling stack.
- Inverse materials design is limited by synthesizability and by the physics of
  what materials can exist, not by the speed of a gradient through a transfer
  matrix.

The gap in each case is between **a numerical method** and **a scientific
result**, and no improvement to the former crosses it. Every claim from this
spec is an empirical benchmark result against a named, validated baseline, and
must be reported as exactly that.

This spec does not claim, imply, or provide evidence for any discovery in
quantum many-body physics, fusion energy, or materials science.
