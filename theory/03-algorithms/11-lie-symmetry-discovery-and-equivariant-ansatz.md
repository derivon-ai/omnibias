# 03-11 Lie symmetry discovery

## 1. Thesis and status

A Lie point symmetry's determining equations are **linear** in the infinitesimal
generator's coefficients once the prolongation is known, and prolongation is
exactly a jet computation — so with exact towers, symmetry discovery becomes a
linear nullspace problem instead of a symbolic computation.

- **Status**: designed
- **Depends on**: 01-01, 01-10, 02-03, 02-09
- **Blocks**: 02-13, 07-06

## 2. Where it lands

`packages/omnibias-symbolic/src/omnibias/symbolic/symmetry/` — the symbolic
package already does neural-jet equation discovery; symmetry discovery is the
same machinery applied to a different determining system.

## 3. Prior art in omnibias

- `packages/omnibias-symbolic/` — neural-jet equation discovery: library-free
  SINDy, AutoML surrogates, PDE coefficient recovery, Blasius; uses
  `omnibias-jax` fastpaths.
- `omnibias.{torch,jax}.jet_mv` — `mlp_jet_mv`, `jet_partials`, `jet_gradient`,
  `jet_hessian`: every mixed partial to total order `N` in one pass. This is a
  prolongation engine.
- `omnibias.core.multi_index` — pure-Python multi-index ordering and the
  Cauchy-product table.
- `omnibias.holonomic` — the Ore skew-polynomial algebra, Gosper and creative
  telescoping, and Lean-certified binomial identities. Symmetry algebras are
  operator algebras, so there is a real connection.
- `omnibias.geometry` — Lie algebra machinery in the gauge submodule.

**Confirmed gap.** No symmetry machinery: no infinitesimal generators, no
prolongation formula, no determining equations, no conservation laws from
Noether.

## 4. Mathematics

### Generators and prolongation

A one-parameter point transformation group on `(x, u)` has infinitesimal
generator

```
X = xi^i(x, u) d/dx^i + eta(x, u) d/du
```

To act on a PDE involving derivatives, `X` must be *prolonged* to the jet space.
The first prolongation coefficient is

```
eta^{(i)} = D_i( eta - xi^j u_j ) + xi^j u_{ji}
```

and higher prolongations follow the recursion

```
eta^{(J,i)} = D_i( eta^{(J)} - xi^j u_{J,j} ) + xi^j u_{J,j,i}
```

with `D_i` the total derivative. **Every term is a jet coordinate**, and the
total derivative operator acting on a jet is a shift-and-combine operation the
jet machinery already implements.

### The determining equations

A PDE `F(x, u, u_x, ...) = 0` admits `X` as a symmetry when

```
pr^{(n)} X ( F ) = 0    whenever    F = 0
```

Expanding, this is a linear expression in `xi`, `eta` and their partial
derivatives, with coefficients that are functions of the jet coordinates.
Splitting by monomials in the free jet coordinates gives an **overdetermined
linear PDE system for `xi` and `eta`**: the determining equations.

Two solution strategies:

1. **Ansatz plus nullspace.** Posit `xi, eta` as linear combinations of basis
   functions (polynomials, or omnibias packs). The determining equations become
   a homogeneous linear system in the coefficients, and the symmetry algebra is
   its nullspace. This is a **singular value decomposition**, and its numerical
   rank is the algebra's dimension.
2. **Symbolic elimination**, the classical route, requiring a computer-algebra
   system.

Route 1 is what makes this fit omnibias: no symbolic algebra, just exact jets
and a nullspace.

### Why exact jets matter

The determining system is assembled by evaluating prolongation coefficients at
sample points. If the derivatives are approximate, the linear system is
perturbed and its **numerical rank is wrong** — one either misses symmetries
(rank too high) or hallucinates them (rank too low). Since the answer is a rank,
and rank is discontinuous, derivative accuracy is not a matter of precision but
of correctness.

That is the sharpest argument for this spec: **symmetry discovery is a rank
determination, and rank determination needs exact inputs.**

### From symmetry to use

- **Reduction.** A symmetry reduces the number of independent variables by one
  (similarity reduction), turning a PDE into an ODE.
- **Invariant solutions.** Solutions fixed by the symmetry are found by solving
  the reduced problem, which is often tractable in closed form and connects to
  spec 02-09's ansatz classes.
- **Conservation laws.** For variational problems, Noether's theorem maps
  symmetries to conserved currents, and `omnibias-variational` already has the
  Euler-Lagrange and Noether surface to receive them.
- **Integrability evidence.** An infinite-dimensional symmetry algebra, or the
  existence of higher (generalized) symmetries, is strong evidence for
  integrability and points at spec 02-13's transforms.

### The honest limits

- Point symmetries only, unless generalized (Lie-Bäcklund) symmetries are
  explicitly implemented; the two are different and the distinction is
  standard.
- Numerical rank needs a threshold. That threshold is a real modelling choice
  and must be reported with a sensitivity study, not hidden.
- The ansatz restricts what can be found: a symmetry whose coefficients are
  outside the basis will be missed. Report the basis.

## 5. Worked example

**The heat equation's scaling symmetry, recovered.**

`u_t = u_xx`. Known symmetries include translations in `x` and `t`, the scaling
`(x, t, u) -> (lambda x, lambda^2 t, u)`, the Galilean boost, the dilation in
`u`, and the projective symmetry — a six-dimensional algebra plus the infinite
family from linearity.

Take the scaling generator

```
X = x d/dx + 2 t d/dt
```

so `xi^x = x`, `xi^t = 2t`, `eta = 0`.

Prolongation coefficients:

```
eta^t = D_t( eta - xi^x u_x - xi^t u_t ) + xi^x u_{xt} + xi^t u_{tt}
```

With `eta = 0`, `xi^x = x`, `xi^t = 2t`:

```
eta - xi^x u_x - xi^t u_t = -x u_x - 2t u_t
D_t of that = -x u_{xt} - 2 u_t - 2t u_{tt}
eta^t = -x u_{xt} - 2 u_t - 2t u_{tt} + x u_{xt} + 2t u_{tt} = -2 u_t
```

Similarly `eta^x = -u_x` and, one more prolongation,

```
eta^{xx} = -2 u_{xx}
```

Now apply to `F = u_t - u_{xx}`:

```
pr X (F) = eta^t - eta^{xx} = -2 u_t - ( -2 u_{xx} ) = -2 ( u_t - u_{xx} ) = -2 F
```

which vanishes on `F = 0`. **Confirmed symmetry**, and the computation used
nothing but jet coordinates and total derivatives.

**As a nullspace problem.** Posit `xi^x = a_1 + a_2 x + a_3 t`,
`xi^t = b_1 + b_2 x + b_3 t`, `eta = c_1 + c_2 u`. Assembling
`pr X (F) mod F` at enough sample points and splitting by the free jet monomials
gives a homogeneous linear system in `(a, b, c)`. Its nullspace contains, among
others, the vector

```
a = (0, 1, 0),  b = (0, 0, 2),  c = (0, 0)
```

which is exactly `X = x d/dx + 2 t d/dt`. The scaling symmetry falls out of a
singular value decomposition.

**Rank sensitivity, the thing to watch.** With this affine ansatz the expected
nullspace dimension is 5:

```
d/dx,   d/dt,   x d/dx + 2t d/dt,   d/du,   u d/du
```

The heat equation's full point-symmetry algebra is larger — the Galilean boost
`2t d/dx - x u d/du` and the projective generator need terms this ansatz does
not contain — which is exactly the point of reporting the basis with the result.
If the jets carry `1e-6` error, the singular values that should
be exactly zero come out near `1e-6`, and if the smallest nonzero singular value
of the true system is also near `1e-6`, the rank is ambiguous and the answer is
garbage. With exact jets the zero singular values are at `1e-15` and the
separation is unambiguous. **Nine orders of magnitude of separation is the
practical difference between working and not working**, and it is a direct
consequence of exact derivatives.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/symbolic/symmetry/_core.py
@dataclass(frozen=True)
class Generator:
    xi: tuple[Callable, ...]          # one per independent variable
    eta: tuple[Callable, ...]         # one per dependent variable
    label: str = ""

def prolong(gen: Generator, *, order: int) -> ProlongedGenerator:
    """Uses the jet machinery; no symbolic algebra."""

def determining_matrix(
    pde: PDESpec, basis: SymmetryBasis, *, samples: FloatArray, order: int,
) -> FloatArray:
    """Rows are (sample, jet monomial); columns are basis coefficients."""

@dataclass
class SymmetryResult:
    generators: tuple[Generator, ...]
    singular_values: FloatArray
    rank_threshold: float             # reported, never hidden
    separation: float                 # ratio of smallest kept to largest dropped
    basis: SymmetryBasis              # what was searched; what was missed is unknown

def discover_symmetries(pde, basis, *, samples, threshold="auto") -> SymmetryResult: ...
def noether_current(gen: Generator, lagrangian) -> ConservedCurrent:
    """Feeds omnibias-variational's existing Noether surface."""
def similarity_reduction(pde, gen: Generator) -> PDESpec: ...
```

Reporting `separation` alongside the rank is the key design decision: a rank
determination without its separation is not interpretable.

## 7. Practical use cases

1. **Automatic reduction.** Discover a symmetry, reduce a PDE to an ODE, solve
   the ODE — the classical pipeline, automated.
2. **Conservation-law discovery** through Noether, feeding
   `omnibias-variational`.
3. **Validating discovered equations.** A law discovered by SINDy that has no
   symmetries where physics expects some is probably wrong; symmetry is a cheap
   consistency check on discovery output.
4. **Integrability screening** ahead of spec 02-13's transforms.
5. **Physics-informed architecture design.** Knowing the symmetry group tells
   you what equivariance to build in (spec 02-08).

## 8. Acceptance gates

Baselines: a symbolic computer-algebra symmetry package on the same equations
(as a correctness oracle), and the same nullspace pipeline fed by
finite-difference derivatives.

- **G1 known-algebra recovery.** For at least eight classical equations (heat,
  wave, Burgers, KdV, Laplace, Korteweg-de Vries, Fisher, Boussinesq), the
  discovered algebra's dimension matches the published value exactly, and the
  generators match the published ones up to a change of basis.
- **G2 rank separation.** With exact jets, the ratio of the smallest retained
  singular value to the largest discarded one exceeds `1e6` on every equation in
  the suite.
- **G3 exactness matters, demonstrated.** The same pipeline fed by
  finite-difference derivatives gets the rank wrong on at least one equation in
  the suite, and the benchmark records it. This is the spec's central claim and
  must be shown, not asserted.
- **G4 threshold sensitivity.** The discovered dimension is stable across at
  least two decades of threshold choice; where it is not, the result is reported
  as ambiguous rather than resolved by picking a threshold.
- **G5 Noether consistency.** For variational equations, the conserved currents
  derived from discovered symmetries are conserved to `<= 1e-10` along numerical
  solutions.
- **G6 negative control.** For an equation with only the trivial symmetry
  algebra, the method returns exactly that and does not hallucinate generators.

## 9. Benchmark plan

- `benchmarks/symmetry_discovery.py`: the eight-equation recovery table with
  separations, the finite-difference comparison arm, threshold sensitivity,
  Noether checks, negative control.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/symmetry/`.

## 10. Honesty and scope

- Jets come from the founding bias collapse (`delta -> 0`) tower. No temperature
  collapse appears.
- **Point symmetries only** unless generalized symmetries are explicitly
  implemented and labelled. The distinction is standard and must not be blurred.
- **The ansatz bounds what can be found.** A symmetry outside the basis is
  invisible, and the result object carries the basis so that limitation is
  visible. "No symmetries found" means "none in this basis".
- **Rank determination requires a threshold**, which is a modelling choice.
  `rank_threshold` and `separation` are required output fields, and G4 requires
  a sensitivity study.
- Lie symmetry analysis is classical (Lie, Ovsiannikov, Olver, Bluman). The
  contribution is the exact-jet numerical route that replaces symbolic
  elimination, and the honest rank reporting.
- No certificate tier. A discovered symmetry can be *verified* exactly by
  symbolic substitution (spec 02-09's verifier), which is a finite algebraic
  fact and a candidate for the Lean obligation class of spec 01-11.

## 11. Open questions and risks

- **Sample-point conditioning.** The determining matrix's conditioning depends
  on where it is sampled; degenerate sample sets give a wrong rank. Needs a
  designed sampling strategy, not random points.
- **Infinite-dimensional algebras.** Linear PDEs have infinite symmetry algebras
  (superposition), which a finite ansatz sees as a large nullspace. Detect and
  report this case explicitly rather than listing hundreds of generators.
- **Verification loop.** Every discovered generator should be verified
  symbolically before being reported; discovery without verification is where
  hallucinated symmetries come from.
- **Falsifier.** If the finite-difference arm gets the same ranks as the exact
  arm on every equation (G3 fails), the central claim about exactness is
  unfounded and the spec reduces to a reimplementation of a known method.

## 12. Implementation checklist

- [ ] `packages/omnibias-symbolic/src/omnibias/symbolic/symmetry/_core.py`
- [ ] Prolongation built on `jet_mv` and `multi_index`; no symbolic algebra
- [ ] Symbolic verification of every discovered generator before reporting
- [ ] Eight-equation recovery suite against published algebra dimensions
- [ ] Finite-difference comparison arm demonstrating G3
- [ ] Threshold sensitivity study
- [ ] Infinite-algebra detection and explicit reporting
- [ ] Designed sample-point strategy with a conditioning test
- [ ] Noether bridge into `omnibias-variational`
- [ ] `benchmarks/symmetry_discovery.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
