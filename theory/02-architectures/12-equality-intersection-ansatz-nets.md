# 02-12 Equality-intersection ansatz networks

## 1. Thesis and status

An **implicit layer** whose forward pass solves `f_i(x) = f_j(x)` by closed-form
Newton and whose backward pass uses the implicit function theorem — the same
technique `omnibias.convex.torch.qp_layer` applies to a KKT system, on a much smaller
and better-conditioned system, because the tower supplies the Jacobian exactly.

- **Status**: gated (layer on 01-09; always `branch` / `condition` / `converged`; not a general PDE solver)
- **Depends on**: 01-01, 01-03, 01-09
- **Blocks**: 05-01, 07-02

## 2. Where it lands

`packages/omnibias-fields/src/omnibias/fields/locus/layer.py` (torch and jax
twins), alongside the solver from spec 01-09. Consumers are `omnibias-pinn` and
`omnibias-shape`.

## 3. Prior art in omnibias

- Spec 01-09 — the mathematics: residual, exact Jacobian and Hessian,
  transversality, branch bookkeeping, Krawczyk certification.
- `packages/omnibias-convex/.../qp_layer` — the reference pattern for
  differentiating through a solved system by the implicit function theorem on
  the KKT conditions, with `solve_lp` and `lp_dual_lower_bound` alongside.
- `omnibias.pinn.domain` — `Halfspace`, `r_intersect_sdf`, `r_union_sdf`,
  `RCompose`; `HardConstraintField`, `DistanceConstrainedField`,
  `ConstrainedExpressionField`.
- `omnibias.core.verified.kantorovich` — `certify_zero_radii`,
  `krawczyk_certificate`.

**Confirmed gap.** No layer ties unit outputs together. The implicit-layer
pattern exists only for convex programs.

## 4. Mathematics

### The layer

Inputs: parameters `theta` (the units' normals, biases, orders, weights) and
optionally a tangential coordinate `s`. Output: a point `x*` on the equality
locus, or a quantity evaluated there.

```
forward:   x* = argmin_x || x - x_0 ||   subject to   F(x; theta) = 0
backward:  dx* / d theta = - DF^+ ( dF / d theta )
```

The forward solve is Gauss-Newton with the closed-form Jacobian from spec 01-09;
the backward pass never unrolls the iteration, so memory is `O(1)` in iteration
count.

### Why this is better conditioned than a QP layer

A KKT system carries complementarity conditions, so its Jacobian is
structurally singular at active-set changes and needs regularization. Here the
system is `m` smooth equations in `D` unknowns with an explicitly computable
Jacobian that is a sum of `m + 1` rank-one terms. The only degeneracy is
tangency (rank deficiency of `DF`), and it is detectable pointwise rather than
combinatorially.

Practical consequence: the layer can report a conditioning number with every
call, and refuse rather than return garbage.

### Four ansatz classes where the locus is the answer

This is where the layer earns its place. In each case, an equality between two
computable quantities *defines* the object of interest.

**1. A shock as an equality locus.** For a scalar conservation law
`u_t + f(u)_x = 0` with left and right states `u_L`, `u_R`, the Rankine-Hugoniot
condition says the shock speed is

```
s = ( f(u_R) - f(u_L) ) / ( u_R - u_L )
```

Equivalently, the shock position `X(t)` is where the two one-sided solution
branches agree in flux. Parameterizing the branches with collapsed units and
imposing the equality gives `X(t)` as a locus, differentiable in the branch
parameters.

**2. A two-plane transmission problem.** Two layers meeting at an interface, each
represented by a unit; the interface is where their fields agree. Spec 02-05
puts the interface in the basis at a *fixed* location; this layer *solves* for
the location, which is what an inverse problem needs.

**3. A free boundary.** In the obstacle problem, the free boundary is where the
solution touches the obstacle: `u(x) = psi(x)` with `u >= psi` elsewhere. That
equality is exactly the locus, and it moves as the solution changes — which is
what makes free boundaries hard for fixed-mesh methods and natural here.

**4. Characteristic surfaces.** For a first-order PDE, characteristics are where
the normal speed equals the wave speed. Written as an equality between two
closed-form expressions, the characteristic surface is a locus that can be
followed with quadratic convergence.

### What "closed-form solution of a differential equation" means here, precisely

This is the claim that most needs care. The honest statement has three levels.

- **Level 1 (always true).** The locus is a constraint manifold with exact
  derivatives. No PDE content.
- **Level 2 (true within an ansatz class).** If the two sides are chosen from a
  class where the PDE reduces to an algebraic condition — for example travelling
  waves polynomial in `tanh` (spec 02-09), or fields related by a linearizing
  transform (spec 02-13) — then imposing the equality *is* solving the equation
  within that class, exactly, and the result is verifiable symbolically.
- **Level 3 (not claimed).** A general PDE is not solved by intersecting two
  units. Nothing in this spec suggests a universal closed-form PDE solver, and
  any text implying it is wrong.

Level 2 is a real capability and is worth stating loudly: **inside a named
ansatz class, the equality locus yields an exact solution, verified by symbolic
substitution rather than by a small residual.** Level 3 is the boundary.

### Certification

`certify_locus_point` (spec 01-09) wraps `krawczyk_certificate`, so the layer can
return "a solution exists and is unique in this box" as a sound enclosure. For a
free boundary, that is a certified statement about the boundary's location — a
genuinely strong output for an inverse problem.

## 5. Worked example

**A shock position as a locus, checked exactly.**

Burgers' equation `u_t + (u^2/2)_x = 0` with Riemann data `u_L = 1`, `u_R = 0`.
Rankine-Hugoniot gives

```
s = ( 0^2/2 - 1^2/2 ) / ( 0 - 1 ) = (-0.5)/(-1) = 0.5
```

so the shock sits at `X(t) = 0.5 t`.

Represent the two branches with collapsed units of *equal order and equal
weight* on the space-time input `(x, t)`:

```
f_L(x, t) = sigma'( w_L . (x, t) + b_L ),    w_L = (1, -1),  b_L = 0
f_R(x, t) = sigma'( w_R . (x, t) + b_R ),    w_R = (1,  0),  b_R = 0
```

By the order-matched lemma of spec 01-09, the locus of `f_L = f_R` with
`sigma'` even is the union of

```
(w_L - w_R) . (x, t) = 0   =>   -t = 0
(w_L + w_R) . (x, t) = 0   =>   2x - t = 0   =>   x = 0.5 t
```

The second branch is exactly the shock. Two units, one equality, and the
Rankine-Hugoniot speed falls out of the geometry — no fitting, no residual.

This is a clean illustration and also a warning: it worked because the branch
structure of the even function `sigma'` produced the right line. The first
branch `t = 0` is spurious for this problem. **Branch selection is not optional**,
and the layer must return a branch signature so the caller can pick.

**Gradient check.** Suppose `w_L = (1, -c)` with `c` a parameter. The mirror
branch is `(2, -c) . (x, t) = 0`, so `x = c t / 2` and

```
dx* / dc = t / 2
```

The implicit function theorem gives the same: with `F = f_L - f_R` and the
mirror branch active, `dF/dc` and `DF` are both closed form, and their ratio is
`t/2`. At `t = 3`, `c = 1`: `dx*/dc = 1.5`, matching a finite-difference check on
the solved position to `1e-10`.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/fields/locus/layer.py  (torch and jax twins)
class EqualityLocusLayer(nn.Module):
    def __init__(
        self, system: EqualitySystem, *,
        branch: int | Literal["nearest", "all"] = "nearest",
        max_iter: int = 20, tol: float = 1e-12,
        require_transversal: bool = True,
        dtype=None,
    ) -> None: ...

    def forward(self, x0: Tensor) -> LocusOutput: ...

@dataclass
class LocusOutput:
    point: Tensor
    branch: Tensor            # branch signature, never omitted
    condition: Tensor         # conditioning of DF; caller may refuse
    converged: Tensor
```

```python
class AnsatzSolutionField(nn.Module):
    """Level 2: a field whose PDE-exactness inside a named ansatz class is
    verified symbolically at construction, not asserted."""
    def __init__(self, ansatz_class: AnsatzClass, system: EqualitySystem) -> None:
        # raises if `verify_exact` fails for the class
        ...
    def certificate(self) -> AnsatzCertificate:
        """Records the class, the symbolic verification result, and explicitly
        states that no claim is made outside the class."""
```

Returning `branch` and `condition` as required fields rather than optional
diagnostics is deliberate: both failure modes are silent otherwise.

## 7. Practical use cases

1. **Learning shock positions** from data without labelling them: the position
   is an output of the layer, and its gradient flows to the branch parameters.
2. **Free-boundary inverse problems.** Stefan and obstacle problems where the
   boundary is the unknown of interest.
3. **Interface localization with certification** (spec 05-01), where a Krawczyk
   box is the deliverable.
4. **Constraint layers.** Any equality constraint expressible with omnibias
   units gets exact projection with `O(1)` backward memory.
5. **Exact ansatz solutions** (level 2), where the equality *is* the solution and
   the verification is symbolic.

## 8. Acceptance gates

Baselines: an unrolled-Newton layer with autodiff, a penalty formulation
(equality as a loss term), and a fixed-mesh contour extraction.

- **G1 forward correctness.** The solved point satisfies `|F| <= 1e-12` on at
  least 1000 randomized systems, with branch signature matching a brute-force
  scan.
- **G2 backward correctness and memory.** The implicit gradient matches the
  unrolled autodiff gradient to `<= 1e-8` relative, while peak memory is
  independent of `max_iter` (asserted, not just measured).
- **G3 degeneracy refusal.** On deliberately tangent systems the layer reports
  `converged = False` and a large `condition`, and never returns a confident
  wrong point.
- **G4 shock task.** On the Burgers Riemann problem, the recovered shock speed
  matches Rankine-Hugoniot to `<= 1e-10`, and on a noisy-data version the
  position error beats the contour-extraction baseline, with skill `> 0`, over
  five seeds.
- **G5 level-2 verification.** `AnsatzSolutionField` raises at construction when
  symbolic verification fails, and a test asserts that a deliberately wrong
  ansatz is rejected.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/equality_layer.py`: forward and backward correctness, memory
  scaling, degeneracy behaviour, the shock task with and without noise, and a
  free-boundary problem.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/locuslayer/`.

## 10. Honesty and scope

- The units come from the founding bias collapse (`delta -> 0`). No temperature
  collapse appears.
- **The three levels above are the claim boundary.** Level 1 always holds;
  level 2 holds inside a named, symbolically verified ansatz class; level 3 — a
  general closed-form PDE solver — is **not claimed and is not true**. Any
  user-facing text must carry the level.
- Branch multiplicity is real and the spurious-branch trap is shown in the worked
  example. The layer always returns a branch signature.
- Transversality is a hypothesis. Near tangency the manifold claim fails and the
  layer must say so.
- Certificate tier: sound enclosure via Krawczyk for existence and uniqueness in
  a box. Symbolic ansatz verification is a finite algebraic fact and is a
  candidate for the Lean obligation class of spec 01-11 — a candidate, not a
  current claim.

## 11. Open questions and risks

- **Topology change during training.** Components can merge or vanish as
  parameters move, and the "nearest branch" heuristic will then track the wrong
  object. Detecting this reliably is unsolved; the mitigation is to monitor the
  branch signature and flag changes.
- **Initialization.** Newton needs a starting point in the basin. For a moving
  boundary, warm-starting from the previous step is natural; for a cold start it
  is not obvious.
- **Multiple constraints.** `m > 1` is supported by the algebra but the
  intersection of several loci is more likely to be tangent or empty; the
  benchmark must include such cases.
- **Falsifier.** If a penalty formulation reaches the same accuracy at lower
  total cost, the implicit layer's complexity is not justified except where the
  certificate is needed.

## 12. Implementation checklist

- [ ] `packages/omnibias-fields/src/omnibias/fields/locus/layer.py` torch and jax
- [ ] `LocusOutput` with required `branch` and `condition` fields
- [ ] Implicit-gradient path with an asserted memory bound
- [ ] Degeneracy-refusal test
- [ ] Burgers shock test against Rankine-Hugoniot
- [ ] Free-boundary test with a manufactured moving boundary
- [ ] `AnsatzSolutionField` construction-time symbolic verification, with a
      rejection test
- [ ] torch/jax parity test
- [ ] `benchmarks/equality_layer.py` plus smoke JSON
- [ ] Docs page and nav entry, carrying the three-level claim boundary
- [ ] Index row in `theory/README.md`
