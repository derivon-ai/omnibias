# 07-06 Validated dynamics and orbits

## 1. Thesis and status

Validated integration is limited by two things — the wrapping effect and the
quality of the Jacobian enclosure — and the derivative tower fixes the second
exactly, while high-order jets attack the first by letting each step be longer
at the same width.

- **Status**: designed
- **Depends on**: 01-01, 03-10, 03-11, 07-01, 07-05
- **Blocks**: none

## 2. Where it lands

`omnibias.dynamics` for the orbit and Lyapunov layer, `omnibias.core.verified`
for the flow primitives. Both exist; the additions are a jet-based stepper and
exact Jacobian enclosures.

## 3. Prior art in omnibias

- `omnibias.core.verified.lohner` — `LohnerSet`, `lohner_step`, `lohner_flow`,
  `qr_gram_schmidt`, `interval_matrix_exp`, `linear_field`,
  `constant_jacobian`, `naive_interval_flow` (the baseline the QR method beats).
- `omnibias.core.verified.kantorovich` — `radii_polynomial_certificate` with
  `RadiiCertificate`, `krawczyk_certificate` and `krawczyk_search` with
  `KrawczykCertificate`, `newton_kantorovich_bounds` with `NKBounds`,
  `certify_zero_radii`.
- `omnibias.dynamics._core.orbits` — `prove_periodic_orbit`,
  `PeriodicOrbitCertificate`.
- `omnibias.dynamics._core.lyapunov` — `certified_lyapunov_exponent`,
  `LyapunovBounds`.
- `omnibias.dynamics._core.poincare` — `PoincareSection`, `PoincareCrossing`,
  `poincare_map`, with guaranteed-crossing logic.
- `omnibias.dynamics._core.{variational,jet_bridge,fields}` — the jet bridge
  already exists.
- `omnibias.core.verified.jet` — verified jets.

**Confirmed gap.** `constant_jacobian` is the supplied Jacobian helper; for a
general vector field the Jacobian enclosure must be provided by the caller and
is typically obtained numerically or by interval-differentiating an expression
tree. There is no path that takes the Jacobian and higher derivatives from the
closed-form tower, and `lohner_step` is a fixed low-order step.

## 4. Mathematics

### The two limits on validated integration

**Wrapping.** An interval box propagated through a rotation grows even when the
true flow is volume-preserving, because a rotated box is enclosed by a larger
axis-aligned box. QR-Lohner mitigates this by carrying an orthogonal frame, and
`lohner.py` implements exactly that. The residual growth per step is set by the
local expansion and the step's truncation error.

**Jacobian quality.** The variational equation needs an enclosure of `Df` over
the current box. A wide `Df` enclosure widens the propagated set multiplicatively
every step, so Jacobian width compounds.

### What the tower fixes exactly

For a vector field whose components are built from the activation dictionary —
which covers every neural vector field, every learned dynamical system, and
many analytically specified ones — the tower supplies `Df` and all higher
derivatives in closed form from a single `sigma` evaluation per order. Combined
with `Interval` arithmetic on the coefficients, the Jacobian enclosure width
reduces to rounding plus the width of the input box, with **no differentiation
error at all**.

This is a clean statement and it is the spec's main content: for this class of
fields, one of the two limits on validated integration disappears.

### Attacking wrapping with high-order steps

The Lohner step's truncation error is `O(h^{p+1})` for an order-`p` method.
Since the tower gives Taylor coefficients of the solution to arbitrary order at
no extra activation cost (spec 01-01 and the existing jet kernels), the step can
be raised to high `p` cheaply. At fixed truncation tolerance `eps`:

```
h ~ ( eps )^{1/(p+1)}
```

so raising `p` from `4` to `12` at `eps = 1e-14` grows the step from
`(1e-14)^{1/5} = 1.6e-3` to `(1e-14)^{1/13} = 8.4e-2`, a factor of about `53`.
Fewer steps means fewer wrapping applications, and since wrapping compounds
per step, the accumulated width improves super-linearly in the step-count
reduction.

That is the mechanism, and it is the standard argument for high-order validated
integrators (Taylor-model methods use it). The omnibias-specific part is that
the Taylor coefficients come from the closed-form tower rather than from
automatic differentiation of an expression tree, which is both faster and free
of the AD tool's own error handling.

### Singularity proximity as a step controller

Spec 03-10's jet-Padé machinery locates the nearest complex singularity of the
solution from its Taylor coefficients. The radius of convergence bounds the
usable step, so a step controller that knows the singularity distance can take
the largest provably valid step rather than a heuristically safe one. This is a
natural pairing and, as far as the prior-art check shows, not something the
existing stepper does.

## 5. Worked example

**A validated orbit of the Lorenz-like field, counting the improvement.**

Take a smooth vector field on `R^3` built from tanh units — a learned or
analytically specified system in the class the tower covers — with a numerically
observed periodic orbit of period `T = 1.5586` and a local expansion rate near
`0.9` per unit time.

*Baseline.* Order-4 Lohner, `Df` supplied by interval differentiation of the
expression tree with a relative width of `1e-10`, truncation tolerance
`1e-14`, so `h = 1.6e-3` and `n = T / h = 974` steps.

Per-step width growth has two parts: the Jacobian width contribution, roughly
`(1 + 1e-10)` per step, and the wrapping contribution from the QR frame, say a
factor `w` per step. Over `974` steps the Jacobian contribution alone
multiplies the width by

```
(1 + 1e-10)^974 = 1 + 9.7e-8
```

which is negligible here — a useful thing to discover, because it says the
Jacobian is **not** the binding constraint in this configuration, and the exact
Jacobian buys nothing.

*Where the exact Jacobian does matter.* Repeat with a stiffer field where the
interval Jacobian's relative width is `1e-4` (typical when the expression tree
has cancellation, which is common for fields with near-cancelling terms). Then

```
(1 + 1e-4)^974 = e^{0.0974} = 1.102
```

a `10%` width inflation from Jacobian error alone, which the tower removes
entirely. Push to `1e-3` width and it becomes `e^{0.974} = 2.65`, a factor of
`2.65`. So the exact-Jacobian contribution matters exactly when the field is
one whose derivative is hard to enclose tightly — which is a statement about the
field, not the method, and must be reported per problem.

*High-order steps.* Order-12 at the same tolerance gives `h = 8.4e-2` and
`n = 19` steps instead of `974`. If the per-step wrapping factor is `w = 1.001`,
the accumulated wrapping goes from `1.001^974 = 2.65` to `1.001^19 = 1.019`:
**the wrapping inflation drops from a factor of `2.65` to `2%`.**

That is the dominant effect in this example, and it is worth being explicit that
it comes from the order increase rather than from anything unique to omnibias —
except that the order increase is cheap here because the Taylor coefficients are
closed form.

*The reporting rule this suggests.* Every validated run should report the width
budget split into truncation, Jacobian, wrapping and rounding, exactly as spec
07-02 requires for enclosures. Without it, an improvement to a non-dominant term
looks the same as a real one.

## 6. Proposed API

```python
# omnibias/core/verified/lohner.py    -- additions
def tower_jacobian(field_spec) -> JacobianEnclosure:
    """Exact closed-form Jacobian enclosure for activation-dictionary fields.
    Width is rounding plus input-box width; no differentiation error."""

def lohner_step_jet(
    state: LohnerSet, field_spec, *, order: int, h: float,
) -> LohnerSet:
    """Order-p validated step using closed-form Taylor coefficients."""

def adaptive_step_from_singularity(
    field_spec, state, *, order: int, safety: float = 0.5,
) -> float:
    """Largest provably valid step from the jet-Pade singularity radius
    (spec 03-10), rather than a heuristic."""

@dataclass(frozen=True)
class WidthBudget:
    truncation: float
    jacobian: float
    wrapping: float
    rounding: float
    dominant: str
```

Pure Python in `omnibias.core.verified`, consistent with the pure-core rule.

## 7. Practical use cases

1. **Proving periodic orbits of learned dynamical systems**, which is currently
   awkward because the Jacobian of a neural field is exactly the thing
   expression-tree interval differentiation handles worst.
2. **Two-sided certified Lyapunov exponents** over longer horizons, feeding
   `certified_lyapunov_exponent`.
3. **Certified Poincaré return maps** with guaranteed crossings over more
   returns, since fewer steps means less width at the section.
4. **Certified reachability** for control systems (pairing with
   `omnibias.control`'s safety filter), where horizon length is the binding
   constraint.
5. **Validated continuation** of orbit families, where each continuation step
   needs a fresh certificate and cost dominates.

## 8. Acceptance gates

Baselines: the existing `lohner_flow` at its current order with caller-supplied
Jacobians, and `naive_interval_flow` as the floor.

- **G1 width budget.** Every validated run emits a `WidthBudget` with its
  dominant term named. No improvement claim is accepted without one.
- **G2 exact Jacobian correctness.** `tower_jacobian`'s enclosure contains the
  true Jacobian on `1000` random boxes for `10` fields, at `100%`. A single miss
  is a bug.
- **G3 horizon extension.** At fixed final enclosure width, the jet stepper
  reaches at least `5x` the time horizon of the order-4 baseline on a suite of
  `10` fields, with per-field results reported.
- **G4 orbit proof.** `prove_periodic_orbit` succeeds on at least `3` orbits
  where it currently fails due to accumulated width, with the certificates
  replayable.
- **G5 step controller soundness.** Steps chosen from the singularity radius
  never exceed the true radius of convergence, checked against analytically
  known cases. This is a soundness gate, not a performance one.
- **G6 no regression.** Existing `lohner_flow` results are bit-unchanged when
  the new paths are not selected. Asserted by test.

## 9. Benchmark plan

- `benchmarks/validated_dynamics.py`: width budgets, horizon comparison across
  the `10`-field suite, orbit-proof successes, step-controller soundness.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/validated_dynamics/`.

## 10. Honesty and scope

- Everything here is about **one field, one initial box, one finite horizon**.
  Validated integration produces enclosures of trajectories, not statements
  about attractors, invariant measures, or asymptotic behaviour.
- The exact-Jacobian benefit applies to fields **in the activation dictionary**.
  For a general field it does not apply, and the worked example shows it can be
  negligible even when it does — which is why G1 comes first.
- A `5x` horizon extension is a statement about a benchmark suite at a fixed
  width, not a general property.
- Existing `PeriodicOrbitCertificate` and `LyapunovBounds` semantics are
  unchanged; this spec makes them reachable on more problems, not stronger on
  the problems they already reach.
- The founding `delta -> 0` bias collapse supplies the exact Jacobian and the
  Taylor coefficients. No temperature collapse appears.
- Certificate tier: **sound enclosure**, with existence obligations
  (radii-polynomial, Krawczyk) already producing certificates that can escalate
  to the Lean kernel where the obligation is finite and rational.

## 11. Open questions and risks

- **Wrapping may still dominate.** High-order steps reduce the *number* of
  wrapping applications but not the per-step mechanism. For strongly hyperbolic
  systems the improvement may be much smaller than the step-count ratio
  suggests, and G1 will show it.
- **High-order interval Taylor coefficients can be wide.** Closed-form
  coefficients are exact as formulas, but evaluating them on a wide input box
  can produce wide intervals, and at high order the dependency problem worsens.
  Affine or Taylor-model arithmetic may be needed and is not free.
- **Step controllers from Padé are heuristic unless bounded.** G5 makes the
  controller's output subject to a soundness check; if a rigorous bound on the
  convergence radius is unavailable, the controller must be used only to
  *suggest* a step that is then validated, never to justify one.
- **The `1e-4` Jacobian width in the example is illustrative.** Whether real
  fields of interest have Jacobian enclosures that wide is an empirical
  question, and if they do not, the exact-Jacobian contribution is small.
- **Falsifier.** If G1's budgets show truncation and Jacobian are both
  negligible against wrapping on every problem of interest, then only the
  step-count reduction matters, and the spec should be rewritten as a
  high-order-stepper spec with the tower as an implementation detail.

## 12. Implementation checklist

- [ ] `tower_jacobian` and `lohner_step_jet` in
      `packages/omnibias-core/src/omnibias/core/verified/lohner.py`
- [ ] `WidthBudget` emitted by every validated run
- [ ] `1000`-box Jacobian containment test over `10` fields
- [ ] `adaptive_step_from_singularity` with the soundness check of G5
- [ ] Affine or Taylor-model arithmetic evaluated for the high-order
      coefficients, with the choice justified by measurement
- [ ] `10`-field horizon suite with per-field reporting
- [ ] Three previously failing orbit proofs, with replayable certificates
- [ ] Bit-unchanged regression test for existing `lohner_flow` paths
- [ ] `benchmarks/validated_dynamics.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: rigorous computer-assisted proof in dynamical systems** — the family
that includes Smale's 14th problem (the Lorenz attractor's existence as a
strange attractor, resolved by Tucker for the classical parameters), the
existence of chaos in specific systems, and the general programme of
computer-assisted proofs of global dynamical structure.

The reason the general programme stays external is that this machinery proves
statements about **finitely many trajectories over finite horizons from finite
initial boxes**. Statements about attractors, invariant measures, structural
stability, or the global topology of a phase portrait quantify over all initial
conditions and all time. Bridging that requires a covering argument — a finite
set of boxes shown to cover the relevant region together with a proven
trapping or hyperbolicity property — and constructing such an argument is the
actual mathematical work in every computer-assisted proof of this kind. The
integrator is a tool inside it, not a substitute for it.

Concretely: a certified periodic orbit is a proof that *one* orbit exists. It is
not a proof that the system is chaotic, that its attractor has a given
structure, or that a bifurcation occurs.

This spec does not claim, imply, or provide evidence for any global dynamical
statement.
