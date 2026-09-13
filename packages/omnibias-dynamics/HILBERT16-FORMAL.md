# Formal verification used by the Hilbert-sixteenth argument

The checked statements below support the written passage arguments. The
actual physical hypotheses and uniform singular remainders are proved in
the linked analytic notes and have not been formalized. None of these
builds verifies full graphic coverage or Hilbert XVI, and the curve/surface
classification is a separate program.

| Checked source | What its Lean theorem establishes | What must still connect it to the physical count |
| --- | --- | --- |
| [Hilbert16Rolle.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Rolle.lean) | An actual derivative with at most one or two zeros excludes three or four ordered function zeros. A strictly negative derivative at every zero implies at most one zero. | The actual fixed-field displacement, its derivative hypotheses, and its connected admission interval |
| [Hilbert16Parabola.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Parabola.lean) | Fourteen exact algebraic statements, including the physical field restriction, rational Darboux identity, cubic-speed/cofactor cancellation, and signed endpoint cancellation | Existence and capture of the regular passage, uniform tail estimates, matched-endpoint sensitivity, and the sharp multiplier estimate |
| [Hilbert16Resonance.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Resonance.lean) | Seven abstract statements: strict-convex zero exclusion, positive-factor zero equivalence, the joint-interval three-zero implication and its positive-curvature specialization, a curvature error budget, an increasing-core whole-interval count, and construction of an increasing core from a nonpositive convex gap | The actual singular and regular derivative estimates; continuity and connected admission; selection of the clipped sublevel interval and its outer negative-at-zero comparisons |
| [Hilbert16ReturnMap.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ReturnMap.lean) | Eight conditional real-variable implications: exponential normalization, the actual weighted derivative, first and mixed event-time identities, first/second derivative zero bounds, interval-sign transfer, and non-isolation of an identity's interior zeros | Actual differentiability, event existence and uniqueness, a nonzero event normal velocity, and the asserted derivative enclosures; no general passage-class membership theorem is inferred |
| [Hilbert16Scale.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16Scale.lean) | Eight statements about the actual exponential parameter derivative and diagonal limit, fixed-product paths and their derivatives, the second weighted polynomial expression, cancellation, and a pole margin | The directed primitive remainder bounds and membership of actual singular passages in the represented class; physical uniformity is additional |
| [Hilbert16ChiScale.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ChiScale.lean) | Seven exact linear-saddle statements: the χ identification, χ and kappa derivatives, scaled-kappa and reciprocal-separation identities, the sensitivity ratio, and the frozen-exponent obstruction | Membership of an actual quadratic passage in the linear model; uniform physical remainders; G1 and G4 |
| [Hilbert16SaddleNode.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16SaddleNode.lean) | Six exact identities: the double-root quadratic, the wall at the equilibrium, vanishing-separation linear exit, vanishing χ, the `sigma * kappa` factorization on a χ-locus, and the shrinking-root product | Physical C2 remainders, the fold-scale tension, outgoing continuation as `r1 -> 0`, G1 and G4 |
| [Hilbert16TwoBlowup.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16TwoBlowup.lean) | Four exact W-coordinate identities: `epsilon * W = h^epsilon`, the outgoing W-ratio, `d log W / d tau = -u`, and the two-scale product | Physical C2 remainders, a covering of `sep = exp(-1/epsilon^2)`, G1 and G4 |
| [Hilbert16ScaleDichotomy.lean](../../formal/omnibias-analytic/OmnibiasAnalytic/Dynamics/Hilbert16ScaleDichotomy.lean) | Nine exact identities: blow-up height and height ratio, fold-scale `epsilon^4`, the affine leading event exponent with first derivative and vanishing second difference, the joint-axis sum and exclusion, and `epsilon log(1/sep) = 1/epsilon` on the kill sequence | Physical C2 remainders, a third compact scale, G1 and G4 |
| [hilbert16_formal_replay.py](../../benchmarks/hilbert16_formal_replay.py) | Six sealed rational interval-sign obligations from the declared first-root saddle rectangle | The interval evaluator, membership of the actual canonical coefficients in that envelope, and the physical passage proof |

The nine Mathlib Hilbert modules passed their targeted builds and are imported
by the full analytic project. The previous five-module axiom audit reported
only `propext`, `Classical.choice`, and `Quot.sound` on 46 theorems; the
χ-scale module adds seven theorems, the saddle-node module adds six, the
two-blow-up module adds four, and the scale-dichotomy module adds nine,
with the same axiom set. They
contain no `sorry`, custom axioms, or `native_decide`. The minimal-kernel
replay made six actual kernel builds and rejected each stale-seal mutation.
The generated checks ran in a private copy of the existing kernel project.

Reproduce the analytic-module builds from `formal/omnibias-analytic`:

```bash
lake build OmnibiasAnalytic.Dynamics.Hilbert16Rolle
lake build OmnibiasAnalytic.Dynamics.Hilbert16Parabola
lake build OmnibiasAnalytic.Dynamics.Hilbert16Resonance
lake build OmnibiasAnalytic.Dynamics.Hilbert16ReturnMap
lake build OmnibiasAnalytic.Dynamics.Hilbert16Scale
lake build OmnibiasAnalytic.Dynamics.Hilbert16ChiScale
lake build OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode
lake build OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup
lake build OmnibiasAnalytic.Dynamics.Hilbert16ScaleDichotomy
```

The implementation evidence command rebuilds all nine Hilbert modules, audits every
named theorem's axioms, and hashes the exact analytic and executable sources:

```bash
uv run --no-sync python -m benchmarks.hilbert16_program --lean
```

Its event, zero-count, curve, and surface certificates are checked by their
Python replay algorithms. These finite certificates do not acquire a Lean
verification tier merely because the conditional analytic modules build.

Reproduce the finite margin replay from the repository root:

```bash
uv run --no-sync python -m benchmarks.hilbert16_formal_replay --output artifacts/hilbert16/formal_margin_replay.json
```

The varying-detuning argument concerns the actual function
`G(kappa)=log D'(ti(kappa))-log Hreg'(ti(kappa))`, with every physical
coefficient held fixed. The
[actual singular estimate](HILBERT16-SINGULAR-KAPPA-JETS.md) controls
the two-scale remainder through two kappa derivatives and gives actual
input-label jets of order u squared. The
[fixed-field regular composition](HILBERT16-REGULAR-KAPPA-JETS.md)
then bounds the regular logarithmic curvature. These written estimates
prove strict convexity of G on the joint admitted interval. The
[synthesis](HILBERT16-VARYING-DETUNING.md) identifies its zeros with
the displacement's critical points there, and joins the outer comparisons
using a single convex sublevel interval on the connected admission domain.
These analytic premises are not inferred from any finite certificate.

The exact-resonance proof uses the separate negative-at-zeros implication.
Its regular value estimate is substituted only at actual displacement
zeros. Such a substitution alone does not prove convexity away from those
zeros. The fixed-field regular section jets supply the derivative
estimate without changing the splitting parameter.

## Joint jets and the fixed-field scale path

The two-scale path has `omega(s)=omega*exp(-s)`, `u(s)=u*exp(s)` and
fixed `epsilon=omega*u`. Thus its derivative operator is
`E=u*partial_u-omega*partial_omega`. Its second derivative includes
the acceleration of that path:

    E^2 R = omega^2 R_omega,omega - 2 omega u R_omega,u
              + u^2 R_u,u + omega R_omega + u R_u.

A straight directional Hessian omits the last two terms. For the actual
scale product `R=omega*u`, it would return `-2*epsilon`, although the
correct second derivative is zero.

The [scale-jet benchmark](../../benchmarks/hilbert16_scale_jets.py) uses
the new live input/parameter jets with both the velocity and acceleration
rows. PyTorch and JAX agree exactly on three dyadic scale pairs. Exact
realization algebra checks the weighted-derivative identities; an
operand-bound replay checks explicit rational coefficient evaluations in
Lean and rejects substitution of a different re-sealed source. The
benchmark deliberately detects the missing-acceleration error.

```bash
uv run --no-sync python benchmarks/hilbert16_scale_jets.py --lean --output artifacts/hilbert16/scale_jets.json
```

This checks the derivative mechanism used by the analytic remainder
estimate. It does not supply the actual ODE error bound. No neural
approximation is used in these proofs. Confluent representations can
stabilize a future fitted passage model, while validated continuation can
certify supported compact branches and joins. Either would still need
sound error enclosures and the singular tail argument before its output
could replace an actual passage estimate. Existing exact polynomial and
inverse-height formulas currently give a direct route without fitting.

## Exact coefficient, moving-event and composition identities

The [singular-kappa benchmark](../../benchmarks/hilbert16_singular_kappa.py)
checks 22 exact identities. These include the second normal-coefficient
jet derived from the physical off-line field, the fixed-epsilon scale
generator, radial height derivatives, the moving-cut second derivative,
and the actual regular-coordinate chain rules. It rejects both an omitted
second entrance-label jet and an unresolved symbolic expression. Normal
and optimized Python reports agree exactly.

```bash
uv run --no-sync python benchmarks/hilbert16_singular_kappa.py --output artifacts/hilbert16/singular_kappa.json
```

These finite symbolic checks establish their displayed identities. They
do not certify the uniform analytic remainder constants, the admissible
small-parameter cutoff, or the physical cycle bound.
