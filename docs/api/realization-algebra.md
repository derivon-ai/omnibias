# Exact realization algebra and finite replay

`omnibias.core.realization` adds an exact coefficient-space register for finite
polynomial neural networks. It complements the trainable realization geometry;
it does not replace the activation derivative tower. The six activation operator
roles and the three named collapses retain their meanings in
[the operator surface](../operator-surface.md).

Numerical rank, a small fitting residual, and membership in an algebraic closure
could not distinguish an attained network from a boundary limit. Exact rational
coefficient maps, real algebraic witnesses, and Laurent parameter paths now
distinguish those cases in `omnibias.core.realization`. Bounded exclusion and
proof replay live in `omnibias.verify.neuroalgebra` and `omnibias.core.proof`.

## Compile coefficients, including biases and depth

`PolynomialNetworkSpec` uses row-major weights followed by biases in each layer.
Each hidden layer has an explicit rational activation polynomial in ascending
powers. The output layer is affine. `biases=False` gives the homogeneous
architectures when the activations are monomials. Input monomial ordering is
returned explicitly; never assume a conventional ordering implicitly.

```python
from fractions import Fraction as Q
from omnibias.core.realization.polynomial import (
    AlgebraBudget, PolynomialNetworkSpec, compile_coefficient_map,
)

spec = PolynomialNetworkSpec((1, 1, 1), ((Q(1, 3), 0, 1),), biases=True)
coefficient_map = compile_coefficient_map(spec)
# f(x) = 3 * (1/3 + (2*x + 1)**2) + 4 = 8 + 12*x + 12*x**2.
assert coefficient_map.input_indices == ((0,), (1,), (2,))
assert coefficient_map.evaluate((2, 1, 3, 4)) == ((Q(8), Q(12), Q(12)),)
assert len(coefficient_map.jacobian((2, 1, 3, 4))) == 3
```

`AlgebraBudget` limits polynomial degree, rational bit size, term count and the
number of products in one polynomial multiplication. These are explicit
representation budgets, not a polynomial-time complexity guarantee. Exceeding
one raises `AlgebraBudgetExceeded`; it does not prove non-membership. Floats and
booleans are rejected as rational coefficients. Use `int` and `Fraction`.

## Complete membership in declared families

The following classifications are complete for their stated real, homogeneous
architecture when their exact computation completes. They make no assertion
about a different width, biases, activation or depth.

| API | Complete scope | Decision and witness |
| --- | --- | --- |
| `classify_linear(C, hidden_widths=())` | Any input/output dimensions; any finite list of linear hidden widths; no biases | Target rank must fit every layer. Rational rank factorization supplies parameters. This image is closed. |
| `classify_scalar_quadratic(A, hidden_width)` | Scalar homogeneous quadratic `x.T @ A @ x`; symmetric rational `A`; one hidden square layer | Symmetric rank must fit the width. Rational signed-square congruence supplies parameters. Output weights can have either sign. |
| `classify_binary_quadratic(C)` | Two inputs, two shared square units, any number of outputs; rows ordered `(x², xy, y²)` | Rank at most one is realizable. Rank above two is excluded. At rank two, the real square-direction discriminant distinguishes attainment, boundary, and exclusion. |

```python
from omnibias.core.realization.membership import (
    classify_linear, classify_scalar_quadratic, classify_binary_quadratic,
)

linear = classify_linear([[1, 2, 3], [0, 1, -1]], hidden_widths=(3, 2))
assert linear.status == "realizable"
assert linear.witness is not None and linear.witness.verify(linear.source)
assert classify_scalar_quadratic([[1, 0], [0, 1]], hidden_width=1).status == "excluded"

algebraic = classify_binary_quadratic([[1, 0, 2], [0, 1, 0]])
assert algebraic.status == "realizable"
assert algebraic.witness is not None and algebraic.witness.verify(algebraic.source)
```

For two independent coefficient rows, let `M12`, `M13`, `M23` be their three
two-column minors and `D = M13**2 - M12*M23`. At coefficient rank two:

- `D > 0` gives two distinct real square directions and a real algebraic witness.
- `D == 0` gives a tangent plane with only one square direction: `closure_only`.
- `D < 0` is outside the Euclidean closure of this real architecture.

These predicates distinguish the real image from its Euclidean and Zariski
closures. There is no general deep-network quantifier-elimination API here.
Other architectures use an explicit witness, or bounded search with an honest
inconclusive outcome.

## Real algebraic values and portable witnesses

`RealAlgebraicField` specifies a square-free rational primitive polynomial and a
rational interval containing exactly one of its real roots. Parameters are
polynomials in that selected primitive element. The isolating interval selects
the intended real embedding; substitution must still satisfy every target equation.

The primitive need not be irreducible. Equality and sign refer to the selected
root. Inversion requires a unit modulo the supplied primitive polynomial; a
nonunit denominator raises `ZeroDivisionError` even if refinement to a smaller
primitive would make it invertible. Refinement is explicit, not guessed.

```python
import json
from omnibias.core.realization.algebraic import RealAlgebraicField
from omnibias.core.realization.witness import witness_from_payload

field = RealAlgebraicField.sqrt(2)
alpha = field.generator
assert (alpha * alpha - 2).is_zero()
assert (alpha - Q(7, 5)).sign() == 1

assert algebraic.witness is not None
payload = json.loads(json.dumps(algebraic.witness.to_payload()))
restored = witness_from_payload(payload, algebraic.source)
assert restored.verify(algebraic.source)
```

Rational, algebraic and Laurent witnesses all carry the exact coefficient-map
fingerprint. Decoding replays against that map. A fingerprint alone is not a
mathematical proof; the actual coefficient equalities are checked as well.

## A closure regression and its relation to bias collapse

The map `(x², xy)` is an unattained boundary point for two shared square units.
For nonzero `epsilon`, use hidden forms `x` and `x + epsilon*y`, and outputs

\[
\left(x^2,\frac{(x+\epsilon y)^2-x^2}{2\epsilon}\right)
=\left(x^2,xy+\frac{\epsilon}{2}y^2\right).
\]

The coefficients diverge while the represented polynomial has a finite limit.
The collision resembles the derivative extraction behind bias collapse.
Replacing the colliding pair by a derivative atom changes the allowed
architecture; it does not establish attainment in the original width-two
square architecture.

```python
boundary = classify_binary_quadratic([[1, 0, 0], [0, 1, 0]])
assert boundary.status == "closure_only"
assert boundary.witness is None and boundary.closure_path is not None
assert boundary.closure_path.verify(boundary.source)
error_bounds = boundary.closure_path.coefficient_error_bounds(boundary.source, Q(1, 10))
assert max(error_bounds) == Q(1, 20)
```

`LaurentClosureWitness` substitutes every parameter path into every exact source
coefficient. It checks that all negative powers cancel and the constant term is
the requested target. Its finite coefficient tail bound applies for
`0 < abs(epsilon) <= radius`. Such a witness proves approach to a target;
only a separate exclusion argument establishes `closure_only`. The binary
classifier supplies that argument through the tangent-plane criterion.

The distinction is a regression target when interpreting Proposition 4.2 of
[the polynomial-network paper, arXiv v2](https://arxiv.org/abs/2402.00949v2).
The endpoint issue is recorded as an independent mathematical check requiring
author review, rather than silently importing a weak discriminant condition as
real-image membership.

A second independent check concerns the two-output endpoint of that paper's
Theorem 6.5. The exact parameter Jacobian has rank six in the six-dimensional
coefficient space:

```python
from omnibias.core.realization.rank import certify_generic_rank

two_output = compile_coefficient_map(PolynomialNetworkSpec.monomial((2, 2, 2)))
generic = certify_generic_rank(two_output.polynomials, (1, 0, 0, 1, 1, 0, 0, 1))
assert generic.complete and generic.lower == generic.upper == 6
```

Over characteristic zero this establishes a full ambient algebraic closure.
The Euclidean distance degree of the full coefficient space is one: the sole
generic critical point is the target itself. This checks the full-space
endpoint independently; it is not an implementation of general learning-degree
or critical-point enumeration. Regression tests reside in
`packages/omnibias-core/tests/test_realization_algebra.py`.

## Bounded search and exact coverage

`bounded_realization_search` accepts any compiled polynomial map, a flat target
in the returned coefficient ordering, and a rational parameter box. Every leaf
of an exclusion proof excludes at least one coefficient equation. Binary splits
cover the entire original box, including shared boundaries.

```python
from omnibias.verify.neuroalgebra import bounded_realization_search

single_weight = compile_coefficient_map(PolynomialNetworkSpec.monomial((1, 1)))
search = bounded_realization_search(single_weight, (2,), ((-1, 1),))
assert search.status == "excluded_on_box"
assert search.certificate is not None and search.certificate.verify(single_weight)

attained = bounded_realization_search(single_weight, (2,), ((1, 3),))
assert attained.status == "realizable_in_box"
assert attained.witness is not None and attained.witness.verify(single_weight)
```

An exact rational midpoint root produces a witness. Exhausted node/depth budgets
or interval dependency produce `inconclusive`, with unresolved boxes. Exclusion
on `[-1, 1]` does not exclude the parameter value `2` globally. Non-rational
roots are handled by algebraic witnesses, not approximate midpoint acceptance.

## Operand-bound finite Lean replay

`omnibias.core.proof.realization_replay` provides:

| Producer | Actual finite operands recomputed by Lean |
| --- | --- |
| `polynomial_evaluation_certificate` | Explicit source polynomials, rational point, target readout |
| `polynomial_inequality_certificate` | Explicit polynomial expressions and rational point; `eq`, `ge`, `gt` against zero |
| `rank_replay_certificate` | Source matrix, full factorization `A = B C`, selected source minor and its nonzero determinant |
| `interval_ldlt_replay_certificate` | Original symmetric interval matrix and every rational interval Schur complement; positive pivot checks |
| `krawczyk_replay_certificate` | Center, enclosed gradient/Hessian, rational preconditioner, box, complete Krawczyk image, strict inclusion, nonzero preconditioner determinant, optional row contraction |
| `strict_box_inclusion_certificate` | Both boxes and every strict endpoint inequality |
| `interval_error_budget_certificate` | Error endpoints and their containment in the stated signed budget |

`omnibias.core.proof.realization_algebra_replay` adds:

| Producer | Actual finite operands recomputed by Lean |
| --- | --- |
| `sturm_isolation_replay_certificate` | Primitive derivative, Euclidean remainder identities, degree descent, non-root endpoints and variation difference |
| `moment_identity_replay_certificate` | Original coefficients, offsets and center; every normalized moment `sum a_i*(b_i-center)**k/k!` |
| `algebraic_realization_replay_certificate` | Source coefficient polynomials substituted with polynomial algebraic parameters; exact divisibility by the primitive; Sturm arithmetic |
| `laurent_closure_replay_certificate` | Source substitution, cancellation of poles, target coefficients and finite coefficient tail budgets |
| `polynomial_box_replay_certificate` | Every rational split, recursive coverage and the actual interval polynomial computation at each excluded leaf |

```python
from omnibias.core.proof.realization_algebra_replay import (
    algebraic_realization_replay_certificate, laurent_closure_replay_certificate,
)
from omnibias.core.proof.realization_replay import verify_replay_certificate

assert algebraic.witness is not None and boundary.closure_path is not None
algebraic_replay = algebraic_realization_replay_certificate(algebraic.source, algebraic.witness)
laurent_replay = laurent_closure_replay_certificate(boundary.source, boundary.closure_path, radius=Q(1, 10))
assert verify_replay_certificate(algebraic_replay)
assert verify_replay_certificate(laurent_replay)
assert search.certificate is not None
box_replay = search.certificate.to_replay_certificate(single_weight)
assert verify_replay_certificate(box_replay)
```

These are sealed, source-bound finite obligations. Re-sealing a false target,
factorization, derivative chain, Laurent cancellation or split does not repair
its arithmetic: the actual generated Lean theorem fails. Supplying
`expected_source_digest=` to `verify_replay_certificate` also prevents a
different re-sealed source from being used for the caller's original claim.

The bridge functions `omnibias.core.proof.lean_check.check_certificate` and
`omnibias.formal.mathlib_check.check_certificate` run real `lake build` commands.
Only a successful build earns the respective result's `verified` flag and the
corresponding consumer tier. Missing toolchains leave verification false.
The bridges lock each project's generated file across threads and processes
and restore it after a build, including failures. Direct external builds and
manual edits do not participate in that advisory lock.

For pair and cluster transitions,
`omnibias.verify.neuromanifold.formal.formalize_confluence` also connects these
checks to the original stored scales, centers and output weights. Its finite
proofs recompute affine biases, common-scale identities, the mean, normalized
moments, pair spread, proposed-coordinate rounding residuals and unused zero
moment slots before checking the final observation budgets. The analytic Taylor
bound and its error propagation remain explicit dependencies.

The emitted theorems use kernel reduction, without native-evaluation axioms.
Their scope remains **finite rational arithmetic**. They do not formalize the
architecture compiler, general Sturm real-root theorem, general interval
soundness theorem, analytic derivative-enclosure provenance, an abstract
rank/positive-definiteness theorem, or a continuum limit. Those dependencies
remain explicit in the mathematical interpretation. Algebraic divisibility
replay rejects a reducible primitive when equality holds only at the selected
root; refine that primitive before requesting this formal family.

The exact algebra and replay producers belong to the permissive core. Bounded
verification consumers and the Mathlib integration retain the verification
license tier. No neural backend imports are introduced into core.
