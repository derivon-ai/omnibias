# Certified Abelian-integral zero count

`omnibias.dynamics.abelian` certifies an exact zero count for one declared
Abelian integral in the cubic elliptic family

\[
H(x,y)=y^2+x^3+p x+q,\qquad
\omega=\alpha(H)y\,dx+\beta(H)x y\,dx.
\]

It combines:

1. an exact Picard--Fuchs syzygy over \(\mathbb Q[h]\);
2. interval quadrature for \(A,B,J_0,J_1\);
3. validated complex continuation of the rank-two Gauss--Manin period basis;
4. an argument-principle upper count;
5. Krawczyk-certified simple real zeros for the lower count.

If the two counts agree, the result is a sealed exact integer for that
instance and contour.

The API returns the four-component view \((A,B,J_0,J_1)\), but for this cubic
it is not a rank-four local system:

\[
A=\frac{3(h-q)}5J_0-\frac{2p}5J_1,\qquad
B=\frac{2p^2}{21}J_0+\frac{3(h-q)}7J_1.
\]

For tighter interval enclosures, the evaluator propagates the overcomplete
four-component representation and checks it against the exact rank-two
reconstruction.

## Build the named cubic instance

The shipped forced-factor smoke problem uses

\[
H=y^2+x^3-x,\qquad
r(h)=h^2-\frac1{64}.
\]

The factor roots \(\pm1/8\) lie in the real period annulus.

```python
from omnibias.dynamics import (
    build_abelian_evaluator,
    named_cubic_abelian_problem,
)

problem = named_cubic_abelian_problem(
    quadrature_panels=64,
    continuation_steps=6,
    continuation_order=12,
)
evaluator = build_abelian_evaluator(problem)

assert evaluator.picard_fuchs.verified
assert evaluator.initial_data.area.lo > 0.0
assert evaluator.picard_fuchs.period_operator.order == 2
```

The period operator on \(J_0=\oint dx/y\) is

\[
15+216hD+4(27h^2-4)D^2.
\]

The area \(A=\oint y\,dx\) is annihilated by that operator composed on
the right with \(D\). Initial values for \(A,A',A''\) come from the
nonsingular cosine turning-point substitution; the implementation does not
numerically integrate \(F^{-3/2}\) through an endpoint.

The mixed evaluator propagates

\[
(A,B,J_0,J_1),\qquad A'=J_0/2,\quad B'=J_1/2,
\]

using the same exact \(2\times2\) system on \((J_0,J_1)\). The base quadrature
for \(B=\oint x y\,dx\) adds the factor \(x(\theta)\) to the nonsingular
action integrand.

For \(s=\sqrt{-p/3}\), \(h_c=q-2s^3\), \(h_s=q+2s^3\), and
\(z=(h-h_c)/(h_s-h_c)\), the distinguished Hamiltonian-oriented action is

\[
A(h)=\frac{4\pi s^{5/2}}{\sqrt3}\,
z\,{}_2F_1\!\left(\frac16,\frac56;2;z\right).
\]

Euler's integral representation makes the hypergeometric factor zero-free
on the cut plane. Before counting, exact rational inequalities must prove
that the whole rectangle satisfies \(h_c<\Re h<h_s\). This excludes the
center zero and the saddle cut and keeps all straight continuation paths on
the distinguished branch.

## Certify the two-sided count

<!-- docs-test: slow -->
```python
from omnibias.dynamics import (
    certify_abelian_zero_count,
    verify_abelian_zero_count,
    verify_abelian_zero_count_formally,
)

count = certify_abelian_zero_count(
    problem,
    half_width=0.2,
    half_height=0.04,
    segments=8,
    max_segments=16,
)

assert count.upper_status == "PROVED"
assert count.action_zero_free_domain.verified
assert count.upper_count == 2
assert len(count.lower_zeros) == 2
assert count.exact_count == 2
assert verify_abelian_zero_count(count)

formal = verify_abelian_zero_count_formally(count)
assert formal.theorem_prover_verified == (
    formal.picard_fuchs.verified and formal.winding_integer.verified
)
```

## Genuine mixed form and finite parameter cover

The genuine named problem uses
\(\alpha(h)=h^2-1/64\), \(\beta(h)=1/1000\). Its certified zero boxes exclude
both old planted roots. This earns `forced_factor_instance=False`; it does not
prove the new roots irrational, because every finite-width interval contains
rational numbers.

<!-- docs-test: slow -->
```python
from fractions import Fraction

from omnibias.dynamics import (
    AbelianCoefficientBox,
    certify_abelian_uniform_cover,
    named_genuine_cubic_abelian_problem,
    verify_abelian_uniform_cover,
)

genuine_problem = named_genuine_cubic_abelian_problem(
    quadrature_panels=64,
    continuation_steps=6,
    continuation_order=12,
)
genuine_count = certify_abelian_zero_count(
    genuine_problem,
    root_guesses=(Fraction(-123, 1000), Fraction(123, 1000)),
    segments=8,
    max_segments=16,
)
assert genuine_count.exact_count == 2
assert genuine_count.seal is not None
assert genuine_count.seal["honesty"]["forced_factor_instance"] is False

coefficient_box = AbelianCoefficientBox.create(
    alpha=(Fraction(-1, 64), 0, 1),
    beta=((Fraction(9, 10_000), Fraction(11, 10_000)),),
)
uniform = certify_abelian_uniform_cover(
    genuine_problem,
    coefficient_box,
    segments=8,
    max_segments=16,
)
assert uniform.uniform_bound == 2
assert verify_abelian_uniform_cover(uniform)
```

`box_cover_tiling` is a Mathlib-free Lean obligation. It reconstructs the
recursive rational bisections and checks each terminal count is at most the
declared bound. Each linked leaf also passes the existing
`winding_integer_isolation` obligation. The analytic winding enclosures remain
trusted inputs.

For an instance certificate, `theorem_prover_verified` is `True` only when the
installed Lean kernel accepts the Picard--Fuchs syzygy and winding isolation.
For a uniform cover it requires the tiling obligation and every linked leaf
isolation. The analytic interval continuation remains a replayed Python
certificate input.

## Negative controls

Two failures are first-class:

- changing a coefficient of the Picard--Fuchs operator fails the exact
  determining-matrix test;
- placing \(h=\pm1/8\) on the counting contour makes its image contain zero,
  so winding returns `BLOCKED`.
- choosing a rectangle not proved strictly between the two critical energies
  is rejected before continuation.

No number is returned when the leading Picard--Fuchs coefficient may vanish
on a continuation path.

## Scope

The exact holonomic layer supports general hyperelliptic de Rham reduction
through `certify_hyperelliptic_picard_fuchs` whenever its exact moment
reduction is nonsingular over \(\mathbb Q(h)\). The phase-one zero-count
consumer supports depressed cubic potentials and forms
\(\alpha(H)y\,dx+\beta(H)x y\,dx\); it does not yet validate arbitrary
hyperelliptic turning-point geometry or arbitrary polynomial one-forms. The
uniform result is a finite rational coefficient box for one Hamiltonian and
contour, not a degree-uniform result. It does not prove finiteness of \(H(2)\)
or \(H(n)\).

The two roots in the named smoke are deliberately inserted through \(r(H)\).
This is a certified forced-factor instance in Petrov's elliptic setting, not
a reproduction of Petrov's parameter-uniform sharp theorem; in particular,
\(H^2y\,dx\) is not a quadratic one-form.

The endpoint neighborhoods of the center and separatrix critical values are
outside the finite regular contour and remain separate local-analysis
obligations. Poincare--Pontryagin transfers a simple Abelian-integral zero to
a nearby limit cycle only for sufficiently small perturbation; this module
does not compute an effective \(\varepsilon_0\).

See also:

- [`HILBERT16-ABELIAN-COUNT.md`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-dynamics/HILBERT16-ABELIAN-COUNT.md)
  for the derivation and obligation boundary;
- [`abelian_zero_count_smoke.json`](../benchmarks/abelian_zero_count_smoke.json)
  for the GA1--GA8 replay;
- [contour winding](contour.md) for the underlying winding collapse.
