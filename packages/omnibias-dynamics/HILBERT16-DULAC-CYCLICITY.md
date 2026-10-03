# Hilbert XVI: exact compactification and finite Dulac models

This note records a computer-assisted **finite-model** result. It does not
claim finite cyclicity of a Roussarie graphic, closure of a DRR case,
\(H(2)<\infty\), or a solution of Hilbert's sixteenth problem.

## 1. Exact source geometry

Let
\[
\dot x=P(x,y),\qquad \dot y=Q(x,y),
\]
where \(P,Q\in\mathbb Q[x,y]\) have declared degree at most \(n\). In the chart
\[
u=y/x,\qquad v=1/x,
\]
the positive time desingularization gives
\[
\begin{aligned}
\dot u&=v^n\{Q(1/v,u/v)-uP(1/v,u/v)\},\\
\dot v&=-v^{n+1}P(1/v,u/v).
\end{aligned}
\]

For a monomial \(a_{ij}x^iy^j\), multiplication by \(v^n\) sends it to
\(a_{ij}u^jv^{n-i-j}\). Because \(i+j\leq n\), no negative exponent remains.
`omnibias.dynamics.compactify` applies this index map directly over
\(\mathbb Q\). The second component has exponent \(n-i-j+1\), so \(v=0\) is
invariant exactly.

The \(U_2\) chart interchanges \(x,y\) and \(P,Q\). The original affine plane is
stored as \(U_3\). Rational singularities of the tangential equator field are
found by the rational-root theorem; remaining simple real roots are isolated
by interval Newton and checked against a distinct-root Sturm count. Multiple
nonrational roots are refused.

The `polynomial_identity_q` obligation makes Lean recompute the cleared
coefficient sums. It does not prove that a chosen global graphic exists.

## 2. Hyperbolic local data

A declared saddle at \(z_0\) is accepted only when
\[
P(z_0)=Q(z_0)=0
\]
exactly and two disjoint eigenvalue enclosures satisfy
\[
\lambda_s<0<\lambda_u,\qquad
\lambda^2-\operatorname{tr}(J)\lambda+\det(J)=0,
\]
with a nonzero derivative of the characteristic polynomial. The
hyperbolicity-ratio enclosure must overlap
\[
r=-\lambda_s/\lambda_u.
\]

This verifies local hyperbolicity. It does not establish a homoclinic
connection. The resonant named example is different: for
\[
\dot x=y,\qquad \dot y=x-x^2,
\]
the Hamiltonian
\[
H(x,y)=\frac{y^2}{2}-\frac{x^2}{2}+\frac{x^3}{3}
\]
is an exact first integral, and \(H=0\) contains the homoclinic loop joining
the saddle to the turning point \(x=3/2\).

## 3. Finite Dulac expansion

The accepted input is a finite declared expansion
\[
L_N(x)=\sum_{\alpha,j}c_{\alpha,j}x^\alpha(\log x)^j.
\]
The displacement model is \(D_N=L_N-x\). Its `remainder_bound` is recorded but
not consumed as an analytic premise; consequently no physical return-map
membership follows.

Set \(x=e^{-\kappa}\). Then
\[
x^\alpha(\log x)^j=(-1)^j\kappa^j e^{-\alpha\kappa}.
\]
For rational \(\alpha\), `dulac_to_confluent` performs this map exactly over
\(\mathbb Q\). `certify_exponential_cyclicity` repeatedly applies
\[
(D-a)\left[p(\kappa)e^{b\kappa}\right]
=\left[p'(\kappa)+(b-a)p(\kappa)\right]e^{b\kappa}.
\]
Choosing the least exponent lowers the finite dimension by exactly one. Rolle
therefore bounds the global number of isolated real zeros by
\[
\dim(D_N)-1.
\]

The exact derivation rows are also passed through `polynomial_identity_q`.
Lean checks those finite identities, not the analytic Rolle theorem.

## 4. Three finite-model outcomes

### Rational ratio

The named \(r=2\) local model has displacement
\[
D_N(x)=2x^2+3x^{5/2}-x^3.
\]
Its first three nonzero coefficients are exactly
\[
(2,0,2),\qquad (5/2,0,3),\qquad (3,0,-1),
\]
where each tuple denotes `(exponent, log_power, coefficient)`. The transformed
finite sum has dimension three and receives the global bound two.

### Resonance

The named \(r=1\) homoclinic model has
\[
D_N(x)=3x^2\log x+2x^2+\tfrac12x^3.
\]
Its first three nonzero coefficients are
\[
(2,1,3),\qquad (2,0,2),\qquad (3,0,1/2).
\]
The two exponent-\(2\) terms merge into
\[
(2-3\kappa)e^{-2\kappa},
\]
so the derivation-division dimension is again three and the bound is two.

### Irrational-ratio enclosure

For the rational linearization
\[
J=\begin{pmatrix}0&1\\1&1\end{pmatrix},
\]
the exact eigenvalues are \((1\pm\sqrt5)/2\). The implementation does not
represent \(\sqrt5\) as a float identity. It verifies rational enclosures
\[
\lambda_s\in[-13/20,-3/5],\quad
\lambda_u\in[8/5,33/20],\quad
r\in[3/8,2/5].
\]

The three displacement terms have exponent boxes
\[
[3/8,2/5],\quad [11/8,7/5],\quad [19/8,12/5].
\]
The first and third coefficient boxes are positive. The middle term is
\(c x^\alpha\log x\) with \(c\in[-3,-2]\); after
\(x=e^{-\kappa}\), its coefficient is \(-c>0\). Every transformed kernel
\(\kappa^j e^{-\alpha\kappa}\) is positive, so the displacement is strictly
positive for every parameter in the box and every \(\kappa>0\). The uniform
isolated-zero bound is zero. The middle term is also evaluated through
`power_compensator`. This is a finite interval-ratio model result, not exact
symbolic arithmetic with an irrational exponent.

## 5. Compensator and confluence

For interval exponents the verified primitive is
\[
C(a,b;x)=\frac{x^a-x^b}{a-b},\qquad
C(a,a;x)=x^a\log x.
\]
`power_compensator` evaluates the entire beta-moment representation and never
divides by \(a-b\). Its derivatives are available through total order sixteen.
The output is an outward-rounded interval, not a symbolic transcendental
expression.

## 6. The open saddle-node-at-infinity target

The named target remains `BLOCKED` for the same reasons as
[HILBERT16-PROGRAM.md](HILBERT16-PROGRAM.md):

1. the proposed coalescing-root chart loses a uniform scale on
   `sep = exp(-1/epsilon^2)`;
2. on `L = 1/n`, the proposed shrinking-root radius becomes nonpositive and
   physical transversality remains unproved;
3. complete first-hit and itinerary capture (G4) has not been opened.

Exact compactification does not resolve these analytic failures. A declared
three-term series is not substituted for the unknown physical passage.

## 7. Formal and honesty boundary

The benchmark gates are:

- GD1: exact Poincare charts and invariant equator;
- GD2: rational-ratio finite-model bound;
- GD3: resonant logarithmic finite-model bound;
- GD4: irrational-ratio parameter-box bound;
- GD5: finite Lean replay;
- GD6: open-case refusal.

Every successful model certificate records

```text
dulac_truncated_model_only = true
physical_return_membership_proved = false
uniform_remainder_proved = false
graphic_finite_cyclicity_proved = false
drr_case_closed = false
full_hilbert16_solved = false
```

The missing theorem is a uniform derivation of the actual return displacement
and remainder on a complete physical chart atlas. Only after that theorem could
the finite nonoscillation certificates be consumed as graphic cyclicity.

## 8. Reproduction

Run the focused tests and benchmark with the prepared environment:

```bash
python -m pytest packages/omnibias-dynamics/tests/test_compactify.py \
  packages/omnibias-dynamics/tests/test_dulac.py \
  packages/omnibias-dynamics/tests/test_graphic.py -q
python -m benchmarks.dulac_cyclicity --lean
```

The smoke artifact is
`docs/benchmarks/dulac_cyclicity_smoke.json`.
