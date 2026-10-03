# Certified Abelian-integral zero counts: genuine and finite-uniform tracks

This note specifies an executable, instance-level result in the
**infinitesimal** Hilbert-sixteenth register. It does not prove that the
Hilbert number \(H(n)\) is finite, does not prove \(H(2)<\infty\), and does
not close the singular-passage or graphic-capture gates in
[HILBERT16-PROGRAM.md](HILBERT16-PROGRAM.md).

The target is a certified count of the zeros of one declared Abelian
integral

\[
 I(h)=\oint_{\Gamma(h)}\omega
\]

on one declared energy domain. The implementation combines an exact
Picard--Fuchs relation, validated complex continuation, the argument
principle, and independently certified simple real zeros. When the upper and
lower counts agree, the surviving object is an integer. This is winding
collapse, not founding bias collapse.

## 1. Supported cubic family

The first implementation uses the elliptic Hamiltonian family

\[
 H(x,y)=y^2+x^3+p x+q,\qquad p,q\in\mathbb Q,\quad p<0,
\]

and one-forms

\[
 \omega=\alpha(H)y\,dx+\beta(H)x y\,dx,\qquad
 \alpha,\beta\in\mathbb Q[h].
\]

Write \(e=h-q\), \(F(x,h)=e-x^3-px\), and let \(\Gamma(h)\) be the real oval
surrounding the local minimum while \(h\) lies strictly between the two
critical values. With the Hamiltonian orientation,

\[
 A(h)=\oint_{\Gamma(h)}y\,dx
     =2\int_{x_1(h)}^{x_2(h)}\sqrt{F(x,h)}\,dx>0,
 \qquad B(h)=\oint_{\Gamma(h)}x y\,dx,
 \qquad I(h)=\alpha(h)A(h)+\beta(h)B(h).
\]

Here \(x_0(h)<x_1(h)<x_2(h)\) are the three real roots of
\(x^3+px+q-h\). The strict positivity of \(A\) on the open period annulus is
an ordinary real-integral fact. It is not inferred from a numerical sample.

The exact holonomic layer also implements the general hyperelliptic
de Rham reduction for \(y^2=h-V(x)\), \(V\in\mathbb Q[x]\), and blocks if
the exact moment reduction is singular over \(\mathbb Q(h)\). The phase-one
zero-count consumer remains cubic because validated turning-point data and
contour selection for arbitrary \(V\), as well as general polynomial
one-forms and general oval parametrizations, require additional geometry.

## 2. Exact Picard--Fuchs derivation

Define the two periods

\[
 J_0(h)=\oint_{\Gamma(h)}\frac{dx}{y},\qquad
 J_1(h)=\oint_{\Gamma(h)}\frac{x\,dx}{y}.
\]

The exact forms \(d(x^k y)\) and \(d(x^k/y)\) have zero integral on the
closed cycle. Reducing those identities gives, with

\[
 \Delta(h)=27e^2+4p^3,
\]

the Gauss--Manin system

\[
 \begin{pmatrix}J_0\\J_1\end{pmatrix}'
 =
 \frac{1}{2\Delta}
 \begin{pmatrix}
 -9e & 6p\\
 2p^2 & 9e
 \end{pmatrix}
 \begin{pmatrix}J_0\\J_1\end{pmatrix}.
\]

Appending \(A'=J_0/2\) and \(B'=J_1/2\) gives the four-component
first-order representation

\[
\frac{d}{dh}
\begin{pmatrix}A\\B\\J_0\\J_1\end{pmatrix}
=
\begin{pmatrix}
0&0&1/2&0\\
0&0&0&1/2\\
0&0&-9e/(2\Delta)&6p/(2\Delta)\\
0&0&2p^2/(2\Delta)&9e/(2\Delta)
\end{pmatrix}
\begin{pmatrix}A\\B\\J_0\\J_1\end{pmatrix}.
\]

It is not a rank-four local system. Exact reduction also gives

\[
A=\frac{3e}{5}J_0-\frac{2p}{5}J_1,\qquad
B=\frac{2p^2}{21}J_0+\frac{3e}{7}J_1.
\]

Thus the cubic Gauss--Manin differential rank is two; the implementation
propagates the overcomplete four-component representation for tighter interval
enclosures and checks it against these rank-two identities. “Rank four” would
require a genuinely larger period module, such as the relevant genus-two
setting. The corresponding minimal scalar equations for the two area moments are

\[
4\Delta A''+15A=0,\qquad 4\Delta B''-21B=0.
\]

Eliminating \(J_1\) yields

\[
 4\Delta J_0''+216eJ_0'+15J_0=0.
\]

Since \(A'=J_0/2\), the period \(A\) is annihilated by

\[
 L_A=
 \left(15+216eD+4\Delta D^2\right)D,\qquad D=\frac{d}{dh}.
\]

This shipped third-order annihilator is valid but nonminimal: it is the
derivative of \(4\Delta A''+15A=0\).

For \(I=rA\), the product annihilator is obtained in the differential Ore
algebra from \(L_A\) and the first-order polynomial annihilator
\(rD-r'\). Operator discovery is only a proposer. Acceptance re-derives the
Gauss--Manin identities over \(\mathbb Q[h]\), clears denominators, and
requires an exact rational syzygy. A perturbed coefficient is rejected.

The cleared determining matrix and integer kernel vector are sealed as an
`integer_matrix_syzygy` obligation. A successful Lean build checks that
finite identity. It does not formalize the analytic continuation or the
Poincare--Pontryagin theorem.

## 3. Base data without endpoint singularities

At a regular base energy whose three turning points are certified, set

\[
 x(\theta)=m+\rho\cos\theta,\quad
 m=(x_1+x_2)/2,\quad \rho=(x_2-x_1)/2.
\]

The cubic factorization gives

\[
 F(x(\theta),h)
 =\rho^2\sin^2\theta\,(x(\theta)-x_0).
\]

Consequently the initial periods have nonsingular integrands:

\[
\begin{aligned}
 A(h)&=2\rho^2\int_0^\pi
       \sin^2\theta\sqrt{x(\theta)-x_0}\,d\theta,\\
 B(h)&=2\rho^2\int_0^\pi
       x(\theta)\sin^2\theta\sqrt{x(\theta)-x_0}\,d\theta,\\
 J_0(h)&=2\int_0^\pi
       \frac{d\theta}{\sqrt{x(\theta)-x_0}},\\
 J_1(h)&=2\int_0^\pi
       \frac{x(\theta)\,d\theta}{\sqrt{x(\theta)-x_0}}.
\end{aligned}
\]

Outward interval quadrature independently encloses these four quantities and
is checked against the two exact reduction identities. Continuation propagates
the overcomplete \((A,B,J_0,J_1)\) representation and rechecks its overlap with
the rank-two reconstruction, without integrating the endpoint-singular
expression \(F^{-3/2}\).

## 4. Branch and contour contract

The critical values are the zeros of \(\Delta(h)\). A continuation path is
admissible only when every path box proves \(0\notin\Delta(h)\). On such a
simply connected cut domain, the chosen initial cycle and its period admit a
single-valued analytic continuation. The companion system of \(L_A\) is
realified and propagated with outward interval matrix exponentials. A path
box meeting a singular value is `BLOCKED`.

Regularity alone does not prove that the distinguished action period is
zero-free. For this depressed cubic, set

\[
 s=\sqrt{-p/3},\qquad h_c=q-2s^3,\qquad h_s=q+2s^3,\qquad
 z=\frac{h-h_c}{h_s-h_c}.
\]

With Hamiltonian orientation, the vanishing-cycle branch has the form

\[
 A(h)=\frac{4\pi s^{5/2}}{\sqrt3}\,
 z\,{}_2F_1\!\left(\frac16,\frac56;2;z\right).
\]

Euler's integral representation proves that the hypergeometric factor has
no zero on \(\mathbb C\setminus[1,\infty)\). Thus \(A\) has only its center
zero there. The implementation accepts a counting rectangle only after exact
rational inequalities prove

\[
 h_c<\Re h<h_s
\]

throughout its closure. This both excludes the center zero and stays to the
left of the saddle cut. Since the real base energy and every target box lie
in this convex strip, their straight continuation paths stay on the same
distinguished branch. This analytic zero-free theorem is declared input; it
is not among the finite Lean obligations.

For a closed contour \(C\) in that regular domain, the interval evaluator
feeds `winding_enclosure_function`. If every image segment excludes zero and
the enclosure of

\[
 \frac{\Delta\arg I}{2\pi}
\]

contains exactly one integer \(N\), the argument principle proves that
\(I\) has \(N\) complex zeros inside \(C\), counted with multiplicity. A
contour that may pass through a zero is `BLOCKED`, never assigned a count.

The smoke instance uses

\[
 p=-1,\quad q=0,\quad
 r(h)=(h+\tfrac18)(h-\tfrac18).
\]

Both rational roots lie strictly inside the real period annulus
\((-2/(3\sqrt3),\,2/(3\sqrt3))\). Krawczyk boxes certify the two simple real
zeros. The smoke rectangle
\[
 |\Re h|\leq\tfrac15,\qquad |\Im h|\leq\tfrac1{25}
\]
lies strictly between the critical energies; its domain check records the
zero-free action theorem, while validated winding directly supplies the
argument-principle upper count. Equality of the lower and upper counts is the
exact-count collapse.

This is a certified **forced-factor** two-zero instance in Petrov's classical
cubic elliptic setting. The roots were inserted through \(r(H)\), and
\(H^2y\,dx\) is not a quadratic one-form. Therefore this is not a machine
reproof or reproduction of Petrov's parameter-uniform sharp Chebyshev
theorem.

## 5. Genuine mixed instance and rational coefficient cover

The GA7 instance retains
\(\alpha(h)=h^2-1/64\) but sets \(\beta(h)=1/1000\). The mixed-period evaluator
certifies exactly two zeros. Their Krawczyk boxes exclude both old declared
factor roots \(h=\pm1/8\), so `forced_factor_instance=False` is earned rather
than asserted.

This exclusion is not an irrationality proof. Every nondegenerate real
interval contains rational numbers. The certificate therefore records
`irrational_zeros_verified=False`; it proves that the zeros are not the two
declared planted factor roots.

GA8 replaces the point coefficient by

\[
  \frac9{10000}\leq\beta_0\leq\frac{11}{10000}.
\]

Interval-coefficient winding encloses every member simultaneously. A blocked
leaf is bisected along its widest rational coefficient interval. The shipped
cover has uniform bound \(N=2\). Its `box_cover_tiling` Lean obligation
reconstructs every axis split over \(\mathbb Q\) and checks every terminal
integer is at most \(N\); a linked `winding_integer_isolation` obligation checks
the integer on every leaf. The analytic interval winding enclosures remain
trusted certificate inputs.

## 6. Endpoint residual

A finite regular contour does not cover the critical values. In particular,
the neighborhoods of the center value \(h_c\) and the homoclinic value
\(h_s\) are separate analytic obligations. Their expected local forms are,
schematically,

\[
 A(h)=c(h-h_c)+O((h-h_c)^2)
\]

at a nondegenerate center, and a logarithm-corrected expansion at the
separatrix. The implementation does not infer those expansions from a
truncated jet and does not claim a zero count on a punctured endpoint
neighborhood unless a separate enclosure is supplied.

## 7. Relation to perturbed limit cycles

For

\[
 dH+\varepsilon\omega=0,
\]

the Poincare--Pontryagin theorem identifies \(I\) as the first displacement
coefficient. A simple zero of \(I\) generates a hyperbolic limit cycle for
all sufficiently small nonzero \(\varepsilon\), under the theorem's stated
hypotheses. The present certificate does not compute a uniform threshold
\(\varepsilon_0\). Its honesty payload therefore records

```text
infinitesimal_hilbert16_instance = true
epsilon_effective = false
uniform_degree_bound_claim = false
h_n_finiteness_claim = false
full_hilbert16_solved = false
```

The constructive degree-uniform bound for the infinitesimal problem is the
published Binyamini--Novikov--Yakovenko theorem. The present track has a
different purpose: a replayable, two-sided exact count for named instances.

## 8. Acceptance gates

- **GA1 -- exact operator:** the Picard--Fuchs syzygy replays over
  \(\mathbb Q\); a perturbed operator fails.
- **GA2 -- sound continuation:** a regular loop returns an enclosure
  containing its initial value, and independent real quadrature agrees at
  checked energies.
- **GA3 -- upper count:** winding collapse isolates one integer; a contour
  through a zero is `BLOCKED`; the rectangle first passes the exact
  zero-free-action strip check.
- **GA4 -- lower count:** disjoint Krawczyk boxes prove simple real zeros.
- **GA5 -- sandwich:** the real lower count equals the complex upper count
  on the named cubic instance.
- **GA6 -- seal:** certificate digests replay, and
  `theorem_prover_verified` is earned only by a genuine Lean build of the
  finite rational obligations.
- **GA7 -- genuine mixed form:** \(\beta\ne0\), the upper/lower count is two,
  and both declared factor roots are excluded from the unique-zero boxes.
  This does not claim the roots are irrational.
- **GA8 -- finite uniform cover:** interval-coefficient winding proves
  \(N=2\) on every rational cover leaf; blocked leaves bisect, and Lean checks
  the exact recursive box tiling, leaf bounds, and leaf winding isolations.

The full Hilbert-sixteenth program remains governed by G1--G6 in
[HILBERT16-PROGRAM.md](HILBERT16-PROGRAM.md). No GA gate changes those
statuses.

The follow-up
[physical-return transfer audit](HILBERT16-ABELIAN-DRR-TRANSFER.md) gives an
exact conditional epsilon threshold once value, derivative, and remainder
margins are supplied.  The open DRR graphics currently provide neither the
Hamiltonian reduction nor those physical remainder and endpoint certificates,
so no DRR ledger status changes.
