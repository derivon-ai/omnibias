# SU(2) vacuum Fourier certificates

These APIs check an actual-Hamiltonian strong-coupling theorem. A passing
family certificate gives a neutral spectral gap uniform over **all finite
graphs in its stated structural class**, at fixed coupling and unit electric
weights. It also bounds the Hessian of the actual logarithmic vacuum and,
when a separate influence gate passes, actual variance factorization.
It does not construct an infinite-volume or continuum theory and does not
certify a static-source string tension. The full analytic proof follows the
API examples below; the finite arithmetic and the written proof have distinct
verification scopes.

## Structural family gate

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.vacuum_fourier import (
    su2_vacuum_fourier_family,
    su2_vacuum_fourier_bounds,
    replay_su2_vacuum_fourier_certificate,
)

family = su2_vacuum_fourier_family(64, 4)
assert family['volume_uniform_finite_graph_family_verified']
assert family['volume_uniform_factorization_bound_verified']
assert Fraction(family['witness']['arithmetic']['neutral_gap_lower']) == 14
assert replay_su2_vacuum_fourier_certificate(family['certificate'])
```

The structural class has weighted plaquette incidence at most four per
edge, cycle length at most four, and cycle diameter at most two in the
ambient ordinary line graph. These are hypotheses defining the quantified
class. The family call does not claim to have inspected a particular graph.
Changing the number of vertices or edges does not change its gap constant.
No graph embedding or spatial dimension is inferred.

For exponential spatial tails, choose a rational locality weight greater
than one. The stronger forcing budget below passes at coupling 128:

```python
weighted = su2_vacuum_fourier_family(
    128, 4, decay_base=2, tail_radius=3,
)
bounds = weighted['witness']['arithmetic']
assert Fraction(bounds['neutral_gap_lower']) == 28
assert Fraction(bounds['actual_log_vacuum_hessian_tail_row_upper']) == Fraction(1, 256)
assert Fraction(bounds['actual_tv_influence_tail_upper']) == Fraction(121, 1568)
assert Fraction(bounds['actual_factorization_constant_upper']) == Fraction(196, 75)
```

The reported tail bounds the sum of all blocks whose ambient line-graph
distance is strictly greater than the nonnegative integer tail_radius.
The proof also permits the larger set at distance at least that radius, so
the API convention is conservative. With decay_base equal to one the same
expression is only a total row bound; the exponential-decay flag stays false.

## Checking a concrete graph

```python
square = su2_vacuum_fourier_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)],
    plaquettes=[[1, 2, 3, 4]], kappa=64,
    weighted_incidence_cap=4, max_cycle_length=4, max_cycle_diameter=2,
)
assert square['explicit_graph_neutral_gap_verified']
assert square['witness']['structural_caps_verified']
assert square['witness']['cycle_ambient_line_graph_diameters'] == [2]
assert replay_su2_vacuum_fourier_certificate(square['certificate'])
```

Cycles use signed one-based edge indices, through the shared graph validator.
Weights must be exact nonnegative integers or fractions, one per cycle.
Electric weights are fixed at one. Distances are computed in the **ambient**
line graph, including shortcuts outside a cycle; the implementation does not
substitute half the cycle length when a shorter route exists. Disconnected
components factor and are handled componentwise.

Omitted structural caps default to the exact values on the supplied graph.
Explicit caps are checked against every cycle and edge. A direct graph
forcing gate can pass while a deliberately coarse structural-family gate
fails; the two verdicts remain distinct. If supplied caps are violated,
status is INCONCLUSIVE and the family claim remains false, even if the
direct graph calculation separately proves a gap.

The factorization gate is also separate: Banach contraction permits radii
less than 3/32, while the displayed rational influence bound additionally
needs (1936/147) radius less than one. A larger permitted radius may prove
the curvature gap while leaving the factorization flag false.

All proof inputs are exact integers or fractions; floats and Boolean integer
substitutes are refused. A failed arithmetic inequality returns INCONCLUSIVE,
not a claim that the Hamiltonian is gapless. The replay function returns true
only for a canonical **passing** certificate. Faithfully stored failed
reports and altered certificates both return false; the failed records remain
useful diagnostics. Replay regenerates the entire graph metric, forcing,
structural hypotheses, exact inequalities, scope and honesty envelope.

No Lean or Mathlib verification is implied by these finite certificates.

## Full analytic implication

11 September 2026. Written proof independently reviewed during this campaign. This
argument supplies an explicit strong-coupling estimate, uniform over finite
graphs with a bounded local interaction budget. It controls the actual
logarithmic groundstate, its full Hessian, conditional influence, and a
weighted interaction tail. No claim of novelty, infinite-volume construction,
continuum Yang--Mills or formal verification is made.

The independent review checked Fourier normalization and pinching, the
volume-independent bilinear constant, finite-volume reconstruction,
orientation-independent Wilson coefficients, the covariant Hessian estimate,
and the heatbath variance-factorization argument. Executable arithmetic
checks remain separate from this analytic review.

## Statement and model

Use a finite connected graph with nonempty edge set \(\mathcal E\),
one independent SU(2) variable per edge, and normalized product Haar measure.
The electric metric has Casimir \(j(j+1)\), so each group factor is the
round three-sphere of radius two. All electric weights are one. Let

\[
A_\kappa=\frac\kappa2 C+\frac2\kappa
\sum_p v_p(2-\chi_{1/2}(U_p)),\qquad
C=-\sum_e\Delta_e,\quad v_p\ge0,\quad \kappa>0.
\]

Every potential cycle traverses each of its edges once. Let its length be
\(\ell_p\), and give the ordinary graph line graph its shortest-path metric.
Thus two distinct edges sharing an endpoint are adjacent. Let \(D_p\) be
the diameter of the cycle's edge support in this ambient metric; shortcuts
elsewhere in the graph may reduce this diameter.

Put \(g=4/\kappa^2\) and choose a locality weight \(b\ge1\). Define

\[
\mathcal A=g\sup_{e\in\mathcal E}
\sum_{p:\,e\in p}2^{\ell_p-1}v_p b^{D_p},\qquad B=\frac{16}{3}.
\]

Suppose a radius \(r>0\) satisfies

\[
\boxed{\qquad \mathcal A+Br^2\le r,\qquad 2Br<1.\qquad}
\tag{1}
\]

Then the true positive groundstate \(\psi_0\) obeys, with
\(S=\log\psi_0\) up to an additive normalization,

\[
\|\operatorname{Hess}S\|_{\rm op,\infty}\le\frac23r.
\tag{2}
\]

There are nonnegative symmetric bounds \(M_{ef}\) on its Hessian blocks with

\[
\sup_U\|\operatorname{Hess}_{ef}S(U)\|_{\rm op}\le M_{ef},
\qquad
\sup_e\sum_f b^{\operatorname{dist}(e,f)}M_{ef}\le\frac23r.
\tag{3}
\]

In particular the entire compact-graph spectrum above the unique vacuum,
and hence the gauge-invariant neutral spectrum, has gap at least

\[
\boxed{\quad
\Delta_{A_\kappa}\ge\frac\kappa2\left(\frac12-\frac43r\right)>0.
\quad}
\tag{4}
\]

Condition (1) gives \(r<3/32\), so the parenthesis is strictly positive.
Every constant here is independent of the number of vertices and edges.
The proof constructs a separate groundstate in each finite volume; it does
not assume or construct an infinite product wavefunction in Haar \(L^2\).

For four-edge plaquettes, \(D_p\le2\). If
\(d_*=\sup_e\sum_{p:e\in p}v_p\), the simple sufficient choice is

\[
r=\frac3{64},\qquad
g d_* b^2\le\frac9{2048}.
\tag{5}
\]

It gives contraction factor \(1/2\), Hessian bound \(1/32\), and
\(\Delta_{A_\kappa}\ge7\kappa/32\).
For ordinary unweighted three-dimensional cubic volumes \(d_*\le4\),
\(\kappa\ge64\) suffices with \(b=1\); \(\kappa\ge128\) suffices with \(b=2\).
These are deliberately conservative strong-coupling examples.

## Fourier conventions and the complete Banach space

Index irreducible product representations by
\(\mathbf j=(j_e)_{e\in\mathcal E}\), with
\(j_e\in\{0,\tfrac12,1,\ldots\}\). Write

\[
\rho_{\mathbf j}(U)=\bigotimes_e D^{j_e}(U_e),\qquad
E_{\mathbf j}=\sum_e j_e(j_e+1),\qquad
d_{\mathbf j}=\prod_e(2j_e+1).
\]

Use coefficients normalized by the expansion

\[
f(U)=\sum_{\mathbf j}\operatorname{Tr}
[F_{\mathbf j}\rho_{\mathbf j}(U)].
\tag{6}
\]

In this convention Parseval's identity is
\(\|f\|_2^2=\sum_{\mathbf j}
\|F_{\mathbf j}\|_{\rm HS}^2/d_{\mathbf j}\).
There is no omitted representation dimension multiplying the trace norm
in what follows. The coefficient \(F_{\mathbf0}\) is the Haar mean.

For nonzero \(\mathbf j\), put
\(X_{\mathbf j}=\{e:j_e>0\}\) and
\(w_{\mathbf j}=b^{\operatorname{diam}(X_{\mathbf j})}\), with singleton
diameter zero. Constant modes are removed before any weighted diameter
is used. Define

\[
\mathcal N_b(f)=\max_i
\sum_{\mathbf j\ne0}
j_i E_{\mathbf j}w_{\mathbf j}\|F_{\mathbf j}\|_1,
\tag{7}
\]

where \(\|\cdot\|_1\) denotes matrix trace norm, not an entrywise norm.
The space consists of real, Haar-mean-zero coefficient families with finite
\(\mathcal N_b\).

For each finite edge set this is complete. Indeed, the sum over \(i\) of
the quantities inside the maximum is a weighted \(\ell^1\) direct sum of
finite-dimensional matrix trace-norm spaces. Its norm lies between
\(\mathcal N_b\) and \(|\mathcal E|\mathcal N_b\). Real-valuedness and the
zero constant coefficient are closed conditions. Thus the maximum norm
is also a Banach norm.

This space reconstructs actual \(C^2\) functions. Since every nonzero
representation has some \(j_i\ge1/2\),

\[
\sum_{\mathbf j\ne0}E_{\mathbf j}\|F_{\mathbf j}\|_1
\le2|\mathcal E|\,\mathcal N_b(f).
\tag{8}
\]

Also \(E_{\mathbf j}\ge3/4\). A unit Lie-algebra generator in spin \(j_i\)
has operator norm \(j_i\). First and second derivatives of each term in
(6) are therefore bounded by its trace norm times \(j_i\) or \(j_i j_k\),
both controlled by a constant times \(E_{\mathbf j}\).
Equation (8) gives uniform absolute convergence of the series and these
derivative series on each finite graph. In particular \(Cf\) is the uniformly
convergent series with coefficients \(E_{\mathbf j}F_{\mathbf j}\).

The factor \(|\mathcal E|\) in this regularity check is harmless: it is used
only to reconstruct a function at a fixed finite volume. The contraction,
Hessian, influence and spectral constants below use \(\mathcal N_b\)
directly and contain no such factor.

## Product coefficients: why no representation dimension is lost

For two individual coefficient functions,

\[
\operatorname{Tr}(F\rho_{\mathbf j})
\operatorname{Tr}(G\rho_{\mathbf k})
=\operatorname{Tr}[(F\otimes G)(\rho_{\mathbf j}\otimes\rho_{\mathbf k})].
\]

The SU(2) Clebsch--Gordan decomposition is unitary and multiplicity free
at each edge, hence also for this product of edgewise tensor products.
After that unitary change of basis, the product representation is block
diagonal in labels \(\boldsymbol\ell\). Only the matching diagonal blocks
of the coefficient matrix contribute to its trace. Calling them
\(K_{\boldsymbol\ell}\), trace-norm pinching gives

\[
\sum_{\boldsymbol\ell}\|K_{\boldsymbol\ell}\|_1
\le\|F\otimes G\|_1=\|F\|_1\|G\|_1.
\tag{9}
\]

Pinching is an average of unitary conjugations, so the inequality follows
from convexity and unitary invariance of trace norm. The allowed output
spins satisfy
\(\ell_i\le j_i+k_i\), and their support is contained in
\(X_{\mathbf j}\cup X_{\mathbf k}\).

For terms coupled by a common differentiated edge, the two input supports
intersect. In any graph metric,

\[
\operatorname{diam}(X\cup Y)\le
\operatorname{diam}(X)+\operatorname{diam}(Y)
\quad\text{when }X\cap Y\ne\varnothing.
\]

Internal connectedness of the sets is unnecessary. Consequently
\(w_{\boldsymbol\ell}\le w_{\mathbf j}w_{\mathbf k}\) for every output
appearing in the gradient product below. The inverse of \(C\) only rescales
Fourier coefficients; it never enlarges a support.

## The exact quadratic estimate

Let

\[
\Gamma(f,h)=\sum_{e,a=1}^3(X_{e,a}f)(X_{e,a}h),\qquad
T(f,h)=C_0^{-1}\Pi_0\Gamma(f,h),
\]

where \(\Pi_0\) removes the Haar mean and \(C_0^{-1}\) multiplies each
remaining coefficient by \(1/E_{\mathbf j}\).
For a fixed pair of input representations, each derivative coefficient
has trace norm at most \(j_e\|F_{\mathbf j}\|_1\) or
\(k_e\|G_{\mathbf k}\|_1\). Summing the three generator directions and
using (9) bounds the fused coefficient norm by

\[
3\sum_e j_e k_e\|F_{\mathbf j}\|_1\|G_{\mathbf k}\|_1.
\]

In the norm of \(T\), the output energy multiplier cancels \(C_0^{-1}\)
exactly. Constant outputs have already been removed. Using
\(\ell_i\le j_i+k_i\) and the support weight inequality gives

\[
\mathcal N_{b,i}(T(f,h))
\le3\sum_{\mathbf j,\mathbf k}
(j_i+k_i)\sum_e j_e k_e\,
w_{\mathbf j}\|F_{\mathbf j}\|_1
w_{\mathbf k}\|G_{\mathbf k}\|_1.
\tag{10}
\]

Two elementary SU(2) inequalities close this sum:

\[
\sum_e j_e\le\frac23E_{\mathbf j},\qquad
\sum_{\mathbf k} k_e w_{\mathbf k}\|G_{\mathbf k}\|_1
\le\frac43\mathcal N_b(h).
\tag{11}
\]

The first uses \(j/(j(j+1))\le2/3\) for nonzero spin; the second uses
\(E_{\mathbf k}\ge3/4\).
The part of (10) with \(j_i\) is therefore at most
\((8/3)\mathcal N_b(f)\mathcal N_b(h)\).
The \(k_i\) part gives the same bound with the inputs exchanged. Thus

\[
\boxed{\quad
\mathcal N_b(T(f,h))
\le\frac{16}{3}\mathcal N_b(f)\mathcal N_b(h).
\quad}
\tag{12}
\]

All sums are absolutely controlled by this estimate, so the finite-polynomial
calculation extends by completion to the Banach space. The bilinear map
preserves real-valuedness.

## The Wilson forcing constant

A cycle character on \(\ell_p\) distinct edges lies entirely in the product
representation with spin \(1/2\) on those edges and zero elsewhere.
Inverse matrix entries are conjugate fundamental entries; SU(2)
pseudoreality keeps them in the same irreducible representation class.
Its representation dimension is \(2^{\ell_p}\).

The holonomy is Haar distributed when any one free cycle link is integrated,
so \(\|\chi_{1/2}(U_p)\|_2^2=1\). Parseval in convention (6) gives
\(\|F_p\|_{\rm HS}=2^{\ell_p/2}\). The trace-norm estimate
\(\|F_p\|_1\le\sqrt{2^{\ell_p}}\|F_p\|_{\rm HS}\) then yields
\(\|F_p\|_1\le2^{\ell_p}\).
This reasoning is orientation independent and does not assume that a
partially transposed permutation matrix remains unitary.

In \(C_0^{-1}gV\), the energy in the norm again cancels the inverse energy.
Each edge in the cycle has spin \(1/2\). The triangle inequality therefore
gives exactly

\[
\mathcal N_b(C_0^{-1}gV)
\le g\max_i\sum_{p:i\in p}2^{\ell_p-1}v_p b^{D_p}
=\mathcal A.
\tag{13}
\]

## Fixed point, regularity and identification of the true vacuum

On the closed radius-\(r\) ball define

\[
\Phi(S)=C_0^{-1}gV+T(S,S).
\]

Equations (12)--(13) and (1) make this a self-map. Moreover,

\[
\mathcal N_b(\Phi(S)-\Phi(R))
\le B(\mathcal N_b(S)+\mathcal N_b(R))\mathcal N_b(S-R)
\le2Br\,\mathcal N_b(S-R).
\]

Banach's theorem supplies a unique fixed point in this ball. It is a real
\(C^2\) function by (8), and satisfies

\[
CS=gV+\Gamma(S,S)-\langle\Gamma(S,S)\rangle_H.
\tag{14}
\]

Consequently, for \(\psi=e^S>0\),

\[
(C-gV)\psi
=-\langle\Gamma(S,S)\rangle_H\,\psi.
\]

Elliptic regularity upgrades this positive \(C^2\) eigenfunction to a smooth
one, since \(V\) is smooth. It is the actual groundstate: integration by
parts gives, first for smooth \(f\) and then on the full form domain,

\[
\langle\psi f,(C-gV-E')\psi f\rangle
=\int\psi^2|\nabla f|^2\,dU\ge0,\qquad
E'=-\langle\Gamma(S,S)\rangle_H.
\tag{15}
\]

Multiplication by \(\psi\) preserves the fixed-graph form domain because it
and its inverse are smooth and bounded there. Equality in (15) requires
\(f\) constant on the connected product manifold. Thus the vacuum is unique.
Normalizing \(\psi\) only adds a constant to \(S\), affecting none of the norms
or derivatives in the proof. Gauge invariance follows from uniqueness and
the gauge-invariant operator.

## Hessian rows, curvature and the actual spectral gap

Set

\[
M_{ef}=\sum_{\mathbf j\ne0}j_ej_f\|S_{\mathbf j}\|_1.
\]

For a product tangent vector \(v=(v_e)\), use the product geodesic with
initial velocity \(v\). A spin-\(j_e\) directional generator has norm
\(j_e|v_e|\), so differentiating the coefficient series twice along this
geodesic gives

\[
|\operatorname{Hess}S(v,v)|
\le\sum_{\mathbf j}\|S_{\mathbf j}\|_1
\left(\sum_e j_e|v_e|\right)^2
=\sum_{e,f}M_{ef}|v_e||v_f|.
\tag{16}
\]

This computes the covariant Hessian, including the connection: geodesic
second derivatives are its quadratic form. For distinct edge blocks the
two derivatives commute; for a diagonal block the Hessian is the symmetrized
pair of generators. The same generator norm estimate gives
\(\|\operatorname{Hess}_{ef}S\|_{\rm op}\le M_{ef}\).

Whenever a coefficient contributes to \(M_{ef}\), its support contains
both edges, so \(b^{\operatorname{dist}(e,f)}\le w_{\mathbf j}\). Using (11),

\[
\sum_f b^{\operatorname{dist}(e,f)}M_{ef}
\le\sum_{\mathbf j}j_e w_{\mathbf j}\|S_{\mathbf j}\|_1
\sum_f j_f
\le\frac23\mathcal N_{b,e}(S)\le\frac23r.
\tag{17}
\]

The symmetric nonnegative matrix \(M\) has operator norm bounded by its
maximum row sum. Equations (16)--(17) prove (2)--(3). In particular the omitted
row beyond distance \(R\ge0\), with distance at least \(R\), is at most
\((2r/3)b^{-R}\). This is an analytic tail bound, not a fitted decay curve.

For \(d\mu=\psi_0^2\,dU\), the groundstate diffusion is
\(L=\Delta+2\nabla S\cdot\nabla\). Its Bochner tensor is
\(\operatorname{Ric}-2\operatorname{Hess}S\), hence at least
\(\rho I\), where \(\rho=1/2-4r/3>0\).
The integrated Bochner identity on a positive-eigenvalue eigenfunction of
\(-L\) gives \(\lambda^2\|f\|_2^2\ge\rho\lambda\|f\|_2^2\);
therefore \(\lambda\ge\rho\).
Compact ellipticity supplies a complete spectrum. The unitary groundstate
transform (15) and the scaling by \(\kappa/2\) prove (4).
Restricting to the invariant physical subspace cannot introduce lower
excited spectrum. A graph with no cycles may have only the vacuum in that
subspace; this does not constitute a nontriviality witness.

## The same estimate supplies actual variance factorization

For \(e\ne f\), let \(c_{ef}\) be the supremum total-variation distance between
the two conditional laws of \(U_e\) when exterior configurations differ only
at \(U_f\). Two coordinate geodesics, each of length at most \(2\pi\), give
the four-point mixed oscillation of \(2S\) at most \(8\pi^2M_{ef}\).
Conditional normalizing constants cancel from this oscillation.

If the log-likelihood ratio of two probability densities has oscillation
at most \(\omega\), their total variation is at most
\(\tanh(\omega/4)\le\omega/4\). To see the first bound, place their density
ratio in \([m,M]\) with mean one and \(M/m\le e^\omega\). Convexity bounds
the positive-part expectation by
\((M-1)(1-m)/(M-m)\); maximizing under the ratio constraint gives
\(\tanh(\omega/4)\). Consequently

\[
c_{ef}\le2\pi^2M_{ef},\qquad
\sup_e\sum_{f\ne e}c_{ef}\le\eta:=\frac{4\pi^2}{3}r.
\tag{18}
\]

When \(\eta<1\), this proves the factorization inequality

\[
\boxed{\quad
\operatorname{Var}_\mu(f)
\le\frac1{1-\eta}\sum_e
\mathbb E_\mu\operatorname{Var}_{\mu(\cdot\,|\,U_{e^c})}(f).
\quad}
\tag{19}
\]

Here is a finite-volume proof of that implication. Let \(P_e\) be conditional
expectation with the outside of edge \(e\) held fixed. For a bounded function,
write \(\delta_i(f)\) for its oscillation when only coordinate \(i\) changes.
Then \(\delta_e(P_e f)=0\), and, for \(i\ne e\),

\[
\delta_i(P_e f)\le\delta_i(f)+c_{ei}\delta_e(f).
\]

Use the Markov Euler step
\(I+h\sum_e(P_e-I)\), with \(0\le h\le|\mathcal E|^{-1}\).
Summing its coordinate-oscillation bounds gives contraction by at most
\(1-h(1-\eta)\). Iterating and taking the bounded-generator limit proves

\[
\sum_i\delta_i(T_t f)
\le e^{-(1-\eta)t}\sum_i\delta_i(f),\qquad
T_t=\exp\!\left[t\sum_e(P_e-I)\right].
\]

The full oscillation is at most the sum of coordinate oscillations, and
\(\mu\) is invariant. Thus the centered \(L^2(\mu)\) norm decays at this
rate for every bounded function, with finite prefactor
\(\sum_i\delta_i(f)\). The generator is self-adjoint: each \(P_e\) is an
orthogonal projection in \(L^2(\mu)\). Spectral positivity and the decay
estimate exclude spectrum below \(1-\eta\) on this dense class of centered
functions. Its Dirichlet form is exactly
\(\sum_e\mathbb E_\mu\operatorname{Var}(f\,|\,U_{e^c})\), proving (19).
The prefactor may depend on volume; the spectral lower bound does not.

The extra condition \(\eta<1\) is not automatic for every radius allowed
by (1). At the explicit radius \(r=3/64\), however,
\(\eta=\pi^2/16<1\). Using \(\pi<22/7\) gives the rational bounds
\(\eta\le121/196\) and \(C\le196/75\) in (19).
For example \(\pi<22/7\) follows from the positive integral
\(\int_0^1 x^4(1-x)^4/(1+x^2)\,dx=22/7-\pi\).

The weighted version of (18) gives an influence tail at distance at least
\(R\) bounded by \((4\pi^2r/3)b^{-R}\).
Both this factorization and the curvature proof concern the actual vacuum
measure. The curvature gap (4) is the sharper direct conclusion of this
argument; the factorization also discharges the previously explicit
local-to-global premise in this same strong-coupling regime.

## Scope, disconnected graphs and precedent

For disconnected graph components the Hamiltonian is a tensor sum and the
positive vacuum factors. Apply the proof on each component and use the
largest local forcing bound. One must not assign a finite diameter to
arbitrary modes spanning disconnected components; the componentwise
construction avoids those modes altogether. This extension also covers
isolated free edges, whose logarithmic groundstate contribution is constant.

The Fourier coefficient algebra is classical; see
[Eymard (1964)](https://numdam.org/articles/10.24033/bsmf.1607/).
The precise trace-norm and fusion estimates used here were proved above.
The curvature-to-gap step uses the classical Bochner framework of
[Bakry--Emery, Proposition 3](https://www.numdam.org/item/SPS_1985__19__177_0.pdf).
Strong-coupling existence and gap stability are established research regimes;
this note asserts no first proof of that regime.

The interaction norm controls every collective Hessian direction and an
actual summable influence row; when the locality weight exceeds one, it also
controls an exponentially decreasing spatial tail. A coordinatewise gradient
bound alone would not give these conclusions.

The result remains a theorem about all finite graphs in a specified
strong-coupling family. The criterion fails along \(\kappa\to0\), so it
does not control the continuum trajectory. Infinite-volume limit
identification, continuum renormalization, OS reconstruction and non-free
Yang--Mills identification remain separate. A finite arithmetic implementation
may check the explicit forcing and radius inequalities; the analytic
implication above is not thereby Lean- or Mathlib-verified.
