# SU(3) vacuum Fourier certificates

The exact rational gate checks a written strong-coupling theorem for the
actual SU(3) Hamiltonian vacuum, over all spin representations and every
finite graph in a stated structural family. Its separate linear charged
bounds apply to a fundamental/conjugate static-source pair with the Gauss
condition at every vertex and no dynamical fundamental matter.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer import (
    su3_vacuum_fourier_family,
    su3_vacuum_fourier_bounds,
    replay_su3_vacuum_fourier_certificate,
)

family = su3_vacuum_fourier_family()
assert family["status"] == "PASS"
assert family["witness"]["arithmetic"]["neutral_gap_lower"] == "102"
assert family["volume_uniform_charged_linear_bound_verified"]
assert replay_su3_vacuum_fourier_certificate(family["certificate"])

square = su3_vacuum_fourier_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)],
    plaquettes=[(1, 2, 3, 4)],
    weighted_incidence_cap=4,
    max_cycle_length=4,
    max_cycle_diameter=2,
)
assert square["finite_gate_verified"]
assert square["witness"]["forcing_by_edge"] == ["1/256"] * 4
assert replay_su3_vacuum_fourier_certificate(square["certificate"])

localized = su3_vacuum_fourier_family(kappa=576, decay_base=2, tail_radius=3)
assert localized["witness"]["arithmetic"][
    "actual_log_vacuum_hessian_tail_row_upper"
] == "1/384"
assert localized["witness"]["arithmetic"][
    "actual_factorization_constant_upper"
] == "32/31"
assert Fraction(localized["witness"]["arithmetic"]["neutral_gap_lower"]) == 204
```

Inputs accept exact integers and fractions, excluding floats and booleans.
The family gate defaults to kappa 288, incidence cap 4, maximum cycle
length 4, maximum ambient line-graph cycle diameter 2, decay base 1,
fixed-point radius 1/48, and tail radius 0. These caps define the quantified
family; they do not claim membership of an uninspected graph.

The graph gate uses signed one-based edge indices for simple oriented
cycles and nonnegative magnetic weights. It computes actual weighted
incidence, cycle length and ambient diameter, then checks every supplied
cap; omitted caps default to those computed values. The direct forcing
gate may pass when a deliberately coarse family cap does not. Conversely,
a false structural cap makes the overall graph certificate inconclusive
even when the direct graph arithmetic passes. Explicit graph and family
flags preserve this distinction.

The coefficient upper bound on charged energy holds at all couplings;
its positive lower bound is earned only by the fixed-point gate. No
source vertices are selected by these APIs: the reported coefficients
are quantified over every distinct connected pair in the stated charged
sector. A source pair in different graph components is outside that claim.

The v1 certificate includes exact inputs, forcing, all bounds, model and
charge-sector normalization, scope, and honesty flags. Canonical replay
recomputes the entire envelope. A rehashed altered bound or scope fails;
a faithfully stored INCONCLUSIVE result also replays false. Digests do
not verify the infinite-dimensional argument below. The scalar
static_confinement_claim and string_tension_claim remain false because no
infinite-volume asymptotic observable is asserted. The narrower finite
and volume-uniform-family charged flags record the earned result.

The complete analytic implication follows. This API and proof are
SU(3)-specific and do not change the existing SU(2) v1 certificate.

11 September 2026. This is a written strong-coupling argument for all
representation sectors on each finite graph, with constants independent
of graph volume. The constants and representation estimates were checked
independently during the campaign. Executable rational gates check its
stated inequalities; they do not formally verify this analysis. There is
no novelty claim, infinite-volume construction, continuum limit, or Clay
Yang–Mills discharge in this result.

## Model and theorem

Let a finite graph have nonempty edge set, distinct endpoints on every
edge, normalized product SU(3) Haar measure, and simple oriented
interaction cycles \(p\) of length \(\ell_p\). Parallel edges are allowed;
repeated edges or vertices within one cycle are excluded. Every electric
edge weight is one and \(v_p\ge0\). Set

\[
A_\kappa=\frac\kappa2C+\frac2\kappa
\sum_pv_p(3-\operatorname{Re}\operatorname{Tr}U_p),\qquad
C=-\sum_e\Delta_e,\quad g=\frac4{\kappa^2},\quad \kappa>0.
\tag{1}
\]

These are dimensionless \(aH\) energy units. The generator normalization is
\(\operatorname{Tr}_F(T_aT_b)=\delta_{ab}/2\). Consequently
\(C_F=4/3\), the adjoint Casimir is \(3\), and the group metric has
\(\operatorname{Ric}=3/4\). These conventions agree with the defining and
adjoint normalization in
[Haber, arXiv:1912.13302](https://arxiv.org/abs/1912.13302).

Use the shortest-path metric of the ordinary line graph: distinct edges
sharing a vertex are adjacent. Let \(D_p\) be the diameter of the cycle
support in that ambient metric, including shortcuts outside the cycle.
For \(b\ge1\), define

\[
\mathcal A=g\max_e\sum_{p:e\in p}3^{\ell_p}v_pb^{D_p}.
\tag{2}
\]

Suppose \(r>0\) obeys

\[
\boxed{\mathcal A+12r^2\le r,\qquad 24r<1.}
\tag{3}
\]

Then the actual positive groundstate has logarithm \(S\), up to its
normalization, with Hessian block majorants \(M_{ef}\ge0\) satisfying

\[
\|\operatorname{Hess}_{ef}S\|_{\infty,\mathrm{op}}\le M_{ef},
\qquad \max_e\sum_f b^{d(e,f)}M_{ef}\le r.
\tag{4}
\]

In particular, with \(\rho=3/4-2r>0\),

\[
\Delta_{A_\kappa}\ge\frac{\kappa\rho}{2}.
\tag{5}
\]

This bounds the whole scalar product-group spectrum above its unique
vacuum and therefore its gauge-invariant neutral subspace. The actual
vacuum measure \(\mu=\psi_0^2\,dU\) has total-variation conditional
influence row sum at most \(\eta=3r/2<1\), and

\[
\operatorname{Var}_\mu f\le\frac1{1-3r/2}
\sum_e\mathbb E_\mu[\operatorname{Var}(f\mid U_{\ne e})].
\tag{6}
\]

Hessian and influence row tails at distances greater than \(R\) are at
most \(rb^{-R}\) and \((3r/2)b^{-R}\), respectively.

Impose the Gauss condition at every vertex, introduce a fundamental
static source at \(s\) and its conjugate at \(t\), and include no
dynamical fundamental matter or boundary that absorbs flux. For distinct
vertices in the same component, with graph distance \(d=d(s,t)\), the
vacuum-subtracted charged energy satisfies

\[
\boxed{\frac{\kappa\rho}{2}\,d
\le E_{s,t}-E_0\le\frac{2\kappa}{3}\,d.}
\tag{7}
\]

This is a linear finite-graph static-source bound, uniform in the volume.
It does not assert existence of an asymptotic string-tension limit.

For four-edge cycles, \(D_p\le2\), and weighted incidence
\(\max_e\sum_{p:e\in p}v_p\le d_*\), a convenient sufficient choice is

\[
r=\frac1{48},\qquad gd_*b^2\le\frac1{5184}.
\tag{8}
\]

It gives contraction \(1/2\), \(\rho=17/24\), neutral gap
\(17\kappa/48\), and factorization constant at most \(32/31\).
For unweighted cubic spatial graphs \(d_*\le4\), the choice
\(\kappa=288,b=1\) reaches the self-map boundary exactly and gives gap
\(102\) in dimensionless units and charged energy between \(102d\)
and \(192d\). Choosing \(\kappa=576,b=2\) gives exponential spatial
tails and twice those energy bounds. These are conservative fixed
strong-coupling regimes, not trajectories toward the ultraviolet limit.

## Representation normalization, fusion and multiplicity

For an SU(3) irrep of highest weight \((p,q)\),

\[
C_{p,q}=\frac{p^2+pq+q^2+3p+3q}{3}.
\]

Every nontrivial irrep therefore has \(C_\lambda\ge c_*=4/3\), with
equality for the defining representation and its dual. Put
\(q_\lambda=\sqrt{C_\lambda/c_*}\), and \(q_0=0\). The formula is also
recorded in the primary study
[SU(N) rotator Hamiltonian, PTEP 2020, 073B04](https://academic.oup.com/ptep/article/2020/7/073B04/5870248).
Only this representation label \(q_\lambda\), not a running physical
coupling, is denoted by \(q\) in the following estimates.

There are eight orthonormal generators. A unit directional generator
in irrep \(\lambda\) has operator norm at most
\(\sqrt{C_\lambda}=\sqrt{c_*}\,q_\lambda\), by Cauchy–Schwarz and
\(\sum_aT_a^2=C_\lambda I\). If \(\nu\) occurs in
\(\lambda\otimes\mu\), the same Casimir identity and Minkowski's inequality
in the direct sum of eight representation spaces give

\[
\sqrt{C_\nu}
=\left(\sum_a\|(T_a^\lambda\otimes I+
I\otimes T_a^\mu)v\|^2\right)^{1/2}
\le\sqrt{C_\lambda}+\sqrt{C_\mu}
\]

for any unit vector \(v\) in one copy of \(\nu\). Thus
\(q_\nu\le q_\lambda+q_\mu\), including tensor products with multiplicity.

Write product-group Fourier series in the convention

\[
f(U)=\sum_{\boldsymbol\lambda}
\operatorname{Tr}[F_{\boldsymbol\lambda}
\rho_{\boldsymbol\lambda}(U)],\qquad
\|f\|_2^2=\sum_{\boldsymbol\lambda}
\frac{\|F_{\boldsymbol\lambda}\|_{\rm HS}^2}
{\dim\rho_{\boldsymbol\lambda}}.
\tag{9}
\]

The matrix norm \(\|\cdot\|_1\) below is the trace norm. Under
Clebsch–Gordan decomposition, tensor products split as
\(\bigoplus_\nu(\rho_\nu\otimes I_{m_\nu})\). The product coefficient
in \(\nu\) is obtained by unitary conjugation of \(F\otimes G\),
pinching onto this block, and taking its ordinary partial trace over
the multiplicity space. Both operations are trace-norm contractions,
and the sum of output trace norms is at most \(\|F\|_1\|G\|_1\).
For completeness, partial trace is contractive because its trace-norm
dual tests \(Y\) lift to \(Y\otimes I\) with the same operator norm.
No representation dimension or multiplicity factor is lost.
No invariance of trace norm under partial transpose is used.

## Complete coefficient space and bilinear estimate

Let \(E_{\boldsymbol\lambda}=\sum_e C_{\lambda_e}\),
\(X_{\boldsymbol\lambda}=\{e:\lambda_e\ne0\}\), and
\(w_{\boldsymbol\lambda}=b^{\operatorname{diam}X_{\boldsymbol\lambda}}\).
Remove the constant coefficient before assigning this weight. Define

\[
\mathcal M_b(f)=\max_i\sum_{\boldsymbol\lambda\ne0}
q_{\lambda_i}E_{\boldsymbol\lambda}w_{\boldsymbol\lambda}
\|F_{\boldsymbol\lambda}\|_1.
\tag{10}
\]

The real, Haar-mean-zero coefficient families with finite norm form a
Banach space on every finite connected graph. The sum of the finitely
many anchored norms is a weighted matrix-valued \(\ell^1\) norm, and lies
between their maximum and the number of edges times their maximum.
Reality and zero mean are closed conditions; conjugation preserves
the Casimir and support weights.

Since every active \(q_{\lambda_i}\ge1\),

\[
\sum_{\boldsymbol\lambda\ne0}
E_{\boldsymbol\lambda}\|F_{\boldsymbol\lambda}\|_1
\le|\mathcal E|\mathcal M_b(f).
\tag{11}
\]

The derivative generator bounds imply absolute uniform convergence of
the function and its first and second derivative series, hence an actual
\(C^2\) function. This finite-volume reconstruction bound depends on
\(|\mathcal E|\); the contraction and Hessian constants below do not.
Disconnected graphs are treated componentwise, with separate mean-zero
logarithms and product groundstates. Infinite distances between
components are never substituted into a weight.

Let \(\Pi_0\) remove the Haar constant, \(C_0^{-1}\) divide nonconstant
coefficients by \(E_{\boldsymbol\lambda}\), and
\(\Gamma(f,h)=\sum_{e,a}(X_{ea}f)(X_{ea}h)\). The key estimate is

\[
\mathcal M_b(C_0^{-1}\Pi_0\Gamma(f,h))
\le12\mathcal M_b(f)\mathcal M_b(h).
\tag{12}
\]

Here are all constants in (12). Each derivative pair costs at most
\(8c_*\sum_e q_{\lambda_e}q_{\mu_e}\) in trace norm. Output energy cancels
the inverse \(C_0^{-1}\), and output anchor is at most
\(q_{\lambda_i}+q_{\mu_i}\). Terms differentiated at \(e\) require
\(e\in X_{\boldsymbol\lambda}\cap X_{\boldsymbol\mu}\), so the union's
diameter is at most the sum of the two diameters. Fusion can only shrink
the support. Therefore the output weight is at most \(w_\lambda w_\mu\).

Use \(E_\lambda\ge c_*\),
\(\sum_e q_{\lambda_e}\le E_\lambda/c_*\), and

\[
\sum_\mu q_{\mu_e}w_\mu\|G_\mu\|_1
\le\mathcal M_b(h)/c_*.
\]

The term anchored by \(q_{\lambda_i}\) is then at most
\((8/c_*)\mathcal M_b(f)\mathcal M_b(h)=6\mathcal M_b(f)\mathcal M_b(h)\).
The other anchor contributes the same amount, proving (12).
First establish these inequalities for finite Fourier sums and then
extend by Banach completion. This also justifies every product in the
fixed-point equation.

## Wilson forcing and the actual groundstate

A cycle with distinct edges has \(\chi_p=\operatorname{Tr}U_p\) in one
product fundamental/dual representation of dimension \(3^{\ell_p}\).
Its Haar \(L^2\) norm is one, because the product holonomy is Haar and a
fundamental character has norm one. Equation (9) gives coefficient
Hilbert–Schmidt norm \(3^{\ell_p/2}\) and trace norm at most \(3^{\ell_p}\).
The real part is half this term plus half its complex conjugate, so
its total coefficient trace norm is still at most \(3^{\ell_p}\).
This argument handles every edge orientation and does not assume that
partial inversion preserves trace norm.

Every active fundamental or dual anchor has \(q_\lambda=1\).
Consequently
\(\mathcal M_b(C_0^{-1}gV)\le\mathcal A\), where
\(V=\sum_pv_p\operatorname{Re}\operatorname{Tr}U_p\) has zero Haar mean.
The map

\[
\mathcal T(S)=C_0^{-1}gV+
C_0^{-1}\Pi_0\Gamma(S,S)
\tag{13}
\]

maps the closed radius-\(r\) ball into itself by (3) and is Lipschitz
with constant \(24r<1\). It has a unique real fixed point there.
Its reconstructed \(C^2\) function satisfies

\[
CS=gV+\Gamma(S,S)-\langle\Gamma(S,S)\rangle_H,\qquad
(C-gV)e^S=-\langle\Gamma(S,S)\rangle_H e^S.
\tag{14}
\]

Elliptic regularity makes \(e^S\) smooth. The groundstate transform gives
for every smooth scalar \(f\)

\[
\langle e^Sf,(C-gV-E)e^Sf\rangle_H
=\int e^{2S}|\nabla f|^2\,dU,\qquad
E=-\langle\Gamma(S,S)\rangle_H.
\tag{15}
\]

Thus the positive eigenfunction is the actual groundstate; no lower
eigenvalue exists, and its kernel is one-dimensional on the connected
product manifold. Gauge transformations commute with this Hamiltonian
and preserve positivity and normalization, so the groundstate is
gauge invariant. No fitted Gibbs density is substituted for \(\psi_0^2\).

## Hessian, curvature and conditional variance

Define symmetric nonnegative block bounds

\[
M_{ef}=c_*\sum_{\boldsymbol\lambda}
q_{\lambda_e}q_{\lambda_f}\|S_{\boldsymbol\lambda}\|_1.
\tag{16}
\]

For a product geodesic with tangent blocks \(v_e\), differentiating
each representation twice bounds its second derivative by
\(c_*(\sum_eq_{\lambda_e}|v_e|)^2\|S_\lambda\|_1\).
This is the covariant Hessian along a geodesic, so no connection term
is dropped. It gives the block bounds (16). Since
\(b^{d(e,f)}\le w_\lambda\) when \(e,f\in X_\lambda\), and
\(\sum_fq_{\lambda_f}\le E_\lambda/c_*\), their weighted row sum is
at most \(\mathcal M_b(S)\le r\). The symmetric block Schur estimate
then gives \(\|\operatorname{Hess}S\|_{\rm op}\le r\).

For the bi-invariant metric \(Q(X,Y)=-2\operatorname{Tr}(XY)\),
the Killing form of SU(3) is \(-3Q\). The Levi-Civita formula
\(\nabla_XY=[X,Y]/2\) yields \(\operatorname{Ric}=-B_{\rm Kill}/4=3Q/4\).
For \(\mu=e^{2S}dU/Z\), weighted Ricci curvature is
\(\operatorname{Ric}-2\operatorname{Hess}S\ge\rho\).
The integrated Bochner identity applied to each nonconstant eigenfunction
gives weighted-Laplacian eigenvalue at least \(\rho\); density gives the
Poincare inequality. Equation (15), multiplied by \(\kappa/2\), proves (5).
The same argument applies to every conditional block measure with the
outside variables fixed: restriction takes a principal Hessian block
and preserves this lower curvature bound independently of block size.
This is the compact diffusion criterion of
[Bakry–Emery, 1985](https://www.numdam.org/item/SPS_1985__19__177_0.pdf).

For completeness, a separate Fourier estimate proves (6). A mixed
four-point difference of \(\log\mu=2S+\text{constant}\), changing only
coordinates \(e,f\), is bounded by
\(8\sum_{\lambda:e,f\in X_\lambda}\|S_\lambda\|_1\).
The total-variation influence \(c_{ef}\) of changing \(f\) on the
conditional law at \(e\) is therefore at most
\(2\sum_{\lambda:e,f\in X_\lambda}\|S_\lambda\|_1\).
Indeed, conditional log-density ratio has this four-point oscillation
\(a\), and normalization gives total variation at most
\(\tanh(a/4)\le a/4\). Since
\(|X_\lambda|\le E_\lambda/c_*\) and active \(q_{\lambda_e}\ge1\),

\[
\sum_{f\ne e}b^{d(e,f)}c_{ef}\le
\frac2{c_*}\mathcal M_b(S)\le\frac32r=\eta.
\tag{17}
\]

One can obtain factorization without an additional premise. Let \(P_e\)
be conditional expectation over coordinate \(e\), and
\(L=\sum_e(P_e-I)\), a bounded self-adjoint heatbath generator.
For coordinate oscillations \(\delta_i f\), maximal coupling gives
\(\delta_i(P_ef)\le\delta_if+c_{ei}\delta_ef\) when \(i\ne e\);
\(\delta_e(P_ef)=0\). Apply this to
\(I+hL=(1-nh)I+h\sum_eP_e\), \(0<h\le1/n\), and sum over \(i\).
Equation (17) gives contraction \(1-h(1-\eta)\) of
\(\sum_i\delta_if\). Iteration gives rate \(e^{-(1-\eta)t}\).
For bounded centered \(f\), its \(L^2\) norm is bounded by its total
oscillation, so the same rate holds with a finite prefactor depending
on \(f,n\). Spectral calculus for the self-adjoint semigroup excludes
spectral support in \((0,1-\eta)\); bounded functions are dense in
\(L^2(\mu)\). Hence
\(-L\ge(1-\eta)(I-P_{\rm const})\), which is precisely (6).
No volume-uniform oscillation prefactor is needed for this spectral step.

## Charged cut proof and exact vacuum subtraction

Represent the source pair by matrix-valued functions satisfying
\(F(U^h)=h_sF(U)h_t^{-1}\). The charged wavefunction is \(\psi_0F\).
Applying (15) componentwise identifies its vacuum-subtracted energy
form exactly as

\[
E_{s,t}-E_0=\frac\kappa2
\inf_F\frac{\int\sum_e\|\nabla_eF\|_{\rm HS}^2\,d\mu}
{\int\|F\|_{\rm HS}^2\,d\mu}.
\tag{18}
\]

For a vertex set \(W\) containing \(s\) but not \(t\), perform the gauge
transformation \(h_v=\zeta I\) in \(W\) and \(I\) elsewhere, where
\(\zeta=e^{2\pi i/3}\). It fixes all noncut variables and multiplies
each cut link by \(\zeta\) or \(\zeta^{-1}\), according to orientation.
The conditional measure on \(B=\delta(W)\) is invariant, whereas
\(F\mapsto\zeta F\). Since \(\zeta\ne1\), every matrix component has
conditional mean zero. Conditional Poincare therefore gives
\(\int\sum_{e\in B}\|\nabla_eF\|^2d\mu\ge\rho\int\|F\|^2d\mu\).

Take \(W_k=\{v:d(s,v)\le k\}\), \(k=0,\ldots,d-1\).
Every cut separates the sources. Adjacent vertex distances differ by
at most one, so each graph edge lies in at most one of these cut
boundaries. Sum the \(d\) inequalities and use (18) for the lower bound
in (7). Smooth equivariant functions are a form core: smooth
approximation followed by compact gauge-group averaging preserves the
constraint. Thus the bound holds on the full charged form domain.

For the upper bound choose a shortest path \(\gamma\) and
\(F=U_\gamma/\sqrt3\). Its Hilbert–Schmidt norm is identically one and
\(\sum_e\|\nabla_eF\|_{\rm HS}^2=(4/3)d\) pointwise. Substitution in
(18) proves the upper bound, with exact cancellation of the interacting
vacuum energy and no extensive plaquette penalty.

The same proof also bounds a specifically defined Hamiltonian rectangle
amplitude. For \(T\ge0\) and normalized \(\Phi_\gamma=\psi_0U_\gamma/\sqrt3\), let
\(C_\gamma(T)=\langle\Phi_\gamma,
e^{-T(A_{\kappa,\mathrm{charged}}-E_0)}\Phi_\gamma\rangle\).
The positive spectral measure, lower spectral edge, and Jensen's
inequality at its exact mean give
\(e^{-2\kappa dT/3}\le C_\gamma(T)\le e^{-\kappa\rho dT/2}\).
Here \(T\) is dimensionless Hamiltonian time, with physical time \(aT\).
No identification with an unspecified four-dimensional Euclidean Wilson
coupling or construction of a limiting string tension is assumed.

## Scope and relation to earlier work

The Hamiltonian framework originates with
[Kogut–Susskind, 1975](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.11.395).
Strong-coupling confinement is an established regime; this note supplies
an explicitly bounded argument for the stated normalization rather than
a priority claim. The finite-family result does not construct a common
infinite-volume Hilbert representation or an ultraviolet limit.
Those require separate analytic work. In particular, nothing here proves
that the constants persist as the physical lattice spacing tends to zero.
The proof uses an exact representation series and compact spectral
analysis, not neural fitting, finite-grid residuals, or truncated spins.
Its certificate flags for continuum Yang–Mills and formal verification
must remain false.
