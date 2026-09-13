# Invariant vacuum Fourier correction

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    invariant_vacuum_fourier_family,
    invariant_vacuum_fourier_bounds,
    replay_invariant_vacuum_fourier_certificate,
)

su2 = invariant_vacuum_fourier_family("su2", 27, correction_radius=Q(1, 9))
su3 = invariant_vacuum_fourier_family("su3", 48, correction_radius=Q(1, 4))
assert replay_invariant_vacuum_fourier_certificate(su2["certificate"])
assert replay_invariant_vacuum_fourier_certificate(su3["certificate"])
# A second proof route passes where the global curvature criterion fails.
su3_conditional = invariant_vacuum_fourier_family(
    "su3", 45, correction_radius=Q(9, 20), gap_method="factorization",
)
assert su3_conditional["witness"]["arithmetic"]["neutral_gap_lower"] == "415459/72000"
assert replay_invariant_vacuum_fourier_certificate(su3_conditional["certificate"])

# An actual square graph: signed, one-based edge indices.
square = invariant_vacuum_fourier_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)],
    group="su3", kappa=48, correction_radius=Q(1, 4),
    plaquettes=[(1, 2, 3, 4)],
)
assert square["explicit_graph_neutral_gap_verified"]
```

`family` checks a theorem for graphs satisfying the declared hypotheses;
`bounds` inspects a concrete graph, including all electric edges when
computing girth. Self-loops and parallel edges are refused. A failed cap
or analytic gate returns `INCONCLUSIVE`. Replay recomputes the full
canonical certificate, including structural hypotheses and honesty fields;
a failed certificate is not replayed as a proof. Inputs must be exact
integers or fractions, not floating-point estimates.

`correction_radius` bounds the unknown correction to the explicit Wilson
seed, not the whole logarithmic vacuum. The SU(2) API uses the sharper
spin-norm variant below; the SU(3) API uses the Casimir norm. `decay_base`
is a coefficient-support weight at least one. The emitted Hessian and
influence rows are **unweighted**; the API emits no tail claim from them.
`gap_method="curvature"` is the default. `gap_method="factorization"`
uses conditional Haar comparison and actual-vacuum influence bounds,
as proved below. Both arithmetic lower bounds are reported separately;
`neutral_gap_lower` and the verdict refer to the selected method. The
factorization route also controls every conditional block, so the
charged cut bound remains justified. Its exponential lower bound is
computed as a finite rational power, without a transcendental evaluation.
All energy bounds use dimensionless `aH`. The unit-time resolvent is for
`aH - E0`; it does not fix a common physical time as lattice spacing varies.

The flags for the infinite-volume limit, the continuum, Yang–Mills, and
formal verification remain false. The following written implication
includes every representation on each finite graph; rational replay alone
does not machine-check that implication. Existing v1 Fourier certificates
are unchanged.

## A local residual correction to an invariant reference vacuum

11 September 2026. This note proves a volume-uniform sufficient criterion
for compact-link Hamiltonians by constructing the correction to a positive
reference vacuum. It strengthens the earlier
[Fourier estimate](gauge-vacuum-fourier.md) in three concrete ways: the generator
sum is bounded as one operator, gauge invariance supplies a support-energy
floor, and the reference Hessian is evaluated directly. The residual norm
is an anchored Fourier norm; no sum of global residual oscillations is
discarded. The target vacuum and its gap are conclusions, not premises.

The argument is a written theorem in a restricted lattice regime. Its
novelty has not been established. An exact rational certificate can replay
the final inequalities, but does not formally verify this analysis.
Neither a continuum theory nor cutoff-uniform physical Yang–Mills follows.
The older SU(2) and SU(3) v1 certificates retain their original constants.

## Configuration space, normalization and graph hypotheses

Let \(G=SU(N)\), with \(N=2\) or \(3\). Let \(\mathcal G\) be a finite
simple graph with a nonempty edge set. There are no self-loops or parallel
undirected edges. Every actual graph cycle has length at least \(g_0\),
where \(g_0\ge3\). Forest components satisfy this condition vacuously.
This is the girth of the entire graph, including edges that occur in no
potential term; a four-edge plaquette list does not exclude a triangular
shortcut elsewhere.

Each edge carries an independent group variable with normalized Haar
measure. Gauge transformations act at every vertex by
\(U_{uv}\mapsto h_uU_{uv}h_v^{-1}\); no boundary vertex is exempted.
The real gauge-invariant potential is
\(V=\sum_pv_p\operatorname{Re}\operatorname{Tr}_F U_p\), where
every cycle \(p\) is simple, has length \(\ell_p\), and \(v_p\ge0\).
The full scalar Hamiltonian is

\[
A_\kappa=\frac\kappa2C+\frac2\kappa
\sum_pv_p(N-\operatorname{Re}\operatorname{Tr}_F U_p),
\qquad C=-\sum_e\Delta_e,\qquad g=\frac4{\kappa^2}.
\tag{1}
\]

Its constant-shifted rescaling is \(C-gV\). Every electric weight is one,
and energies in (1) are dimensionless \(aH\) units. Use
\(\operatorname{Tr}_F(T_aT_b)=\delta_{ab}/2\); then

\[
c_*=\min_{\lambda\ne0}C_\lambda
=\begin{cases}3/4&SU(2),\\4/3&SU(3),\end{cases}
\qquad
\rho_0=\operatorname{Ric}
=\begin{cases}1/2&SU(2),\\3/4&SU(3).\end{cases}
\tag{2}
\]

For SU(2), \(C_j=j(j+1)\); for SU(3),
\(C_{p,q}=(p^2+pq+q^2+3p+3q)/3\), which directly prove the
nontrivial-irrep lower endpoints in (2). For the chosen metric,
\(\nabla_XY=[X,Y]/2\) and
\(\operatorname{Ric}=-B_{\rm Kill}/4=N/4\).
The group generator normalization is consistent with
[Haber, arXiv:1912.13302](https://arxiv.org/abs/1912.13302).

The proof is componentwise for disconnected graphs; the positive vacuum
and Hamiltonian factor across components. Distances between different
components are not inserted into weights. Charged conclusions concern
distinct source vertices in the same component.

## The invariant Fourier space

Write

\[
f(U)=\sum_{\boldsymbol\lambda}
\operatorname{Tr}[F_{\boldsymbol\lambda}
\rho_{\boldsymbol\lambda}(U)],\qquad
E_{\boldsymbol\lambda}=\sum_eC_{\lambda_e},\qquad
q_\lambda=\sqrt{C_\lambda/c_*},\quad q_0=0.
\tag{3}
\]

The coefficient convention has Parseval identity
\(\|f\|_2^2=\sum_\lambda\|F_\lambda\|_{\rm HS}^2/
\dim\rho_\lambda\). Let \(X_\lambda=\{e:\lambda_e\ne0\}\),
and use the ordinary ambient line-graph distance between graph edges.
For \(b\ge1\), put \(w_\lambda=b^{\operatorname{diam}X_\lambda}\)
on nonconstant modes only. Define

\[
\mathcal M_b(f)=\max_i\sum_{\lambda\ne0}
q_{\lambda_i}E_\lambda w_\lambda\|F_\lambda\|_1.
\tag{4}
\]

Here \(\|\cdot\|_1\) is matrix trace norm. Work in the real,
mean-zero, gauge-invariant subspace of this coefficient space.
For each finite graph it is complete: the sum of its finitely many
anchored norms is a weighted \(\ell^1\) direct sum of finite-dimensional
matrix spaces, equivalent to the maximum norm. Reality is closed.
Gauge transformations act by left and right coefficient unitaries;
averaging them is a bounded projection preserving the norm bound.
Consequently invariance and the zero constant mode are closed conditions.

Because every active \(q_\lambda\ge1\),
\(\sum_{\lambda\ne0}E_\lambda\|F_\lambda\|_1
\le|\mathcal E|\mathcal M_b(f)\).
Unit generator norms are at most \(\sqrt{C_\lambda}\); first and second
derivative coefficient norms are thus bounded by a constant times
\(E_\lambda\|F_\lambda\|_1\). This proves absolute uniform convergence
through two derivatives on each finite graph. The reconstruction yields
an actual \(C^2\) function. Its intermediate bound contains the finite
number of edges, but none of the contraction or Hessian constants below
does.

Each product Peter–Weyl projection commutes with the vertex gauge
transformations. If exactly one active irrep is incident to a vertex in
a proposed Fourier label, averaging that vertex action multiplies one
side of its coefficient by \(\int_G\rho_\lambda(h)\,dh=0\).
The invariant coefficient must therefore vanish. Thus every nonzero
invariant Fourier coefficient has a support graph in which each incident
vertex has degree at least two. Every finite nonempty graph of minimum
degree at least two contains a cycle. By the actual girth assumption,

\[
|X_\lambda|\ge g_0,\qquad
\boxed{E_\lambda\ge E_{\min}:=g_0c_*}
\quad\text{for every nonzero invariant mode.}
\tag{5}
\]

No such support can occur in a forest, where the invariant scalar
functions are constant. One must not apply (5) to charged coefficient
functions or to a graph with an unchecked shorter cycle.

## Group the derivative directions before taking norms

For two input irreps on edge \(e\), define
\(\Omega_e=\sum_aT_a^\lambda\otimes T_a^\mu\).
Regard this as the product of the operator row
\([T_a^\lambda\otimes I]_a\) and column
\([I\otimes T_a^\mu]_a\). Their squared norms are
\(C_\lambda\) and \(C_\mu\), respectively, by the Casimir identities.
Therefore

\[
\boxed{\|\Omega_e\|_{\rm op}
\le\sqrt{C_\lambda C_\mu}
=c_*q_\lambda q_\mu.}
\tag{6}
\]

Taking the triangle inequality separately for every Lie-algebra direction
would introduce an unnecessary factor equal to its dimension.
For coefficient terms \(F_\lambda,G_\mu\), the corresponding derivative
product coefficient before decomposition is
\(-(F_\lambda\otimes G_\mu)\Omega_e\).
Equation (6) bounds its trace norm directly.

Under Clebsch–Gordan decomposition, coefficient output is obtained by
unitary conjugation, pinching onto irreducible blocks, and ordinary partial
trace over multiplicity spaces. The sum of output trace norms is at most
the input trace norm: pinching is contractive, and partial trace is
contractive by trace-norm duality since a test \(Y\) lifts to \(Y\otimes I\)
with the same operator norm. This handles SU(3) multiplicities.
No invariance under partial transpose is used.

Casimir-vector Minkowski gives
\(q_\nu\le q_\lambda+q_\mu\) for every allowed output irrep.
Let \(\Pi_0\) remove the constant mode and let \(C_0^{-1}\) divide each
remaining coefficient by its energy. Set

\[
T(f,h)=C_0^{-1}\Pi_0\Gamma(f,h),\qquad
\Gamma(f,h)=\sum_{e,a}(X_{ea}f)(X_{ea}h).
\]

This map preserves invariance: \(C\), products, and its spectral inverse
commute with gauge transformations, and
\(2\Gamma(f,h)=fCh+hCf-C(fh)\).
It preserves reality and zero mean as well.

For a nonzero derivative pair the input supports intersect, so
the diameter of their union is at most the sum of their diameters.
Fusion only shrinks the support. Thus the output weight is at most
\(w_\lambda w_\mu\), even when the supports are internally disconnected.
Output energy cancels \(C_0^{-1}\) in (4). Equation (6) gives

\[
\mathcal M_{b,i}(T(f,h))
\le c_*\sum_{\lambda,\mu}
(q_{\lambda_i}+q_{\mu_i})
\sum_eq_{\lambda_e}q_{\mu_e}
w_\lambda\|F_\lambda\|_1w_\mu\|G_\mu\|_1.
\tag{7}
\]

Use
\(\sum_eq_{\lambda_e}\le E_\lambda/c_*\) and, by (5),
\(\sum_\mu q_{\mu_e}w_\mu\|G_\mu\|_1
\le\mathcal M_b(h)/E_{\min}\).
Each anchor in (7) contributes at most
\(\mathcal M_b(f)\mathcal M_b(h)/E_{\min}\). Hence

\[
\boxed{\mathcal M_b(T(f,h))
\le B\mathcal M_b(f)\mathcal M_b(h),\qquad
B=\frac2{g_0c_*}.}
\tag{8}
\]

The finite-sum calculation extends by completion; it does not require a
spin cutoff. Without the invariant-support hypothesis, the same grouped
calculation gives \(B=2/c_*\), still an improvement over separately
counting generator directions.

## General candidate-centered residual theorem

Choose a real mean-zero gauge-invariant candidate \(S_*\) in (4).
It may be a finite character series or a series with a proved tail.
Suppose the following three bounds are established independently:

\[
\mathcal M_b(S_*)\le v,\qquad
\mathcal M_b\!\left(
C_0^{-1}\Pi_0[CS_*-\Gamma(S_*,S_*)-gV]\right)\le\varepsilon,
\qquad
\operatorname{Hess}S_*\le h_*I.
\tag{9}
\]

The last bound is allowed to be sharper than the general norm bound.
It must hold on the whole compact configuration manifold.
Suppose \(r>0\) satisfies

\[
\boxed{\varepsilon+2Bvr+Br^2\le r,\qquad
2B(v+r)<1,\qquad \rho_0-2h_*-2r>0.}
\tag{10}
\]

Then the actual positive logarithmic vacuum is \(S=S_*+U\), up to
normalization, where \(\mathcal M_b(U)\le r\), and

\[
\boxed{\Delta_{A_\kappa}\ge
\frac\kappa2(\rho_0-2h_*-2r).}
\tag{11}
\]

Proof: write
\(\mathcal F(S)=C_0^{-1}gV+T(S,S)\).
The correction map
\[
U\longmapsto \mathcal F(S_*)-S_*+2T(S_*,U)+T(U,U)
\]
maps the radius-\(r\) ball into itself and has Lipschitz constant at most
\(2B(v+r)<1\), by (8)–(10). The fixed point reconstructs a \(C^2\)
function satisfying
\[
CS=gV+\Gamma(S,S)-\langle\Gamma(S,S)\rangle_H,\qquad
(C-gV)e^S=-\langle\Gamma(S,S)\rangle_H e^S.
\]
Elliptic regularity makes the positive eigenfunction smooth.
The exact groundstate transform
\[
\langle e^Sf,(C-gV-E)e^Sf\rangle_H
=\int e^{2S}|\nabla f|^2\,dU,\qquad
E=-\langle\Gamma(S,S)\rangle_H
\]
shows that it is the true groundstate, with no lower spectrum and a
one-dimensional kernel. This identity holds for every scalar \(f\), not
merely invariant \(f\); smooth functions are a form core.

For the correction, the symmetric Hessian majorants
\(M_{ef}=c_*\sum_\lambda q_{\lambda_e}q_{\lambda_f}\|U_\lambda\|_1\)
have weighted row sum at most \(\mathcal M_b(U)\le r\).
Differentiation along product geodesics proves these are covariant
Hessian bounds, including the same-edge connection term.
The block Schur estimate gives \(\|\operatorname{Hess}U\|_{\rm op}\le r\).
Consequently the actual measure
\(\mu=\psi_0^2dU\) has weighted Ricci tensor
\(\operatorname{Ric}-2\operatorname{Hess}S
\ge(\rho_0-2h_*-2r)I\).
The compact integrated Bochner identity and spectral decomposition give
Poincare with that constant, and the groundstate transform proves (11)
on the entire scalar Hilbert space. Restricting afterward to its neutral
gauge-invariant sector preserves this lower bound.

This is a local residual criterion: (9) uses the maximum over anchors of
a summable Fourier interaction budget. There is no factor of volume.
It is not enough to bound the residual pointwise or to fit it on a grid.
Omnibias jets can compute candidate derivatives, but the norm and tail in
(9), the Hessian bound, and every structural assumption must be verified.
If \(2Bv\ge1\), these inequalities fail to provide a correction theorem.
A sharper method would need a proved bounded inverse for the linearized
map \(I-2T(S_*,\cdot)\); its stability may not be silently assumed.

## Explicit Wilson reference and its much smaller Hessian

For the potential in (1), take the exact reference

\[
S_*=C_0^{-1}gV
=\sum_p\frac{gv_p}{\ell_pc_*}
\operatorname{Re}\operatorname{Tr}_F U_p.
\tag{12}
\]

Each cycle is an eigenfunction of \(C\) with energy \(\ell_pc_*\).
Its Haar mean is zero. The character lies in a product fundamental/dual
representation of dimension \(N^{\ell_p}\), has Haar norm one, and hence
coefficient trace norm at most \(N^{\ell_p}\) by Parseval and
\(\|\cdot\|_1\le\sqrt{\dim}\|\cdot\|_{\rm HS}\).
Taking the real part adds two half coefficients when the dual differs,
not an extra factor of two. The argument is orientation independent.
Consequently (9) is satisfied with

\[
v=\mathcal A:=g\max_e\sum_{p:e\in p}N^{\ell_p}v_pb^{D_p},
\qquad \varepsilon=B\mathcal A^2.
\tag{13}
\]

The residual is quadratic because the linear forcing cancels exactly.
However, using \(h_*=\mathcal A\) would lose useful structure.
For any pair of unit tangent directions on two active edges,

\[
\left|\operatorname{Hess}_{ef}
\operatorname{Re}\operatorname{Tr}_F U_p\right|\le\frac12.
\tag{14}
\]

For different edges, its matrix expression inserts the two generators
among unitary factors, so Hilbert–Schmidt Cauchy–Schwarz bounds it by
\(\|T_X\|_{\rm HS}\|T_Y\|_{\rm HS}=1/2\).
For the same edge, the covariant Hessian along group geodesics inserts
\((T_XT_Y+T_YT_X)/2\); the same two Hilbert–Schmidt bounds apply.
Inverse-oriented edges change positions and signs but not the bounds.
This proves the full block operator estimate, not just coordinate
derivatives or a Hessian of class functions.

Let
\[
d_*=\max_e\sum_{p:e\in p}v_p,\qquad
d_{*,b}=\max_e\sum_{p:e\in p}v_pb^{D_p}.
\]
Summing the \(\ell_p\) possible Hessian blocks in a row cancels the
\(\ell_p\) in (12). Therefore

\[
\boxed{h_*=\frac{gd_*}{2c_*},\qquad
\max_e\sum_fb^{d(e,f)}
\|\operatorname{Hess}_{ef}S_*\|_{\infty,\rm op}
\le\frac{gd_{*,b}}{2c_*}.}
\tag{15}
\]

The unweighted curvature uses \(d_*\), even when \(b>1\) is used to certify
the residual. This distinction improves the bound further. For distances
larger than every potential support diameter, the reference Hessian is
zero and the entire Hessian row tail is bounded by \(rb^{-R}\).

## Two exact examples in the Casimir norm

Both examples concern simple cubic spatial graphs of actual girth at
least four, with four-edge cycles and weighted incidence \(d_*\le4\).
They use \(b=1\), so exponential tails are not claimed for these numbers.

For SU(2), set \(\kappa=32\), so \(c_*=3/4\), \(B=2/3\), and choose
\[
v=\frac14,\quad \varepsilon=\frac1{24},\quad r=\frac1{12}.
\]
The self-map bound is \(2/27<1/12\), the contraction is \(4/9\),
and \(h_*=1/96\). Thus \(h_*+r=3/32\),
\[
\rho=\frac5{16},\qquad
\boxed{\Delta_{A_{32}}\ge5.}
\tag{16}
\]

For SU(3), set \(\kappa=48\), so \(c_*=4/3\), \(B=3/8\), and choose
\[
v=\frac9{16},\quad\varepsilon=\frac{243}{2048},\quad r=\frac14.
\]
The self-map bound is \(507/2048<1/4\), the contraction is \(39/64\),
and \(h_*=1/384\). Thus \(h_*+r=97/384\),
\[
\rho=\frac{47}{192},\qquad
\boxed{\Delta_{A_{48}}\ge\frac{47}{8}.}
\tag{17}
\]

The bare SU(3) curvature-ball proof would require a total norm radius
below \(3/8\), although its forcing alone is \(9/16\). Hence (17) is a
strict improvement over that sufficient criterion, earned by controlling
the explicit reference separately. These dimensionless numbers do not
provide a physical mass in GeV or a continuum renormalization trajectory.

## Actual conditional influence can also be split

This is an optional consequence, not a prerequisite for (11).
Suppose the mixed four-point oscillations of \(2S_*\) give a weighted
conditional-influence row bound \(\eta_*\).
Each Fourier term of the correction contributes at most
\(8\|U_\lambda\|_1\) to a mixed four-point oscillation of \(2U\).
A conditional density whose log ratio has oscillation \(a\) changes by
total variation at most \(\tanh(a/4)\le a/4\). Since
\(|X_\lambda|\le E_\lambda/c_*\) and active anchors have \(q\ge1\),
the correction's weighted influence row is at most \(2r/c_*\).
Adding the two mixed oscillation bounds proves an influence row bound
for the actual measure:

\[
\eta\le\eta_*+\frac{2r}{c_*}.
\tag{18}
\]

If this is below one, the conditional heatbath semigroup has spectral gap
at least \(1-\eta\). To see this directly, let \(P_e\) be conditional
expectation, \(L=\sum_e(P_e-I)\), and \(\delta_i f\) the coordinate
oscillation. A maximal coupling yields
\(\delta_i(P_ef)\le\delta_i f+c_{ei}\delta_e f\) for \(i\ne e\)
and \(\delta_e(P_ef)=0\).
Summing after a step \(I+hL\), \(h\le1/|\mathcal E|\), gives contraction
\(1-h(1-\eta)\) of \(\sum_i\delta_i f\).
Pass to continuous time. The oscillation bound implies exponential
\(L^2\) decay on bounded centered functions with a finite prefactor.
Self-adjoint spectral calculus excludes any smaller positive spectral
support; bounded functions are dense. Therefore
\[
\operatorname{Var}_\mu f\le\frac1{1-\eta}
\sum_e\mathbb E_\mu\operatorname{Var}(f\mid U_{\ne e}).
\tag{19}
\]

For (12), use
\(|\operatorname{Re}\operatorname{Tr}_F U|\le N\).
A mixed four-point difference of this trace is at most \(4N\).
It vanishes unless both coordinates belong to the cycle.
Thus a valid weighted candidate row bound is

\[
\eta_{*,b}=
g\max_e\sum_{p:e\in p}
\frac{2N(\ell_p-1)}{\ell_pc_*}\,v_pb^{D_p}.
\tag{20}
\]

At the SU(2) parameters (16), \(\eta_*\le1/16\) and
\(\eta\le41/144<1\), giving factorization constant \(144/103\).
At the SU(3) parameters (17), \(\eta_*\le3/128\) and
\(\eta\le51/128<1\), giving factorization constant \(128/77\).
These statements concern \(\psi_0^2\), not a substituted classical
Wilson Gibbs density. For \(b>1\), (18)–(20) also give row tails
\(\eta b^{-R}\); all weighted forcing inequalities must be checked again.

## Charged sources and defined rectangle amplitudes

Conditional restriction to any cut block retains the actual curvature
lower bound \(\rho=\rho_0-2h_*-2r\), independently of cut size.
With a fundamental source at \(s\), its conjugate at \(t\), no dynamical
fundamental matter and the Gauss condition at all vertices, a center
gauge transformation on a separating vertex set fixes outside links
and multiplies every charged matrix component by a nonidentity center
phase. Its conditional mean over the cut is therefore zero.
Conditional Poincare gives energy at least \(\rho\) per cut.
The boundaries of the distance balls centered at \(s\), from radius zero
through \(d(s,t)-1\), are pairwise edge disjoint: adjacent graph distances
differ by at most one. The exact vacuum groundstate transform then gives

\[
\frac{\kappa\rho}{2}d(s,t)\le E_{s,t}-E_0
\le\frac{\kappa c_*}{2}d(s,t).
\tag{21}
\]

The upper bound uses the vacuum-dressed shortest-path matrix
\(U_\gamma/\sqrt N\), whose norm is identically one and derivative energy
is \(c_*d\). Smooth equivariant functions are dense in the charged form
domain by smooth approximation followed by compact gauge averaging.
Thus no omitted representation sector or extensive vacuum subtraction
enters either bound.

For normalized \(\Phi_\gamma=\psi_0U_\gamma/\sqrt N\) and \(T\ge0\), define
the Hamiltonian amplitude
\[
C_\gamma(T)=
\langle\Phi_\gamma,
e^{-T(A_{\kappa,\mathrm{charged}}-E_0)}\Phi_\gamma\rangle.
\]
Its positive spectral measure, lower edge (21), and Jensen's inequality
at the exact path energy mean imply
\[
e^{-\kappa c_*dT/2}\le C_\gamma(T)
\le e^{-\kappa\rho dT/2}.
\]
This is a stated Hamiltonian observable, with physical time \(aT\).
No unspecified Euclidean Wilson coupling or asymptotic string tension
limit is substituted.

## Optional sharper SU(2) spin norm

For SU(2) one may retain the earlier norm
\(\mathcal N_b(f)=\max_i\sum_{\mathbf j}j_iE_{\mathbf j}w_{\mathbf j}
\|F_{\mathbf j}\|_1\).
The original three-generator estimate together with
\(\sum_ej_e\le2E_{\mathbf j}/3\) and the invariant floor
\(E_{\min}=3g_0/4\) yields
\[
\mathcal N_b(T(f,h))\le B_{\rm spin}\mathcal N_b(f)\mathcal N_b(h),
\qquad B_{\rm spin}=\frac4{E_{\min}}.
\]
Each of the two anchors contributes \(2/E_{\min}\).
The coefficient-space, invariance, and correction proof is unchanged.
Its correction Hessian bound is \(2r/3\), and its influence row bound is
\(16r/3\). The reference forcing is
\(g\max_e\sum_{p:e}2^{\ell_p-1}v_pb^{D_p}\), while the directly computed
reference Hessian (15) and influence (20) do not change.

Thus replace the last inequality in (10) by
\(\rho_0-2h_*-4r/3>0\), and the correction term in (18) by \(16r/3\).
On the same cubic class at \(\kappa=27\), choose
\[
B_{\rm spin}=\frac43,\quad v=\frac{128}{729},
\quad\varepsilon=\frac43v^2,\quad r=\frac19.
\]
The self-map sum is \(174724/1594323<1/9\), and the contraction bound is
\(1672/2187<1\). With \(h_*=32/2187\),
the actual Hessian is at most \(194/2187\). Consequently
\[
\rho=\frac{1411}{4374},\qquad
\Delta_{A_{27}}\ge\frac{1411}{324}.
\]
The actual influence row is at most \(496/729<1\), giving factorization
constant at most \(729/233\). This is a separate sufficient estimate,
not a retroactive modification of a sealed SU(2) v1 witness.

## What remains beyond this criterion

The result constructs a separate true vacuum on each finite graph, with
uniform local estimates. A thermodynamic-limit theorem can use these
bounds only after checking its own convergence, locality and state
construction hypotheses. Existing bounds for another coupling window
must be recomputed before being transferred to these examples.

A continuum bridge still requires ultraviolet control, surviving
finite-energy states, and identification of an interacting relativistic
Yang–Mills theory. These fixed-coupling estimates do not supply that bridge.

## Exact curvature-radius window for the invariant reference correction

This calculation characterizes the global-curvature sufficient criterion in
the invariant-vacuum proof above, not the physical
phase diagram. Failure excludes this particular majorant argument, not a
Yang–Mills gap. All quantities below are exact scalar bounds.

For the first Wilson reference, put \(A=\|S_*\|_{\rm upper}\), let \(B>0\)
be its bilinear constant, and use correction Hessian factor \(\alpha>0\).
The required inequalities are

\[
F(r):=B(A+r)^2-r\le0,\quad 2B(A+r)<1,\quad
h_*+\alpha r<\rho_0/2,\qquad r>0.
\]

Assume \(A\ge0\), and set

\[
r_v=\frac1{2B}-A,\qquad
r_c=\frac{\rho_0/2-h_*}{\alpha}.
\]

There exists an admissible real radius if and only if

\[
\boxed{A<\frac1{4B},\qquad r_c>0,\qquad
[\ r_c\ge r_v\ \text{or}\ F(r_c)<0\ ].}
\]

Indeed, \(F'(r)=2B(A+r)-1\); strict contraction is exactly \(r<r_v\).
The discriminant is \(1-4BA\). If it is positive, the lower root
\[
r_- = \frac{1-2BA-\sqrt{1-4BA}}{2B}
\]
is nonnegative and \(F\) is strictly decreasing until \(r_v\).
The feasible interval begins at \(r_-\), while curvature requires
\(r<r_c\). Thus feasibility requires \(r_c>r_-\), which is the boxed
test. If \(A=0\), choose any sufficiently small positive radius.
If the discriminant is zero, the only root has contraction exactly one
and is refused. The strict tests also exclude a zero curvature endpoint.
For rational input, feasibility permits a rational radius by density;
the test itself uses only rational arithmetic.

### Cubic SU(2)

Use four-edge cycles, incidence cap four, \(b=1\), and the spin norm.
For \(x=\kappa^2\),

\[
A=128/x,\quad B=4/3,\quad h_*=32/(3x),\quad
\alpha=2/3,\quad \rho_0=1/2.
\]

Here \(r_c=3/8-16/x>r_v=3/8-128/x\), so the exact window is

\[
\boxed{\kappa^2>2048/3.}
\]

Consequently 27 is the smallest positive integer coupling admitted by
this gap criterion. At \(\kappa=27\), \(r=1/9\) also earns the optional
conditional variance factorization gate. That extra gate is not included
in the all-radius equivalence above and is checked separately.

### Cubic SU(3)

With the Casimir norm,

\[
A=1296/x,\quad B=3/8,\quad h_*=6/x,\quad
\alpha=1,\quad\rho_0=3/4.
\]

The discriminant requires \(x>1944\). Curvature has endpoint
\(r_c=3/8-6/x\). When \(r_c<r_v\), its additional test is

\[
512x^2F(r_c)=-3P(x)<0,\qquad
P(x)=55x^2-62944x-106502400.
\]

At \(\kappa=45\), \(P(2025)=-8429625\), so **no real radius** passes
both contraction and curvature in this criterion. For smaller positive
couplings, either the discriminant fails or \(P\) is smaller: \(P\) is
strictly increasing for \(x\ge1944\), and throughout
\(1944<x\le2025\) one has \(r_c<r_v\).

At \(\kappa=46\), \(P(2116)=6568176>0\), and the explicit radius
\(r=7/20\) gives

\[
\rho=1/20-3/529=469/10580,\qquad
\boxed{\Delta_{A_{46}}\ge469/460.}
\]

Thus 46 is the smallest positive integer coupling admitted here. The
less marginal \(\kappa=48,r=1/4\) example gives the larger lower gap
\(47/8\). Both optional influence gates are checked independently.

This explains what radius optimization of the curvature criterion can
achieve. The conditional-block argument below supplies another coercivity
estimate and passes at coupling 45. No amount of radius sampling changes
these exact bounds. A continuum trajectory additionally requires its
own measure, reconstruction, nontriviality and physical-scale estimates.

## A gap from the corrected vacuum without positive global curvature

11 September 2026. This note supplies a second spectral conclusion from
the actual vacuum constructed in
the invariant Fourier construction above.
The fixed-point argument and its representation, graph-girth, gauge and
domain hypotheses remain necessary. Positive global weighted Ricci
curvature is not required for the argument here. Instead, a verified
conditional influence bound and a bounded-density one-link comparison
give a gap on every conditional block, uniformly in graph volume.

This is a written compact-link theorem with exact rational sufficient
gates. It is not a novelty claim, formal verification, thermodynamic-limit
construction, continuum Yang–Mills theorem, or a measured physical mass.
The earlier all-radius obstruction for the *curvature* criterion does
not obstruct this different gap implication.

## Premises already earned by the correction theorem

Use the finite simple graph and Hamiltonian normalization
\[
A_\kappa=\frac\kappa2C+\frac2\kappa
\sum_pv_p(N-\operatorname{Re}\operatorname{Tr}_F U_p),
\qquad g=\frac4{\kappa^2},
\]
with \(SU(2)\) or \(SU(3)\), unit electric weights, gauge transformations
at every vertex, and nonnegative simple-cycle weights.
Let the entire electric graph have girth at least \(g_0\); forest
components are allowed. Set \(c_*=3/4\) for SU(2), \(c_*=4/3\) for
SU(3), and \(E_{\min}=g_0c_*\).

The previous construction uses
\[
S_*=\sum_p\frac{gv_p}{\ell_pc_*}
\operatorname{Re}\operatorname{Tr}_F U_p,\qquad
\psi_0=Z^{-1/2}e^{S_*+U},\qquad
d\mu=\psi_0^2\,dU_{\rm Haar}.
\tag{1}
\]
It identifies this as the actual unique positive groundstate if its
correction ball satisfies
\[
B(A+r)^2\le r,\qquad 2B(A+r)<1.
\tag{2}
\]
For the implemented SU(2) spin norm, \(B=4/E_{\min}\) and
\(\mathcal N_b(U)\le r\). For the SU(3) Casimir norm,
\(B=2/E_{\min}\) and \(\mathcal M_b(U)\le r\).
The forcing \(A\) includes the specified locality weights \(b\ge1\).
Both inequalities in (2) concern the invariant Fourier space and include
every representation. No curvature assumption enters the existence or
groundstate-identification step.

Suppose the configuration-uniform conditional total-variation influence
bounds have row sum at most \(\eta<1\). The previous proof gives the
explicit choice
\[
\eta=\eta_*+
\begin{cases}
16r/3&SU(2)\text{ spin norm},\\
3r/2&SU(3)\text{ Casimir norm},
\end{cases}
\tag{3}
\]
where
\[
\eta_*=g\max_e\sum_{p:e\in p}
\frac{2N(\ell_p-1)}{\ell_pc_*}\,v_p.
\tag{4}
\]
These unweighted expressions follow from the stronger weighted norm
when \(b>1\). For a family with incidence cap \(d_*\) and cycle length
at most \(\ell_{\max}\), replace (4) by
\(2Ngd_*(\ell_{\max}-1)/(\ell_{\max}c_*)\).
The resulting bound is valid for the actual measure in (1), not a
substituted Wilson Gibbs density.

## One-link logarithmic oscillation from the same local norm

Fix the values of all links except \(e\), and compare two values of
that link. A Fourier coefficient with trivial label on \(e\) cancels.
Any other coefficient contributes at most \(2\|U_\lambda\|_1\) to the
oscillation of \(U\). For SU(3), its anchored Casimir norm has
\(q_{\lambda_e}\ge1\) and \(E_\lambda\ge E_{\min}\), so
\[
\sum_{\lambda:e\in X_\lambda}\|U_\lambda\|_1
\le r/E_{\min}.
\]
For the SU(2) spin norm, \(j_e\ge1/2\) instead gives
\[
\sum_{\lambda:e\in X_\lambda}\|U_\lambda\|_1
\le 2r/E_{\min}.
\]
Since \(\log\mu\) differs from \(2S_*+2U\) by a constant, the correction
contributes at most \(4r/E_{\min}\) and \(8r/E_{\min}\), respectively,
to the conditional log-density oscillation.

The reference term uses
\(\operatorname{Re}\operatorname{Tr}_F U_p\in[-N,N]\).
For a fixed link it therefore contributes at most
\[
\Omega_*=
4Ng\max_e\sum_{p:e\in p}\frac{v_p}{\ell_pc_*}.
\tag{5}
\]
Since every actual cycle length is at least \(g_0\), a family bound is
\(\Omega_*\le4Ngd_*/(g_0c_*)\).
Consequently, uniformly over the chosen link and all exterior values,
\[
\boxed{
\operatorname{osc}\log\mu_e(\,\cdot\mid U_{\ne e})
\le\Omega:=\Omega_*+
\begin{cases}
8r/E_{\min}&SU(2)\text{ spin norm},\\
4r/E_{\min}&SU(3)\text{ Casimir norm}.
\end{cases}}
\tag{6}
\]
The conditional normalizing factor is independent of the free link and
does not alter this oscillation. No sum over the entire graph appears.

## Elementary bounded-density Poincare comparison

Let a probability density \(w\) on one group factor, relative to
normalized Haar measure \(dH\), satisfy \(0<m\le w\le M\).
Haar has spectral gap \(c_*\) on the full scalar \(L^2\) space.
For any smooth complex \(f\), using its Haar mean as a variance trial
constant gives
\[
\begin{aligned}
\operatorname{Var}_{wH}f
&=\inf_z\int|f-z|^2w\,dH\\
&\le M\operatorname{Var}_Hf
\le\frac{M}{c_*}\int|\nabla f|^2dH
\le\frac{M}{mc_*}\int|\nabla f|^2w\,dH.
\end{aligned}
\]
This proves the bounded-density comparison directly; no external
stability theorem is assumed. Since \(M/m\le e^\Omega\) by (6), every
conditional one-link measure has Poincare constant
\[
\boxed{\gamma\ge c_*e^{-\Omega}.}
\tag{7}
\]
The compact smooth positive conditional densities ensure finite positive
extrema. Closure extends the estimate to the full Sobolev form domain.

One can avoid a transcendental gate altogether. For an integer
\(m>\Omega\), the elementary inequality \(e^{-x}\ge1-x\) gives
\[
e^{-\Omega}\ge L_m(\Omega):=(1-\Omega/m)^m>0.
\tag{8}
\]
The deterministic choice \(m=\lfloor\Omega\rfloor+1\) is a positive
integer strictly larger than \(\Omega\), including when \(\Omega\) is an
integer. Thus an exact rational lower bound exists for every finite
nonnegative rational \(\Omega\). No unjustified first-order lower bound
is used when \(\Omega\ge1\). For \(0\le\Omega<1\), \(m=1\) reduces (8)
to \(1-\Omega\).

## Factorization applies to every conditional block

Let \(B\) be any subset of graph links and fix arbitrary outside values.
The one-site conditional law of a link \(e\in B\), after all other links
in \(B\) are specified, is exactly the original conditional law with
additional coordinates fixed. Therefore (7) holds with the same
\(\gamma\).

Likewise the influence coefficient between \(e,f\in B\) is bounded by
the original configuration-uniform coefficient \(c_{ef}\). Restricting
the influence matrix to the principal submatrix indexed by \(B\)
cannot increase its nonnegative row sums. Its row bound is still
\(\eta<1\).

For clarity, the variance-factorization proof needs no unearned
uniform mixing premise. For the conditional measure on \(B\), let
\(P_e\) be one-site conditional expectation and
\(L_B=\sum_{e\in B}(P_e-I)\). For bounded functions, maximal coupling
gives
\(\delta_i(P_ef)\le\delta_i f+c_{ei}\delta_e f\) for \(i\ne e\), while
\(\delta_e(P_ef)=0\). Summing coordinate oscillations for
\(I+hL_B\), \(0<h\le1/|B|\), gives contraction
\(1-h(1-\eta)\). Iteration and the semigroup limit give exponential
rate \(1-\eta\). The bound on centered \(L^2\) norms has a finite
prefactor depending on the function and block size. Self-adjoint
spectral calculus rules out smaller positive spectral support;
density of bounded functions gives
\[
\operatorname{Var}_{\mu_B}f\le\frac1{1-\eta}
\sum_{e\in B}
\mathbb E_{\mu_B}\operatorname{Var}(f\mid U_{B\setminus\{e\}}).
\]
Apply (7) to each conditional variance and integrate. For every block,
every exterior configuration and every form-domain function,
\[
\boxed{
\int\sum_{e\in B}|\nabla_ef|^2\,d\mu_B
\ge\gamma(1-\eta)\operatorname{Var}_{\mu_B}f.}
\tag{9}
\]
The empty block is vacuous. For all nonempty blocks the same constants
work regardless of their size. This is the required local-to-global
estimate, obtained from the true corrected vacuum.

## Neutral spectrum, charged cuts and amplitude

Take \(B\) to be the entire finite edge set. Its exact groundstate
transform, valid for all scalar functions, implies
\[
\boxed{
\Delta_{A_\kappa}\ge
\frac{\kappa c_*}{2}(1-\eta)e^{-\Omega}
\ge\frac{\kappa c_*}{2}(1-\eta)L_m(\Omega)>0.}
\tag{10}
\]
The vacuum is unique by the existing positive-groundstate construction.
No positive global curvature is needed in (10).
When the curvature criterion also passes, the larger of the two earned
lower bounds may be reported.

The same constant bounds the charged sector linearly in separation.
Consider a fundamental source at \(s\), its conjugate at \(t\), Gauss
constraints at every vertex, no dynamical fundamental matter, and no
flux-absorbing boundary. For a separating vertex set \(W\), a center
gauge transformation equal to a nonidentity center element inside \(W\)
and identity outside fixes every noncut link. It preserves \(\mu\)
and multiplies each component of the charged matrix function by that
nonidentity phase. Its conditional mean over the cut
\(B=\delta(W)\) is therefore zero.

Equation (9) gives charged energy at least \(\gamma(1-\eta)\) on each
such cut. The boundaries of the graph-distance balls centered at \(s\),
with radii \(0,\ldots,d(s,t)-1\), are pairwise edge disjoint and all
separate the sources. Summing and using the exact interacting-vacuum
groundstate transform gives
\[
\frac{\kappa c_*}{2}(1-\eta)L_m(\Omega)\,d(s,t)
\le E_{s,t}-E_0
\le\frac{\kappa c_*}{2}\,d(s,t).
\tag{11}
\]
The upper bound uses the shortest-path trial matrix \(U_\gamma/\sqrt N\).
It has constant norm one and derivative energy \(c_*d(s,t)\).
Smooth equivariant functions are a form core by averaging smooth
approximants over the compact gauge group.

For the specifically defined Hamiltonian amplitude
\(C_\gamma(T)=\langle\psi_0U_\gamma/\sqrt N,
e^{-T(A_{\kappa,\mathrm{charged}}-E_0)}
\psi_0U_\gamma/\sqrt N\rangle\), \(T\ge0\),
its positive spectral measure and Jensen's inequality yield the
corresponding two-sided area exponent with lower slope (10) and upper
slope \(\kappa c_*/2\).
This uses dimensionless Hamiltonian time, physical time \(aT\).
No identification with an unspecified Euclidean lattice action or
asymptotic string-tension limit is asserted.

## Exact SU(3) example that the curvature criterion misses

For simple cubic spatial graph families of girth at least four, four-edge
cycles and incidence at most four, choose \(b=1\), \(\kappa=45\).
Then \(g=4/2025\), \(E_{\min}=16/3\), and choose
\[
A=\frac{16}{25},\qquad B=\frac38,\qquad r=\frac9{20}.
\]
The correction inequalities are
\[
B(A+r)^2=\frac{35643}{80000}<\frac9{20},
\qquad 2B(A+r)=\frac{327}{400}<1.
\]
Thus (1) is the actual vacuum. Its influence and oscillation bounds are
\[
\eta_*=\frac2{75},\quad
\eta=\frac2{75}+\frac{27}{40}=\frac{421}{600}<1,
\]
\[
\Omega_*=\frac4{225},\quad
\Omega=\frac4{225}+\frac{27}{80}
=\frac{1279}{3600}<1.
\]
Using \(m=1\), equation (10) gives the exact rational result
\[
\boxed{
\Delta_{A_{45}}\ge
\frac{179}{20}e^{-1279/3600}
\ge\frac{415459}{72000}>0.}
\tag{12}
\]
The rational floor is approximately \(5.77026\) in dimensionless
energy units. The charged source bound has the same lower slope per
graph edge and upper slope \(30\).

In contrast, the seed Hessian bound is \(2/675\), so the global-curvature
candidate \(\rho_0-2(h_*+r)=3/4-2(2/675+9/20)\) is negative.
The curvature sufficient criterion supplies no positive gap here;
the factorization and conditional-density argument earns (12).
Neither conclusion states that the true gap vanishes where one
sufficient criterion fails.

## Remaining limits

The gain is a volume-independent finite-graph estimate for an interacting
compact-link Hamiltonian in a larger proven window. It does not remove
the fixed-point smallness condition or construct the ultraviolet theory.
A continuum bridge still requires
cutoff-uniform convergence, surviving finite-energy excitations, and
identification of a nontrivial relativistic Yang–Mills theory.
