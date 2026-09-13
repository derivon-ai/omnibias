# SU(2) Wilson large-field energy and moment certificates

12 September 2026. This note proves an all-coupling, all-spin first-moment
and local form estimate for the actual quantum vacuum on finite periodic
three-dimensional cubic lattices. Its constants are independent of volume.
It does not prove a weak-coupling mass gap, exponentially small polymers,
exterior-uniform conditionals, or a continuum limit. The variational and
localization methods are standard; no novelty or formal-verification claim
is made.

## 1. Hamiltonian, geometry and why a noninvariant trial is allowed

Use the microscopic Hamiltonian convention

\[
\widehat H_\kappa=
\frac\kappa2 C+\frac2\kappa\sum_{p\in P}A_p,\qquad
A_p=2-\operatorname{Tr}U_p,\qquad
C=-\sum_{e\in E}\Delta_e,\qquad \kappa>0.
\tag{1}
\]

The SU(2) metric has \(C_{1/2}=3/4\). Each elementary square uses four
distinct original link variables, with their actual orientations. The
public family is the isotropic periodic cubic torus of integer side
\(L\ge3\), with \(|E|=|P|=3L^3\), no external sources and Gauss law at
every vertex. The first trial estimate below applies to more general
finite square graphs, but the per-plaquette conclusion uses the stated
cubic symmetry. The physical Hamiltonian is \(\widehat H_\kappa/a\)
at spatial spacing \(a\); \(\kappa=g_H^2\) is the microscopic Hamiltonian
coupling, not an automatically matched bare Euclidean coupling.
The canonical Hamiltonian formulation originates with
[Kogut and Susskind (1975)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.11.395).

On the compact connected manifold \(SU(2)^E\), (1) has a lowest scalar
eigenfunction. Compact Sobolev embedding and the variational principle
supply a minimizer; replacing it by its absolute value cannot increase
the form. Elliptic regularity and the strong positivity principle give
a smooth strictly positive normalized minimizer \(\psi_0\). It is unique
up to a scalar. One direct uniqueness proof uses its groundstate form:
for any second smooth lowest eigenfunction \(\phi\),

\[
0=\langle\phi,(\widehat H_\kappa-E_0)\phi\rangle
=\frac\kappa2\int\psi_0^2
   \left|\nabla(\phi/\psi_0)\right|^2dU,
\]

so \(\phi/\psi_0\) is constant. Gauge transformations preserve the operator
and positivity, hence preserve the normalized positive eigenfunction.
Thus the scalar bottom equals the neutral gauge-invariant physical bottom.

Consequently **any** scalar trial function bounds this physical \(E_0\)
from above. The trial below need not satisfy Gauss law itself. We do not
assert that gauge-projecting an arbitrary trial preserves its Rayleigh
quotient, or identify its probability density with the actual vacuum.

## 2. A one-link mean bound without a transcendental remainder

Write \(U=q_0I+i\boldsymbol q\cdot\boldsymbol\sigma\), where
\(q_0^2+|\boldsymbol q|^2=1\), and \(\chi(U)=2q_0\).
For \(c>0\), let \(\nu_c\) be normalized Haar measure weighted by
\(e^{cq_0}\), and let \(m(c)=\mathbb E_{\nu_c}q_0\).
Symmetry of Haar measure gives \(0<m(c)<1\). Differentiating the finite
compact integral and integrating the spherical divergence gives

\[
m'(c)=\mathbb E_c q_0^2-m(c)^2,\qquad
c\,\mathbb E_c(1-q_0^2)=3m(c),
\]

and therefore

\[
m'=1-m^2-\frac{3m}{c}.
\tag{2}
\]

The metric factors cancel: \(\Delta q_0=-3q_0/4\) and
\(|\nabla q_0|^2=(1-q_0^2)/4\).

For \(c\ge3/2\), compare (2) with \(b(c)=1-3/(2c)\). At contact,

\[
\left(1-b^2-\frac{3b}{c}\right)-b'
=\frac{3}{4c^2}>0.
\tag{3}
\]

At \(c=3/2\), \(m>0=b\). The scalar differential comparison therefore
gives \(m(c)\ge b(c)\) for all \(c\ge3/2\). Explicitly,
\(d=m-b\) satisfies
\(d'=-(m+b+3/c)d+3/(4c^2)\); its integrating-factor formula keeps it
positive. For \(0<c\le3/2\), \(m\ge0\) supplies the same inequality.
Thus, for every \(c>0\),

\[
\boxed{1-m(c)\le\frac{3}{2c}.}
\tag{4}
\]

This is an analytic inequality, not a floating Bessel-ratio enclosure.
The finite arithmetic certificate never evaluates \(m\), an exponential,
or a Bessel function.

## 3. Volume-uniform variational energy

Take the product-link trial

\[
\Phi_t(U)=Z_t^{-1/2}\prod_{e\in E}e^{t\chi(U_e)},
\qquad t>0.
\tag{5}
\]

Its squared density is a product of the one-link measures in Section 2
with \(c=4t\). Integration by parts gives

\[
\mathbb E_{\Phi_t^2}|\nabla\chi|^2
=\mathbb E_c(1-q_0^2)=\frac{3m(c)}{4t}.
\]

Therefore its kinetic energy is
\(3\kappa t\,m(c)|E|/8\le3\kappa t|E|/8\).
Central symmetry gives \(\mathbb E_c U=m(c)I\) and
\(\mathbb E_c U^{-1}=m(c)I\). Independence of the four **distinct**
link variables of each square gives

\[
\mathbb E_{\Phi_t^2}\operatorname{Tr}U_p=2m(c)^4.
\]

Since \(0\le m\le1\), equations (4)--(5) imply

\[
\begin{aligned}
E_0
&\le\frac{3\kappa t}{8}|E|
  +\frac4\kappa |P|(1-m^4)\\
&\le\frac{3\kappa t}{8}|E|+\frac6{\kappa t}|P|.
\end{aligned}
\tag{6}
\]

For positive edge and plaquette counts, optimizing this elementary upper
bound gives \(E_0\le3\sqrt{|E||P|}\). The rational choice \(t=4/\kappa\)
suffices for

\[
E_0\le\frac32(|E|+|P|).
\tag{7}
\]

On the periodic cubic family, (7) is \(E_0\le3|P|\). The constant Haar
trial separately gives \(E_0\le4|P|/\kappa\). Both bound the same actual
operator; neither requires the current strong-coupling Fourier source
gate or an isolated-plaquette gap.

## 4. Actual per-plaquette moments and large-field events

Put \(d\mu_\kappa=\psi_0^2dU\). Magnetic positivity in (1) gives

\[
\frac2\kappa\sum_p\mathbb E_{\mu_\kappa}A_p\le E_0.
\]

Uniqueness of the positive vacuum preserves translations and cubic axis
permutations. These act transitively on elementary plaquettes, so all
means coincide. Combining the two trials proves

\[
\boxed{\mathbb E_{\mu_\kappa}A_p
\le m_\kappa:=\min\{3\kappa/2,2\},\qquad 0\le A_p\le4.}
\tag{8}
\]

This is uniform in \(L\), including arbitrarily small positive \(\kappa\).
For nontransitive finite square graphs, (7) instead bounds the sum of
plaquette means; it does not by itself bound each one by the cubic
constant. No boundary independence or infinite-volume existence follows
from (8). If a limit state is independently obtained, the bounded local
observable \(A_p\) inherits this inequality under local weak convergence.

For any set \(B\) of \(n\) distinct marked plaquettes, a threshold
\(0<s\le4\), and \(1\le k\le n\), define

\[
K_B=\sum_{p\in B}\mathbf1_{\{A_p\ge s\}}.
\]

Pointwise \(sK_B\le\sum_{p\in B}A_p\). Taking expectations and applying
Markov gives the exact finite-family bounds

\[
\boxed{\mathbb E K_B\le\min\{n,nm_\kappa/s\},\qquad
\mu_\kappa(K_B\ge k)\le\min\{1,nm_\kappa/(ks)\}.}
\tag{9}
\]

A geometric gauge-invariant large-field event can be expressed this way:
if \(U_p\) has class angle \(\theta_p\in[0,\pi]\), then
\(A_p=4\sin^2(\theta_p/2)\). The Riemannian distance from the identity
in the radius-two SU(2) metric is \(2\theta_p\). A certified angle-to-action
threshold may be used; the API takes the rational action threshold itself.

For the event that **all** marked plaquettes are bad, \(k=n\), equation
(9) gives \(\min\{1,m_\kappa/s\}\), not its \(n\)-th power.
For \(\kappa=1/100,s=1/2,n=8\), the mean action bound is \(3/200\),
the expected bad count bound is \(6/25\), and the all-eight probability
bound is \(3/100\). None of these is an independence estimate.

## 5. Exact localized-groundstate and IMS form estimates

For any bounded real Lipschitz gauge-invariant \(F\), the actual
groundstate transform gives

\[
\mathcal E_0(F):=
\langle F\psi_0,(\widehat H_\kappa-E_0)F\psi_0\rangle
=\frac\kappa2\int\sum_e|\nabla_eF|^2d\mu_\kappa.
\tag{10}
\]

First establish this on smooth functions by integration by parts, and
then approximate in the Sobolev form norm. On each fixed finite graph
the positive smooth density is bounded above and below, so this form
closure introduces no singular-measure assumption.

For one plaquette, differentiating its four distinct links gives the
pointwise identity

\[
\sum_e|\nabla_eA_p|^2=4A_p-A_p^2.
\tag{11}
\]

Each individual link derivative contributes \(1-(\operatorname{Tr}U_p)^2/4\);
multiplication and inversion act isometrically on its tangent space.
If \(\bar a=\mathbb E A_p\), variance positivity and (8) imply
\(\mathbb E(4A_p-A_p^2)\le4\bar a-\bar a^2\le m_\kappa(4-m_\kappa)\),
because \(0\le\bar a\le m_\kappa\le2\). Hence

\[
\boxed{\mathcal E_0(A_p)
\le\frac\kappa2m_\kappa(4-m_\kappa).}
\tag{12}
\]

For \(F_B=n^{-1}\sum_{p\in B}A_p\), every edge belongs to at most
\(d_B:=\min\{4,n\}\) marked squares. Cauchy--Schwarz at each edge gives

\[
\sum_e|\nabla_eF_B|^2
\le\frac{d_B}{n^2}\sum_{p\in B}\sum_e|\nabla_eA_p|^2,
\]

so

\[
\boxed{\mathcal E_0(F_B)
\le\frac{\kappa\,d_B}{2n}m_\kappa(4-m_\kappa).}
\tag{13}
\]

Centering either observable does not change its form energy.
For a scalar cutoff \(f\) with Lipschitz constant \(1/w\), the chain
rule multiplies (12) or (13) by at most \(1/w^2\). A clipped ramp,
zero below an action threshold and one above a second threshold
separated by \(w\), is one such gauge-invariant localization.

For a finite Lipschitz partition \(\sum_j\chi_j^2=1\), expansion of
the product gradients proves the IMS identity

\[
\sum_j\langle\chi_j u,\widehat H_\kappa\chi_j u\rangle
=\langle u,\widehat H_\kappa u\rangle+
 \frac\kappa2\int\sum_{j,e}|\nabla_e\chi_j|^2|u|^2dU.
\tag{14}
\]

In particular \(u=\psi_0\) yields an exact controlled localization cost.
One must retain the derivatives of every partition member; a ramp and
\(\sqrt{1-\text{ramp}^2}\) need not have uniformly bounded derivatives.

At weak coupling, (12) is \(O(\kappa^2)\), and (13) is
\(O(\kappa^2/n)\) when the incidence cap stays fixed. They are actual
upper form bounds. They do not give a lower bound on the variance of
these observables. The physical energy divides them by the spatial
spacing. They could feed the variance/energy survival corollary in a
scale-compatible reconstruction argument
only after the matching actual variance floor and physical scaling
have been established.

## 5.1. An actual gauge-invariant small-field localization

Fix an embedded open cubic block of side \(b\ge1\), with
\(n=3b^2(b+1)\) elementary faces, inside a periodic torus with
\(L\ge\max\{3,b+1\}\). For \(0<\varepsilon<2\pi\), put

\[
S=\sum_{p\subset B}A_p,\qquad
s=\left(\frac{7\varepsilon}{44b}\right)^2,\qquad
d_B=\min\{4,n\}.
\tag{15}
\]

Choose the Lipschitz function
\(\theta(x)=0\) for \(x\le s/2\),
\(\theta(x)=\pi(x-s/2)/s\) for \(s/2<x<s\), and
\(\theta(x)=\pi/2\) for \(x\ge s\). Define

\[
\chi_{\mathrm g}=\cos\theta(S),\qquad
\chi_{\mathrm b}=\sin\theta(S).
\tag{16}
\]

Both are gauge invariant, take values in \([0,1]\), and satisfy
\(\chi_{\mathrm g}^2+\chi_{\mathrm b}^2=1\).
Their nonzero good region has \(S<s\), and the closed support of
\(\chi_{\mathrm g}\) has \(S\le s\). Since every \(A_p\ge0\), every
configuration in this closed support satisfies \(A_p\le s\) at every
face of the block.

The following finite axial-tree construction therefore gives a representative with every block link logarithm of
norm at most \(\varepsilon\). Explicitly, set all \(z\)-links,
all \(y\)-links at \(z=0\), and all \(x\)-links at \(y=z=0\) to the
identity. Each remaining link is a product of at most \(2b\)
conjugated plaquette holonomies. The elementary distance bound
\(d(I,U)\le\pi\sqrt{2-\operatorname{Tr}U}\le(22/7)\sqrt{2-\operatorname{Tr}U}\)
then gives \(2b(22/7)\sqrt{s}=\varepsilon\).
This assertion concerns a representative of each orbit, rather than
support in one product of link balls in the original variables.

Because \(\chi_{\mathrm b}=0\) for \(S\le s/2\), Markov's inequality
gives the actual localized norm bound

\[
\|\chi_{\mathrm b}\psi_0\|^2
\le\min\left\{1,\frac{2n m_\kappa}{s}\right\},\qquad
\boxed{\|\chi_{\mathrm g}\psi_0\|^2
\ge v_{\mathrm g}:=\max\left\{0,1-\frac{2n m_\kappa}{s}\right\}.}
\tag{17}
\]

The derivative cost retains both partition members exactly:
\[
\Gamma(\chi_{\mathrm g})+\Gamma(\chi_{\mathrm b})
=(\theta'(S))^2\Gamma(S),\qquad
|\theta'|\le\frac{\pi}{s}\le\frac{22}{7s}
\quad\text{almost everywhere}.
\]
Equation (13), multiplied by \(n^2\), yields
\(\mathbb E_{\mu_0}\Gamma(S)\le n d_B m_\kappa(4-m_\kappa)\).
Applying (14) to the actual vacuum, and subtracting \(E_0\) times
the identity \(\|\chi_{\mathrm g}\psi_0\|^2+
\|\chi_{\mathrm b}\psi_0\|^2=1\), proves

\[
\begin{aligned}
&\langle\chi_{\mathrm g}\psi_0,
        (\widehat H_\kappa-E_0)\chi_{\mathrm g}\psi_0\rangle+
 \langle\chi_{\mathrm b}\psi_0,
        (\widehat H_\kappa-E_0)\chi_{\mathrm b}\psi_0\rangle\\
&\hspace{1cm}\le
\boxed{\mathcal C_B:=
 \left(\frac{22}{7s}\right)^2
 \frac{\kappa}{2}\,n d_B\,m_\kappa(4-m_\kappa).}
\end{aligned}
\tag{18}
\]

This is a bound on the sum of the two vacuum-relative forms, not an
uncontrolled subtraction of the extensive energy. Each summand is
nonnegative. If \(v_{\mathrm g}>0\), the normalized good state is
therefore a physical gauge-invariant trial satisfying

\[
\frac{\langle\chi_{\mathrm g}\psi_0,\widehat H_\kappa
                    \chi_{\mathrm g}\psi_0\rangle}
     {\|\chi_{\mathrm g}\psi_0\|^2}
\le E_0+\frac{\mathcal C_B}{v_{\mathrm g}}.
\tag{19}
\]

The piecewise linear \(\theta\) is legitimate in the form domain by
the Lipschitz chain rule. Equivalently one may use smooth approximants
and pass to the Sobolev limit; no differentiability assertion at its
two breakpoints is needed.

For the rational weak-coupling scaling
\(\kappa=t^5,\ \varepsilon=t^2,\ 0<t\le1\), define
\(c_b=(7/(44b))^2\). Then \(s=c_b t^4\) and
\(m_\kappa=3t^5/2\). Equations (17)--(18) imply

\[
\boxed{
\|\chi_{\mathrm g}\psi_0\|^2
 \ge\max\left\{0,1-\frac{3n}{c_b}t\right\},\qquad
\mathcal C_B\le
 3n d_B\left(\frac{22}{7c_b}\right)^2t^2.}
\tag{20}
\]

For every fixed block, the good norm tends to one and the total
localization cost tends to zero. The explicit constants grow with \(b\);
the statement alone does not control a block size growing arbitrarily
fast as \(t\to0\). The good state need not be orthogonal to the vacuum,
so neither (19) nor (20) is a spectral-gap or surviving-excitation bound.
They are dimensionless Hamiltonian estimates; physical costs divide by
the spatial spacing.

A stronger, pointwise estimate is available for the same partition.
Cauchy--Schwarz at each original edge and (11) give

\[
\Gamma(S)\le d_B\sum_{p\subset B}\Gamma(A_p)
=d_B\sum_{p\subset B}(4A_p-A_p^2)
\le4d_B S.
\tag{21}
\]

The derivative \(\theta'(S)\) vanishes outside the transition region
\(s/2<S<s\), almost everywhere. It follows pointwise, without using a
vacuum expectation, that

\[
0\le\frac\kappa2
       \left(\Gamma(\chi_{\mathrm g})+\Gamma(\chi_{\mathrm b})\right)
\le
\boxed{\mathcal C_B^{\mathrm{univ}}:=
       \frac{968\,\kappa\,d_B}{49s}.}
\tag{22}
\]

For every state \(u\) in the full quadratic-form domain, including every
physical gauge-invariant state, the IMS identity therefore yields

\[
0\le
 \sum_{j\in\{\mathrm g,\mathrm b\}}
       \langle\chi_j u,\widehat H_\kappa\chi_j u\rangle
       -\langle u,\widehat H_\kappa u\rangle
\le\mathcal C_B^{\mathrm{univ}}\|u\|^2.
\tag{23}
\]

The displayed expectations mean quadratic forms when \(u\) is not in
the operator domain. Lipschitz multiplication preserves that form
domain and, since the cutoffs are gauge invariant, also preserves Gauss
law. This estimate is uniform over all ambient torus sizes admitting
the block; it assumes no particular distribution of \(u\). Equation
(18) remains the sharper vacuum-specific option in regimes where its
expectation bound is smaller.

The closed support of \(\chi_{\mathrm b}\) lies in \(S\ge s/2\).
Thus the actual local magnetic multiplication operator, and hence the
full nonnegative Hamiltonian form on the bad localized state, obey

\[
\frac2\kappa S\ge\frac{s}{\kappa}
 \quad\hbox{on }\operatorname{supp}\chi_{\mathrm b},\qquad
\langle\chi_{\mathrm b}u,\widehat H_\kappa\chi_{\mathrm b}u\rangle
\ge\frac{s}{\kappa}\|\chi_{\mathrm b}u\|^2.
\tag{24}
\]

This is an **uncentered** energy bound. Subtracting the ambient vacuum
energy changes its right-hand coefficient to \(s/\kappa-E_0\), which
need not be positive uniformly in ambient volume. The universal IMS
bound does not supply the missing local vacuum-energy cancellation.

For the explicit growing-block sequence
\(b\in\mathbb N,\ b\to\infty,\ t=b^{-6},\
\kappa=b^{-30},\ \varepsilon=b^{-12}\),
one has \(s=(7/44)^2b^{-26}\) and \(d_B=4\). Consequently,

\[
\boxed{
\mathcal C_B^{\mathrm{univ}}
 \le\frac{3872}{49}\left(\frac{44}{7}\right)^2b^{-4},
\qquad
\frac{s}{\kappa}=\left(\frac7{44}\right)^2b^4.}
\tag{25}
\]

The universal localization cost tends to zero while the uncentered
bad-region potential floor increases. These are dimensionless lattice
estimates on growing blocks. No physical block size, scale iteration,
exterior-conditioned kinetic comparison, or continuum conclusion is
asserted.

This closes an actual, gauge-compatible vacuum localization step.
It does not identify the kinetic operator after choosing the axial
representative with an independent scalar small-link chart operator.
A comparison of that gauge-reduced form, exterior-conditional
estimates, and the subsequent scale iteration still require their own
arguments.

## 6. What remains missing from a large-field expansion

First moments allow perfect correlation. For example, a Bernoulli
mixture can put probability \(p\) on a configuration where every marked
action equals four and probability \(1-p\) where every action is zero.
Its one-plaquette mean is \(4p\), but its all-marked probability is \(p\),
not \(p^n\). On an even periodic cubic lattice, the all-negative central
plaquette configuration is geometrically possible: take

\[
U_i(x)=(-1)^{\,\sum_{j<i}x_j}I.
\]

Every elementary square has holonomy \(-I\). Averaging each of the two
configuration orbits over the vertex gauge group makes the example
gauge invariant. This is a counterexample to inference from moments
alone, **not** a claim that this singular mixture is a Wilson groundstate.
The actual groundstate equation contains additional information that an
exponential estimate would have to use.

Similarly, IMS by itself does not cancel an extensive vacuum energy.
On a state localized to \(r\) bad plaquettes, magnetic positivity gives
an absolute potential floor \(2rs/\kappa\). Subtracting the global
\(E_0\), which is proportional to the whole volume, can destroy that
bound for a local cluster. One needs a justified comparison with the
exterior energy, or genuine conditional/relative estimates, before
calling it a local excitation cost.

A useful next input would be an exterior-uniform conditional estimate
for the **actual** vacuum measure, or a controlled polymer activity
bound that handles shared plaquettes and the Gauss constraints. A
conditional bound must be proved in the same actual measure; neither
the product trial density nor an isolated plaquette's vacuum can replace
it. The all-coupling result (8) supplies a polynomial weak-coupling
starting bound, not that missing exponential estimate.

The parent obligations remain separate: weak-coupling volume-uniform
spectral control, a scale-compatible interacting continuum family,
renormalized observable bounds, full OS positivity/regularity/covariance,
and a surviving finite-energy nontrivial theory. Equation (8) controls
a bounded lattice observable. Multiplication by negative powers of
spacing to form a continuum curvature observable may destroy its bound.
No Clay parent or infinite-volume existence flag is earned here.

## 7. API and exact examples

The public functions are
`su2_wilson_large_field(kappa, *, threshold=1, marked_count=1, required_bad_count=1)`
and `replay_su2_wilson_large_field_certificate`. The function quantifies
over all periodic cubic tori of side \(L\ge3\) with at least
`marked_count` plaquettes, and every set of that many distinct plaquettes.
It does not infer membership of a supplied arbitrary graph.

The coupling and action threshold must be exact integers or
`fractions.Fraction` values, with \(\kappa>0\) and \(0<s\le4\).
Counts must be ordinary positive integers satisfying
`required_bad_count <= marked_count`. Floats, booleans, strings and
invalid domains are rejected.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.wilson_large_field import (
    su2_wilson_large_field,
    replay_su2_wilson_large_field_certificate,
)

result = su2_wilson_large_field(
    Fraction(1, 100),
    threshold=Fraction(1, 2),
    marked_count=8,
    required_bad_count=8,
)
assert result["plaquette_action_mean_upper"] == "3/200"
assert result["large_field_probability_upper"] == "3/100"
assert result["actual_plaquette_mean_bound_verified"]
assert not result["exponential_polymer_bound_verified"]
assert not result["spectral_gap_claim"]
assert replay_su2_wilson_large_field_certificate(result["certificate"])
```

Changing the event to at least one bad plaquette gives the union bound,
while requesting all eight does not exponentiate a single-site probability.

```python
union = su2_wilson_large_field(
    Fraction(1, 100),
    threshold=Fraction(1, 2),
    marked_count=8,
    required_bad_count=1,
)
assert union["large_field_probability_upper"] == "6/25"
assert result["witness"]["arithmetic"]["plaquette_action_form_energy_upper"] == "2391/8000000"
assert result["witness"]["arithmetic"]["marked_average_action_form_energy_upper"] == "2391/16000000"
```

A probability bound equal to one is sound but explicitly noninformative.
It does not invalidate the underlying mean or form estimates.

```python
coarse_bound = su2_wilson_large_field(
    10, threshold=1, marked_count=8, required_bad_count=1,
)
assert coarse_bound["status"] == "PASS"
assert coarse_bound["large_field_probability_upper"] == "1"
assert not coarse_bound["probability_bound_nontrivial"]
```

The certificate records the Hamiltonian normalization, periodic-family
quantifier, actual-vacuum variational argument, trial parameter, mean,
count probability, and width-free localization coefficients. Replay
recomputes every field and refuses rehashed changes of normalization,
family, arithmetic or scope. All values are rational; the runtime does
not evaluate an exponential, a Bessel function, or a numerical vacuum.

This certificate earns actual finite-volume vacuum moment and form bounds
uniform in volume. It does not earn an exterior-uniform conditional law,
a polymer exponential, a gap, a variance lower bound, infinite-volume
existence, a continuum claim, or a Yang--Mills claim. Formal tiers stay
false; any separately checked finite algebra does not verify the analytic
vacuum or ODE argument.

Dedicated regressions include deterministic and seeded one-link
Haar-integral enclosures, exact quaternion gradients, the cubic incidence
count, Markov counterexamples, input guards and canonical replay.
