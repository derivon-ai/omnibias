# Static-source confinement from the actual SU(2) vacuum

A replayed Fourier vacuum certificate supplies a uniform conditional-curvature
bound. The charged-sector center parity and edge-disjoint distance cuts then
give a **linear static-source lower bound**, with an independent Wilson-path
upper bound. All spins are included and the subtraction uses the actual
neutral vacuum. This is a finite-graph strong-coupling theorem, not a
continuum or asymptotic string-tension claim.

## A checked graph and source pair

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.vacuum_fourier import (
    su2_vacuum_fourier_bounds,
    su2_vacuum_fourier_family,
)
from omnibias.geometry.gauge.transfer.charged_confinement import (
    su2_static_confinement_bounds,
    su2_static_confinement_family,
    replay_su2_static_confinement_certificate,
)

vacuum = su2_vacuum_fourier_bounds(
    4, [(0, 1), (1, 2), (2, 3), (3, 0)],
    plaquettes=[[1, 2, 3, 4]], kappa=64,
)
charged = su2_static_confinement_bounds(vacuum['certificate'], 0, 2)
assert charged['witness']['graph_distance'] == 2
assert charged['witness']['static_energy_enclosure'] == ['28', '48']
assert charged['witness']['path'] == [1, 2]
assert replay_su2_static_confinement_certificate(charged['certificate'])
```

The witness contains all vertex distance labels, each separating vertex set,
its complete edge cut, cut multiplicities, source parity, the oriented
shortest path, and the exact curvature inherited from the verified parent.
Curvature and honesty flags are never caller inputs. The charged Hamiltonian
has one external fundamental and one antifundamental source, Gauss law at
every other vertex, unit electric weights, and zero added source rest energy.
Graphs must be connected and source vertices distinct. Parallel edges are
retained as separate kinetic degrees of freedom.

## A uniform structural family

```python
vacuum_family = su2_vacuum_fourier_family(64, 4)
charged_family = su2_static_confinement_family(vacuum_family['certificate'])
assert charged_family['finite_graph_family_confinement']
assert charged_family['witness']['linear_energy_coefficients'] == ['14', '24']
assert not charged_family['asymptotic_string_tension_claim']
assert not charged_family['continuum_claim']
assert replay_su2_static_confinement_certificate(charged_family['certificate'])
```

This means 14 times graph distance is a lower bound on the dimensionless
static energy, and 24 times distance is an upper bound, for **every connected
finite graph in the parent's stated structural class** and every pair of
distinct source vertices. Ordinary cubic finite volumes have plaquette
incidence at most four and satisfy this family at the displayed coupling.
The positive finite_graph_family_confinement flag has this precise scope.
It does not assert existence of an infinite-volume static potential or of a
limiting string tension. The continuum and formal-verification flags stay
false.

The conditional-curvature proof does not need the separate Dobrushin
factorization gate. If the direct graph Fourier gate passes but a coarser
structural-family gate does not, an explicit graph result can still pass
while its family flag remains false. A failed, malformed, or wrong-scope
upstream certificate is refused. Full replay checks the complete Fourier
parent and regenerates every distance cut, source condition, energy bound,
scope and honesty field; rehashing edited fields cannot pass.

The full analytic implication follows. Strong-coupling lattice confinement
and the Hamiltonian setting have classical precedents in
[Wilson (1974)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.10.2445)
and [Kogut--Susskind (1975)](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.11.395).
No originality claim is made for this regime or for the proof method.

## Full analytic implication

11 September 2026. This independently checked argument uses the actual
log-vacuum Hessian estimate in [the Fourier vacuum theorem](gauge-vacuum-fourier.md).
It gives a distance-growing charged energy bound, uniform in finite graph
volume, in the explicitly certified strong-coupling regime. It also gives
two-sided exponential bounds for a precisely defined Hamiltonian
Wilson-line amplitude. It establishes no continuum theory, novelty or
asymptotic string-tension limit.

## Model and the stronger input

Work on a finite connected graph with independent SU(2) edge variables and
the unweighted electric convention

\[
A_\kappa=\frac\kappa2\sum_e(-\Delta_e)
+\frac2\kappa\sum_pv_p(2-\chi_{1/2}(U_p)).
\]

This is the dimensionless Hamiltonian \(aH^{\rm phys}\). Each group metric
is the round three-sphere of radius two, with
\(\operatorname{Ric}_e=\tfrac12g_e\). The potential cycles and coupling
must satisfy the assumptions of the parent Fourier estimate.

Let \(E_0\) be the true neutral ground energy and \(\psi_0>0\) its normalized
gauge-invariant groundstate. Set \(S=\log\psi_0\) and
\(d\mu=\psi_0^2\,dU\). The hypothesis used below is the pointwise estimate

\[
\operatorname{Hess}S\le hI,\qquad
\rho=\frac12-2h>0.
\tag{1}
\]

The Fourier proof supplies \(h=2r/3\) on its radius-\(r\) ball. Its
explicit choice \(r=3/64\) therefore gives \(h=1/32\) and \(\rho=7/16\).

This is stronger input than a neutral spectral gap alone. Its usefulness
here is that it controls the conditional measure of **every** set of links,
including large separating cuts.

Place distinct external fundamental and antifundamental sources at \(s,t\).
Impose Gauss law at every vertex, including graph boundaries. There is no
dynamical fundamental matter and no boundary where flux can end freely.
Represent the charged Hilbert space by matrix-valued functions

\[
\Psi(g\cdot U)=g_s\Psi(U)g_t^{-1}
\]

with Haar Hilbert--Schmidt norm. The Hamiltonian acts componentwise; call
its restriction \(A_{\kappa,s,t}\), its lowest energy \(E_{s,t}\), and
\(V_\Gamma(s,t)=E_{s,t}-E_0\).

Multiplication by \(\psi_0\) preserves charged covariance. Its exact
groundstate transform gives the variational identity

\[
V_\Gamma(s,t)
=\frac\kappa2\inf_{F\ne0}
\frac{\displaystyle\int\sum_e\|\nabla_e F\|_{\rm HS}^2\,d\mu}
{\displaystyle\int\|F\|_{\rm HS}^2\,d\mu},
\quad F(g\cdot U)=g_sF(U)g_t^{-1}.
\tag{2}
\]

The infimum is over the charged covariant Sobolev form domain.
Centering is at the actual \(E_0\), not at a numerical approximation to it.

## Every conditional link block has the same curvature floor

Fix a nonempty edge subset \(B\), and hold all exterior links fixed.
Its conditional measure is

\[
d\mu_B(U_B\,|\,U_{B^c})
=Z_B(U_{B^c})^{-1}
e^{2S(U_B,U_{B^c})}\,dU_B.
\]

The slice \(SU(2)^B\) is a totally geodesic product submanifold.
The Hessian of restricted \(S\) is the corresponding principal block of
the full covariant Hessian. No second fundamental form is added.
Its Ricci tensor is still \(\tfrac12 I\), independent of \(|B|\).
Consequently its weighted Bochner tensor is at least \(\rho I\).

The integrated Bochner argument on this compact connected slice proves
the conditional Poincare inequality

\[
\int\sum_{e\in B}|\nabla_e f|^2\,d\mu_B
\ge\rho\,\operatorname{Var}_{\mu_B}(f)
\tag{3}
\]

for every exterior configuration and every function in its form domain.
It applies separately to the real and imaginary parts of each matrix
component. The normalization \(Z_B\) is constant along the slice and
contributes no derivatives.

This step uses the actual \(\psi_0^2\), not a presumed local Gibbs formula
for it. It also does not require a variance-factorization theorem:
the Hessian estimate already controls arbitrary block sizes directly.

## A separating cut forces conditional mean zero

Let \(W\) be a vertex set containing \(s\) but not \(t\), and let
\(B=\partial W\) be its edge boundary. Apply the center-valued gauge
transformation

\[
g_v=\begin{cases}-I,&v\in W,\\ I,&v\notin W.\end{cases}
\]

An edge with both endpoints inside \(W\), or both outside it, is unchanged.
Every cut edge is sent to \(-U_e\), independently of its chosen orientation.
Thus the transformation fixes the entire exterior configuration \(U_{B^c}\)
and acts as simultaneous central inversion on the cut coordinates.

The positive vacuum is gauge invariant, so this transformation preserves the
conditional density \(d\mu_B\); Haar measure is preserved as well. On the
charged function, its action is exactly

\[
F(g\cdot U)=-F(U),
\]

because precisely one of the two source vertices lies in \(W\).
Changing variables in the conditional integral gives
\(\int F\,d\mu_B=-\int F\,d\mu_B\), hence

\[
\int F\,d\mu_B=0.
\tag{4}
\]

Apply (3) componentwise, use (4), and then integrate over the exterior links:

\[
\int\sum_{e\in\partial W}\|\nabla_eF\|_{\rm HS}^2\,d\mu
\ge\rho\int\|F\|_{\rm HS}^2\,d\mu.
\tag{5}
\]

For Sobolev functions, covariance holds almost everywhere and Fubini gives
the conditional statement for almost every exterior configuration. Smooth
charged functions are also dense in the form domain: smoothing by the
product heat semigroup respects the gauge action. Closure therefore extends
the argument from smooth functions without assuming pointwise regularity
of every trial state.

## Disjoint distance cuts give the linear factor

Let \(d=\operatorname{dist}_\Gamma(s,t)\) be ordinary unweighted graph
distance. For \(k=0,\ldots,d-1\), set

\[
W_k=\{v:\operatorname{dist}_\Gamma(s,v)\le k\},\qquad B_k=\partial W_k.
\]

Every \(W_k\) contains \(s\) and excludes \(t\).
Distances from \(s\) at the endpoints of any graph edge differ by at most
one. An edge crossing a level cut therefore joins levels \(k\) and \(k+1\),
and crosses exactly that one cut. The sets \(B_k\) are pairwise disjoint.
Parallel edges remain distinct and do not change this fact.

Sum (5) over these \(d\) cuts. All unused edge energies are nonnegative, so

\[
\int\sum_e\|\nabla_eF\|_{\rm HS}^2\,d\mu
\ge \rho d\int\|F\|_{\rm HS}^2\,d\mu.
\]

Inserting this into (2) proves

\[
\boxed{\quad
V_\Gamma(s,t)\ge\frac{\kappa\rho}{2}
\operatorname{dist}_\Gamma(s,t).
\quad}
\tag{6}
\]

No separately estimated vacuum energy has been subtracted. The entire
vacuum energy disappeared in the exact groundstate transform before the
cut inequalities were applied. Large cut boundaries cause no deterioration:
their conditional curvature floor is independent of their cardinality.

## Upper bound and explicit strong-coupling interval

For a shortest simple path \(\gamma:s\to t\), the charged trial function
\(\Phi_\gamma=\psi_0 U_\gamma/\sqrt2\) has norm one. Its pointwise squared
norm is \(\psi_0^2\). The electric product-rule cross term vanishes because
\(\operatorname{Tr}(U_\gamma^\dagger\nabla_eU_\gamma)=0\), and each used
edge contributes fundamental Casimir \(3/4\). Thus

\[
\langle\Phi_\gamma,
(A_{\kappa,s,t}-E_0)\Phi_\gamma\rangle
=\frac{3\kappa d}{8}.
\tag{7}
\]

The variational principle and (6) give the two-sided bound

\[
\boxed{\quad
\frac{\kappa\rho d}{2}
\le V_\Gamma(s,t)\le\frac{3\kappa d}{8}.
\quad}
\tag{8}
\]

For the explicit Fourier radius \(3/64\), it reads

\[
\frac{7\kappa d}{32}\le V_\Gamma(s,t)\le\frac{3\kappa d}{8}.
\]

The sufficient four-edge-plaquette condition is
\(g d_* b^2\le9/2048\), where \(g=4/\kappa^2\) and \(d_*\) is the
weighted number of plaquettes incident to an edge, not source distance.
In three-dimensional cubic volumes with unweighted plaquettes,
\(d_*\le4\); \(\kappa\ge64\) suffices with \(b=1\).
All spins and all finite volumes meeting these assumptions are covered.
Separating bridges are unnecessary; ordinary bulk cuts now supply the
lower bound.

These are bare lattice charged-minus-vacuum energies. An additional source
self-energy renormalization is not included. In physical units,
with spacing \(a\) and \(R=ad\), (8) bounds the bare energy between
\(\kappa\rho R/(2a^2)\) and \(3\kappa R/(8a^2)\).
The strong-coupling criterion does not remain valid along \(\kappa\to0\).

## A defined Hamiltonian rectangle amplitude

For the normalized path state above, define

\[
C_\gamma(T)=
\langle\Phi_\gamma,
e^{-T(A_{\kappa,s,t}-E_0)}\Phi_\gamma\rangle,\qquad T\ge0.
\tag{9}
\]

This is a scalar amplitude obtained by inserting an open fundamental line,
propagating its static charged sector, and closing with the same line.
It is positive by spectral calculus and has \(C_\gamma(0)=1\).
For a straight spatial path it is the Hamiltonian construction of a
spatial--temporal rectangle. Equation (9) is the definition used here;
no equality with an unspecified isotropic Euclidean Wilson measure is assumed.

Its spectral probability measure is supported above \(\kappa\rho d/2\)
by (6). Therefore
\(C_\gamma(T)\le e^{-\kappa\rho dT/2}\).
That measure has finite energy mean \(3\kappa d/8\) by (7).
Since \(\lambda\mapsto e^{-T\lambda}\) is convex, Jensen's inequality gives
the converse bound
\(C_\gamma(T)\ge e^{-3\kappa dT/8}\).
Thus

\[
\boxed{\quad
e^{-3\kappa dT/8}
\le C_\gamma(T)
\le e^{-\kappa\rho dT/2},
\qquad T\ge0.
\quad}
\tag{10}
\]

These are two-sided exponentials in the rectangle's \(dT\) area variable
for this defined Hamiltonian amplitude. They cover every spin and every
finite propagation time, with constants independent of graph volume.
They do not assert existence of a limiting string tension as distance or
volume tends to infinity.

The time \(T\) is dimensionless and conjugate to \(aH^{\rm phys}\);
\(t_{\rm phys}=aT\). Hence \(dT=R t_{\rm phys}/a^2\).
Identifying (9) with a continuum or Euclidean lattice observable would
require the corresponding transfer/reconstruction theorem and its coupling
and time normalization. No such identification is inferred from a
Landau-gauge data file or an arbitrary Wilson inverse coupling.

## Why this argument does not extend automatically

The decisive input is uniform curvature of every conditional link block,
proved in the parent Fourier smallness regime. A neutral gap alone does not
imply that property. Replacing the conditional estimate by a density
comparison whose exponent counts the entire cut would lose the distance
bound in large volume.

The center transformation also uses exactly the stated charge content and
Gauss law. Dynamical fundamental matter, freely terminating boundary flux,
or a different external representation require another argument.
The proof is formulated on the full product of link groups and does not
silently treat gauge-fixed coordinates as independent cut links.

The result supplies a uniform finite-volume linear confinement bound and
a corresponding amplitude estimate in a known strong-coupling regime.
It does not supply the weak-coupling/ultraviolet induction, a continuum
limit, a relativistic reconstruction, or an interacting continuum
Yang--Mills mass gap. Its analytic proof and any finite arithmetic replay
also remain distinct from actual Lean or Mathlib verification.
