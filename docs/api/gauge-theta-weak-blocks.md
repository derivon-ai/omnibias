# Weak-coupling theta gaps and actual conditional blocks

`omnibias.geometry.gauge.transfer.theta_weak_blocks` proves the following
finite-graph statement. The graph has seven unit-weight original links and
two square plaquettes sharing a link. At every real
\(0<\kappa\le1/64\), its full reduced \(SU(2)^2\) Hamiltonian has gap at
least \(2/5\). The same bound holds on the original physical Hilbert space.
Two specified forest blocks have actual conditional quantum gaps
\(2/15\) and \(4/25\), uniformly over their frozen exteriors **within this
graph**, imposing internal Gauss invariance and retaining every boundary
representation.

The API seals exact rational arithmetic and the canonical one-plaquette
trial-energy source. The operator, form-domain and conditional-measure
arguments below are written analysis, not formally verified Lean theorems.
No supplied or previously computed theta gap is a premise.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.theta_weak_blocks import (
    replay_su2_theta_weak_block_certificate,
    su2_theta_weak_block_gaps,
)

result = su2_theta_weak_block_gaps(Fraction(1, 64))
assert result["status"] == "PASS"
assert result["arithmetic"]["full_reduced_gap_lower"] == "2/5"
assert result["arithmetic"]["conditional_A_quantum_gap_lower"] == "2/15"
assert result["arithmetic"]["conditional_B_quantum_gap_lower"] == "4/25"
assert replay_su2_theta_weak_block_certificate(result["certificate"])

outside = su2_theta_weak_block_gaps(Fraction(1, 32))
assert outside["status"] == "INCONCLUSIVE"
assert outside["arithmetic"]["full_reduced_gap_lower"] is None
assert outside["actual_forest_correlation_lower_verified_in_written_analysis"]
assert not outside["continuum_claim"]
```

Positive exact inputs outside the declared interval return `INCONCLUSIVE`;
this is not an absence-of-gap conclusion. The correlation **lower** bound in
Section 7 has a separate all-positive-coupling scope. Floats and booleans are
rejected. Replay reconstructs the entire payload, including nested evidence,
normalizations, sector restrictions and both successful and inconclusive
decisions.

## 1. Original links, the reduced operator and its vacuum

The oriented original edges, numbered from one, are
\[
 (01),(12),(34),(45),(03),(14),(25).
\]
With normalized Haar measure and \(C=-\Delta_{SU(2)}\), use generators
\(\operatorname{Tr}(T_aT_b)=\delta_{ab}/2\), so \(C_{1/2}=3/4\).
Write
\[
 a=e_1^{-1},\quad b=e_2,\quad s=e_6,\quad
 P=e_5e_3,\quad Q=e_7e_4^{-1},\qquad
 x=s^{-1}aP,\quad y=s^{-1}bQ.                         \tag{1}
\]
Both plaquette words are based at vertex 4. Original gauge-invariant
functions correspond isometrically to simultaneous-conjugation-invariant
functions of \((x,y)\) in product Haar measure. Integrating the five tree
variables gives unit Haar factors, so no volume factor occurs.

Let \(C_\Sigma=C_x+C_y\) and let \(C_d\) be the Casimir for simultaneous
left multiplication of \(x,y\). Direct original-link differentiation gives
\[
 C_\theta=3C_\Sigma+C_d,\qquad
 H_\theta=\frac\kappa2 C_\theta+
       \frac2\kappa\bigl(A(x)+A(y)\bigr),\qquad A(U)=2-\operatorname{Tr}U.
                                                               \tag{2}
\]
These are exact operator/form identities on physical functions. Each of the
three nonshared links of a plaquette contributes one individual Casimir;
the shared link differentiates both words by the same left generator.

Equation (2) also defines an operator on **all** of \(L^2(SU(2)^2)\).
Its closed quadratic form has domain \(H^1\), since
\(3C_\Sigma\le C_\theta\le5C_\Sigma\), and its smooth real potential is
bounded on this compact connected manifold. Standard elliptic positivity
gives a simple strictly positive smooth ground state \(\Phi\), and compact
resolvent. The operator commutes with simultaneous conjugation, so uniqueness
makes \(\Phi\) conjugation invariant. Consequently its ground energy equals
the physical original-graph ground energy, and its normalized lift is the
actual original vacuum. A full reduced spectral lower bound therefore also
bounds physical excitations. It does **not** bound arbitrary non-Gauss
functions on the unreduced seven-link product.

## 2. A trial-energy upper bound of six

The [canonical one-plaquette source](gauge-weak-plaquette.md) proves, in the
four-edge normalization,
\[
 h_{\rm plaq}=2\kappa C+\frac2\kappa A,
 \qquad E_0(h_{\rm plaq})\le\min(3,4/\kappa).                  \tag{3}
\]
The first bound is obtained from the explicit radial trial
\(\exp[-8(1-\cos(\theta/2))/\kappa]\), whose radial Dirichlet representative
lies in \(H^2\cap H^1_0\); the constant trial gives the second. Thus (3)
requires no assumption about a gap. The antipodal cusp of the first class
trial is harmless in the form domain.

Choose a normalized real central one-rotor trial \(f\) with Rayleigh quotient
at most the appropriate bound. For every invariant derivative \(L_a\), Haar
integration gives \(\int fL_af=\frac12\int L_a(f^2)=0\), valid by
approximation for this \(H^1\) trial. On the product \(f(x)f(y)\), the
cross terms in \(C_d\) therefore have zero expectation. The diagonal part
of (2) is exactly four individual Casimirs. This physical product trial gives
\[
 E_0(H_\theta)\le2\min(3,4/\kappa)\le6.                    \tag{4}
\]
The runtime consumes only the sealed upper-energy field of a dynamically
replayed canonical source at the requested coupling. Detached summaries and
its existing one-plaquette gap are not used.

## 3. Complete angular comparison by a positive radial function

Discarding the nonnegative \(C_d\) gives
\[
 H_\theta\ge h_{\rm sep}=h(x)+h(y),\qquad
 h=\frac{3\kappa}{2}C+\frac2\kappa A.                        \tag{5}
\]
Write \(U=\cos\theta+i\sin\theta\,\omega\cdot\sigma\),
\(0<\theta<\pi\), \(\omega\in S^2\). This parametrization includes all
rotor functions, not merely characters. In the spherical harmonic sector
\(\ell\ge0\), the radial operator is
\[
 h_\ell=\frac{3\kappa}{8}
 \left[-\partial_\theta^2-2\cot\theta\,\partial_\theta
             +\frac{\ell(\ell+1)}{\sin^2\theta}\right]
       +\frac4\kappa(1-\cos\theta).                         \tag{6}
\]
Its Friedrichs domain is inherited from the full smooth compact rotor. Put
\(a_0=3\kappa/8\), \(t=4/(\sqrt3\kappa)\) and
\(f_\ell(\theta)=\sin^\ell\theta\,e^{t\cos\theta}\). The radial
function is positive in the interior. Moreover
\(\sin^\ell\theta\,Y_{\ell m}(\omega)\) is a homogeneous harmonic
polynomial in the vector quaternion coordinates, so it has the correct
smooth behavior at both poles. Direct differentiation yields
\[
 \frac{h_\ell f_\ell}{f_\ell}
 =a_0\ell(\ell+2)+\frac{(2\ell+3)\sqrt3}{2}\cos\theta
                   +\frac2\kappa(1-\cos\theta)^2.            \tag{7}
\]
For \(B_\ell=(2\ell+3)\sqrt3/2\), completing the square in
\(z=1-\cos\theta\) gives
\[
 \frac{h_\ell f_\ell}{f_\ell}
 \ge B_\ell-\frac{(12\ell+27)\kappa}{32}.                    \tag{8}
\]
The positive radial ground-state identity proves the corresponding form
bound first for interior test functions and then on the Friedrichs form
domain by closure. No finite angular or radial truncation is used.

In particular the ground radial sector has lower bound
\(3\sqrt3/2-27\kappa/32\). All \(\ell\ge1\) sectors are bounded below
by the \(\ell=1\) ground operator, because their centrifugal terms increase.
Their common lower bound is \(5\sqrt3/2-39\kappa/32\).

## 4. The radial second level and the two-rotor first excitation

For \(\ell=0\), multiplication by \(\sin\theta\) is a radial unitary,
up to a common normalization, onto \(L^2(0,\pi)\). Equation (6) becomes
\[
 -\frac{3\kappa}{8}\partial_\theta^2+
       \frac4\kappa(1-\cos\theta)-\frac{3\kappa}{8},          \tag{9}
\]
with Dirichlet endpoints. Concavity of sine gives
\(1-\cos\theta\ge2\theta^2/\pi^2\). Extending Dirichlet form functions
by zero embeds this interval problem in the half-line oscillator. Its odd
levels are \((4n+3)\sqrt3/\pi\), \(n\ge0\), before the constant shift.
Thus the second radial level is at least
\(7\sqrt3/\pi-3\kappa/8\).

The one-rotor ground is positive and central. Its first nonvacuum level is
either the second radial level or a level with \(\ell\ge1\). Tensor-product
spectral ordering and (8) therefore give
\[
 E_1(h_{\rm sep})\ge\min\left\{
 \left(\frac32+\frac7\pi\right)\sqrt3-\frac{39\kappa}{32},
 4\sqrt3-\frac{33\kappa}{16}\right\}.                       \tag{10}
\]
Here \(E_1\) counts the first excitation on the entire \(SU(2)^2\) space,
including nontrivial simultaneous-conjugation representations. By min–max,
(5) gives the same absolute lower bound for \(E_1(H_\theta)\).

Use \(\sqrt3>19/11\) and \(\pi<22/7\). The latter follows, for example,
from \(\int_0^1 x^4(1-x)^4/(1+x^2)\,dx=22/7-\pi>0\). On
\(0<\kappa\le1/64\), the two lower branches in (10) exceed
\[
 \frac{779}{121}-\frac{39}{2048}>\frac{32}{5},\qquad
 \frac{76}{11}-\frac{33}{1024}>\frac{32}{5}.                  \tag{11}
\]
Combining (4) and (11) proves the conservative uniform result
\[
 \operatorname{gap}(H_\theta;L^2(SU(2)^2))\ge\frac25.         \tag{12}
\]
No vacuum gap is used in the comparison, and no large-energy tail is omitted.

## 5. Exact conditional fibers and original kinetic weights

Partition original links into
\[
 A=\{e_1,e_2,e_6\},\qquad B=\{e_3,e_4,e_5,e_7\}.
\]
The internal vertices are \(1\) for \(A\), and \(3,5\) for \(B\).
Boundary vertices are \(0,2,4\). Fixing \(B\), the transformation
\((a,b,s)\leftrightarrow(x,y,s)\) from (1) preserves product Haar measure:
each inverse transformation is a fixed left/right translation. The internal
gauge at vertex 1 acts freely only on the redundant variable \(s\).
Consequently its invariant conditional functions are **arbitrary** functions
of \((x,y)\); no simultaneous-conjugation constraint is imposed on this
conditional space. Original derivatives give
\[
 C_A=C_\Sigma+C_d.                                          \tag{13}
\]
For fixed \(A\), the two internal bivalent gauges of \(B\) leave the path
products \(P,Q\). Each path has two original edges, hence
\[
 C_B=2C_\Sigma.                                             \tag{14}
\]
In both cases the conditional density is exactly
\(\Phi(x,y)^2\,dx\,dy\). The normalization is one independently of the
frozen exterior, since \(\Phi\) was normalized in product Haar measure.
These are conditionals of the **actual interacting theta vacuum**, not
ground states of independently frozen magnetic Hamiltonians.

For cotangent vectors \(v_x,v_y\), simultaneous-left differentiation gives
\(|v_x+v_y|^2\le2(|v_x|^2+|v_y|^2)\); thus \(C_d\le2C_\Sigma\) as
forms, and the following are pointwise carré-du-champ inequalities:
\[
 C_A\ge\frac13C_\theta,\qquad C_B\ge\frac25C_\theta.        \tag{15}
\]
They remain valid when integrated against the positive density \(\Phi^2\).

## 6. Actual conditional Poincare and quantum gaps

The exact ground-state transform of (2) is
\[
 \langle\Phi f,(H_\theta-E_0)\Phi f\rangle
 =\frac\kappa2\int\Gamma_\theta(f)\,\Phi^2.
\]
Equation (12) bounds this by \((2/5)\operatorname{Var}_{\Phi^2}f\)
from below on the complete reduced scalar form domain. Applying (15) gives
\[
 \frac\kappa2\int\Gamma_A(f)\,d\mu_A\ge\frac{2}{15}
        \operatorname{Var}_{\mu_A}f,\qquad
 \frac\kappa2\int\Gamma_B(f)\,d\mu_B\ge\frac{4}{25}
        \operatorname{Var}_{\mu_B}f.                         \tag{16}
\]
Accordingly the diffusion/Poincare gaps are \(4/(15\kappa)\) and
\(8/(25\kappa)\). The quantum gaps in (16) use dimensionless Hamiltonian
units. Internal Gauss invariance removes only the indicated internal gauge
orbits; all boundary flux representations survive. Smooth functions are a
core, and the Haar coordinate reductions and positive smooth density extend
the estimates to their closed physical form domains.

## 7. A separately earned correlation obstruction at every coupling

The two blocks are forests. A gauge-invariant marginal on a forest is product
Haar, because the vertex gauge action is transitive on its link assignments.
Let \(u=s^{-1}a\), a function of \(A\), and retain \(P\), a function of
\(B\). Their characters both have mean zero and variance one. Averaging the
actual vacuum measure under the boundary gauge at vertex 0 gives the exact
character identity
\[
 \mathbb E[\chi(u)\chi(P)]=\frac12\mathbb E\chi(uP)
                          =\frac12\mathbb E\chi(x).         \tag{17}
\]
Indeed that gauge sends \(u\mapsto uh^{-1}\), \(P\mapsto hP\), and
\(\int\chi(uh^{-1})\chi(hP)\,dh=\chi(uP)/2\).

Let \(\delta(A,B)\) be maximal correlation on the unrestricted \(L^2\)
spaces of the two forest marginals.
Uniqueness and exchange symmetry of (2) give
\(\mathbb E A(x)=\mathbb E A(y)\). Positivity of kinetic energy and (4)
imply \(\mathbb E[A(x)+A(y)]\le\kappa E_0/2\le3\kappa\).
Consequently, for **every** \(\kappa>0\),
\[
 \delta(A,B)\ge\frac12|\mathbb E\chi(x)|
       \ge\max\left(0,1-\frac{3\kappa}{4}\right).            \tag{18}
\]
This is a lower bound. It cannot be used as an overlap upper bound or a
contraction certificate. In particular the same fixed split has correlations
approaching one as \(\kappa\downarrow0\), despite the conditional gaps (16).
At the default coupling the certified lower bound is \(253/256\).

This is not automatically the angle of projections restricted to globally
physical functions. A globally gauge-invariant function depending only on
one forest is constant, by the same transitive gauge action. The unrestricted
marginal spaces used in (17)–(18) retain boundary-dependent observables and
must not be exchanged for that smaller space.

The conditional laws of this two-face graph have the exact special charts
above. Adding plaquettes in an ambient lattice changes its true vacuum and
its conditional density. Neither (16) nor (18) identifies those larger-graph
conditionals, proves an exterior-uniform ambient theorem, or establishes
spatial-volume, refinement or continuum control.
