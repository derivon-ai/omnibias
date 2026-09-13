# Physical blocks and conditional overlap

`omnibias.geometry.gauge.transfer.physical_blocks` consumes an actual,
all-spin SU(2) vacuum on the seven-edge theta graph. Its default certificate
proves a physical block comparison floor \(1/5\), hence a dimensionless
Hamiltonian gap \(3/5\), at \(\kappa=6\). This is a new block-level verification
of an existing finite-graph vacuum. The existing coefficient comparison
already gives a stronger gap on this graph; this API does not claim a better
coupling window or an infinite-volume result.

The generic overlap APIs verify rational implications whose analytic inputs
remain explicit premises. They do not certify a caller's density estimate.
All paths are `CERTIFIED` finite arithmetic plus the written implications
below. No derivative is approximated numerically, and no analytic theorem is
claimed to have been formalized in Lean.

## 1. Physical conditional Poincare theorem

Let a finite connected original-link configuration space be a product of
compact SU(2) groups, with its original product metric, and let

\[
 aH=\frac{\kappa}{2}C+V,\qquad
 \psi_0>0,\qquad \mu=\psi_0^2\,dU,\qquad S=\log\psi_0.
\]

The smooth positive ground state is normalized and gauge invariant. The exact
ground-state transform is

\[
 \langle\psi_0f,(aH-E_0)\psi_0f\rangle
 =\frac{\kappa}{2}\int\sum_e|\nabla_ef|^2\,d\mu.
\tag{1}
\]

Partition original links into disjoint blocks \(B_i\). A vertex is *internal*
to a block if its complete original star lies in that block. Assume:

1. For every exterior configuration, the conditional block measure has
   Poincare constant at least \(\gamma_i>0\) on scalar functions invariant
   under the block's internal vertex gauge group. No condition is imposed
   on its boundary flux representations.
2. The mixed logarithmic-vacuum Hessian satisfies
   \(\|\operatorname{Hess}_{B_i,B_j}S\|\le c_{ij}\), uniformly in the full
   configuration, with \(c_{ij}=c_{ji}\ge0\).
3. The symmetric matrix
   \(H_{ii}=\gamma_i, H_{ij}=-2c_{ij}\) satisfies \(H\ge\delta I\),
   \(\delta>0\).

Then the globally gauge-invariant scalar space satisfies

\[
 \operatorname{Var}_\mu f\le\delta^{-1}
       \int\sum_i|\nabla_{B_i}f|^2\,d\mu,
 \qquad \operatorname{gap}_{\rm physical}(aH)\ge\kappa\delta/2.
\tag{2}
\]

Here is the domain argument that permits an internal-Gauss conditional gap
instead of an unrestricted scalar gap. Let

\[
 A=-\Delta-2\nabla S\cdot\nabla,
 \qquad Au=g-\mu g.
\]

For this fixed compact manifold, a nonuniform Poincare bound from the positive
smooth density ensures existence of the centered Poisson solution. This does
not assume the desired bound \(\delta\). Gauge covariance and uniqueness make
\(u\) globally gauge invariant. At each frozen exterior its block restriction
therefore belongs to the internal-Gauss scalar subspace. The conditional
generator \(A_i\) preserves that subspace, so both \(u\) and \(A_i u\) belong to it. Its scalar spectral inequality \(A_i^2\ge\gamma_iA_i\) gives

\[
 \int(A_i u)^2\,d\mu\ge\gamma_i\int|d_i u|^2\,d\mu.
\tag{3}
\]

Conditional Bochner integration identifies the left side with the sum of
the within-block connection energy and
\(\operatorname{Ric}_i-2\operatorname{Hess}_{ii}S\) on the **exact form**
\(d_i u\). It does not assert a spectral bound on arbitrary one-forms.
In the differentiated global Poisson equation the other-block connection
energies are nonnegative. Setting
\(x_i=\|d_i u\|_{L^2(\mu)}\) and
\(b_i=\|d_i g\|_{L^2(\mu)}\) gives \(Hx\le b\) componentwise, after dividing
the \(i\)-th identity by \(x_i>0\); a zero component satisfies this inequality
directly. A positive definite symmetric matrix with nonpositive off-diagonal
entries has an entrywise nonnegative inverse. Thus \(x\le H^{-1}b\), and

\[
 |\operatorname{Cov}(f,g)|
 \le a^T H^{-1}b,
 \quad a_i=\|d_i f\|_2.
\]

Taking \(f=g\) proves (2); smooth density and closure extend it to the physical
form domain. This is the block and symmetry-restricted version of the
covariance/Poincare comparison mechanism in
[Menz, *A Brascamp–Lieb type covariance estimate*](https://arxiv.org/abs/1402.5160).
The actual source, symmetry restriction and metric identification still need
the separate checks that follow.

## 2. Original theta geometry and all boundary flux

The edges, in order, are

\[
 (01),(12),(34),(45),(03),(14),(25).
\]

Let \(A=\{e_1,e_2,e_6\}\), \(B=\{e_3,e_4,e_5,e_7\}\).
The internal vertices are \(\{1\}\) for \(A\), and \(\{3,5\}\) for \(B\).
Vertices \(0,2,4\) are boundary vertices. A six-edge outer cycle gains the
shared edge \(e_6\), so this refinement adds one genuine loop mode.

Write

\[
 a=e_1^{-1},\quad b=e_2,\quad s=e_6,\quad
 P=e_5e_3,\quad Q=e_7e_4^{-1},\qquad
 x=s^{-1}aP,\quad y=s^{-1}bQ.
\tag{4}
\]

Every physical wavefunction is a function \(\Phi(x,y)\) invariant under
simultaneous conjugation. For fixed \(B\), the map from \(A\) to \((s,x,y)\)
preserves product Haar measure; the internal gauge at vertex 1 acts only on
\(s\). Thus all internally invariant conditional functions are arbitrary
functions of \((x,y)\). Their original electric form is

\[
 C_A=C_x+C_y+C_{\rm simultaneous\ left}\ge C_x+C_y.
\tag{5}
\]

For fixed \(A\), reduction by the internal gauges at 3 and 5 leaves \(P,Q\).
Each is a two-edge path, giving

\[
 C_B=2(C_x+C_y).
\tag{6}
\]

Both conditional densities in these coordinates are proportional to the
**same actual** \(|\Phi(x,y)|^2\), with no exterior-dependent normalization
after the Haar change of coordinates. These are not isolated or frozen-
potential ground states. In applying the conditional inequalities to
arbitrary functions of \(x,y\), we retain all boundary representations;
simultaneous conjugation is not imposed as an extra conditional restriction.

## 3. Source-derived conditional gaps and mixed Hessian

The source is the canonical all-spin adjacent vacuum from
[the cone source](gauge-adjacent-cone-vacuum.md) and
[its coefficient comparison](gauge-adjacent-gap-comparison.md). It proves

\[
 S=S_*+U,\quad S_*={g\over3}(\chi_x+\chi_y),
 \quad g={4\over\kappa^2},\quad N_{\rm original}(U)\le r.
\]

The complete coefficient dual, not a finite spin sample, gives
\(\sum|u_\lambda|\le13r/72\). Since each normalized theta basis function is
bounded by one,

\[
 \operatorname{osc}(2U)\le13r/18.
\]

Relative to product Haar on \(x,y\), the seed density is the product
\(e^{(2g/3)\chi_x}e^{(2g/3)\chi_y}\). A one-group density oscillation is
\(8g/3\). Haar has gap \(3/4\). Tensorization followed by bounded-density
comparison, on the entire scalar space of \(x,y\), gives

\[
 \gamma={3\over4}\exp[-8g/3-13r/18],
 \quad \gamma_A\ge\gamma,\quad\gamma_B\ge2\gamma.
\tag{7}
\]

The original mixed derivative of a once-traversed fundamental character has
operator norm at most \(1/2\) for two unit original-link directions. The seed
cross incidence matrix between the ordered blocks is, up to permutations,

\[
 {g\over6}
 \begin{pmatrix}1&1&0&0\\0&0&1&1\\1&1&1&1\end{pmatrix}.
\]

The unscaled integer matrix has squared singular values \(0,2,6\); consequently the displayed matrix has norm at most
\(5g/12\). The original Fourier Hessian bound for \(U\) is \(2r/3\). Hence

\[
 c_{AB}\le5g/12+2r/3.
\tag{8}
\]

At \(\kappa=6,r=6/25\), use the rational exponential lower bound
\((1-\Omega/4)^4\le e^{-\Omega}\). Then

\[
 \gamma={3\over4}(2383/2700)^4,
 \quad c=557/2700,
 \quad
 \det\left[\begin{pmatrix}\gamma&-2c\\-2c&2\gamma\end{pmatrix}
                 -{I\over5}\right]
 ={27448811209921731753555841\over2510484768720000000000000000}>0.
\]

Both shifted diagonal entries are positive, establishing (2) with
\(\delta=1/5\). Source replay earns the actual vacuum; the source's previously
computed gap is never used as a premise. A failed matrix gate does not erase
a separately earned actual source or overlap consequence.

## 4. Actual joint overlap and projection iteration

Both \(A\) and \(B\) are forests. The marginal of a gauge-invariant probability
measure on any forest is product Haar: gauge transformations act transitively
on its edge configurations and preserve its marginal. The explicit changes
of coordinates (4) give a second verification here. Thus the actual joint
density \(p(A,B)\) has both marginals equal to one relative to product Haar.

The global bound is

\[
 \operatorname{osc}\log p\le\Omega={16g\over3}+{13r\over18}.
\]

Since \(\int p=1\), \(p\ge e^{-\Omega}\ge m>0\), where the API takes the
source's rational exponential minorant. Decompose
\(p=m+(1-m)q\). The nonnegative residual kernel \(q\) has both marginals one;
its conditional Markov operator is an \(L^2\) contraction by Jensen. Therefore
the maximal correlation between the two blocks satisfies

\[
 \delta_{AB}\le1-m.
\tag{9}
\]

At the default source, \(m=(2183/2700)^4\). Let \(P_A,P_B\) be conditional
expectations onto functions of the indicated block and let \(\Pi\) project
onto constants. The standard two-projection argument, restricted to the
centered space, gives

\[
 \|(P_AP_B)^n-\Pi\|\le\delta_{AB}^{2n-1},\qquad
 \operatorname{Var}f\le
 {\mathbb E\operatorname{Var}(f\mid A)+\mathbb E\operatorname{Var}(f\mid B)
  \over1-\delta_{AB}}.
\tag{10}
\]

Indeed \(\|P_AP_B\|\le\delta\) and
\(\|P_AP_BP_A\|\le\delta^2\); iterating proves the first assertion. The
spectral lower bound of \((I-P_A)+(I-P_B)\) is \(1-\delta\), which proves the
second. This is a rate-one **heat-bath** generator. Its gap is not the gap of
the electric Hamiltonian in (1). These statements concern this same finite
joint measure, not a succession of different coarse theories.

## 5. Generic density-error and overlap-transfer gates

For two probability base measures and a normalized joint density \(p\), let
\(p_X,p_Y\) denote its own marginals. Assume

\[
 p_X\ge m_X>0,\quad p_Y\ge m_Y>0,
 \quad\|p-p_Xp_Y\|_{L^2({\rm base})}\le\epsilon.
\]

For centered \(f,g\), Cauchy–Schwarz bounds their covariance by
\(\epsilon\|f\|_{L^2({\rm base})}\|g\|_{L^2({\rm base})}\), hence

\[
 \delta\le\min\{1,\epsilon/\sqrt{m_Xm_Y}\}.
\tag{11}
\]

An \(L^\infty\) residual bound suffices because the base measures are
probabilities. Total variation does not suffice. `conditional_overlap_budget`
rounds \(\sqrt{\epsilon^2/(m_Xm_Y)}\) upward to a dyadic rational. It may be inconclusive
near one even when the exact ratio is smaller. It never earns the premises.

For a larger block suppose there are \(s\ge1\) covering decompositions, each
with the same conditional local Poincare lower bound \(\gamma\) and angle
upper bound \(\delta<1\), uniformly in every exterior. If

\[
 {1\over s}\sum_{j=1}^s(\mathcal E_{A_j}+\mathcal E_{B_j})
 \le(1+1/s)\mathcal E_{\rm larger},
\tag{12}
\]

then averaging (10) and using the local inequalities gives

\[
 \gamma_{\rm larger}\ge{\gamma(1-\delta)\over1+1/s}.
\tag{13}
\]

Pairwise disjoint overlap regions can establish (12); a single arbitrary pair
does not establish it for an arbitrary \(s\). The generic
`conditional_overlap_transfer_budget` leaves (12), the actual conditional
measures, exterior uniformity and compatibility of internal-Gauss form
domains as unverified premises. Iterating rational inequalities does not
construct those premises at the next scale.

## 6. Discriminating controls

**Pinned Gaussian overlap.** On sites \(0,\ldots,L\) with endpoints pinned to
zero, the massless nearest-neighbor Gaussian is a normalized probability.
For balanced wings ending at \(a=(L-w)/2\) and beginning at \(b=(L+w)/2\), the
Green function \(G(i,j)=\min(i,j)-ij/L\) and the Markov property reduce maximal
correlation to the endpoints. Thus

\[
 \delta=\sqrt{G(a,b)^2/[G(a,a)G(b,b)]}={L-w\over L+w}.
\]

For Gaussian linear subspaces the maximal correlation of all square-
integrable functions equals the largest canonical linear correlation: expand
in orthogonal Hermite chaoses, whose higher singular values are products of
the linear ones. Conditional independence reduces the wings to one endpoint
pair. At fixed \(w\), this tends to one as \(L\to\infty\). For the precision
with an added unit mass, its Green function is a product of hyperbolic sines;
the endpoint correlation is at most \(r^w\), where
\(r=(3-\sqrt5)/2<2/5\). Hence \(\delta\le(2/5)^w\), independently of \(L\).
Equivalently the finite inverse is determined by
\(D_0=1,D_1=3,D_n=3D_{n-1}-D_{n-2}\); the tests check the bound with exact
rational Green functions. The implemented artifact caps \(w\le512\) only to
bound serialization size; the displayed theorem has no such restriction.

**Strictly positive rare events.** For \(0<\epsilon\le1/2\), take

\[
 \mu_{00}=1-\epsilon-2\epsilon^3,\quad
 \mu_{01}=\mu_{10}=\epsilon^3,\quad\mu_{11}=\epsilon.
\]

All four probabilities are positive. With \(u=\epsilon+\epsilon^3\),

\[
 \operatorname{TV}(\mu,\mu_X\mu_Y)=2(\epsilon-u^2)\longrightarrow0,
 \qquad
 \delta=1-{\epsilon^3\over u(1-u)}\longrightarrow1.
\]

Thus a small total-variation error cannot supply a volume-independent strict
overlap margin, even with full support.

**Gaussian cycle gap.** Use the probability normalization \(\mu\propto\exp(-x^TQx/2)\) for positive \(Q\). For every \(N\ge3\),

\[
 x^T(m^2I+L_{\rm cycle})x=m^2\sum_i x_i^2+\sum_i(x_i-x_{i+1})^2.
\]

The constant vector attains the lower bound, giving exact gap \(m^2\) for
\(m^2>0\). The massless precision has a zero mode and is not itself a full-
space probability. The normalized sequence \(m_N^2=N^{-2}\) still has gaps
tending to zero. Passing each finite positivity test is therefore insufficient.

**Original-link Gaussian Hessian obstruction.** For a periodic cubic Gaussian
reference with \(N\ge3\), let
\(\Omega_\varepsilon=\sqrt{\operatorname{curl}^*\operatorname{curl}+\varepsilon^2I}\).
Its scalar conditional comparison, apart from \(2/\kappa\), is
\(D_{ii}=\Omega_{ii}, D_{ij}=-|\Omega_{ij}|\).
Axis-constant torons give the same-axis row sum \(\varepsilon\). A cross-axis
Fourier mode \(k=(q,q,0)\),
\(q=2\pi\lfloor N/2\rfloor/N\), has \(s=4\sin^2(q/2)\ge3\) and cross-entry
modulus \( (\sqrt{2s+\varepsilon^2}-\varepsilon)/2\). Fourier inversion and the
triangle inequality give

\[
 \sum_jD_{ij}\le{3\varepsilon-\sqrt{6+\varepsilon^2}\over2}
 \le{3\varepsilon-2\over2}<0
 \quad(0<\varepsilon\le1/2).
\]

The all-ones Rayleigh quotient is negative. Positive diagonal weights cannot
turn this symmetric comparison matrix into a positive definite matrix. This
is a reference-model obstruction to that absolute-Hessian method, not a
theorem ruling out a nonlinear Wilson-vacuum estimate by another method.

**Internal-star full-scalar cap.** A gauge-invariant conditional measure at
a complete internal star has Haar marginal on each incident link. The gauge-
variant function \(\operatorname{Tr}U/2\) has variance \(1/4\) and energy \(3/16\),
so its unrestricted conditional gap is at most \(3/4\). This function is
excluded by internal Gauss invariance. It does not cap (2).

## 7. API and replay

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.physical_blocks import (
    conditional_overlap_budget,
    conditional_overlap_transfer_budget,
    physical_block_controls,
    replay_su2_theta_physical_block_certificate,
    su2_theta_physical_block_gap,
)

actual = su2_theta_physical_block_gap()
assert actual["physical_gap_lower"] == "3/5"
assert replay_su2_theta_physical_block_certificate(actual["certificate"])
conditional = conditional_overlap_budget(Q(1, 8), Q(1, 4), 1)
assert conditional["actual_vacuum_verified"] is False
transfer = conditional_overlap_transfer_budget(Q(3, 4), Q(1, 3), decompositions=4)
assert transfer["arithmetic"]["larger_block_gap_lower"] == "2/5"
controls = physical_block_controls(chain_length=16, overlap_width=4)
assert controls["pinned_gaussian_overlap"]["massless_maximal_correlation"] == "3/5"
```

Exact inputs accept integers and `Fraction`, refusing booleans, floats and
strings. Serialized rationals use canonical spelling. Every replay rebuilds
the full certificate, including nested source certificates, claims, metadata,
geometry and honesty flags. A rehashed forged flag is rejected. Actual source,
matrix gap and joint-overlap acceptance remain separate. No generic budget
earns an actual vacuum, a uniform family, a continuum theory or a Clay parent.
