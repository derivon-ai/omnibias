# A quantitative SU(2) bosonic commutator gap

The source certifies an excitation gap for a finite number of noncompact
matrix coordinates. It includes their complete configuration space and
every gauge-singlet angular channel. It adds no quadratic mass. The
result is a written analytic theorem supported by fully replayed rational
Sturm calculations; the infinite-domain and representation arguments
are not asserted as Lean theorems.

```python
from fractions import Fraction
from omnibias.core.verified.airy_sturm import (
    airy_half_line_lower_bounds,
    replay_airy_sturm_certificate,
)
from omnibias.geometry.gauge.transfer.commutator_matrix import (
    su2_commutator_matrix_gap,
    replay_su2_commutator_matrix_gap_certificate,
)

scalar = airy_half_line_lower_bounds()
assert scalar["status"] == "PASS"
assert replay_airy_sturm_certificate(scalar["certificate"])
assert [s["dirichlet_neumann_count_below_energy"] for s in scalar["shooting"]] == [0, 1]

matrix = su2_commutator_matrix_gap()
assert replay_su2_commutator_matrix_gap_certificate(matrix["certificate"])
assert Fraction(matrix["arithmetic"]["ground_energy_upper"]) == Fraction(1701, 200)
assert Fraction(matrix["arithmetic"]["first_singlet_excited_energy_lower"]) == Fraction(43, 5)
assert Fraction(matrix["arithmetic"]["singlet_gap_lower"]) == Fraction(19, 200)
assert matrix["actual_matrix_singlet_gap_verified_in_written_analysis"]
assert not matrix["compact_lattice_gap_verified"]
assert not matrix["yang_mills_mass_gap_claim"]
```

## 1. Operator, domain and claim

Write the three SU(2) adjoint coordinates as real color vectors
\(x_1,x_2,x_3\in\mathbb R^3\). On \(L^2(\mathbb R^9)\), let
\[
h_3=-\sum_{i=1}^3\Delta_{x_i}
       +\sum_{1\le i<j\le3}|x_i\times x_j|^2.                 \tag{1}
\]
Its self-adjoint realization is the Friedrichs operator of the
nonnegative closed form initially defined on \(C_c^\infty(\mathbb R^9)\).
The physical subspace is invariant under the simultaneous action
\((x_1,x_2,x_3)\mapsto(Rx_1,Rx_2,Rx_3)\), \(R\in SO(3)\).
This is the adjoint SU(2) singlet condition. It does not impose a
separate singlet condition on each vector, or on rotations of the
three spatial labels.

There is a unique scalar groundstate, which belongs to this subspace.
If \(E_0\) is its energy and \(E_{1,\mathrm{sing}}\) is the next
eigenvalue in the complete singlet subspace, then
\[
\boxed{
 E_0\le\frac{1701}{200},\qquad
 E_{1,\mathrm{sing}}\ge\frac{43}{5},\qquad
 E_{1,\mathrm{sing}}-E_0\ge\frac{19}{200}.}                 \tag{2}
\]
The quartic potential vanishes along noncompact commuting valleys.
Its confinement mechanism is transverse quantum zero-point energy.
That mechanism is classical mathematical physics, rather than a claim
of a newly discovered general principle: see
[Simon, *Some quantum operators with discrete spectrum but classically continuous spectrum* (1983)](https://authors.library.caltech.edu/records/4ahvp-yy937).
The explicit normalization, lower constants and replayed source used
here are proved below.

## 2. All-space transverse confinement

For fixed \(x_i\), split \(x_j\) into its longitudinal and two transverse
coordinates. A one-dimensional oscillator satisfies
\(-a\partial_y^2+b|x_i|^2y^2\ge\sqrt{ab}|x_i|\).
Dropping only the nonnegative longitudinal kinetic energy gives
\[
 -\frac14\Delta_{x_j}+\frac12|x_i\times x_j|^2
 \ge\frac{|x_i|}{\sqrt2}.                                \tag{3}
\]
The inequality holds also at \(x_i=0\). Fubini and the scalar oscillator
inequality justify it as a form bound without differentiating a moving
frame. Allocate (3) to all six ordered pairs \(i\ne j\). Each original
quartic term occurs twice with coefficient one half, and each coordinate
kinetic term occurs twice with coefficient one quarter. The remaining
kinetic coefficient is one half. Therefore
\[
 h_3\ge G:=\sum_{i=1}^3
       \left(-\frac12\Delta_{x_i}+\sqrt2|x_i|\right).       \tag{4}
\]

The form domain of \(G\) embeds compactly into \(L^2(\mathbb R^9)\):
Rellich compactness applies on bounded sets, and the growing potential
controls the exterior norm uniformly. The form inequality (4) makes
the form-domain embedding of \(h_3\) compact as well. Thus \(h_3\) has
compact resolvent. Its scalar Schrödinger semigroup is positivity
improving on connected \(\mathbb R^9\), as follows for example from
the strictly positive Brownian-bridge integrand for its finite
nonnegative continuous potential. Consequently its groundstate is
unique and positive. Rotation invariance then puts that groundstate
in the simultaneous singlet subspace. No spectral gap is used to
establish these facts.

## 3. Complete radial and angular comparison

For one summand of (4), separate spherical harmonics of angular momentum
\(l\in\mathbb N_0\), use the radial wavefunction multiplied by its radius,
and rescale the radius by \(r=s/\sqrt2\). The resulting operator is exactly
\[
 A_l=-\frac{d^2}{ds^2}+s+\frac{l(l+1)}{s^2},
 \qquad s>0,                                             \tag{5}
\]
with its Friedrichs condition at zero. There is no residual multiplicative
scale in (5). Section 4 proves for \(A_0\)
\[
 a_0>\frac{23}{10},\qquad a_1>4,                          \tag{6}
\]
where the indices begin at zero. These are the positive absolute values
of the first two negative Airy zeros, but the numerical source never
evaluates those zeros. The usual Airy notation is described in
[NIST DLMF §9.9](https://dlmf.nist.gov/9.9).

For \(l=1\), take the positive Barta function
\(u(s)=s^2\exp(-s^{3/2}/2)\). Direct differentiation gives
\[
 \frac{A_1u}{u}=\frac{7s}{16}+\frac{27}{8\sqrt{s}}.
                                                               \tag{7}
\]
Writing the second term as two equal terms and applying arithmetic–
geometric mean shows that the cube of (7) is at least
\[
 27\frac7{16}\left(\frac{27}{16}\right)^2
 =\frac{137781}{4096}>\left(\frac{16}{5}\right)^3.         \tag{8}
\]
For compactly supported test functions away from zero, the groundstate
transform with this positive \(u\) proves \(A_1\ge16/5\). Closure extends
the bound to its full Friedrichs form domain. Centrifugal monotonicity
then gives the same floor for every \(l\ge1\); no angular cutoff is used.

A triple of spherical-harmonic representations can contain a simultaneous
singlet only if their angular momenta satisfy the SU(2)/SO(3) triangle
conditions. In particular exactly one nonzero angular momentum cannot
occur. The all-zero-angular ground of \(G\) is unique. Any other singlet
therefore has one of the following lower energies:

| Channel | Lower energy |
| --- | --- |
| All angular momenta zero, at least one radial excitation | \(2a_0+a_1\ge43/5\) |
| Exactly two nonzero angular momenta | \(a_0+2(16/5)\ge87/10\) |
| Three nonzero angular momenta | \(3(16/5)=48/5\) |

All higher radial levels only increase these lower bounds. Restricting
(4) to the singlet subspace and applying min–max proves
\(E_{1,\mathrm{sing}}\ge43/5\). The comparison also gives
\(E_0\ge3a_0>69/10\).

## 4. Rational Sturm calculation and its half-line implication

Let \(L=6\), \(h=1/32\), and use 192 cells with the lower potential
\[
 V_h(r)=jh\quad\text{on }[jh,(j+1)h),\quad j=0,\ldots,191.
                                                               \tag{9}
\]
Insert a Neumann cut at \(L\) in \(A_0\). This enlarges its form domain,
and its exterior summand has form floor \(L=6\). Replace the interior
potential by (9), which is pointwise at most \(r\). It remains to count
the eigenvalues below each energy \(E=23/10,4\) of the finite
Dirichlet-at-zero/Neumann-at-six operator with potential (9).

Use the shooting initial conditions \(u(0)=0,u'(0)=1\).
On cell \(j\), put \(q_j=jh-E\). The exact transfer has the form
\[
 \begin{pmatrix}u(r+h)\\u'(r+h)\end{pmatrix}
 =\begin{pmatrix}C&S\\q_jS&C\end{pmatrix}
   \begin{pmatrix}u(r)\\u'(r)\end{pmatrix},\qquad
 C=\sum_{k\ge0}\frac{q_j^kh^{2k}}{(2k)!},\quad
 S=\sum_{k\ge0}\frac{q_j^kh^{2k+1}}{(2k+1)!}.              \tag{10}
\]
For order \(m\), put \(z=|q_j|h^2\). Provided the displayed ratios are
less than one, absolute factorial tails after \(k=m\) are bounded by
\[
 e_C=\frac{z^{m+1}}{(2m+2)!}
       \frac1{1-z/((2m+3)(2m+4))},\qquad
 e_S=\frac{hz^{m+1}}{(2m+3)!}
       \frac1{1-z/((2m+4)(2m+5))}.                         \tag{11}
\]
All later term ratios are smaller. Thus (11) is valid for either sign
of \(q_j\); it does not depend on cancellation in an alternating series.
The implementation uses \(m=6\), encloses both series by exact rational
intervals, and propagates (10). It rounds interval endpoints outward
to multiples of \(2^{-128}\) using integer floor/ceiling division.
This is denominator control, not a floating-point approximation.
Every transfer enclosure, factorial remainder and shooting endpoint
is in the sealed payload and is recomputed during replay.

No interior zero can be missed. When \(q_j<0\), consecutive zeros of
a nonzero solution are separated by \(\pi/\sqrt{-q_j}\). The exact check
\((-q_j)h^2<9\), together with \(\pi>3\), proves that each cell has at
most one zero. For \(q_j\ge0\), a nonzero hyperbolic or affine solution
also has at most one zero. Every right endpoint has a strictly resolved
sign, and \(u'(0)=1\) fixes the initial positive sign. Sign changes
therefore count all the interior nodes.

For clarity, the replayed endpoint enclosures imply the following
looser rational intervals; these decimal fractions are consequences
of the rational source, not its inputs:

| Energy | Node cells, zero based | \(u(6)\) | \(u'(6)\) |
| --- | --- | --- | --- |
| \(23/10\) | none | \([229/100,230/100]\) | \([421/100,422/100]\) |
| \(4\) | 56 | \([-633/1000,-632/1000]\) | \([-661/1000,-659/1000]\) |

The endpoint Prüfer phase \(\theta=\arg(u'+iu)\), continuously lifted
from zero, lies between \(n\pi\) and \(n\pi+\pi/2\) when \(u,u'\)
have the same sign after \(n\) nodes. Sturm oscillation with the Neumann
right condition then counts exactly \(n\) eigenvalues below \(E\).
Both source endpoints have that same-sign property: the counts are
zero and one. More generally the code also handles the opposite-sign
case, whose count is \(n+1\). An unresolved endpoint sign cannot pass.

The exterior Neumann summand starts at 6, above both thresholds.
Operator ordering and min–max now prove the **full half-line** inequalities
(6). The finite interval and staircase have not been substituted for
the original half-line operator.

The API permits alternative cell counts, series orders and dyadic
precision within stated resource bounds. A mesh that is too coarse or
an enclosure that cannot resolve the required signs returns
`INCONCLUSIVE`. That status does not refute (6).

## 5. Gaussian upper bound and spectral separation

Take the normalized simultaneous-rotation-invariant Gaussian
\(\phi_a\propto\exp[-a\sum_i|x_i|^2/2]\). Its coordinate variances
under \(|\phi_a|^2\) are \(1/(2a)\). For independent color vectors,
\(\mathbb E|x_i\times x_j|^2=3/(2a^2)\). Hence
\[
 \langle\phi_a,h_3\phi_a\rangle
 =\frac{9a}{2}+\frac{9}{2a^2}.
                                                               \tag{12}
\]
Choosing the rational width \(a=5/4\) gives
\[
 E_0\le\frac{45}{8}+\frac{72}{25}
       =\frac{1701}{200}.
                                                               \tag{13}
\]
The complete singlet min–max lower bound in Section 3 and the variational
upper bound (13) concern the same operator and the same physical sector.
Their difference is exactly \(19/200\), proving (2).

## 6. Scaling and remaining scope

For an explicitly **defined homogeneous model**, let \(\kappa>0\) and
let \(N\) be a positive integer, and set
\[
 H_{\mathrm{hom}}(\kappa,N)=
 \frac{\kappa}{2N^3}\left(-\sum_i\Delta_{A_i}\right)
 +\frac{N^3}{2\kappa}\sum_{i<j}|A_i\times A_j|^2.
                                                               \tag{14}
\]
The unitary dilation \(A_i=\kappa^{1/3}x_i/N\) gives exactly
\[
 H_{\mathrm{hom}}(\kappa,N)
 \cong\frac{\kappa^{1/3}}{2N}h_3,
 \qquad
 \Delta_{\mathrm{hom,sing}}
 \ge\frac{19\kappa^{1/3}}{400N}.
                                                               \tag{15}
\]
This definition does not identify homogeneous functions as an invariant
subspace of a spatial lattice Hamiltonian. In particular its displayed
scale depends on \(N\); it supplies no spatial-volume-uniform positive
gap. A compact commutator model also has distinct center sectors and
requires its own localization, metric and spectral comparison.

The certificates earn written-analysis flags only after the scalar
source and its exact cell counts replay. Full dictionary replay rejects
altered sources, scope fields and resealed arithmetic. A separate formal
runner may check finite rational implications, but this producer sets
`theorem_prover_verified`, `mathlib_verified`, and
`analytic_proof_formally_verified` to false. It also leaves compact
lattice, continuum, and Yang–Mills mass-gap claims false.
