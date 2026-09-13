# An actual small-coupling SU(2) corner-vacuum inequality

This page proves explicit ground-energy comparisons for one finite graph:
three corner-adjacent squares with nine original links. The entire local
Hilbert space is covered; no spin truncation or assumed spectral gap enters.
For every \(0<\kappa\le2^{-40}\), the actual normalized positive vacuum obeys
\[
\langle A(D)-S\rangle_0\le-\frac{19}{210}\kappa<0,\qquad
S=A(U_1)+A(U_2)+A(U_3),\quad D=U_1U_2U_3.             \tag{1}
\]
Here \(A(U)=2-\operatorname{Tr}U\). The finite certificate checks the
rational budgets in this written analytic argument. Neither Lean
formalization of the analytic argument nor exterior-uniform or continuum
conclusions follow automatically.

The method is the nondegenerate-well harmonic comparison with IMS
localization; see [Simon (1983), Theorem1.1, Lemma3.1 and Section6](https://www.numdam.org/item/AIHPA_1983__38_3_295_0.pdf).
All constants needed here are supplied below, rather than taken from an
unspecified asymptotic remainder. This is an application of that established
method, not a claim of a new general semiclassical theorem.

## 1. Original graph and exact rooted operator

Use vertices \(0,1,2,3,4,5,6\) labelled by their three coordinate bits. The
oriented edges are
\[
01,02,04,13,23,26,46,45,15.
\]
The coherently oriented faces have signed edge indices
\((1,4,-5,-2),(2,6,-7,-3),(3,8,-9,-1)\).
Fix the six-edge rooted tree with indices \(1,2,3,5,7,9\). The remaining
chords \(4,6,8\) are exactly \(U_1,U_2,U_3\); Haar invariance pushes the
rooted physical measure to independent Haar on \(SU(2)^3\). The boundary
word \((-9,4,-5,6,-7,8)\) is \(D=U_1U_2U_3\) in these coordinates.
Residual Gauss invariance is simultaneous conjugation of the three variables.

Let \(L_i,R_i\) denote the triples of left and right infinitesimal fields,
with fundamental Casimir \(3/4\). Each of the six outer links contributes
one \(C_i\), twice for each face. The three original root spokes differentiate
as \(L_1-R_3,L_2-R_1,L_3-R_2\). Thus the exact electric form is
\[
\Gamma(f)=2\sum_i|L_i f|^2+
 \sum_i|L_i f-R_{i-1}f|^2.                            \tag{2}
\]
No product-face kinetic substitution has been made. Define
\[
H_\kappa=\frac\kappa2 C+\frac{2S}{\kappa},\qquad
\widetilde H_\kappa=\frac\kappa2 C+
 \frac{5S-A(D)}{2\kappa}.                            \tag{3}
\]
The chord terms make this a uniformly elliptic operator on the compact
connected rooted space. Both potentials are smooth and real. The positive
scalar ground state is unique, by elliptic positivity and the ground-state
transform, and residual gauge transformations preserve it. Consequently
these scalar ground energies equal the neutral physical ground energies.
This argument does not replace a charged exterior conditional by a neutral
vacuum.

The unitary chord triangle inequality gives
\[
A(U_1U_2U_3)\le3\sum_i A(U_i)=3S.                    \tag{4}
\]
Indeed \(A(U)=\|U-I\|_F^2/2\); telescope the product and use Cauchy--Schwarz.
Thus the tilted numerator \(V_1=(5S-A(D))/2\) is at least \(S\), globally.
Both potentials have their unique zero at \((I,I,I)\).

## 2. Exact local chart and conservative bounds

On the positive hemispheres write \(U_i=(s_i,z_i)\) in unit quaternions,
\(s_i=\sqrt{1-|z_i|^2}\). Put \(R_z^2=\sum_i|z_i|^2\). Up to a constant
which cancels from Rayleigh quotients, Haar density is
\(w(z)=\prod_i(1-|z_i|^2)^{-1/2}\).
For \(S\le r^2\), \(r\le1/2\), these coordinates are valid and
\[
R_z^2\le S\le(1+r^2)R_z^2,\qquad 1\le w\le1+4r^2.  \tag{5}
\]
The first pair follows from
\(A_i-|z_i|^2=(1-s_i)^2\le |z_i|^4\).
For the density, \(\prod_i(1-|z_i|^2)\ge1-R_z^2\), so
\(w\le(1-r^2)^{-1/2}\le1+4r^2\); squaring verifies the last elementary
inequality throughout \(0\le r^2\le1/4\).

The symbol \(G\) in (2) is explicitly evaluable from quaternion multiplication.
At zero,
\[
G_0=\tfrac14 M\otimes I_3,\qquad
M=5I_3-\mathbf1\mathbf1^T,\quad\operatorname{spec}M=\{2,5,5\}.
\]
A left/right field coefficient is \((s_iI\pm[z_i]_\times)/2\).
Its difference from \(I/2\) has norm at most \(|z_i|\).
Each original derivative has at most two face incidences, and each face
has four incidences. Hence, writing \(G=B^TB\), Cauchy--Schwarz gives
\(\|B-B_0\|\le\sqrt8 r<4r\). Since \(\|B_0\|=\sqrt5/2<2\),
\[
\|G-G_0\|\le16r+16r^2\le32r,
\quad (1-64r)G_0\le G\le(1+64r)G_0.                 \tag{6}
\]
We only use the lower bound when \(r\le1/128\).

An explicit potential remainder is
\[
\left|A(D)-|z_1+z_2+z_3|^2\right|\le32rR_z^2.       \tag{7}
\]
To check it, set \(h_i=U_i-I\). Then
\(\|h_i\|\le\sqrt2|z_i|\), and the difference between
\(D-I\) and the pure quaternion \((0,z_1+z_2+z_3)\) has norm at most
\(4R_z^2\): the scalar linear remainder contributes at most \(R_z^2\),
the three pair products at most \(2R_z^2\), and the triple at most
\(2rR_z^2\le R_z^2\). Comparing squared norms proves (7), since
\(8\sqrt3 r+16r^2\le32r\).

The tilted quadratic potential is
\[
V_{1,0}=\frac52R_z^2-\frac12|z_1+z_2+z_3|^2\ge R_z^2.
\]
Equations (5)--(7) imply
\[
V_1\ge(1-18r)V_{1,0},\qquad
2S\le(1+r^2)2R_z^2.                                 \tag{8}
\]
For the first estimate the absolute remainder is bounded by
\(((5/2)r^2+16r)R_z^2\le18rR_z^2\).

## 3. Oscillator energies and Gaussian moments

The untilted quadratic form is
\(-\kappa\nabla\cdot G_0\nabla/2+2R_z^2/\kappa\).
Its ground energy and squared-ground-state covariance are
\[
e_* =\frac32(\sqrt2+2\sqrt5)<U_*:=\frac{7417}{840},
\qquad \operatorname{Cov}(z)=\frac\kappa8\sqrt M\otimes I_3.
                                                               \tag{9}
\]
Thus \(\mathbb E R_z^2<9\kappa/4\). Rational bounds
\(\sqrt2<99/70\) and \(\sqrt5<161/72\) prove the displayed upper energy.
The tilted quadratic potential commutes with \(M\); its frequencies are
\(1,5/2,5/2\) per color. Its exact ground energy is \(9\).
The untouched energy margin is \(9-U_*=143/840\).

The Gaussian and the cutoff below are invariant under simultaneous color
rotation. They therefore give physical trial states. The lower comparison
on all rooted scalar functions is also a valid physical lower bound.

## 4. Global localization and the tilted lower energy

Set \(R=\sqrt S\). Each old square has
\(\Gamma(A_i)=4A_i-A_i^2\), and at most two marked faces meet an edge.
Therefore \(\Gamma(S)\le8S\), and \(\Gamma(R)\le2\) almost everywhere.
Choose \(\theta(R)=0\) for \(R\le r/2\), linear up to \(\pi/2\) at \(r\),
and constant thereafter; put \(\chi_g=\cos\theta\), \(\chi_b=\sin\theta\).
They are Lipschitz form-domain multipliers, \(\chi_g^2+\chi_b^2=1\), and
\[
\frac\kappa2\bigl(\Gamma(\chi_g)+\Gamma(\chi_b)\bigr)
\le\frac{\pi^2\kappa}{r^2}<\frac{10\kappa}{r^2}.     \tag{10}
\]
The good part is supported within the hemisphere chart. Extending it by
zero to \(\mathbb R^9\), the weighted form and norm bounds (5)--(8) give
its tilted Rayleigh floor \(9(1-64r)/(1+4r^2)\).
On the bad part, \(S\ge r^2/4\), so (4) supplies the floor
\(r^2/(4\kappa)\). IMS now proves
\[
\widetilde E_0\ge
\min\left\{\frac{9(1-64r)}{1+4r^2},\frac{r^2}{4\kappa}\right\}
 -\frac{10\kappa}{r^2}.                             \tag{11}
\]
Every function is in the same original-link quadratic form. No extensive
energy has been subtracted from an independently localized block.

## 5. Cut Gaussian upper energy

Let \(g_\kappa\) be the normalized oscillator Gaussian from (9), and use
\(\chi_g g_\kappa\) as the compactly supported physical trial. On the ball \(R_z^2\le r^2/5\), the direct identity
\(A_i-|z_i|^2=(1-s_i)^2\le|z_i|^4\) implies
\(S\le(1+r^2)R_z^2\le r^2/4\). Thus Markov and (9) give
\[
\int\chi_g^2g_\kappa^2dz\ge1-\frac{45\kappa}{4r^2}
 \ge1-\frac{12\kappa}{r^2}.                          \tag{12}
\]
On the support of the cutoff derivative,
\( |\nabla_z\sqrt S|^2\le1/(1-r^2)\le2\) and
\(G_0\le5I/4\). The exact oscillator ground-state identity yields its
cutoff Rayleigh cost at most
\[
\frac{5\pi^2\kappa}{4r^2(1-12\kappa/r^2)}
 <\frac{13\kappa}{r^2(1-12\kappa/r^2)}.              \tag{13}
\]
This bound does not replace a cutoff eigenfunction by an uncut one.
Taking the density and metric upper ratios in (5)--(8), with the lower
bound1 for the density in the norm, proves
\[
E_0\le (1+4r^2)(1+64r)
 \left(U_*+\frac{13\kappa}{r^2(1-12\kappa/r^2)}\right).
                                                               \tag{14}
\]
The normalization constant of Haar measure cancels. Working with forms
avoids an omitted half-density derivative term.

## 6. A rational interval and actual vacuum conclusion

Take \(\kappa=\tau^5\), \(r=\tau^2\). The upper expression in (14) is
increasing in \(\tau\) while \(12\tau<1\); both branches and the negative
error term in (11) are decreasing in \(\tau\). At \(\tau=1/256\), exact
rational arithmetic gives
\[
E_0\le U_*+\frac1{16},\qquad
\widetilde E_0\ge9-\frac1{16},\qquad
\widetilde E_0-E_0\ge\frac{19}{420}.                 \tag{15}
\]
All chart, norm and barrier conditions also hold at that endpoint and
improve for smaller positive \(\tau\). Thus (15) covers the full interval
\(0<\kappa\le2^{-40}\), not only dyadic samples.

Apply the variational principle to the actual old normalized vacuum:
\[
\widetilde E_0\le E_0+\frac{\langle S-A(D)\rangle_0}{2\kappa}.
\]
This proves (1). The actual vacuum density was never replaced by a heat
profile, a Wilson Gibbs distribution, or the oscillator density.

Combined with the independently proved three-increment heat-bridge bound,
(1) gives a strict averaged seam comparison
\[
\langle W\rangle_0\le\frac{8667}{512}
 +\frac2\kappa\frac{363}{392}\langle S\rangle_0.
\]
This is a moment/upper-form conclusion. It provides no lower spectral gap.
The next external premise is the analogous local *relative* energy estimate
in the actual exterior-coupled vacuum, uniform in the ambient volume.

## API and trust boundary

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.corner_vacuum import (
    replay_corner_vacuum_certificate,
    search_corner_vacuum_interval,
    su2_corner_vacuum_bound,
)

result = search_corner_vacuum_interval()
assert result['status'] == 'PASS'
row = result['accepted']
assert row['inputs']['tau_upper'] == '1/256'
assert row['arithmetic']['kappa_upper'] == str(Q(1, 2**40))
assert replay_corner_vacuum_certificate(row['certificate'])
assert row['physical_gap_claim'] is False
assert su2_corner_vacuum_bound(Q(1, 128))['status'] == 'INCONCLUSIVE'
```

`corner_reduced_metric` provides the exact9x9 symbol at three rational unit
quaternions. The search records every failed dyadic budget without treating
a failed enclosure as a false mathematical statement. Canonical replay
includes the claim, metadata and all arithmetic flags. All transcendental
constants in the finite budget have rational bounds; its backend is
`not_used`. The written analytic proof and conditional Mathlib implications
remain distinct registers.
