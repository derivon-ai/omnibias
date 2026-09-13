# Adjacent SU(2) reference inverse with complete spin tails

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.adjacent_resolvent import (
    su2_adjacent_linearized_inverse,
    replay_su2_adjacent_linearized_inverse_certificate,
)

result = su2_adjacent_linearized_inverse(5, cutoff=3)
assert result["status"] == "PASS"
assert Fraction(result["original_N_inverse_upper"]) < 6
assert result["actual_vacuum_verified"] is False
assert replay_su2_adjacent_linearized_inverse_certificate(result["certificate"])

refusal = su2_adjacent_linearized_inverse(4, cutoff=1)
assert refusal["status"] == "INCONCLUSIVE"
assert refusal["original_N_inverse_upper"] is None
assert replay_su2_adjacent_linearized_inverse_certificate(refusal["certificate"])
```

`kappa` is an exact positive integer or `Fraction`; `cutoff` is a positive
integer. The retained cube includes every admissible doubled spin through
the cutoff and excludes only the Haar constant. Floats and booleans are
refused. The stronger `Fraction(14, 5), cutoff=7` example passes despite an
actual original-norm unpreconditioned column bound above one.

`scalar_M_inverse_upper` and `original_N_inverse_upper` distinguish the two
norms and their factor-3/2 conversion. All finite columns and the complete
omitted tail are rebuilt on canonical replay. Faithful inconclusive reports
replay as refusals; resealed changes to arithmetic, inputs or scope fail.
No target-vacuum, physical-gap, locality, volume-uniform, continuum or formal
flag is earned. The full analytic implication follows.


12 September 2026. The exact seven-edge SU(2) reference linearization admits
finite inverse certificates beyond an actual unpreconditioned operator-norm
obstruction. At \(\kappa=14/5\), a complete block of 153 nonconstant theta
states and an analytic bound on every omitted spin certify an inverse bound
below 162 in the original anchored Fourier norm. This is a fixed compact
reference problem; the target vacuum and a continuum mass gap are separate
obligations.

## 1. Physical operator and complete invariant basis

Fix two adjacent square plaquettes with all horizontal edges oriented right
and vertical edges up. Their two outer paths have three electric edges each;
the shared path has one. Write doubled spins \(a,b,s\in\mathbb N_0\),
\(a+b+s\) even and \(|a-b|\le s\le a+b\). Each admissible triple supports
one normalized intertwiner at each trivalent vertex and equal-spin
contractions at the four bivalent vertices. These Peter–Weyl states form the
complete neutral invariant sector, with no extra multiplicity.

In independent Haar loop coordinates \(U,V\), use
\[
b_{a,b,s}=\frac{\operatorname{Tr}[P_s(D_{a/2}(U)\otimes D_{b/2}(V))]}{s+1},
\qquad b_{a,b,s}(I,I)=1,
\qquad \|b_{a,b,s}\|_H^2=\frac1{(a+1)(b+1)(s+1)}.
\tag{1}
\]
Here \(P_s\) projects the tensor product onto spin \(s/2\). The electric
operator on the original seven edges is
\[
Cb_{a,b,s}=E_{a,b,s}b_{a,b,s},\qquad
E_{a,b,s}=\frac{3a(a+2)+3b(b+2)+s(s+2)}4.
\tag{2}
\]
Thus \(E_{1,0,1}=3\) and \(E_{1,1,0}=9/2\). Neither the shared edge nor
the original electric weights may be replaced by two independent rotors.

The convention is
\[
aH_\kappa=\frac\kappa2C+\frac2\kappa(4-W),\qquad
W=\chi_p+\chi_q,\qquad g=4/\kappa^2,\qquad t=g/3,
\qquad S_*=tW.
\tag{3}
\]
On zero-Haar-mean functions set
\[
\mathcal L=I-K,\qquad K=2C_0^{-1}\Pi_H\Gamma(S_*,\cdot).
\tag{4}
\]
The constant is removed before \(C_0^{-1}\). The positive explicit drift
density \(e^{2S_*}dH/Z\) is a reference measure, not the unknown ground-state
density of (3). No diagonal skew similarity is assumed. On the normalized fundamental
inner/outer loop pair the adjacent operator has off-diagonal entries
\(-3t/4\) and \(-t/6\). Their positive product precludes a positive
diagonal skew similarity, so the [one-plaquette construction](gauge-plaquette-resolvent.md)
cannot be copied to this basis.

## 2. Exact multiplication and linearization

Multiplication by \(\chi_p\) changes \(a,s\) independently by \(\pm1\),
keeps spectator \(b\) fixed, and retains only admissible outputs. Its
coefficient in the normalization (1) is
\[
m_{o,i}^{(p)}=(a_o+1)(s_o+1)
\begin{Bmatrix}a_o/2&s_o/2&b/2\\s_i/2&a_i/2&1/2\end{Bmatrix}^{\!2}.
\tag{5}
\]
The other plaquette exchanges \(a,b\). This is the two-endpoint Haar/CG
contraction: the unnormalized bilinear integral is the same squared Racah
symbol divided by \(b+1\); dividing by the output Gram in (1) gives (5).
The independently audited Haar identification is the one used by the
physical two-plaquette Hamiltonian and exact vacuum recurrence.

All coefficients are nonnegative rational numbers. Each multiplication has
at most four admissible outputs. Evaluating its complete character identity
at the identity gives
\[
\sum_o m_{o,i}^{(p)}=2,\qquad
\sum_o m_{o,i}^{(q)}=2.
\tag{6}
\]
The constant output belongs in this mass sum even though (4) subsequently
removes it. These are exact representation identities, not numerical
positivity or a finite-cut extrapolation.

Since \(CW=3W\), the electric product rule gives
\[
2\Gamma(W,b_i)=3Wb_i+W(Cb_i)-C(Wb_i).
\]
Consequently, for every nonconstant output,
\[
\boxed{K_{o,i}=t\,\frac{3+E_i-E_o}{E_o}\,m_{o,i},
\qquad m=m^{(p)}+m^{(q)}.}
\tag{7}
\]
The sign of \(K_{o,i}\) is retained. Taking an absolute value of the magnetic
coefficient is harmless because it is positive; replacing the energy
factor by an absolute value while forming the finite matrix would change
its inverse.

## 3. Original coefficient norm and an explicit norm conversion

For arbitrary admissible spins, not only the lowest channels, the
original-edge coefficient matrix of (1) has
\[
\boxed{\|F_{a,b,s}\|_1=(a+1)^2(b+1)^2.}
\tag{8}
\]
To track the normalization, use Hilbert–Schmidt-normalized trivalent
invariant tensors. The canonical endpoint flattenings isolate outer legs
of dimensions \(a+1\) and \(b+1\), respectively. Schur's lemma makes
their singular values \(1/\sqrt{a+1}\) and \(1/\sqrt{b+1}\), with the
corresponding numbers of repetitions. On each outer path one bivalent
flattening is mixed, with singular values one, and one is a vector of norm
\(\sqrt{a+1}\) or \(\sqrt{b+1}\). The tensor product therefore has rank
\((a+1)^2(b+1)^2\), all nonzero singular values one. Its identity value
one matches (1), so no extra shared-dimension factor remains. This also
works when a leg is trivial. Its squared Hilbert–Schmidt norm agrees with
(1) times the ambient dimension \((a+1)^3(b+1)^3(s+1)\).
This proof fixes the original orientation and uses no partial-transpose
invariance of nuclear norm.

For coefficients \(h_i\), define the three original path-anchored sums
\[
N_e(h)=\sum_{i\ne0}\frac{i_e}2E_i(a_i+1)^2(b_i+1)^2|h_i|,
\quad e\in\{a,b,s\},\qquad N(h)=\max_eN_e(h).
\tag{9}
\]
Every edge along one path has the same anchor. Thus these are precisely the
original seven-edge unweighted source norm, with redundant anchors omitted.
The scalar weighted \(\ell^1\) norm used to certify the finite inverse is
\[
M(h)=\sum_{i\ne0}w_i|h_i|=N_a(h)+N_b(h)+N_s(h),\qquad
w_i=\frac{a_i+b_i+s_i}{2}E_i(a_i+1)^2(b_i+1)^2.
\tag{10}
\]
Each admissible label obeys \(i_e\le i_f+i_g\). Multiplying by the
nonnegative coefficient weights and summing gives \(2N_e\le M\) for
every anchor. Hence
\[
\boxed{2N(h)\le M(h)\le3N(h).}
\tag{11}
\]
Both spaces contain exactly the same coefficient sequences. A bound \(B\)
on an inverse in \(M\) gives the bound \(3B/2\) in \(N\). The conversion is
applied once to the final inverse. No locality weight \(b>1\) or
volume-uniform estimate is implied by (11).

## 4. A bound on every sufficiently high input spin

Let \(J=\max(a,b,s)\ge1\), \(h=a+b+s\). The triangle inequalities imply
\(h\ge2J\). Every magnetic output changes two labels by one, so
\[
\frac{h_o}{h_i}\le1+\frac1J.
\tag{12}
\]
Moreover
\[
E_i\ge\frac{J(5J+16)}8.
\tag{13}
\]
If \(J=s\), use \(a+b\ge s\) and \(a^2+b^2\ge s^2/2\) in (2).
If \(J=a\), minimize \(3b^2+s^2\) under \(b+s\ge a\), obtaining
\(3a^2/4\); the quadratic floor is then the stronger \(15J^2/16\).
The linear term is at least \(2J\) in either case. Exchange \(a,b\)
for the third case. These arguments prove (13) without a spin cutoff.

For one active change \(a_o=a+\epsilon\), \(s_o=s+\delta\),
\(\epsilon,\delta\in\{-1,1\}\), the energy difference satisfies
\[
3+E_i-E_o=\frac{4-3\epsilon(a+1)-\delta(s+1)}2,
\qquad |3+E_i-E_o|\le4+2J.
\tag{14}
\]
The coefficient nuclear ratio for that output is at most four, since only
\((a+1)^2\) changes in (8) and
\((a_o+1)^2/(a+1)^2\le4\). Equations (6) therefore bound the sum of
nuclear-weighted magnetic coefficients by eight for each plaquette and
sixteen for their sum.

The output electric energy in (10) cancels the denominator in (7). Combining
(12)–(14) and the preceding magnetic mass bound gives the full column bound
\[
\frac{\sum_o w_o|K_{o,i}|}{w_i}
\le \tau(J):=
\boxed{\frac{256t(J+1)(J+2)}{J^2(5J+16)}}.
\tag{15}
\]
Removing a constant output can only reduce this sum. The function is strictly
decreasing for positive integer \(J\), since
\[
\frac{\tau(J)-\tau(J+1)}{256t}
=\frac{(J+2)(5J^3+30J^2+68J+21)}
 {J^2(J+1)^2(5J+16)(5J+21)}>0.
\tag{16}
\]
Thus a single rational evaluation of (15) bounds infinitely many columns.
The bound tends to zero. It is not a dense-grid or random-sample assertion.

## 5. Complete block preconditioning and both boundary directions

Let \(P_N\) retain every admissible \(\max(a,b,s)\le N\) except the
constant, \(Q_N=I-P_N\). Form \(L_N=P_N\mathcal LP_N\) using (7).
If its exact rational inverse \(R_N\) exists, define
\[
\widehat R_N=R_N\oplus I_{Q_N},\qquad
Z_N=I-\widehat R_N\mathcal L.
\tag{17}
\]
The production calculation clears rational row denominators, uses
fraction-free Gauss–Jordan elimination with exact-division checks, and
verifies \(L_NR_N=I\) directly. A failed finite inverse is inconclusive.
The defect has three fully specified kinds of columns:

- For retained input, the retained component is zero and the omitted
  component is \(Q_NKP_N\). All its outputs are obtained from (5).
- For input \(\max(a,b,s)=N+1\), the retained component is
  \(R_NP_NK\) and the omitted component is \(Q_NK\). Every such input
  and every output is enumerated. The absolute value is taken after the
  retained matrix-vector multiplication, preserving possible cancellation.
- For input \(\max(a,b,s)\ge N+2\), one magnetic multiplication cannot
  reach the retained cube. Its entire defect is \(K\), bounded by
  \(\tau(N+2)\).

Let \(z_N\) be the maximum of the exact weighted column sums in the first
two classes and \(\tau(N+2)\). This is an upper bound on the complete
infinite defect. It need not equal the exact defect norm because the last
class uses (15). Let
\[
B_N=\max\left(1,\max_{i\in P_N}
 \frac{\sum_{o\in P_N}w_o|(R_N)_{o,i}|}{w_i}\right).
\tag{18}
\]
If \(z_N<1\), the Neumann inverse of \(\widehat R_N\mathcal L\)
proves
\[
\boxed{\|\mathcal L^{-1}\|_{M\to M}\le\frac{B_N}{1-z_N},
\qquad
\|\mathcal L^{-1}\|_{N\to N}\le\frac{3B_N}{2(1-z_N)}.}
\tag{19}
\]
All operators are bounded on this coefficient space by the finite low-spin
part and (15). The block preconditioner is invertible. Thus this argument
proves both injectivity and surjectivity of the full operator; it does not
assume a physical or reference spectral gap.

## 6. Exact witnesses beyond the unpreconditioned norm obstruction

At \(\kappa=5\), \(N=3\), the retained dimension is 22. Its largest retained
leak is \(59/540\), the boundary defect is below \(1/10\), and
\[
z_3=\frac{14336}{25625}<1,\qquad
\|\mathcal L^{-1}\|_N<6.
\tag{20}
\]
The old strip linear majorant \(56g/3=224/75\) already exceeds one.
For an obstruction in the actual norms, use input \(b_{1,0,1}\). Exact
multiplication gives
\[
K b_{1,0,1}=t\left(-\frac38 b_{2,0,2}
 +\frac16 b_{1,1,0}-\frac3{26}b_{1,1,2}\right).
\tag{21}
\]
Inserting its three coefficients in (9)–(10) yields
\[
\|K\|_{N\to N}\ge\frac{13t}2,
\qquad \|K\|_{M\to M}\ge\frac{15t}2.
\tag{22}
\]
At \(\kappa=14/5\), \(t=25/147\), the first lower bound is
\(325/294>1\). Nevertheless the complete retained cube \(N=7\), with
153 nonconstant states, gives
\[
\boxed{z_7=\frac{704000}{726327}<1,
\qquad \|\mathcal L^{-1}\|_{N\to N}<162.}
\tag{23}
\]
This is an exact preconditioned inverse beyond a demonstrably failed
unpreconditioned Neumann test in the original norm. The stated decimals are
not needed: the certificate compares all fractions directly. A smaller
example \(\kappa=3,N=6\) has \(z_6=20/21\) and inverse below 91;
its scalar unpreconditioned norm is at least \(10/9\).
At \(\kappa=4\), cutoff one is inconclusive while cutoff three passes,
which tests that an omitted-spin failure is not silently promoted.

The API is `su2_adjacent_linearized_inverse` with canonical
`replay_su2_adjacent_linearized_inverse_certificate`. It records complete
basis and boundary labels, exact pivots and inverse residual, all finite
defect columns, the analytic far tail, and the factor-3/2 norm conversion.
It accepts faithful passing and inconclusive replays; changed inputs,
arithmetic, orientation, normalization or honesty fields are rejected.
The research artifacts are `artifacts/adjacent_linearized_inverse.py` and
matching JSON, with canonical API certificates generated separately.

This inverse belongs to the constructed Wilson reference. Applying it to
an actual vacuum requires a sufficiently small preconditioned residual and
a complete nonlinear radius bound in the same norm. Assembling many blocks
also requires spatial locality and constants independent of their number.
Neither follows from a fixed two-plaquette inverse. No target-vacuum,
physical-gap, infinite-volume, continuum, all-scale or Clay claim is earned.
