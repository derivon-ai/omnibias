# A compact SU(2) commutator gap in the center-even sector

This page proves a quantitative gap for the actual compact model with one
vertex and three SU(2) selfloops. It retains their nonabelian commutator
interactions and all representations. Its Hilbert space and its center
sectors are specified below; it is not a homogeneous quantum subspace of
an arbitrary larger lattice.

For every \(0<\kappa\le2^{-36}\), the simultaneous-Ad-invariant,
center-even sector has

\[
 \boxed{E_{1,+}(\kappa)-E_0(\kappa)
              \ge\frac{\kappa^{1/3}}{50}.}       \tag{1}
\]

For each of the seven other center characters \(\chi\), the different
conclusion is

\[
 \boxed{0<E_{0,\chi}(\kappa)-E_0(\kappa)
                  \le91\kappa^{2/3}.}           \tag{2}
\]

Thus (1) is not a lower bound for every nonvacuum state in the full physical
space. Equation (2) is an upper bound, not an exponential tunneling law.
The analytic statements have a written proof. Their rational budgets and
the prerequisite matrix source have canonical replay; the analytic proof
is not thereby formalized.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.compact_commutator import (
    su2_compact_commutator_gap,
    replay_compact_commutator_certificate,
)

result = su2_compact_commutator_gap(Fraction(1, 2**36))
assert result["status"] == "PASS"
assert result["arithmetic"]["center_even_gap_lower"] == "1/204800"
assert result["actual_compact_center_even_gap_verified_in_written_analysis"]
assert not result["unrestricted_gap_verified"]
assert replay_compact_commutator_certificate(result["certificate"])
```

## 1. Operator, ground state and sectors

Let Haar measure on each SU(2) factor have mass one, and let
\(C=-\Delta_{SU(2)}\) have fundamental Casimir \(3/4\). With
\(A(U)=2-\operatorname{Tr}U\), the compact operator is

\[
 H_\kappa=\frac\kappa2\sum_{i=1}^3C_i+
 \frac2\kappa\sum_{i<j}A(U_iU_jU_i^{-1}U_j^{-1}).       \tag{3}
\]

Use the Friedrichs form on the compact connected product group. Its
potential is smooth and nonnegative. Ellipticity and positivity of its
heat kernel give compact resolvent and a unique, strictly positive ground
state. Symmetry and uniqueness make this state invariant under simultaneous
conjugation and under each independent center flip \(U_i\mapsto-U_i\).
Therefore its scalar, physical and center-even ground energies agree.

The physical space is the simultaneous-Ad-invariant subspace. The group
\(\mathbb Z_2^3\) of independent center flips commutes with (3) and splits
this space into eight character subspaces. In (1), \(E_{1,+}\) means the
second eigenvalue within the trivial character subspace, counting the
unique vacuum as the first. In (2), \(E_{0,\chi}\) is the lowest eigenvalue
in a specified nontrivial character subspace. Compact resolvent and ground
state uniqueness imply the strict lower inequality in (2), without a
numerical lower bound for that sector difference.

## 2. Exact conditional oscillator comparison on the whole compact group

Write a unit quaternion as \(U_i=(u_{i,0},v_i)\), and put
\(s_i=|v_i|\). Quaternion multiplication gives the global identity

\[
 A(U_iU_jU_i^{-1}U_j^{-1})=4|v_i\times v_j|^2.          \tag{4}
\]

Fix \(U_i\) and, when \(s_i>0\), rotate its vector axis to the third
coordinate. On the other rotor set \(z=|v_{j,\perp}|^2\in[0,1]\).
Since \(C=-\tfrac14\Delta_{S^3}\) in unit-quaternion coordinates,

\[
 C_jF(z)=-z(1-z)F''(z)-(1-2z)F'(z).                 \tag{5}
\]

For any \(a,b>0\), choose the smooth positive function
\(\phi=\exp(-\eta z)\), \(\eta=\sqrt{b/a}\,s_i\).
Its exact local energy is

\[
 \frac{(aC_j+bs_i^2z)\phi}{\phi}
 =\sqrt{ab}\,s_i+a\bigl[(\eta z-1)^2-1\bigr]
 \ge\sqrt{ab}\,s_i-a.                              \tag{6}
\]

The positive-groundstate form identity proves the same operator lower
bound on the entire conditional rotor space. There is no boundary term on
SU(2). When \(s_i=0\), use the constant test function; the same lower
bound holds. Integrating the fixed-coordinate inequality proves it for
arbitrary joint scalar functions, including functions not invariant under
any gauge action.

Now split (3) exactly as

\[
 H_\kappa=\frac\kappa4\sum_iC_i+
 \sum_{i\ne j}\left[\frac\kappa8C_j+
              \frac1\kappa A(U_iU_jU_i^{-1}U_j^{-1})\right].
\]

Each of the six ordered terms uses (6) with \(a=\kappa/8\),
\(b=4/\kappa\). Each \(s_i\) occurs twice, so

\[
 \boxed{H_\kappa\ge\frac\kappa4\sum_iC_i+
                    \sqrt2\sum_i s_i-\frac{3\kappa}{4}.} 
                                                               \tag{7}
\]

This global form inequality supplies a quantum lower bound along the
commuting toron valleys. The commutator potential itself vanishes on those
valleys; a pointwise lower potential bound would not suffice.

## 3. The matrix input and the exact local scaling

The dynamically replayed [matrix source](gauge-commutator-matrix.md) concerns

\[
 h_3=-\sum_i\Delta_{Y_i}+
             \sum_{i<j}|Y_i\times Y_j|^2
 \quad\text{on }L^2((\mathbb R^3)^3),               \tag{8}
\]

restricted to simultaneous SO(3) invariants. It proves

\[
 e_0:=E_0(h_3)\le U_*:=\frac{1701}{200},\qquad
 e_1:=E_{1,\mathrm{phys}}(h_3)\ge L_*:=\frac{43}{5}. \tag{9}
\]

In particular the matrix ground state is unique, positive and invariant.
The first inequality also follows directly from the invariant Gaussian
\(\phi_a(Y)=c_a\exp(-a|Y|^2/2)\), with \(a=5/4\):
its energy is \(9a/2+9/(2a^2)=1701/200\). The matrix lower source includes
its scalar radial eigenvalue controls and all angular sectors; it is not
an unverified numerical truncation.

Discreteness of bosonic commutator matrix Hamiltonians is an established
result: [Simon (1983)](https://www.math.caltech.edu/SimonPapers/158.pdf),
Section 7, Corollary 4, and
[Lüscher (1983)](https://doi.org/10.1016/0550-3213(83)90436-4), Section 5
and Appendix D. The present bridge consumes the explicit bounds (9).

In the chart near any of the eight triples of central elements, write
\(U_i=\epsilon_i(\sqrt{1-|v_i|^2},v_i)\), where
\(\epsilon_i\in\{1,-1\}\). Formula (4) is independent of the signs.
The Haar density and the original electric form are exactly

\[
 w(v)=\prod_i(1-|v_i|^2)^{-1/2},\qquad
 \Gamma_i(F)=\frac14\left(|\nabla_{v_i}F|^2
                      -(v_i\cdot\nabla_{v_i}F)^2\right).       \tag{10}
\]

Constant Haar normalization factors cancel in all Rayleigh quotients.
Set \(\sigma=\kappa^{1/3}\), \(\alpha=\sigma/2\), and
\(v_i=\alpha Y_i\). The Euclidean form obtained by replacing the metric
in (10) by \(I/4\) and its density by one is exactly
\(\alpha h_3\). There is no magnetic Taylor remainder: (4) is exact.

## 4. Center-invariant IMS partition and lower eigenvalue comparison

Put \(\tau=\kappa^{1/6}\), \(x=\tau^2=\sigma\), and
\(\rho=(\sum_i s_i^2)^{1/2}\). Let \(\theta(\rho)\) be zero below
\(\tau\), linear to \(\pi/2\) between \(\tau\) and \(2\tau\), and
constant thereafter. Use \(\chi_g=\cos\theta\), \(\chi_b=\sin\theta\).
These Lipschitz cutoffs are gauge invariant and center even. Smooth form
approximations give the usual IMS identity with the same limiting bound.

If \(z_i=s_i^2\), then \(\Gamma_i z_i=z_i(1-z_i)\). Consequently
\(\Gamma\rho\le1/4\) almost everywhere. Since
\(|\theta'|=\pi/(2\tau)\) on its transition interval, the IMS error is

\[
 I_\kappa\le\frac{\kappa\pi^2}{32\tau^2}
              <\alpha\frac58x.                    \tag{11}
\]

On the bad support, \(\rho\ge\tau\). Equation (7) gives the
conservative floor

\[
 B_\kappa=\tau-\frac34\tau^6.                       \tag{12}
\]

For \(0<\tau\le1/64\), this exceeds \(\alpha L_*\). On the good
support, \(\rho\le2\tau\), and (10) gives metric lower factor
\(1-4x\), metric upper factor one, and

\[
 1\le w\le(1-4x)^{-3/2}\le W:=(1-4x)^{-2}.          \tag{13}
\]

Thus the good Rayleigh comparison has lower factor at least
\((1-4x)^{5/2}\ge f:=(1-4x)^3\).

The eight chart components must be counted correctly. In the center-even
sector they are eight identical copies of one function. Their common
factor eight cancels from both norms and forms, leaving one Euclidean
simultaneous-SO(3)-invariant function. It does not leave eight independent
matrix vacua. The residual color action is retained; no singular orbit-space
coordinate measure is introduced.

Here is the min–max argument with the Haar density made explicit. For a
spectral subspace below \(\lambda\), IMS gives
\(q_g+q_b\le(\lambda+I_\kappa)(n_g+n_b)\). If
\(\lambda+I_\kappa<B_\kappa\), multiplication by \(\chi_g\) is
injective on this subspace and \(q_g\le(\lambda+I_\kappa)n_g\).
Use \(q_g\ge\alpha(1-4x)q_{h_3}\) and the upper Haar bound for
\(n_g\), and then Euclidean min–max. Zero extension from each good chart
belongs to its Euclidean form domain because the cutoff vanishes at the
chart support boundary. This proves

\[
 E_{1,+}(\kappa)\ge
 \min\{\alpha f e_1,B_\kappa\}-I_\kappa
 \ge\alpha\left[\frac{43}{5}f-\frac58x\right].     \tag{14}
\]

This is a comparison of forms and their dimensions, not a claim that
changing Haar density is a derivative-free unitary conjugation.
For the ground-energy lower bound, the same argument works before selecting
a center character: eight model copies still have the same lowest energy
\(e_0\), so \(E_0(H_\kappa)\ge\alpha f e_0-I_\kappa\).

## 5. An explicit Gaussian upper bound with polynomial tail control

Use \(\phi_a\) from Section 3 and the same cutoff in the scaled variables.
Its inner radius is \(T=\tau/\alpha=2/\tau\), and its outer radius is
\(2T\). Repeat the cut Gaussian in all eight central charts with equal
signs; it is an actual center-even physical trial function.

For the normalized Gaussian probability \(\phi_a^2dY\),

\[
 \mathbb E R^2=\frac9{2a}=\frac{18}{5},\qquad
 \mathbb E R^4=\frac{99}{4a^2},\qquad R=|Y|.        \tag{15}
\]

The flat cutoff norm is at least
\(1-\mathbb E R^2/T^2=1-9x/10\). One cannot simply discard the
negative part of the Gaussian's local energy. The exact identity is

\[
 \frac{h_3\phi_a}{\phi_a}=9a-a^2R^2+V_4(Y),
 \qquad V_4=\sum_{i<j}|Y_i\times Y_j|^2.            \tag{16}
\]

The form identity for \(\chi_g\phi_a\) consists of the cutoff-weighted
local energy plus \(\int|\nabla\chi_g|^2\phi_a^2\). Removing part of
the negative term in (16) costs at most

\[
 a^2\mathbb E[R^2\mathbf1_{R\ge T}]
 \le\frac{a^2\mathbb E R^4}{T^2}=\frac{99}{16}x.
\]

The cutoff-gradient contribution is at most
\(\pi^2/(4T^2)<10x/16\). Thus its flat energy numerator is at most
\(1701/200+109x/16\). Equation (10) has metric no larger than the
Euclidean metric, and (13) bounds the density in the numerator by \(W\)
and the density in the denominator below by one. Therefore

\[
 E_0(H_\kappa)\le\alpha\,
 W\frac{1701/200+109x/16}{1-9x/10}.                \tag{17}
\]

There is no exponential-tail assumption and no unknown compact groundstate
estimate in this upper bound.

## 6. Exact interval of couplings

Define the rational functions

\[
 L(x)=\frac{43}{5}(1-4x)^3-\frac58x,\qquad
 U(x)=\frac{1701/200+109x/16}{(1-4x)^2(1-9x/10)}.
                                                               \tag{18}
\]

On \(0\le x\le1/4096\), \(L\) decreases and \(U\) increases: all
denominators are positive, and each positive factor in \(U\) increases.
At the upper endpoint,

\[
 L(1/4096)-U(1/4096)
 =\frac{11389445042978907523}{230083594272878100480}
 >\frac1{25}.                                    \tag{19}
\]

The bad-branch comparison used in (14) follows, throughout the interval,
from
\(1-3\tau^5/4\ge1-3/(4\cdot64^5)\)
and \(L_*\tau/2\le43/640\). Multiplying (19) by
\(\alpha=\kappa^{1/3}/2\) proves (1).

## 7. Separate upper bounds in all other center sectors

This argument uses the actual normalized groundstate \(\psi_*\) of the
matrix Hamiltonian (8), whose existence and positivity are part of the
matrix theorem. Its known coercivity bound is

\[
 h_3\ge-\tfrac12\sum_i\Delta_{Y_i}
                      +\sqrt2\sum_i|Y_i|\ge\sqrt2 R.
\]

Hence \(\mathbb E_{\psi_*^2}R\le e_0/\sqrt2\le U_*\), and the same
cutoff retains norm at least
\(1-U_*/T=1-U_*\tau/2\ge1/2\). Repeat \(\chi_g\psi_*\) in the eight
charts with the signs of any chosen center character. Its norm and energy
are independent of those signs because the supports are disjoint. The
exact matrix groundstate transform now gives, after division by its cutoff
norm, the upper bound

\[
 E_{0,\chi}\le\alpha W\left[
 e_0+\frac{5x/8}{1-U_*\tau/2}\right].              \tag{20}
\]

This use of a true matrix groundstate is an existence proof for a trial;
the runtime does not pretend to compute that wavefunction. Subtract the
all-sector compact ground lower bound after (14). On the declared interval,

\[
 W-1\le9x,\quad1-f\le12x,\quad W\le2,
 \quad1-U_*\tau/2\ge\tfrac12.
\]

For example, the first inequality follows by multiplying its positive
denominator and using \(1-56x+144x^2\ge1-56/4096>0\).
The second follows directly from expanding \((1-4x)^3\).
Consequently

\[
 \begin{split}
 E_{0,\chi}-E_0
 &\le\alpha x\left(21U_*+\frac{25}{8}\right)\\
 &=\frac{18173}{200}\kappa^{2/3}
 <91\kappa^{2/3}.                                \tag{21}
 \end{split}
\]

Uniqueness of the positive compact vacuum supplies strict positivity for
nontrivial characters, proving (2). This does not give a positive lower
coefficient for those seven differences.

## 8. Certificate contract and limits

`su2_compact_commutator_gap(kappa)` accepts positive exact integers or
fractions. A positive input above \(2^{-36}\) is inconclusive for this
criterion. It constructs and canonically replays the matrix source, reads
the sealed spectral bounds and written-analysis flags, and seals that
source inside the compact certificate. Caller-supplied spectral or vacuum
premises are not accepted.

For a requested rational coupling, a normalized integer cube-root algorithm
encloses \(\kappa^{1/3}\) on an 80-bit relative dyadic grid. It returns a
positive rational lower bound for (1) and a rational upper bound for (2).
Exact dyadic cubes, including the default endpoint, produce point
enclosures. The symbolic coefficients \(1/50\) and \(91\) remain explicit.
All arithmetic uses integers and fractions; no floating root determines a
gate. Canonical replay reconstructs both passing and inconclusive outcomes.

Only the declared compact sector earns the gap flag. Unrestricted gap,
exterior-uniform conditional gap, volume-uniform lattice gap, exponential
tunneling, continuum and formal analytic-proof flags remain false. An
embedding into a larger quantum lattice would additionally require a
proved vacuum-subtracted comparison and control of the modes it discards.
