# Coarse holonomy refinement on an SU(2) theta graph

`su2_theta_refinement` eliminates an actual new gauge-invariant sector of
a finite graph. It computes exact rational entries of the resulting
energy-dependent operator and bounds its remainder on the **entire** coarse
Hilbert space. No representation cutoff enters that remainder. This is a
finite spatial graph result with a written analytic proof, not a continuum
construction or a vacuum-preserving renormalization map.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer import (
    su2_theta_refinement,
    replay_su2_theta_refinement_certificate,
)

result = su2_theta_refinement(
    electric=2, path_weights=(3, 3, 1),
    magnetic_left=Q(1, 2), magnetic_right=Q(1, 2),
    spectral_parameter=0, coarse_two_j_max=4,
)
assert result["status"] == "ENCLOSED"
w = result["witness"]
assert w["coarse_electric_gap"] == "9"
assert w["complement_electric_floor"] == "6"
assert w["magnetic_cross_block_norm"] == "1"
assert w["second_order_matrix"][0][0] == "1/12"
assert w["self_energy_remainder_upper"] == "1/48"
assert replay_su2_theta_refinement_certificate(result["certificate"])
assert not result["continuum_claim"]
```

## Operator and physical space

Take three internally disjoint paths between two vertices, with total
positive electric weights \(A,B,S\). Subdivision of each path is allowed;
Gauss's law is imposed also at every internal bivalent vertex. Write path
holonomies \(X,Y,Z\), coarse holonomy \(W=XY^{-1}\), and fundamental characters

\[
\chi_L=\operatorname{Tr}(XZ^{-1}),\qquad
\chi_R=\operatorname{Tr}(YZ^{-1}).
\]

The fine Hamiltonian is the self-adjoint bounded perturbation

\[
H=H_0-M,\quad
H_0=\alpha(AC_a+BC_b+SC_s),\quad
M=v_1\chi_L+v_2\chi_R,
\]

with \(\alpha,A,B,S>0\), \(v_1,v_2\ge0\). The quadratic Casimir is

\[
C_j=j(j+1),\qquad C_{1/2}=3/4.
\]

Adding \(2(v_1+v_2)I\) gives the nonnegative Wilson potential convention.
The spectral parameter \(z\) always refers to the displayed \(H\), with
that additive constant removed. All energies use the same input unit.

The gauge-invariant Haar Hilbert space has one normalized spin-network
vector for each admissible triple \((a,b,s)\): triangle inequalities and
integer \(a+b+s\). Let \(P\) be conditional Haar expectation over \(Z\),
and \(Q=I-P\). Then \(P\) selects precisely \(s=0,a=b=j\). Its orthonormal
basis consists of the characters \(\chi_j(W)\). This identifies the entire
coarse Hilbert space; \(Q\) contains every newly added representation.

## Exact electric and magnetic blocks

The electric operator reduces this decomposition. Its coarse excited gap
and complementary floor are exactly

\[
d_{\rm old}=\frac{3\alpha}{4}(A+B),\qquad
d_{\rm new}=\frac{3\alpha}{4}(S+\min(A,B)).
\]

For the second formula, \(s>0\) forces at least one of \(a,b\) to be
nonzero. Each nonzero Casimir is at least \(3/4\). The bound is attained
by \((a,b,s)=(1/2,0,1/2)\) or \((0,1/2,1/2)\).
Thus equal path weights give equal old and new floors. On two adjacent
square plaquettes, \((A,B,S)=(3,3,1)\), the new floor is lower. A high-energy
buffer cannot be inferred from the word "refinement."

Character orthogonality and Haar convolution give

\[
PMP=0,\qquad
PM^2P=(v_1^2+v_2^2)I+v_1v_2\chi_{1/2}(W).
\]

Indeed each squared small-loop character integrates to one, and the mixed
product integrates to \(\chi_{1/2}(W)/2\). Since \(\chi_{1/2}\) has essential
range \([-2,2]\), with \(b=v_1+v_2\),

\[
\|QMP\|=b,\qquad \|M\|\le2b.
\]

This sharp cross-block norm covers arbitrary coarse vectors, not just a
trial pack. The coarse constant is generally not carried to the fine
ground state: \(QHP1=-M\ne0\) whenever \(b>0\).

## Feshbach operator and parity-improved enclosure

For real \(z<d_{\rm new}-2b\), set

\[
d=d_{\rm new}-z>2b,\quad q=2b/d<1,
\quad R_0=(QH_0Q-z)^{-1}.
\]

The complementary block \(QHQ-z\) is strictly positive. The
Schur/Feshbach operator on the entire coarse space is

\[
F(z)=PH_0P-z-\Sigma(z),\qquad
\Sigma(z)=PMQ(QHQ-z)^{-1}QMP.
\]

For example, an eigenvector at energy \(z\) corresponds to a nonzero coarse
vector in \(\ker F(z)\), with fine complement

\[
\psi_Q=(QHQ-z)^{-1}QMP\psi_P.
\]

This energy-dependent reconstruction is not an isometric embedding of
every coarse state preserving an already-known fine vacuum.

The exactly computed second-order operator is

\[
\Sigma_0(z)=PMQ R_0 QMP.
\]

Put \(T=R_0^{1/2}QMQ R_0^{1/2}\) and \(L=R_0^{1/2}QMP\). The center flip

\[
Z\longmapsto-Z
\]

commutes with \(P,Q,H_0,R_0\), makes \(M,T\) odd, and makes every vector in
\(\operatorname{ran}L\) odd. Consequently \(L^*T^{2k+1}L=0\). The norm-convergent
resolvent series, or its even functional calculus, yields

\[
\Sigma-\Sigma_0=L^*T^2(I-T^2)^{-1}L\ge0.
\]

Because \(\|L\|^2\le b^2/d\) and \(\|T\|\le q<1\),

\[
\boxed{\quad
0\le\Sigma(z)-\Sigma_0(z)\le
\frac{b^2}{d}\frac{q^2}{1-q^2}I
=\frac{4b^4}{d(d^2-4b^2)}I.
\quad}
\]

All inverse and square-root manipulations with the unbounded \(H_0\) are
understood through its closed quadratic form and the bounded perturbation
\(M\). The strict domain ensures bounded resolvents throughout. The parity
argument is specific to this SU(2) center flip and this pair of fundamental
plaquette terms; it has not been asserted for SU(3) or arbitrary added
interactions.

The enclosure on \(F(z)\) reverses the self-energy inequality:

\[
PH_0P-z-\Sigma_0(z)-\varepsilon I
\le F(z)\le PH_0P-z-\Sigma_0(z),
\]

where \(\varepsilon=4b^4/[d(d^2-4b^2)]\). Every finite compression inherits
these inequalities. A finite compression alone does not settle the
spectrum of the full coarse operator.

## Exact finite entries

The report displays doubled coarse spins \(n=0,\ldots,N\). Only
intermediate spins \(s=1/2\), \(a=b\pm1/2\) can occur in \(\Sigma_0\). They
are included exactly even when their labels exceed \(N\). This finite
intermediate sum computes entries of the infinite operator; it is not a
truncation assumption about \(Q\).

The rational projector basis has squared norm

\[
g_{a,b,s}=\frac1{(2a+1)(2b+1)(2s+1)},\qquad
\chi_j(W)=(2j+1)\phi_{j,j,0}.
\]

Fundamental multiplication is given by the square of a Racah \(6j\) symbol,
with both trivalent vertices included. Summing the resulting rational
bilinear products divided by \(g_{a,b,s}(E_{a,b,s}-z)\) gives exact entries.
The finite Racah sum simplifies to the following closed Jacobi formula
for every doubled coarse spin \(n\ge0\). Write \(c_n=n(n+2)/4\), and

\[
D_A^\pm(n)=\alpha[Ac_{n\pm1}+Bc_n+3S/4]-z,\qquad
D_B^\pm(n)=\alpha[Ac_n+Bc_{n\pm1}+3S/4]-z.
\]

The normalized upward fundamental coupling has amplitude
\(v_A\sqrt{(n+2)/(2(n+1))}\), and the downward coupling has amplitude
\(v_A\sqrt{n/(2(n+1))}\). The latter is absent at \(n=0\).
They follow from the spin-\(j\) tensor spin-\(1/2\) projector dimensions
\(n+2\) and \(n\), divided by the full dimension \(2(n+1)\).
The right-path amplitudes replace \(v_A=v_1\) by \(v_B=v_2\).

Consequently the entire infinite operator is tridiagonal:

\[
(\Sigma_0)_{nn}=\sum_{\eta=A,B}v_\eta^2
\left[\frac{n+2}{2(n+1)D_\eta^+(n)}
+\mathbf1_{n>0}\frac{n}{2(n+1)D_\eta^-(n)}\right],
\]

\[
(\Sigma_0)_{n,n+1}=\frac{v_1v_2}{2}
\left[\frac1{D_A^+(n)}+\frac1{D_B^+(n)}\right],\qquad
(\Sigma_0)_{nm}=0\quad (|n-m|>1).
\]

The minus denominator is not evaluated at \(n=0\). Production uses these
closed rational entries; independent tests compare them to the Racah
contraction. No floating-point eigensolver or nested differentiation is used.

For \(A=B=3,S=1,z=0,v_1=v_2=v\), the entries simplify further:

\[
(\Sigma_0)_{00}=\frac{2v^2}{3\alpha},\qquad
(\Sigma_0)_{nn}=\frac{4v^2}{3\alpha(n+1)^2}\ (n\ge1),\qquad
(\Sigma_0)_{n,n+1}=\frac{2v^2}{3\alpha(n+1)(n+2)}.
\]

For the example above the first entries are

\[
\Sigma_0(0)=
\begin{pmatrix}
1/12&1/24&0&\cdots\\
1/24&1/24&1/72&\cdots\\
0&1/72&1/54&\cdots\\
\vdots&\vdots&\vdots&\ddots
\end{pmatrix},\qquad
0\le\Sigma-\Sigma_0\le\frac1{48}I.
\]

The varying diagonal and off-diagonal coefficients already rule out
identifying this second-order term with a constant plus one Wilson
character multiplication operator. A theory space for repeated
elimination must accommodate the additional dependence on electric spin
and spectral parameter.

## Certified error for displaying only finitely many coarse spins

Let \(P_N\) retain \(n\le N\), and extend the displayed matrix
\(P_N\Sigma_0P_N\) by zero on all other coarse spins. Define

\[
t_N=\alpha[\max(A,B)c_N+\min(A,B)c_{N+1}+3S/4]-z>0,\quad
e_N=(\Sigma_0)_{N,N+1}.
\]

For a vector supported on \(n\ge N+1\), a single magnetic multiplication
reaches only intermediate labels \((n\pm1,n,1)\) or their outer-path swap.
Their electric resolvent denominators are at least \(t_N\).
Consequently

\[
\|(I-P_N)\Sigma_0(I-P_N)\|\le b^2/t_N.
\]

The only cross-boundary Jacobi entry is \(e_N\), giving a self-adjoint
off-block of norm exactly \(e_N\). Hence

\[
\|\Sigma_0-P_N\Sigma_0P_N\|\le b^2/t_N+e_N,
\]

\[
\boxed{\|\Sigma-P_N\Sigma_0P_N\|
\le\varepsilon+b^2/t_N+e_N.}
\]

Every term is rational. For fixed parameters the extra display error
is \(O(N^{-2})\); enlarging the display does not remove the interacting
remainder \(\varepsilon\). The certificate records both errors separately.
This is a bound on the whole coarse operator, not a gap inferred from a
finite eigenvalue calculation.

## Scope and refusal behavior

- `ENCLOSED` earns the stated finite-graph identities and all-spin
  self-energy enclosure in the strict domain.
- `INCONCLUSIVE_DOMAIN` retains the exact electric and cross-block
  identities, but returns no self-energy matrix or remainder. It does not
  assert a pole, a vanishing mass, or infeasibility of other estimates.
- Both statuses can be replayed. Replay recomputes the complete canonical
  certificate and rejects altered claims even if their digest is resealed.
- Neither status earns a spectral gap, actual vacuum alignment, all-scale
  estimates, a continuum limit, or a formal theorem-prover tier.

The Hamiltonian framework follows
[Kogut and Susskind (1975)](https://doi.org/10.1103/PhysRevD.11.395).
The identities and error estimates above are supplied as explicit
derivations for this finite graph; they are not attributed to that paper
as new or previously stated theorems.
