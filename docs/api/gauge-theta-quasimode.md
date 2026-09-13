# Quantitative smooth approximation of the actual theta vacuum

For the actual two-plaquette, seven-link SU(2) Hamiltonian, a global smooth
trial gives an explicit error without a spatial cutoff or an unspecified
elliptic regularity constant. For \(0<\kappa\le1/64\), let \(\psi_\kappa\)
be its positive normalized reduced ground state. The trial defined below
satisfies
\[
\|(H_\kappa-\lambda)v_\kappa\|_2\le83\kappa,\qquad
\|\psi_\kappa-v_\kappa\|_2\le332\kappa,\qquad
\|\psi_\kappa^2-v_\kappa^2\|_1\le664\kappa.                 \tag{1}
\]
All norms here use normalized product Haar measure. The last norm is
unhalved L1 distance. The bounds remain valid under the corresponding
unitary or density-preserving coordinate changes.

The source is
omnibias.geometry.gauge.transfer.theta_quasimode.su2_theta_quasimode_error.
It consumes the dynamically replayed
[actual weak-theta certificate](gauge-theta-weak-blocks.md), including its
absolute first excited energy bound. It does not replace the actual vacuum
by a Gibbs density or infer a normalized-kernel error from L1 distance alone.
This is a fixed graph result, with no ambient-volume, refinement or
continuum claim. The analytic proof below is written mathematics;
canonical rational replay is distinct from theorem-prover verification.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.theta_quasimode import (
    su2_theta_quasimode_error,
    replay_su2_theta_quasimode_certificate,
)

row = su2_theta_quasimode_error(Fraction(1, 1024))
assert row["status"] == "PASS"
assert row["arithmetic"]["residual_l2_upper"] == "83/1024"
assert row["arithmetic"]["vacuum_density_l1_upper"] == "83/128"
assert replay_su2_theta_quasimode_certificate(row["certificate"])
assert not row["density_error_is_normalized_kernel_error"]
assert not row["continuum_claim"]
```

## 1. Operator, global trial and domain

Use the exact two-holonomy reduction of the original seven unit-weight
electric links, retaining the full reduced space \(L^2(G^2)\), where
\(G=\mathrm{SU}(2)\):
\[
H_\kappa=\frac{\kappa}{2}C_\theta+\frac2\kappa S,\qquad
C_\theta=3(C_x+C_y)+C_{\mathrm{diag,left}},\qquad
C_{\mathrm{fund}}=\frac34.                                \tag{2}
\]
Here \(S=A(x)+A(y)\), \(A(U)=2-\operatorname{Tr}U\), and the diagonal
Casimir uses simultaneous left translations of both variables. Its
carré du champ is
\[
\Gamma_\theta f=3\bigl(|D_x f|^2+|D_y f|^2\bigr)
                    +|D_x f+D_y f|^2.                    \tag{3}
\]
These are the exact reduced original-link coefficients, rather than a
product kinetic reference. The compact scalar elliptic operator has a
unique positive ground state, which is invariant under simultaneous
conjugation. Its Friedrichs realization has discrete spectrum.

Write unit quaternions \(x=(s,X)\), \(y=(t,Y)\), and set
\[
U=|X|^2=1-s^2,\quad V=|Y|^2=1-t^2,\quad W=X\cdot Y,\qquad
a=\frac1{\sqrt3}+\frac1{\sqrt5},\quad
b=2\left(\frac1{\sqrt5}-\frac1{\sqrt3}\right).
\]
Define
\[
F=aS+bW,\qquad Z_\kappa=\int_{G^2}e^{-2F/\kappa}\,dH^2,\qquad
v_\kappa=Z_\kappa^{-1/2}e^{-F/\kappa},\qquad
\lambda=\frac32(\sqrt3+\sqrt5)<6.                         \tag{4}
\]
The function \(F\) is a global polynomial in quaternion coordinates.
Consequently \(v_\kappa\) is smooth, positive and in the operator domain,
including at either antipode. There is no coordinate-boundary condition.
Since \(|W|\le S/2\),
\[
\frac89S\le\frac2{\sqrt5}S\le F
   \le\frac2{\sqrt3}S\le\frac65S.                         \tag{5}
\]
In particular its only zero is \(x=y=I\). We will also use
\(a\le11/10\), \(|b|\le1/3\).
The quadratic Taylor polynomial of \(2F/\kappa\), under
\(x=\exp(i\sqrt\kappa\,u\cdot\sigma/2)\) and the analogous \(y\), is
\(z^\mathsf T(M^{-1/2}\otimes I_3)z\), where
\(M=\left(\begin{smallmatrix}4&1\\1&4\end{smallmatrix}\right)\).
This identifies the oscillator energy in (4), but the estimates below are
global polynomial inequalities.

## 2. Exact differentiated identities

For the left generator normalized by \(i\sigma_j/2\), direct quaternion
multiplication gives
\[
D_xF=aX+\frac b2(sY-X\times Y),\qquad
D_yF=aY+\frac b2(tX+X\times Y).
\]
Changing the quaternion cross-product convention changes both displayed
cross signs and leaves every contracted expression below unchanged.
Differentiating a second time gives
\[
C_\theta S=3S-12,\qquad C_\theta W=5W-\frac32st,\qquad
C_\theta F=a(3S-12)+b\left(5W-\frac32st\right).           \tag{6}
\]
Thus \(-C_\theta F(I,I)/2=6a+3b/4=\lambda\).
The identities
\[
4a^2+ab+b^2=4,\qquad 2a^2+8ab+b^2/2=0
\]
cancel the entire quadratic error in the gradient square, giving
\[
\begin{split}
\Gamma_\theta F-4(U+V)
={}&ab\bigl[(t-1)U+(s-1)V+4(s+t-2)W\bigr]\\
&+\frac{b^2}{4}\bigl[-2UV-6W^2+2(st-1)W\bigr].          \tag{7}
\end{split}
\]
For example, the separate-gradient contribution before this cancellation is
\[
|D_xF|^2+|D_yF|^2
=(a^2+b^2/4)(U+V)+ab(s+t)W-b^2W^2/2,
\]
and the diagonal contribution is
\(\left|(a+bt/2)X+(a+bs/2)Y\right|^2\).
These formulas also check the diagonal generator and its sign.

Set \(A_x=2(1-s)\), \(A_y=2(1-t)\). Then
\[
U\le A_x,\quad V\le A_y,\quad UV,W^2\le S^2/4,\quad
1-st\le S/2,\quad |W|\le S/2.
\]
The absolute value of the first bracket in (7) is at most \(5S^2/4\).
The absolute value of the second bracket, after division by four, is at
most \(5S^2/8\). Hence
\[
|\Gamma_\theta F-4(U+V)|\le
 \left(\frac54|ab|+\frac58b^2\right)S^2
 \le\frac{19}{36}S^2.                                   \tag{8}
\]
Also \(4(S-U-V)=A_x^2+A_y^2\le S^2\). Equation (6) and
\(1-st\le S/2\) imply
\[
\left|-\frac12C_\theta F-\lambda\right|
\le\frac12\left(3a+\frac{13}{4}|b|\right)S
\le\frac52S,\qquad
\left|2S-\frac12\Gamma_\theta F\right|
\le\frac{55}{72}S^2\le S^2.                              \tag{9}
\]
The exact exponential differentiation rule for the positive Casimir is
\[
\frac{(H_\kappa-\lambda)e^{-F/\kappa}}{e^{-F/\kappa}}
=-\frac12C_\theta F-\lambda
       +\frac{2S-\Gamma_\theta F/2}{\kappa}.
\]
Therefore, pointwise on all of \(G^2\),
\[
|(H_\kappa-\lambda)v_\kappa|
       \le\left(\frac52S+\frac{S^2}{\kappa}\right)v_\kappa. \tag{10}
\]

## 3. Explicit normalized Haar moments

The pushforward of normalized SU(2) Haar measure under \(A=2-\operatorname{Tr}U\)
has density
\[
\frac{\sqrt{A(4-A)}}{2\pi}\,dA,\qquad 0\le A\le4.
\]
Bounding this density above by \(\sqrt A/\pi\), then using
\(\int_0^1\sqrt{u(1-u)}\,du=\pi/8\), gives, for nonnegative \(g\),
\[
\int_{G^2}g(S)\,dH^2
 \le\frac1{8\pi}\int_0^\infty s^2g(s)\,ds.              \tag{11}
\]
For a lower bound restrict to \(S\le\kappa\le1/64\). On this simplex each
single density is at least \(3\sqrt A/(4\pi)\). By (5),
\(e^{-2F/\kappa}\ge e^{-12/5}\ge1/27\); the latter follows from \(e<3\).
Thus
\[
Z_\kappa\ge\frac9{128\pi}\frac{\kappa^3}{3\cdot27}
 =\frac{\kappa^3}{1152\pi}\ge\frac{\kappa^3}{4096}.       \tag{12}
\]
This is a lower bound for the normalization of the actual smooth trial,
including its signed mixed term.

Using (5), (11), (12) and \(\pi\ge3\), its normalized moments obey
\[
\int S^m v_\kappa^2\,dH^2
\le c_m\kappa^m,\qquad
c_m=\frac{512}{3}(m+2)!\left(\frac9{16}\right)^{m+3}.    \tag{13}
\]
The needed exact values are
\[
c_2=\frac{59049}{256},\qquad
c_3=\frac{2657205}{4096},\qquad
c_4=\frac{71744535}{32768}.
\]
Squaring (10) and integrating therefore gives
\[
\|(H_\kappa-\lambda)v_\kappa\|_2^2
\le\left(\frac{25}{4}c_2+5c_3+c_4\right)\kappa^2
=\frac{225271935}{32768}\kappa^2
<83^2\kappa^2.                                         \tag{14}
\]

## 4. Transfer to the actual vacuum

The canonical [weak-theta proof, Section 4](gauge-theta-weak-blocks.md)
establishes the absolute first full-reduced excited energy
\(E_1\ge32/5\) throughout \(0<\kappa\le1/64\). Since \(\lambda<6\), the
distance from \(\lambda\) to every excited spectral value is at least
\(2/5\). This absolute statement, rather than only a relative gap,
licenses the following projection estimate.

Let \(P=|\psi_\kappa\rangle\langle\psi_\kappa|\). By the spectral theorem,
\[
\|(1-P)v_\kappa\|_2
\le\frac52\|(H_\kappa-\lambda)v_\kappa\|_2.
\]
Both vectors are positive and normalized, so
\(c=\langle\psi_\kappa,v_\kappa\rangle>0\). Consequently
\[
\|\psi_\kappa-v_\kappa\|_2^2
=2(1-c)\le2(1-c^2)
=2\|(1-P)v_\kappa\|_2^2
<16\|(H_\kappa-\lambda)v_\kappa\|_2^2.                   \tag{15}
\]
No overlap lower bound and no small-residual eigenvalue-identification
assumption enter this argument. Even when the displayed error exceeds
the trivial distance bound, it remains a valid upper bound.
Cauchy--Schwarz then gives the density estimate in (1):
\[
\|\psi_\kappa^2-v_\kappa^2\|_1
\le2\|\psi_\kappa-v_\kappa\|_2\le664\kappa.
\]
This estimate supplies quantitative L1 input to a separate normalized-kernel
comparison. Control of its own-marginal denominators and of the normalized
kernel tails remains necessary for such a comparison.

## 5. Exact API and scope

The callable signature is
su2_theta_quasimode_error(kappa: int | Fraction = Fraction(1, 64)).
It returns a JSON-serializable report with the same fields in its sealed
certificate payload. The nested weak-theta source is rebuilt and replayed;
its absolute excited-energy bound, actual-vacuum identification, coupling
and full reduced sector are checked.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.theta_quasimode import (
    replay_su2_theta_quasimode_certificate,
    su2_theta_quasimode_error,
)

report = su2_theta_quasimode_error(Fraction(1, 1024))
assert report["status"] == "PASS"
assert report["arithmetic"]["vacuum_density_l1_upper"] == "83/128"
assert replay_su2_theta_quasimode_certificate(report["certificate"])
```

Positive rational inputs outside the proved interval return INCONCLUSIVE
with null actual error bounds and false actual-error flags. Malformed or
nonpositive inputs are rejected. The coefficient budgets remain visible
for review. Canonical replay rebuilds all fields, including the nested
source and sector flags. Digest resealing does not make altered evidence
canonical.

The proof uses all reduced representations. It does not assert a gap on
the unrestricted original seven-link scalar space, an ambient conditional
theorem, a volume-uniform result or a continuum limit. Neither rational
arithmetic nor these written operator arguments set the Mathlib or
theorem-prover flags.
