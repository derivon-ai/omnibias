# Periodic corner comparisons and a global upper trial

The module `omnibias.geometry.gauge.transfer.periodic_corner` supplies two
volume-uniform harmonic results and an actual upper bound for the periodic
SU(2) ground-energy density. A separate arithmetic function checks the
implication of a proposed nonlinear lower bound. That function does not
verify its input hypotheses.

```python
from fractions import Fraction as Q
from omnibias.geometry.gauge.transfer.periodic_corner import (
    periodic_corner_energy_budget,
    replay_periodic_corner_certificate,
    su2_periodic_corner_harmonic,
    su2_periodic_upper_density,
)

harmonic = su2_periodic_corner_harmonic()
upper = su2_periodic_upper_density(Q(1, 8192))
conditional = periodic_corner_energy_budget(Q(1, 64), Q(1, 64))
assert harmonic["arithmetic"]["harmonic_density_secant_lower"] == "29/112"
assert upper["arithmetic"]["kappa_upper"] == str(Q(1, 2**39))
assert upper["actual_periodic_upper_density_verified"]
assert conditional["arithmetic"]["conditional_action_difference_over_kappa_lower"] == "15/56"
assert not conditional["actual_periodic_vacuum_decorrelation_verified"]
for row in (harmonic, upper, conditional):
    assert replay_periodic_corner_certificate(row["certificate"])
```

Every rational input is an integer or `Fraction`; booleans, floats, and
strings are refused by the public API. Serialized rational inputs have a
canonical fraction spelling. Replay rebuilds the entire certificate,
including its model, scope, status, and unearned claims.

## 1. Original model and coherent corner

Let the spatial lattice be the periodic cubic graph of side \(N\ge3\).
There are \(V=N^3\) sites, \(3V\) original links, and \(3V\) elementary
plaquettes, counted once. Product Haar measure is normalized. The model is

\[
 H_\kappa=\frac\kappa2\sum_e C_e+
       \frac2\kappa\sum_p(2-\operatorname{Tr}U_p),
 \qquad C_{1/2}=\frac34.
 \tag{1}
\]

At each site \(x\), use the three faces in the coherent order
\(xy,yz,zx\). Their transported product has the six-link boundary \(D_x\).
Write \(S_x\) for their action sum and \(A_{D,x}\) for its boundary action.
Every plaquette belongs to exactly one such anchored triple. Consider

\[
 H_{\kappa,t}=H_\kappa+
       \frac{t}{2\kappa}\sum_x(S_x-A_{D,x}),
 \qquad e_{\kappa,N}(t)=V^{-1}\inf\operatorname{spec}H_{\kappa,t}.
 \tag{2}
\]

The nonabelian chord triangle inequality gives \(A_{D,x}\le3S_x\).
Consequently the magnetic part of (2) is at least \(1-t/2\) times that of
(1), and is positive for the whole interval \(0\le t\le1/4\).

For each finite lattice the full scalar compact elliptic Hamiltonian has
a unique positive groundstate. Gauge transformations commute with it;
uniqueness and positivity therefore make that groundstate gauge invariant.
Thus its scalar and physical ground energies agree. An arbitrary scalar
trial gives an upper bound on the physical ground energy. This reasoning
does **not** assert that gauge projection decreases a trial's Rayleigh
quotient.

## 2. Exact identity-background harmonic symbol

In original link coordinates \(U_e=\exp(iA_e\cdot\sigma/2)\), the
quadratic Hamiltonian for one color is

\[
 \frac\kappa2(-\Delta_A)+\frac1{2\kappa}A^*KA,
 \qquad K=B^*B,
 \tag{3}
\]

where \(B\) is the oriented original-link curl. At momentum
\(k_i=2\pi n_i/N\), put \(z_i=e^{ik_i}-1\) and
\(\lambda=\sum_i|z_i|^2\). Its coherent face rows are

\[
 b_{xy}=(-z_2,z_1,0),\quad b_{yz}=(0,-z_3,z_2),\quad
 b_{zx}=(z_3,0,-z_1).
 \tag{4}
\]

For nonzero momentum, \(B^*B\) has eigenvalues \(\lambda,\lambda,0\).
At zero momentum all three eigenvalues vanish. Per color the common
kernel therefore has \(N^3-1\) gauge modes and three constant toron modes.
The Gaussian reference is defined on the orthogonal complement of this
kernel; it has \(2N^3-2\) oscillators per color.

Let \(v=(1,1,1)^T\), and let \(P(k)\) project onto the two-dimensional
range of the face curl. Put \(p(k)=\|P(k)v\|^2\in[0,3]\). The two
nonzero squared frequencies of the averaged tilt are

\[
 \lambda(1+t/4),\qquad
 \lambda\{1+t(1-p)/4\}.
 \tag{5}
\]

The second expression also equals
\((1-t/2)\lambda+(t/4)|z_1+z_2+z_3|^2\). All zero modes remain zero.
With \(\mathbb E_N\) the normalized finite momentum sum,

\[
 e_{\mathrm q,N}(t)=\frac32\mathbb E_N\sqrt\lambda
  \left[\sqrt{1+t/4}+\sqrt{1+t(1-p)/4}\right].
 \tag{6}
\]

The zero-momentum term in every expression is defined as zero.

## 3. Uniform Gaussian moment and averaged secant

Write \(q_i=|z_i|^2=4\sin^2(k_i/2)\). Discrete orthogonality and
independence of the coordinate sums give, exactly for every \(N\ge3\),

\[
 \mathbb E_Nq_i=2,\qquad \mathbb E_Nq_iq_j=4\ (i\ne j),\qquad
 \mathbb E_N\lambda=6,\qquad 0\le\lambda\le12.
 \tag{7}
\]

Independent momentum reflections cancel sine cross terms. The local
matrix \(BK^{-1/2}B^*\) on the three corner faces has diagonal
\(d_{1/2}\) and off-diagonal \(-c_{1/2}\), where

\[
 d_{1/2}=\frac23\mathbb E_N\sqrt\lambda\le\frac{2\sqrt6}3,
 \qquad
 c_{1/2}=\frac14\mathbb E_N\frac{q_1q_2}{\sqrt\lambda}
       \ge\frac1{\sqrt{12}}.
 \tag{8}
\]

Each color's Gaussian covariance is \(\kappa K^{-1/2}/2\), and each
quadratic face action is \(|b|^2/4\). Summing the three colors gives

\[
 \frac{\langle S_{\mathrm q}-A_{D,\mathrm q}\rangle}{\kappa}
 =\frac94c_{1/2}\ge\frac9{8\sqrt3}>\frac9{14},\qquad
 \frac{\langle A_{D,\mathrm q}\rangle}{\langle S_{\mathrm q}\rangle}
 =1-\frac{2c_{1/2}}{d_{1/2}}
 \le1-\frac1{2\sqrt2}<\frac23.
 \tag{9}
\]

In particular \(e'_{\mathrm q,N}(0)\ge9/28\). Twice differentiating
the explicit scalar square roots in (6), using \(p\in[0,3]\), gives

\[
 e''_{\mathrm q,N}(t)\ge
 -\frac{15\sqrt6}{128}(8/7)^{3/2}>-\frac12
 \quad(0\le t\le1/4).
 \tag{10}
\]

The square of the displayed positive constant is \(675/5488<1/4\).
Integrating (10) twice therefore proves

\[
 \boxed{\frac{e_{\mathrm q,N}(1/4)-e_{\mathrm q,N}(0)}{1/4}
       \ge\frac{29}{112}.}
 \tag{11}
\]

This is a uniform finite-volume theorem, not an extrapolation of
finite-size numerical values.

## 4. Single-corner harmonic energy difference

For a single local defect, put
\(D=(I_3-vv^*)/4\) and \(G(s)=B_c(K+sI)^{-1}B_c^*\).
The same symmetry gives diagonal \(d\), off-diagonal \(-c\), with

\[
 d=\frac23\mathbb E_N\frac\lambda{s+\lambda}\le\frac4{s+6},
 \qquad c=\frac14\mathbb E_N\frac{q_1q_2}{s+\lambda}
       \ge\frac1{s+12},\qquad 0\le c\le d/2.
 \tag{12}
\]

The determinant is \((1-d/2+c)(1+(d+c)/4)^2\). It decreases in \(d\)
and increases in \(c\) on the relevant domain, so it is at least

\[
 f(s)=\left(1-\frac2{s+6}+\frac1{s+12}\right)
       \left(1+\frac1{s+6}+\frac1{4(s+12)}\right)^2.
 \tag{13}
\]

The numerator of \(f-1\), over denominator
\(16(s+6)^3(s+12)^3\), is

\[
 24s^5+993s^4+15509s^3+111654s^2+353484s+344088.
 \tag{14}
\]

Hence \(f>1\) for \(s\ge0\). On \([0,4]\), the sign of
\(f-1083/1024\) is the sign of

\[
 s(6106752+2099232s+189704s^2-6540s^3-1650s^4-59s^5).
 \tag{15}
\]

The three negative terms are at most \(56336s^2\), so (15) is
nonnegative. The common kernel cancels in the finite-rank determinant
identity, and scalar integration gives

\[
 \Delta E_{\mathrm q,N}
 =\frac3{4\pi}\int_0^\infty s^{-1/2}\log\det(I+DG(s))\,ds
 \ge\frac3\pi\log\frac{1083}{1024}
 >\frac{413}{7942}.
 \tag{16}
\]

The integral converges at both endpoints. Positivity allows all of it
outside \([0,4]\) to be discarded; no unestimated ultraviolet or infrared
tail is subtracted. The final rational bound uses
\(\log(1+x)\ge x/(1+x)\) and \(\pi<22/7\).

## 5. Actual global upper trial, including all zero modes

Now \(K\) denotes the full \(9V\)-dimensional curl matrix including
all three colors, all gauge directions, and all constant toron directions.
For \(0<\kappa\le1\), choose

\[
 \eta=\kappa^{2/3},\qquad \Omega=\sqrt{K+\eta I},\qquad
 r=\kappa^{1/4},\qquad a=\frac\pi{2r},
 \qquad u(A)=e^{-A^*\Omega A/(2\kappa)}
             \prod_{i=1}^{9V}\cos(aA_i)
 \tag{17}
\]

on the box \(|A_i|<r\), extended by zero. No zero mode is discarded.
The cosine factors vanish linearly, so \(u\in H_0^1\) of the box.
Each link radius is at most \(\sqrt3r<2\pi\), inside the injective
SU(2) exponential chart. With

\[
 s(\rho)=\frac{\sin(\rho/2)}{\rho/2},\qquad
 J(A)=\prod_e s(|A_e|)^2,
 \qquad \psi(A)=J(A)^{-1/2}u(A),
 \tag{18}
\]

the actual Haar norm is exactly the flat norm of \(u\), up to a harmless
constant measure normalization. Thus no product density-ratio estimate
and no global cutoff union bound occurs.

### 5.1 Dimension-independent moments

The probability density proportional to \(u^2\) is even under the
simultaneous transformation \(A\mapsto-A\). Its negative logarithm is
strongly convex, with Hessian at least
\(2\sqrt\eta I/\kappa=\sigma^{-2}I\), where
\(\sigma^2=\kappa^{2/3}/2\). The Brascamp–Lieb variance inequality yields

\[
 \operatorname{Var}(f)\le\sigma^2\mathbb E|\nabla f|^2.
 \tag{19}
\]

Its convex-support version applies to the cosine barrier: one can first
use smooth convex approximations of the extended potential and then pass
to the limit. Equivalently, the weighted Bochner identity has curvature
at least \(\sigma^{-2}\), with nonnegative convex-domain boundary term.
No physical gap or unknown property of the actual vacuum is an input.
The underlying variance and marginal inequalities are proved in
[Brascamp–Lieb (1976)](https://doi.org/10.1016/0022-1236(76)90004-5).

Applying (19) to each coordinate, whose mean is zero, and then to
\(|A_e|^2\), proves

\[
 \mathbb E|A_e|^2\le3\sigma^2,\qquad
 \mathbb E|A_e|^4\le
 4\sigma^2\mathbb E|A_e|^2+(\mathbb E|A_e|^2)^2
 \le21\sigma^4.
 \tag{20}
\]

### 5.2 Cutoff derivatives without a volume-dependent normalization

Remove the \(i\)-th cosine factor and integrate all other coordinates.
The resulting cavity marginal \(h_i(x)\) is even and log-concave, hence
nonincreasing in \(|x|\). Log-concavity of marginals follows from
[Prékopa (1973)](https://acta.bibl.u-szeged.hu/14411/), Theorem 6.
Against the reference density proportional to \(\cos^2(ax)\) on
\((-r,r)\), the functions \(h_i\) and \(\tan^2(ax)\) have opposite
monotonicity in \(|x|\). The elementary two-copy covariance identity
therefore gives

\[
 \mathbb E\tan^2(aA_i)
 \le\frac{\int_{-r}^r\sin^2(ax)\,dx}
          {\int_{-r}^r\cos^2(ax)\,dx}=1.
 \tag{21}
\]

This uses the exact one-coordinate marginal, not a product assumption.
Let \(\chi\) be the product of cosines and \(\phi_\Omega\) the Gaussian.
The oscillator groundstate-transform identity is

\[
 \frac{q_K[\chi\phi_\Omega]}{\|\chi\phi_\Omega\|^2}
 =\frac12\operatorname{Tr}\Omega
  -\frac\eta{2\kappa}\mathbb E|A|^2
  +\frac\kappa2\mathbb E|\nabla\log\chi|^2.
 \tag{22}
\]

There is no separate cross-term error. Since
\(\sqrt{\lambda+\eta}-\sqrt\lambda\le\sqrt\eta\), (21)–(22) imply

\[
 V^{-1}q_K[u]/\|u\|^2
 \le e_{\mathrm q,N}(0)+\frac92\kappa^{1/3}
                      +\frac{9\pi^2}{8}\kappa^{1/2}.
 \tag{23}
\]

### 5.3 Original SU(2) electric metric

The exact Haar half-density transform on one exponential chart is

\[
 J^{1/2}C J^{-1/2}=-\nabla\cdot(G^{-1}\nabla)-\frac14,
 \qquad
 G^{-1}=P_{\rm radial}+
  \left(\frac\rho{2\sin(\rho/2)}\right)^2P_{\rm tangential}.
 \tag{24}
\]

Indeed \(G^{-1}\nabla s=\nabla s\) and
\(s''+2s'/\rho=-s/4\). This proves the constant term and extends from
the smooth compact core to \(H_0^1\) by form closure. On the support of
(17), sine bounds give

\[
 I\le G^{-1}\le M I,\qquad
 M=(1-r^2/8)^{-2}\le64/49,\qquad
 M-1\le(16/49)\kappa^{1/2}.
 \tag{25}
\]

The positive metric excess must be retained. The negative half-density
term is \(-3\kappa V/8\), which can safely be dropped in an upper bound.

### 5.4 Nonabelian magnetic remainder and its parity

For a four-link plaquette, differentiate the ordered unitary product
\(\prod_{j=1}^4\exp(i t a_j\cdot\sigma/2)\), with orientation already
included in \(a_j\). If \(R=\sum_j|a_j|/2\), the trace's fourth
derivative is bounded by \(2R^4\). Taylor's integral remainder gives

\[
 2-\operatorname{Tr}U_p
 =\frac14\left|\sum_j a_j\right|^2+P_{3,p}(A)+R_{4,p}(A),
 \quad |R_{4,p}|\le\frac{(\sum_j|a_j|)^4}{192}.
 \tag{26}
\]

The exact cubic polynomial \(P_{3,p}\) is odd under \(A\mapsto-A\).
Its mean under the actual trial density is therefore exactly zero;
noncommuting factors do not invalidate this polynomial parity. By
\((\sum_{j=1}^4r_j)^4\le64\sum_jr_j^4\) and (20), the magnetic
error is at most \(14\kappa^{1/3}\) per plaquette, or
\(42\kappa^{1/3}\) per site.

Combining this with (23) and (25), and using positivity of the quadratic
potential, gives

\[
 e_{\kappa,N}(0)\le
 M\left[e_{\mathrm q,N}(0)+\frac92\kappa^{1/3}
             +\frac{9\pi^2}{8}\kappa^{1/2}\right]+42\kappa^{1/3}.
 \tag{27}
\]

Finally \(e_{\mathrm q,N}(0)\le3\sqrt6<15/2\), \(\pi<22/7\), and
\(\kappa^{1/2}\le\kappa^{1/3}\) prove the explicit bound

\[
 e_{\kappa,N}(0)-e_{\mathrm q,N}(0)
 \le\frac{2346}{49}\kappa^{1/3}
       +\frac{40728}{2401}\kappa^{1/2}
 <\boxed{65\kappa^{1/3}}.
 \tag{28}
\]

This is an actual upper bound for all periodic sizes \(N\ge3\) and
\(0<\kappa\le1\). It requires no preferred toron groundstate: a trial
near the identity supplies an upper bound even if another region lowers
the actual energy. The API represents the smaller coupling interval as
\(0<\kappa\le\tau^3\) and returns the rational uniform error \(65\tau\).

## 6. What the conditional energy budget proves

Suppose the same actual periodic model satisfies

\[
 e_{\kappa,N}(0)\le e_{\mathrm q,N}(0)+\varepsilon_0,
 \qquad
 e_{\kappa,N}(1/4)\ge e_{\mathrm q,N}(1/4)-\varepsilon_t.
 \tag{29}
\]

The original vacuum is a trial for (2), so the variational principle and
translation invariance give, without differentiating the actual energy,

\[
 \frac{\langle S_x-A_{D,x}\rangle}{\kappa}
 \ge2\left[\frac{29}{112}-4(\varepsilon_0+\varepsilon_t)\right].
 \tag{30}
\]

For \(\varepsilon_0=\varepsilon_t=1/64\), this is \(15/56>1/4\).
The upper premise has the direct construction above. The conditional
budget does not construct the nonlinear tilted lower premise.

Generic commuting toron backgrounds shift charged-color momentum grids;
their lack of the symmetry used in (8) prevents automatic reuse of that
Gaussian proof. In particular, the second squared dispersion includes
\(+\frac t2\sum_{i<j}\cos(k_i-k_j)\), so scalar positive-hopping
diamagnetism is not an automatic toron-minimization argument. A nonlinear
lower comparison must handle these modes or use a separately justified
open-box argument. No physical gap, continuum theory, or Clay parent is
established by any function on this page.

## 7. A harmonic surface bound and a conditional box budget

`periodic_corner_box_surface(ell)` certifies a separate harmonic comparison
for \(N\ge\max(3,\ell)\), \(\ell\ge2\), and \(0\le t\le1/4\).
Partition each periodic coordinate into consecutive vertex intervals with
lengths in \([\ell,2\ell-1]\). Such a partition exists: put all the
remainder in one interval of length \(\ell+(N\bmod\ell)\). Internal
links are the ordinary consecutive links of these open intervals; even
when there is one interval, its wrapping periodic seam is excluded.

Keep the complete potential of every wholly contained corner cell. For
each crossing cell, use \(V_{x,t}\ge(2-t)S_x\), retain the actions of
its wholly internal faces with coefficient \(2-t\), and discard the
remaining positive terms. Retain every internal original electric term.
This gives a sum of independent compact box operators below the original
operator. It does not substitute a neutral box groundstate for a
conditional exterior state: the scalar bottom of each box is its neutral
bottom by the uniqueness and positivity argument of Section 1.

For the quadratic comparison, extend the direct sum of box matrices by
zero on removed original links, writing it \(K_{\rm split}\). On a
crossing cell the original face coefficient is
\(Q_t=I+t(I-vv^*)/4\); the retained replacement is
\((1-t/2)P\), where \(P\) selects internal faces. Since
\(Q_t\ge(1-t/2)I\), the difference is positive and has rank at most
three after composition with the three original curl rows. Thus, if
\(n_c\) cells cross box boundaries,

\[
 0\le K_{\rm split}\le K_{N,t}\le(51/4)I<16I,\qquad
 \operatorname{rank}(K_{N,t}-K_{\rm split})\le3n_c.
 \tag{31}
\]

Positive rank-\(q\) perturbations of matrices in \([0,16I]\) change
their square-root trace by at most \(4q\), by eigenvalue interlacing and
telescoping. Including the SU(2) factor \(3/2\), (31) costs at most
\(18n_c\) in harmonic energy. All zero modes are included in this
finite-matrix inequality.

A box with vertex side lengths \(n_i\) contains exactly
\(\prod_i(n_i-1)\) complete corner cells. Hence
\(n_c/N^3\le3/\ell\), and

\[
 0\le e^{\rm q}_{N,t}-N^{-3}\sum_B E^{\rm q}_{B,t}
 \le\boxed{54/\ell}.
 \tag{32}
\]

`periodic_corner_box_lower_budget(ell, local_error)` adds this proved
surface term to a **hypothesized** actual box density error. If every
retained rectangular box satisfies
\(E_{B,t}\ge E^{\rm q}_{B,t}-\varepsilon_B|B|\) at the same coupling,
then the variational lower decomposition and (32) imply

\[
 e_{\kappa,N}(t)\ge e^{\rm q}_{N,t}-54/\ell-\varepsilon_B.
 \tag{33}
\]

The helper verifies the arithmetic target \(54/\ell+\varepsilon_B\le1/64\).
It does not validate a supplied \(\varepsilon_B\) as an actual operator
bound, nor acquire a nonlinear lower flag from its numerical value.
