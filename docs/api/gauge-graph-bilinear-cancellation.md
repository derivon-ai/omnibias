# Gauge-polar cancellation in the original Fourier norm

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer import (
    su2_wilson_polar_vacuum,
    replay_su2_wilson_polar_vacuum_certificate,
)

source = su2_wilson_polar_vacuum(15, correction_radius=Fraction(1, 8))
assert source["actual_vacuum_verified"]
assert source["witness"]["arithmetic"]["quadratic_constant"] == "8/9"
assert source["witness"]["arithmetic"]["self_map_slack"] == "49/5625"
assert replay_su2_wilson_polar_vacuum_certificate(source["certificate"])
assert not source["continuum_claim"]
assert not source["theorem_prover_verified"]
```

The analytic estimate below is the proof behind the improved bilinear
constant in [the complete Wilson source](gauge-wilson-polar-source.md).
The finite rational certificate replays the source inequalities and scope;
it does not formally verify this operator proof.

12 September 2026. Let a finite connected oriented SU(2) graph have neutral
Gauss law at every vertex, unit electric edge weights, and a nonconstant
gauge-invariant electric energy floor \(E_{\min}>0\). In the original
edge Fourier norm defined below, for every diameter weight \(b\ge1\),
\[
\boxed{N_b\!\left(C_0^{-1}\Pi_H\Gamma(f,h)\right)
 \le \frac{8}{3E_{\min}}N_b(f)N_b(h).}
\tag{1}
\]
For simple graphs of girth at least four, \(E_{\min}=3\) is a valid
uniform floor, and the constant is **\(8/9\)**. Neither the number of
edges nor the vertex valences enter this bound. Arbitrary signs, complex
coefficients and vertex-intertwiner multiplicities are allowed.

The [theta proof](gauge-theta-bilinear.md) used positive spherical
product coefficients. On a general graph those scalar coefficients need
not be positive. Here a positive *dominating probability pack*, obtained
from the two polar magnitudes of an arbitrary matrix, replaces them.
It retains the exact mean-Casimir cancellation without asserting positivity
of the Fourier matrices themselves.

This is an all-spin bilinear estimate on each finite graph, with a constant
uniform over the stated graph family. An actual-vacuum certificate still
requires its source, residual and contraction gates. No continuum limit,
weak-coupling continuation, or Clay parent follows from (1).

## 1. Original-edge norm and gauge covariance

Use normalized product Haar measure and
\[
 C=\sum_e C_e,\qquad C_e D_{j_e}=j_e(j_e+1)D_{j_e},\qquad
 \Gamma(f,h)=\tfrac12\{fCh+hCf-C(fh)\}.
\tag{2}
\]
The product in \(\Gamma\) is complex bilinear; no conjugation is implicit.
Write the Peter-Weyl expansion in the convention
\[
 f(U)=\sum_{\mathbf j}\operatorname{Tr}
   [F_{\mathbf j}\rho_{\mathbf j}(U)],\qquad
 \rho_{\mathbf j}(U)=\bigotimes_eD_{j_e}(U_e),\qquad
 E_{\mathbf j}=\sum_ej_e(j_e+1).
\tag{3}
\]
The representation dimensions are included in \(F_{\mathbf j}\) under
this convention. The nuclear norm is that of this original-edge matrix;
it is not the smaller norm obtained after fixing a spanning-tree gauge.

For nonconstant labels put \(X_{\mathbf j}=\{e:j_e>0\}\), and let
\(D(X)\) be the diameter of \(X\) in the ambient edge line graph.
Singleton diameter is zero. Define
\[
 w_{\mathbf j}=b^{D(X_{\mathbf j})},\qquad
 N_{b,i}(f)=\sum_{\mathbf j\ne0}
 j_iE_{\mathbf j}w_{\mathbf j}\|F_{\mathbf j}\|_1,
 \qquad N_b(f)=\max_iN_{b,i}(f).
\tag{4}
\]
The constant coefficient is removed. For a disconnected physical graph,
apply the source construction separately to each connected component;
this avoids assigning a finite diameter to modes spanning components.

Under a vertex gauge transformation,
\(U_e\mapsto g_{s(e)}U_eg_{t(e)}^{-1}\). On the fixed product
representation space define
\[
 R_{\rm out}(g)=\bigotimes_eD_{j_e}(g_{s(e)}),\qquad
 R_{\rm in}(g)=\bigotimes_eD_{j_e}(g_{t(e)}).
\]
Uniqueness of each Peter-Weyl coefficient gives the exact intertwining
identity
\[
 F R_{\rm out}(g)=R_{\rm in}(g)F.
\tag{5}
\]
Consequently \(|F|=(F^\dagger F)^{1/2}\) commutes with every outgoing
star action, while \(|F^\dagger|=(FF^\dagger)^{1/2}\) commutes with
every incoming star action. Partial trace over all factors except edge
\(e\), using its source in the first case and its target in the second,
shows that the one-edge reductions commute with the irreducible spin
representation. Their values are therefore
\[
 \operatorname{Tr}_{\ne e}|F|
 =\operatorname{Tr}_{\ne e}|F^\dagger|
 =\frac{\|F\|_1}{2j_e+1}I_{j_e}.
\tag{6}
\]
In particular every traceless spin generator has zero expectation in
either polar magnitude. Mixed incoming/outgoing orientations cause no
exception: the partial trace eliminates the other factors of the chosen
star action. Equation (5) does not require \(F\) to be positive, normal,
rank one, or diagonal in vertex-intertwiner indices.

A nonzero neutral coefficient cannot have a support vertex of degree
one. Averaging over that vertex would annihilate its single nontrivial
irreducible factor. Its nonempty support therefore contains a cycle.
On a graph of girth \(g_0\),
\[
 E_{\mathbf j}\ge 3g_0/4,
 \qquad \sum_ej_e\le\frac23E_{\mathbf j}.
\tag{7}
\]
The second inequality uses \(j(j+1)\ge3j/2\) for half-integer spins.
All spins and all invariant tensors are included in these arguments.

## 2. A polar compression inequality

For any finite square matrix \(A\) and orthogonal projector \(P\),
\[
 \boxed{\|PAP\|_1\le
 \frac12\operatorname{Tr}\!\left[P(|A|+|A^\dagger|)\right].}
\tag{8}
\]
To prove it, extend the partial isometry in \(A=V|A|\) to a unitary.
Then the following block matrix is positive semidefinite:
\[
 \begin{pmatrix}|A^\dagger|&A\\A^\dagger&|A|\end{pmatrix}
 =\begin{pmatrix}V\\I\end{pmatrix}|A|
   \begin{pmatrix}V^\dagger&I\end{pmatrix}.
\tag{9}
\]
Compress by \(P\oplus P\). For any positive block matrix with diagonal
blocks \(C,D\) and off-diagonal block \(B\), choose left and right
singular vectors of \(B\), with phases making their matrix elements
nonnegative. Positivity on the difference of each such pair gives
\(2s_k(B)\le\langle u_k,Cu_k\rangle+
\langle v_k,Dv_k\rangle\). Summing proves
\(2\|B\|_1\le\operatorname{Tr}C+\operatorname{Tr}D\), hence (8).

Both polar magnitudes matter. For example, take
\[
 A=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
 P=\frac15\begin{pmatrix}4&2\\2&1\end{pmatrix}.
\]
Then \(\|PAP\|_1=2/5\), whereas
\(\operatorname{Tr}P|A|=1/5\). The one-sided proposed bound fails.
The right-hand side of (8) is \(1/2\), so the symmetric polar bound
passes. This example also explains why replacing matrix coefficients by
their signs or positive parts is not a justified substitute.

## 3. Complete fusion gives a positive dominating pack

Fix input blocks \(F_{\mathbf j},G_{\mathbf k}\), and regroup
\(A=F_{\mathbf j}\otimes G_{\mathbf k}\) by edges. On each edge use the
complete unitary Clebsch-Gordan decomposition
\[
 V_{j_e}\otimes V_{k_e}
 =\bigoplus_{\ell_e=|j_e-k_e|}^{j_e+k_e}V_{\ell_e}.
\tag{10}
\]
Let \(P_{\boldsymbol\ell}\) denote the resulting orthogonal product
fusion projectors. The Fourier matrix of the product in output block
\(\boldsymbol\ell\) is the compression
\(A_{\boldsymbol\ell}=P_{\boldsymbol\ell}AP_{\boldsymbol\ell}\),
expressed in its coupled basis. Every vertex-intertwiner multiplicity
remains inside this matrix. Off-diagonal representation blocks have zero
trace against the block-diagonal group representation and contribute no
additional coefficient.

Set \(a=\|F_{\mathbf j}\|_1\|G_{\mathbf k}\|_1\), and omit a pair
with \(a=0\). Define
\[
 q_{\boldsymbol\ell}
 =\frac{\operatorname{Tr}[P_{\boldsymbol\ell}
                 (|A|+|A^\dagger|)]}{2a}.
\tag{11}
\]
By (8),
\[
 q_{\boldsymbol\ell}\ge0,\qquad
 \sum_{\boldsymbol\ell}q_{\boldsymbol\ell}=1,\qquad
 \|A_{\boldsymbol\ell}\|_1\le a q_{\boldsymbol\ell}.
\tag{12}
\]
These are dominating weights, not assertions that the output matrices
are positive. The identity
\[
 |A|=|F_{\mathbf j}|\otimes|G_{\mathbf k}|,
 \qquad |A^\dagger|=|F_{\mathbf j}^\dagger|
                       \otimes|G_{\mathbf k}^\dagger|
\tag{13}
\]
will retain cancellation under this domination.

## 4. Exact mean-Casimir cancellation survives domination

Let \(J_e^a,K_e^a\) be Hermitian spin generators and put
\(\Omega_e=\sum_aJ_e^a\otimes K_e^a\). It is scalar on each edge
fusion block, with eigenvalue
\[
 \omega_{e,\boldsymbol\ell}
 =\frac{\ell_e(\ell_e+1)-j_e(j_e+1)-k_e(k_e+1)}2.
\]
Equations (6) and (13) make the expectation of every cross-generator
term zero, separately in both polar products. Thus
\[
 \sum_{\boldsymbol\ell}q_{\boldsymbol\ell}
       \omega_{e,\boldsymbol\ell}=0
 \quad\hbox{for each edge }e.
\tag{14}
\]
The electric product rule gives the output coefficient of \(\Gamma\)
as \(\gamma_{\boldsymbol\ell}A_{\boldsymbol\ell}\), where
\[
 \gamma_{\boldsymbol\ell}
 =\frac{E_{\mathbf j}+E_{\mathbf k}-E_{\boldsymbol\ell}}2
 =-\sum_e\omega_{e,\boldsymbol\ell},\qquad
 \sum_{\boldsymbol\ell}q_{\boldsymbol\ell}
      \gamma_{\boldsymbol\ell}=0.
\tag{15}
\]
The upper fusion bound \(\ell_e\le j_e+k_e\) implies
\[
 \gamma_{\boldsymbol\ell}\ge-Q(\mathbf j,\mathbf k),\qquad
 Q(\mathbf j,\mathbf k)=\sum_ej_ek_e.
\]
It is this *lower bound on \(\gamma\)*, equivalently an upper bound
on \(\sum_e\Omega_e\), which is used. The opposite inequality for
\(\Omega_e\) would be false already for two spin-half singlets.
Zero mean and equality of the mean positive and negative parts give
\[
 \boxed{\sum_{\boldsymbol\ell}q_{\boldsymbol\ell}
       |\gamma_{\boldsymbol\ell}|
 \le2\sum_ej_ek_e.}
\tag{16}
\]
Keep the constant output in this probability pack. Haar centering removes
it only after the zero-mean identity has been used.

## 5. Anchors and diameter weights

If the two input supports are disjoint, all cross generators vanish and
\(\Gamma\) is zero. Otherwise their supports intersect. Every output
support is contained in their union, so the ambient line-graph triangle
inequality gives
\[
 w_{\boldsymbol\ell}\le w_{\mathbf j}w_{\mathbf k},\qquad
 \ell_i\le j_i+k_i.
\tag{17}
\]
The supports themselves need not be internally connected. An intersection
edge supplies the intermediate point in the diameter estimate.

For finite sums, put
\(\alpha_{\mathbf j}=w_{\mathbf j}\|F_{\mathbf j}\|_1\) and
\(\beta_{\mathbf k}=w_{\mathbf k}\|G_{\mathbf k}\|_1\).
The output electric energy in (4) cancels the inverse in
\(T=C_0^{-1}\Pi_H\Gamma\). The constant output has zero anchor.
Taking (17) outside the probability sum *before* applying (16) gives
\[
 N_{b,i}(T(f,h))\le
 2\sum_{\mathbf j,\mathbf k}\alpha_{\mathbf j}\beta_{\mathbf k}
 (j_i+k_i)\sum_ej_ek_e.
\tag{18}
\]
There is no assertion that a diameter- or anchor-reweighted probability
pack still has zero mean.

For the contribution containing \(j_i\), use
\[
 \sum_{\mathbf k} k_e\beta_{\mathbf k}
 \le N_{b,e}(h)/E_{\min}\le N_b(h)/E_{\min}
\]
and (7). This gives
\[
 N_{b,i}(T(f,h))\le\frac{4}{3E_{\min}}
 \{N_b(h)N_{b,i}(f)+N_b(f)N_{b,i}(h)\}.
\tag{19}
\]
Taking the maximum proves (1). The sum over edges is absorbed by the
input electric energy, so it introduces no volume or maximum-degree
factor. For girth \(g_0\) one obtains \(B=32/(9g_0)\).

For completeness, the finite maximum of weighted nuclear-\(\ell^1\)
norms is a Banach norm on centered invariant coefficients. On each finite
graph of \(m\) edges,
\[
 \sum_{\mathbf j\ne0}E_{\mathbf j}\|F_{\mathbf j}\|_1
 \le2\sum_iN_{b,i}(f)\le2mN_b(f).
\tag{20}
\]
Generators and their products of order at most two have operator norm
bounded by a fixed multiple of \(E_{\mathbf j}\). Thus finite Fourier
sums converge in \(C^2\), and (1) extends to the completed coefficient
space. Passing \(CT(f,h)=\Pi_H\Gamma(f,h)\) to the limit identifies the
extension with the actual differential expression. The finite \(m\) in
this reconstruction check does not enter the uniform estimate (1).

## 6. A stronger Wilson linear bound in the same norm

For the canonical orientations of open square strips and rectangular
cubic boxes, let at most \(d\) elementary plaquettes meet any edge.
Write \(g=4/\kappa^2\) and
\[
 S_*=(g/3)\sum_p\chi_p.
\]
The independently derived original square-character nuclear norm is
eight, with square support diameter at most two. Its four spin-half
edges have electric energy three. Therefore
\[
 N_b(S_*)\le v=4dgb^2.
\tag{21}
\]
These canonical tensor normalizations are proved in
[the exact Wilson source calculation](gauge-wilson-residual-source.md); no
partial-transpose invariance of the nuclear norm is assumed here.
Using (1) gives the complete linearized bound
\[
 \boxed{\|2T(S_*,\cdot)\|_{N_b\to N_b}
 \le2Bv=\frac{64}{9}dgb^2.}
\tag{22}
\]
It improves the earlier independent
[fundamental estimate](gauge-wilson-linear-source.md), \(28dgb^2/3\), in
exactly the same norm. The new conclusion is not inferred from a
finite-spin operator norm.

One can reuse an already proved residual bound \(D\) and check
\[
 D+Lr+Br^2\le r,\qquad L+2Br<1,
 \qquad L=2Bv,\quad B=8/9.
\tag{23}
\]
For open strips,
\(D=(6b^2+8b^3)g^2\); for cubic boxes with \(d\le4\),
\(D=(12b^2+112b^3)g^2\). These constants include the Haar removal
of the constant residual and the original-edge tensor norms.

For cubic boxes at \(\kappa=15,b=1,r=1/8\), exact arithmetic yields
\[
 v=\frac{64}{225},\quad D=\frac{1984}{50625},\quad
 L=\frac{1024}{2025},\quad
 r-D-Lr-Br^2=\frac{49}{5625},\quad
 L+2Br=\frac{1474}{2025}<1.
\tag{24}
\]
This is a passing source-radius witness uniform over finite cubic boxes.
At \(\kappa=14,b=1\), the same sufficient scalar criterion has
\[
 (1-L)^2-4BD=-\frac{1487}{194481}<0,
\]
so no scalar radius can satisfy its self-map condition. That failure is
an obstruction to this majorant, not absence of an actual vacuum, a gap,
or confinement. Published certificate versions that used the older
constant remain unchanged; any new consumer must replay all its gates.

## 7. Independent algebra checks and scope

The independent regression file `tests/test_graph_bilinear_algebra.py`
checks the polar compression counterexample, exact mixed-orientation
endpoint intertwiners with two-dimensional multiplicity spaces, scalar
one-edge polar marginals, and complete low-spin fusion. Floating singular
value calculations are explicitly finite diagnostics. The proof above,
including its completion argument, supplies the all-spin implication.

The resulting improvement is applicable to growing interacting neutral
graphs in this fixed-spacing strong-coupling construction. It does not
identify an infinite-volume limit merely from the finite-family bound,
does not supply a running-coupling continuum bridge, and earns no
Yang-Mills parent or formal verification flag by itself.
