# Isolated SU(2) block kinetic comparison

This API proves a finite physical kinetic metric comparison after maximal-tree
reduction of an isolated open cube. It retains the root Gauss law. It does
not identify a block conditioned on an exterior or prove a spectral gap.

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer.isolated_block_kinetic import (
    su2_isolated_block_kinetic,
    replay_su2_isolated_block_kinetic_certificate,
)

report = su2_isolated_block_kinetic(Fraction(1, 1000), block_side=1)
assert report["status"] == "PASS"
assert report["isolated_block_positive_kinetic_comparison_verified"]
assert replay_su2_isolated_block_kinetic_certificate(report["certificate"])
```

`edge_radius` is an exact integer or `Fraction` in `[0,2]` and bounds the
retained **chord** logarithms. `block_side` is a positive integer. Booleans,
floats, and out-of-domain inputs are refused. Every accepted report contains
a sound rational error bound. `PASS` means that error is strictly below one;
`INCONCLUSIVE` still has a replayable bound but earns no positive lower
comparison. The point radius zero is the exact tangent metric identity,
not a nonempty localization domain.

The returned `kinetic_relative_error_upper` and
`kinetic_comparison_lower`/`kinetic_comparison_upper` are rational strings.
Replay rebuilds all counts, constants, hypotheses and flags, including
`INCONCLUSIVE` cases. The mathematical implications below are written proofs;
the certificate does not assert that Lean checked them.

## Complete physical metric proof

This section concerns the **isolated** finite graph consisting of the
open block and its links, with Gauss law at every vertex and no exterior
edges. Let \(v=|V|-1\) be the number of tree links and
\(c=|E|-|V|+1\) the number of chords. For the cubic block,

\[
 v=(b+1)^3-1,\qquad |E|=3b(b+1)^2,\qquad c=|E|-v.
 \tag{18}
\]

Use the tree consisting of all z-edges, all y-edges at z=0, and all x-edges at y=z=0, with positive-axis orientations. For an arbitrary
configuration, let \(P_x\) be its ordered tree transport from the root
to vertex \(x\). The chord coordinate for \(a:x\to y\) is
\(W_a=P_x U_aP_y^{-1}\). Tree gauge sets all tree links to the
identity. A physical scalar function has a unique representation
\(f(U)=h((W_a)_a)\), where \(h\) is invariant under simultaneous
conjugation of all chord variables. The remaining conjugation is the
root Gauss law; it is retained.

Successive Haar changes of variables along the tree give exactly

\[
 \int_{SU(2)^E}|f(U)|^2dU
     =\int_{SU(2)^c}|h(W)|^2\prod_a dW_a.
 \tag{19}
\]

Thus this maximal-tree reduction has product Haar measure on the chords,
restricted to simultaneous-conjugation-invariant functions. No
nonconstant Faddeev–Popov determinant appears in this specific compact
tree change of variables. This does **not** make its electric metric a
product metric.

Let \(p_a\in\mathbb R^3\) be the right gradient of \(h\) at \(W_a\),
using the generators \(i\sigma_j/2\). For a tree edge \(e\), let
\(S_e\) be the descendant side of the cut made by removing it. Define
\(s_{ea}=\mathbf1_{\{\mathrm{source}(a)\in S_e\}}\) and
\(t_{ea}=\mathbf1_{\{\mathrm{target}(a)\in S_e\}}\). Differentiating
the actual original edge, at a tree-gauged configuration, gives the
exact electric form

\[
 \boxed{\displaystyle
 \Gamma_{\rm tree}(h)
  =\sum_a|p_a|^2+
    \sum_{e\in\mathrm{tree}}
       \left|\sum_a
           (s_{ea}\operatorname{Ad}_{W_a}-t_{ea}I)p_a\right|^2.}
 \tag{20}
\]

Indeed, varying that tree link by \(e^{uX}\) changes \(P_x\) by this
factor exactly for \(x\in S_e\), and hence changes a chord to
\(e^{us_{ea}X}W_a e^{-ut_{ea}X}\). The left gradient is
\(\operatorname{Ad}_{W_a}p_a\). Varying an original chord gives its
own right-gradient square, yielding (20). In particular, a chord with
**both** endpoints in \(S_e\) contributes
\((\operatorname{Ad}_{W_a}-I)p_a\); omitting such terms is incorrect.
Gauge invariance transports this computation back to every original
configuration.

At the identity put \(D_{ea}=s_{ea}-t_{ea}\). For an array of real
vectors \(z=(z_a)_a\), define

\[
 \Gamma_0(z)=\sum_a|z_a|^2+\sum_e\left|\sum_a D_{ea}z_a\right|^2,
 \qquad M_0=I+D^TD.
 \tag{21}
\]

In color coordinates the metric is \(M_0\otimes I_3\). Since
\(|D_{ea}|\le1\),
\(I\le M_0\le(1+vc)I\). This is the fundamental-cycle electric
metric, rather than the identity metric on independent chord logs.

Suppose all chord log norms are at most \(0\le\varepsilon\le2\).
The adjoint action is a three-dimensional rotation, so
\(\|\operatorname{Ad}_{W_a}-I\|
 =2|\sin(|A_a|/2)|\le\varepsilon\).
Write the map inside (20), including its chord rows, as \(J(W)\),
and its value at the identity as \(J_0\). Rowwise Cauchy–Schwarz gives

\[
 \|J(W)-J_0\|\le\varepsilon\sqrt{vc},\qquad
 \|J_0\|\le\sqrt{1+vc}.
\]

Consequently, for any array \(p\),

\[
 |\Gamma_{\rm tree}(p)-\Gamma_0(p)|
 \le\delta\,\Gamma_0(p),\qquad
 \delta=(2vc+1)\varepsilon+vc\varepsilon^2.
 \tag{22}
\]

Here \(2\sqrt{vc(1+vc)}\le2vc+1\) and
\(\Gamma_0(p)\ge\sum_a|p_a|^2\) remove all square roots from the
sufficient bound. This estimate remains a valid error bound when
\(\delta\ge1\); a positive lower comparison then has not been earned.

For comparison with Cartesian chord logs, write
\(W_a=\exp(iA_a\cdot\sigma/2)\). If
\(\dot W_a=W_a(iY_a\cdot\sigma/2)\), then
\(Y_a=J_R(A_a)\dot A_a\). For the convention U(A)=exp(+i A·sigma/2),

\[
 J_R(A)^{-1}
   =P_r+\frac r2\cot(r/2)P_t-\frac12(A\times),
 \qquad p_a=J_R(A_a)^{-T}\nabla_{A_a}h.
 \tag{23}
\]

The transpose changes the cross sign. The formula follows either by
integrating the adjoint rotation in the derivative of the exponential,
or by differentiating its quaternion expression. The radial eigenvalue
is one. For \(x=r/2\),
\(\sin x-x\cos x=\int_0^x u\sin u\,du\le x^3/3\).
Using \(\sin x\ge x(1-x^2/6)\) proves

\[
 \|J_R(A)^{-1}-I\|\le\eta,
 \qquad
 \eta=\frac\varepsilon2+
         \frac{\varepsilon^2}{12(1-\varepsilon^2/24)}.
 \tag{24}
\]

Let \(z_a=\nabla_{A_a}h\). The block-diagonal map \(z\mapsto p\)
differs from the identity by norm at most \(\eta\). Thus

\[
 |\Gamma_0(p)-\Gamma_0(z)|
 \le(1+vc)(2\eta+\eta^2)\,\Gamma_0(z).
\]

Combining this with (22) gives the exact rational sufficient criterion

\[
 \boxed{\begin{gathered}
 |\Gamma_{\rm tree}(A,\nabla h)-\Gamma_0(\nabla h)|
       \le\mathfrak e\,\Gamma_0(\nabla h),\\
 \mathfrak e=\delta+(1+\delta)(1+vc)(2\eta+\eta^2).
 \end{gathered}}
 \tag{25}
\]

When \(\mathfrak e<1\), this proves the two-sided physical kinetic
comparison \((1-\mathfrak e)\Gamma_0\le\Gamma_{\rm tree}
\le(1+\mathfrak e)\Gamma_0\) on the isolated block. It integrates
against the same product chord Haar measure, or against that measure
times any nonnegative density. For a comparison with flat Cartesian
measure, the density \(\prod_a s(|A_a|)^2\) must still be retained or
conjugated explicitly. In the coupled metric (20), such a conjugation
does not automatically give the scalar shift of the unreduced product metric.

The chord-log ball is invariant under the residual simultaneous
conjugation. Smooth invariant functions supported in it are legitimate
physical localized functions on the isolated graph. Unlike the
unreduced original-edge ball, this domain has not discarded entire
gauge orbits.

Finally, let \(B\) be the linear face-edge incidence of the open cubic
cell complex. This complex is contractible, so every edge cochain with
zero face curl is a vertex gradient: one can integrate it from the
root, since elementary faces generate its cycle space. If its tree-edge
coordinates also vanish, the vertex potential is constant along the
spanning tree, hence constant everywhere, and all remaining chord
coordinates vanish. Thus \(B\) restricted to tree-zero coordinates
has trivial kernel. The isolated block's quadratic magnetic matrix in
chord coordinates is positive; its kinetic matrix is \(M_0\), and
the global color-singlet restriction remains. This describes a change
of coordinates in the linearized model. It is not an argument for
discarding nonlinear modes, nor does it remove the harmonic flat modes
present on a periodic torus.

**Exterior limitation.** An open block embedded in a larger interacting
graph is not this isolated Hamiltonian. Tree derivatives can act on
exterior-dependent states and additional chord coordinates, and the
actual vacuum is correlated with that exterior. Equations (20)–(25)
do not identify an exterior-conditioned kinetic operator or permit
cross-boundary edges to be dropped. An actual embedded-block small-field probability estimate
and the isolated kinetic comparison are separate earned statements.

