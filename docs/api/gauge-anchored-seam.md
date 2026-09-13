# An anchored three-face SU(2) seam

A trial depending on all three old plaquettes gives the uniform form estimate

\[
 \boxed{J_\kappa^*H_{\rm new}J_\kappa
 \le H_{\rm old}+\frac{21}{2}I+\frac2\kappa S_{\rm old}}
 \qquad(\kappa>0).                                      \tag{1}
\]

Here a corner of one cube is old and the complementary three faces are added,
with one genuinely new vertex and three genuinely new links. The coefficient
of the old action is exactly one. This result is a noncontracting reference,
not an exterior-uniform conditional gap or a continuum construction.

~~~python
from fractions import Fraction as Q
from omnibias.geometry.gauge.stochastic.anchored_seam import (
    anchored_seam_control,
    replay_anchored_seam_certificate,
    su2_anchored_three_face_seam,
)

bound = su2_anchored_three_face_seam(Q(1, 4096))
assert bound["witness"]["arithmetic"]["constant_upper"] == "21/2"
assert bound["actual_anchored_form_comparison_verified"]
assert replay_anchored_seam_certificate(bound["certificate"])

control = anchored_seam_control([Q(2), Q(-1), Q(0)], tilt=Q(1, 3))
assert not control["actual_anchored_form_comparison_verified"]
assert replay_anchored_seam_certificate(control["certificate"])
~~~

The exact arithmetic accepts integers and rational numbers, excluding floats
and booleans. The parameter heat_time_scale=c selects \(t=c\kappa\) and the
more general constant \(12c+9/(4c)\). The default \(c=1/2\) is a convenient
rational choice; the infimum in this family is \(6\sqrt3\), attained at
\(c=\sqrt3/4\). The analytic implication below is a written proof. The
certificate checks its exact geometry and arithmetic; it does not assert
Lean verification.

## 1. Original graph and the retained old operator

Use coordinates \(\{0,1\}^3\), with vertex integer \(x+2y+4z\). The three
faces incident to \(000\) are old. Their union contains seven vertices and
nine links. The new vertex is \(111\), with three incident links \(q_i\)
oriented from

\[
 v_0=101,\qquad v_1=110,\qquad v_2=011
\]

towards \(111\). Write \(P_i:v_i\to v_{i+1}\) for the two-edge parts of the
common boundary, with indices modulo three. Choose the three old paths

\[
\begin{array}{lll}
Q_0:101\to100\to000,&
Q_1:110\to010\to000,&
Q_2:011\to001\to000,\\
P_0:101\to100\to110,&
P_1:110\to010\to011,&
P_2:011\to001\to101 .
\end{array}                                             \tag{2}
\]

The \(Q_i\) are pairwise edge-disjoint. Direct cancellation of adjacent
inverse links shows that

\[
 B_i=Q_i^{-1}P_iQ_{i+1}                                \tag{3}
\]

are exactly the old \(xy,yz,zx\) plaquette holonomies based at \(000\).
For example \(B_0\) reduces to
\(000\to100\to110\to010\to000\). The new face based at \(v_i\) has
holonomy \(P_iq_{i+1}q_i^{-1}\).

The old Hamiltonian is

\[
 H_{\rm old}=\frac\kappa2\sum_{e\in E_{\rm old}}C_e
                 +V_{\rm old},\qquad C=-\Delta_G,\quad C_{1/2}=3/4, \tag{4}
\]

on product normalized Haar measure, where \(V_{\rm old}\) is any real smooth
gauge-invariant multiplication potential on a finite ambient graph.
The new Hamiltonian retains all of (4), adds the three electric terms,
and adds \((2/\kappa)\sum_{i=0}^2 A(P_iq_{i+1}q_i^{-1})\), where
\(A(U)=2-\operatorname{Tr}U\). The listed old corner may be embedded
injectively in a larger old graph, but no other new face, new-link potential,
vertex identification or electric weight is implicit in this theorem.

## 2. Two independent relative increments

For fixed old links set

\[
 Y_0=I,\qquad
 Y_i=Q_i^{-1}q_iq_0^{-1}Q_0\quad(i=1,2),\qquad
 h=Q_0^{-1}q_0.
                                                               \tag{5}
\]

The inverse relation is \(q_i=Q_iY_i h\). Successive left and right Haar
translations show that \((q_0,q_1,q_2)\mapsto(h,Y_1,Y_2)\) preserves product
Haar exactly. Consequently

\[
 \phi_t^2=K_t(Y_1)K_t(Y_2),\qquad
 \int\phi_t^2\,dq_0\,dq_1\,dq_2=1,                    \tag{6}
\]

where \(K_t\) is the SU(2) heat kernel of Haar mass one for \(\Delta_G\).
The normalizer is the constant one, independent of every old link.
The new vertex gauge acts only on \(h\). Under any old vertex gauge,
each \(Y_i\) is conjugated by the gauge at \(000\); centrality of \(K_t\)
therefore makes \(\phi_t\) gauge invariant.

For every \(t>0\), strict positivity and smoothness of \(K_t\) on the compact
group make \(\phi_t\) a smooth real normalized amplitude. Thus \(J_tf=\phi_tf\)
is an isometry preserving physical states. Differentiating (6) proves
\(\int\phi_tD_e\phi_t\,dq=0\) for every old derivative. Expanding the full
original-link form gives

\[
 J_t^*H_{\rm new}J_t=H_{\rm old}+W_{\kappa,t},\qquad
 W_{\kappa,t}=\frac\kappa2 I_t+\frac2\kappa\mathbb E_t S_{\rm new}. \tag{7}
\]

All cross terms with derivatives of \(f\) vanish conditionally, and every
old multiplication potential is preserved. This identity holds first on
smooth functions and then by form closure. The trial law (6) is not asserted
to be the actual new ground-state conditional law.

## 3. Exact magnetic averages

Conjugating each new face by \(q_i\) expresses its trace as

\[
 \operatorname{Tr}(Y_i^{-1}B_iY_{i+1}).                \tag{8}
\]

The two increments are independent in (6), and the fundamental heat
coefficient is
\(\int D_{1/2}(Y)K_t(Y)\,dY=e^{-3t/4}I\). Hence

\[
 \mathbb E_t\operatorname{Tr}U_{{\rm new},i}
 =d_i(t)\operatorname{Tr}B_i,\qquad
 (d_0,d_1,d_2)=(e^{-3t/4},e^{-3t/2},e^{-3t/4}).       \tag{9}
\]

Writing \(S_{\rm old}=\sum_i A(B_i)\), using
\(\operatorname{Tr}B_i\le2\) and \(1-e^{-x}\le x\), gives the global bound

\[
 \mathbb E_t S_{\rm new}-S_{\rm old}
 =\sum_i(1-d_i)\operatorname{Tr}B_i
 \le2\sum_i(1-d_i)\le6t.                              \tag{10}
\]

It covers noncommuting old holonomies and negative traces.

## 4. Every original-link derivative

Let
\[
 F_t=\int K_t(Y)|\nabla\log K_t(Y)|^2\,dY.
\]
Each relative word in (5) has six distinct original links: two in \(Q_i\),
one \(q_i\), one \(q_0\), and two in \(Q_0\). There are twelve total
incidences across the two words, eight old and four new.

An incidence acts on \(Y_i\) by an isometric left or right group derivative,
so its squared score integrates to \(F_t\). The only rows containing both
increments are \(q_0\) and the two links of \(Q_0\). In these rows the
variation right-multiplies both \(Y_1,Y_2\) by the same infinitesimal
group element, transported by old links only. For example varying
\(q_0\mapsto e^{sX}q_0\) gives
\(Y_i\mapsto Y_i Q_0^{-1}e^{-sX}Q_0\).
For each fixed generator the mean score is zero:

\[
 \int K_t(Y)D_X\log K_t(Y)\,dY=\int D_XK_t(Y)\,dY=0.
\]

Independence therefore cancels the mixed score products. No discarded frame
depends on the other increment. Since \(\log\phi_t\) is one half of the
sum of the two log kernels, the exact original-link Fisher identities are

\[
 \boxed{I_{\rm old}=2F_t,\qquad I_{\rm new}=F_t,\qquad I_t=3F_t.} \tag{11}
\]

The integrated Li–Yau inequality on this dimension-three compact group with
nonnegative Ricci curvature gives

\[
 F_t\le\frac3{2t}.
\]

Indeed \(|\nabla\log K_t|^2-\partial_t\log K_t\le3/(2t)\), and the integral
of \(K_t\partial_t\log K_t\) is \(\partial_t\int K_t=0\).
The compact maximum-principle proof, including the Casimir normalization,
is given in the [cube bridge proof](gauge-stochastic-cube.md).
This bound holds at every \(t>0\); no small-time expansion is needed.

Equations (7), (10) and (11) now give

\[
 W_{\kappa,t}\le \frac2\kappa S_{\rm old}
                       +\frac{12t}{\kappa}+\frac{9\kappa}{4t}.   \tag{12}
\]

Taking \(t=\kappa/2\) proves (1).

## 5. Physical energy consequence and limits of the result

For real smooth potentials on the finite connected product group, elliptic
positivity implies a unique positive scalar ground state. Gauge symmetry then
makes it physical, so scalar and gauge-invariant ground energies agree.
Since every added term is nonnegative, (1) therefore implies

\[
 0\le E_{\rm new}-E_{\rm old}
 \le\frac{21}{2}+\frac2\kappa\langle S_{\rm old}\rangle_{\rm old}. \tag{13}
\]

The lower bound uses \(H_{\rm old}\ge E_{\rm old}\) on the full old scalar
space, including fibers of new physical states. The upper bound tests its
actual old ground state with the normalized isometry. It does not substitute
the trial for the actual new ground state.

The coefficient one is significant. If three old faces are equal commuting
roots of a boundary holonomy with angle \(u\), their total action and the
minimum added action both equal \(6(1-\cos(u/3))\). Thus a coefficient
strictly below one with a fixed additive remainder cannot hold uniformly
as \(\kappa\to0\), even for fixed nonzero small \(u\). The present trial keeps
the old face data that specifies those roots; a boundary-only heat bridge
has additional cut-locus derivative costs. Neither a noncontracting
comparison nor small expected action supplies centered excitation coercivity.

## 6. Exact controls and regression scope

The function anchored_seam_point computes a rational positive control using
\(f(Y)=1+c\operatorname{Tr}Y\), \(c=r/(1+r^2)\), \(|r|<1\). Its two relative
increments have constant normalizer one. The integrated fundamental
dampings are \((c/2,c^2/4,c/2)\); (11) holds with

\[
 F_f=\int\frac{c^2(1-\chi^2/4)}{1+c\chi}\,dY.
\]

The function anchored_seam_control reuses the existing exact one-profile
Haar integral and records its replayed source. These controls are not heat
kernels and earn no physical heat-form flag.

The tests check exact noncommuting path identities, all vertex gauges,
all thirty-six original-link generator derivatives, and exact orientation
cubature cancelling the shared rows. Independent Haar quadrature checks the
remaining one-dimensional radial integral. Grid and seeded random samples
check the heat damping estimate. These diagnostics accompany the analytic
proof; they do not replace its infinite-dimensional argument.
Every replayer reconstructs geometry, arithmetic, nested sources and scope;
rehashing a changed field does not license a different theorem.
