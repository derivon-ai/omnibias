# Finite-graph Fourier resolvent and a conditional input tail

The pure rational helper replays a high-electric-energy tail inequality in
an original-link Fourier norm. Its inputs are numerical premises, not a
validated graph or source certificate. The full written proof below also
establishes qualitative finite-graph invertibility for the stated graph
families, but the helper does not compute or certify an inverse norm.

```python
from omnibias.geometry.gauge.transfer.finite_graph_resolvent import (
    su2_finite_graph_linearized_tail,
    replay_su2_finite_graph_linearized_tail_certificate,
)

result = su2_finite_graph_linearized_tail(
    5, plaquette_count=2, plaquettes_per_edge=2,
    energy_root=8, decay_base=1,
)
assert result["operator_tail_upper"] == "7/10"
assert result["numeric_tail_below_one"]
assert replay_su2_finite_graph_linearized_tail_certificate(result["certificate"])
assert not result["actual_graph_verified"]
assert not result["fourier_nuclear_inverse_verified"]
```

Every numerical parameter accepts an exact integer or Fraction except the
counts, which must be exact integers. Booleans and floating-point numbers
are refused. Coupling, counts, and energy_root must be positive;
decay_base must be at least one. The energy cutoff is the square of
energy_root, so the tail uses only rational arithmetic.

For \(g=4/\kappa^2\), a supplied square count \(P\), incidence cap \(q\),
locality weight \(b\), and energy_root \(t\), the report returns

\[
\tau=gb^2(16P/t+12q/t^2).
\]

Its meaning is conditional: if all stated original-link graph, Fourier
normalization, and omitted-energy premises hold, this bounds
\(\|\mathcal K Q_{t^2}\|\), where
\(\mathcal K=2C_0^{-1}\Pi_H\Gamma(S_*,\cdot)\).
A small tail alone gives no estimate of the retained inverse. Equality
\(\tau=1\) is not a strict contraction. A value greater than one does
not invalidate the bound, so status remains CONDITIONAL_BOUND and
numeric_tail_below_one is false.

Canonical replay rebuilds the exact arithmetic, external premises, norm,
projection, and scope. A rehashed change to any of them is rejected unless
it is the complete canonical certificate for its inputs. Actual graph,
reference, finite inverse, Fourier inverse, quantum vacuum, target gap,
volume-uniform inverse, continuum, and formal flags all remain false.
The proof below is a written analytic argument, not a Lean verification.

## Full analytic proof

12 September 2026. This note proves that the Wilson-reference linearized
operator has a bounded inverse in the original invariant Fourier nuclear
norm on each finite open strip or cubic box, for every finite
\(\kappa>0\). It includes the seven-edge graph of two adjacent squares.
The proof uses an explicit high-electric-energy tail and the reference
diffusion only to exclude a kernel. It does not transfer a reference
\(L^2\) norm bound to the Fourier norm.

The constants in the compact tail depend on the finite graph. Neither
the existence of a finite-graph inverse nor the argument below supplies a
volume-independent inverse norm, a nonlinear correction bound, or a
continuum theory. The Fredholm mechanism is classical; see, for example,
the Riesz–Schauder and Fredholm discussion in
[Ralph Chill's functional-analysis notes, Section 5](https://tu-dresden.de/mn/math/analysis/chill/ressourcen/dateien/skripte/FunctionalAnalysisChill.pdf).
A short proof of the needed implication is included below.

## 1. Space, source, and projections

Use a finite open square strip or rectangular cubic graph, with every
original edge pointing in the positive coordinate direction. Let
\(m\) be the number of original edges, \(P\) the number of elementary
squares, and \(q\) an upper bound on the number of squares incident to one
edge. Thus \(P=2,q=2,m=7\) for two adjacent squares. Every square has four
distinct edges. Gauge transformations act at all vertices, including
boundary vertices.

Write original-link Peter–Weyl series in the convention
\[
 u=\sum_{\mathbf k\ne0}\operatorname{Tr}
               (U_{\mathbf k}D^{\mathbf k}),\qquad
 E_{\mathbf k}=\sum_e k_e(k_e+1),\qquad
 w_{\mathbf k}=b^{\operatorname{diam}(\operatorname{supp}\mathbf k)},
 \quad b\ge1,                                          \tag{1}
\]
where \(k_e\) are half-integer spins and distance is the ambient edge
line-graph distance. Constants are omitted. Set
\[
 N_i(u)=\sum_{\mathbf k\ne0}
 k_i E_{\mathbf k}w_{\mathbf k}\|U_{\mathbf k}\|_1,
 \qquad N(u)=\max_i N_i(u).                             \tag{2}
\]
Let \(X_b\) be the real, Haar-centered, gauge-invariant closed subspace
with finite norm (2). Finite arrays are dense; the sum of the finitely
many anchored norms is a weighted \(\ell^1\) norm equivalent to their
maximum. Thus \(X_b\) is a Banach space. Gauge projections preserve each
Peter–Weyl label, so truncating the labels preserves \(X_b\).

At least one \(k_i\ge1/2\) in each nonconstant label. Therefore
\[
 \sum_{\mathbf k\ne0}E_{\mathbf k}\|U_{\mathbf k}\|_1
 \le2\sum_iN_i(u)\le2mN(u).                            \tag{3}
\]
The norm of a unit spin-\(k\) generator is at most \(k\).
Products of two generators and covariant same-edge derivatives have
coefficient bounds controlled by \(E_{\mathbf k}\).
Consequently (3) gives absolute uniform convergence through two
derivatives on the finite compact product group. Each element of \(X_b\)
is an actual \(C^2\) function; this is not just a formal coefficient
sequence. The finite factor \(m\) here is not asserted to be uniform.

Let \(\Pi_H\) remove Haar mean, \(C=-\sum_e\Delta_e\), and let
\(C_0^{-1}\) divide nonconstant modes by \(E_{\mathbf k}\). Put
\[
 g=4/\kappa^2,\qquad S_*=(g/3)\sum_p\chi_{1/2}(U_p),
 \qquad {\cal K}=2C_0^{-1}\Pi_H\Gamma(S_*,\cdot),
 \qquad {\cal L}=I-{\cal K}.                           \tag{4}
\]
The nuclear coefficient norm of each fundamental square is eight in
the fixed original-edge orientation. This is the vertex-tensor result
in [strip-source-window.md](gauge-wilson-residual-source.md), not an assumed
invariance of nuclear norm under partial transpose.

## 2. An explicit high-energy input tail

Let \(Q_R\) retain precisely the nonconstant input labels with
\(E_{\mathbf k}\ge R\), \(R>0\), and put \(P_R=I-Q_R\).
The projection \(P_R\) has finite rank: bounded \(E_{\mathbf k}\) bounds
every spin, and each coefficient block is finite dimensional. Both
projections have norm at most one in (2).

The anchored fundamental fusion calculation in
[wilson-linear-bound.md](gauge-wilson-linear-source.md) gives, for each square
\(p\), input label \(\mathbf k\), and anchor \(i\), the coefficient bound
\[
 G_i(p,\mathbf k)\le
 \frac32 k_i a_p+
 \frac34{\bf1}_{i\in p}\sum_{e\in p\setminus\{i\}}k_e,
 \qquad a_p=\sum_{e\in p}k_e.                           \tag{5}
\]
Its derivation uses the exact eigenvalues \(k/2\) and \(-(k+1)/2\)
of the spin-\(1/2\) generator contraction on the two fusion channels.
In particular the differentiated anchor costs
\(\max_\ell\ell|\Omega_\ell|=k(k+1/2)/2\), not the product
of two separately maximized factors. Unitary fusion and trace-norm
pinching prove (5) for arbitrary coefficient matrices. The generator
contractions commute with the fusion blocks.

If a derivative product is nonzero, its two input supports intersect.
The output support lies in their union, and a square has diameter two.
Hence \(w_{\rm output}\le b^2w_{\mathbf k}\). This remains valid for a
disconnected input support. The output energy in (2) cancels the
\(C_0^{-1}\) denominator; the constant channel has been removed.
The remaining numerical prefactor is \(2(g/3)\cdot8=16g/3\).

Cauchy–Schwarz on a four-edge square gives
\[
 a_p\le2\left(\sum_{e\in p}k_e^2\right)^{1/2}
       \le2\sqrt{E_{\mathbf k}}.                       \tag{6}
\]
For the first term in (5), sum over the \(P\) squares and then over
input labels of energy at least \(R\):
\[
 \sum_{\mathbf k:E_{\mathbf k}\ge R}
   3P k_i\sqrt{E_{\mathbf k}}w_{\mathbf k}\|U_{\mathbf k}\|_1
 \le\frac{3P}{\sqrt R}N_i(u).                          \tag{7}
\]
For the second term, at most \(q\) squares contain \(i\), each with three
other edges counted with multiplicity. For each such edge,
\[
 \sum_{\mathbf k:E_{\mathbf k}\ge R}
        k_e w_{\mathbf k}\|U_{\mathbf k}\|_1
 \le R^{-1}N_e(u).
\]
Its total contribution is at most \(9qN(u)/(4R)\). Multiplying both
contributions by \(16gb^2/3\) proves
\[
 \boxed{\|{\cal K}Q_R\|_{X_b\to X_b}\le
 \tau(R):=gb^2\left(\frac{16P}{\sqrt R}+\frac{12q}{R}\right).}
 \tag{8}
\]
This bound controls every omitted input label, with no sampled
representation tail. The simpler exact rational choice \(R=t^2\),
\(t>0\) rational, gives
\[
 \tau(t^2)=gb^2(16P/t+12q/t^2).                         \tag{9}
\]
The same bound works if a chosen omitted set is a subset of
\(\{E_{\mathbf k}\ge t^2\}\).

There is an optional finite-graph improvement of the first term:
\(\sum_pa_p\le q\sum_e k_e\le q\sqrt{mE_{\mathbf k}}\).
Thus \(16P/\sqrt R\) in (8) can be replaced by
\(\min(16P,8q\sqrt m)/\sqrt R\).
Its graph-size dependence still prevents a volume-uniform compactness
argument. The simpler rational formula (9) suffices below.

## 3. Compactness in exactly the original norm

The map \({\cal K}P_R\) has finite rank. Indeed \(P_R\) has finite rank,
and multiplication and differentiation against the finite source
\(S_*\) produces only finitely many fusion outputs from those inputs.
No assertion that these outputs remain inside \(P_R X_b\) is needed.
Equation (8) gives
\[
 \|{\cal K}-{\cal K}P_R\|\le\tau(R)\longrightarrow0.
 \tag{10}
\]
Therefore \({\cal K}\) is compact on \(X_b\), for any fixed finite graph,
any fixed finite \(g\), and any fixed \(b\ge1\).

The distinction between an input truncation and a square Galerkin
matrix matters: \({\cal K}P_R\) can send a retained input to a larger
energy. Discarding such outputs would require an additional bound.
The finite-rank compact approximation in (10) retains them.

## 4. Elliptic injectivity, with both means explicit

The reference probability is the known positive smooth density
\[
 d\nu=Z_*^{-1}e^{2S_*}dH,\qquad
 A=C-2\Gamma(S_*,\cdot).
\]
It is an auxiliary reference, not the unknown quantum vacuum.
Suppose \({\cal L}u=0\) in \(X_b\). The reconstruction in (3) makes this
a \(C^2\) equation. Multiplying (4) by \(C\) gives
\[
 Cu=2\Pi_H\Gamma(S_*,u),\qquad
 Au=-2H[\Gamma(S_*,u)],                               \tag{11}
\]
where the right side is a constant. Integration by parts gives
\(\nu[Au]=0\), so that constant is zero. A second integration by parts
then gives
\[
 0=\int\overline u\,Au\,d\nu
   =\int\Gamma(\overline u,u)\,d\nu.
 \tag{12}
\]
Since \(\nu\) is strictly positive and the product group connected,
\(u\) is constant. Its Haar mean is zero, so \(u=0\).
Thus \({\cal L}\) is injective on \(X_b\).

Only integration by parts and positivity were used. In particular,
no quantitative reference gap, no Fourier inverse bound, and no
target-vacuum gap was assumed in excluding the kernel.

## 5. Surjectivity and a bounded Fourier inverse

For completeness, the needed compact-operator argument can be given
directly. If \(I-K\) is injective and \(K\) compact on a Banach space,
then \(I-K\) is bounded below. Otherwise take norm-one \(u_n\) with
\((I-K)u_n\to0\); a subsequence of \(Ku_n\) converges by compactness,
so \(u_n\) converges to a nonzero kernel vector. Thus its range is closed.

Let \(X_j=(I-K)^jX\). Each \(X_j\) is closed by repeating that argument
on the previous closed, \(K\)-invariant subspace. If the initial range
were proper, injectivity would make every inclusion
\(X_{j+1}\subset X_j\) proper. Choose \(u_j\in X_j\) with norm one and
distance at least \(1/2\) from \(X_{j+1}\). For \(n>j\),
\(Ku_n\in X_{j+1}\) and \(u_j-Ku_j\in X_{j+1}\).
It follows that \(\|Ku_n-Ku_j\|\ge1/2\), contradicting compactness.
Therefore \(I-K\) is onto; the bounded-below estimate bounds its inverse.

Apply this argument to (10) and (12). We obtain
\[
 \boxed{\text{For each specified finite graph and each }\kappa>0,\quad
 {\cal L}:X_b\longrightarrow X_b
 \text{ is a bounded isomorphism}.}                   \tag{13}
\]
This is an actual inverse in the original Fourier norm, beyond merely
an \(L^2(\nu)\) inverse. The proof has not computed its norm.
The finite graph and fixed \(b\) are part of the quantifiers in (13).

On smooth Haar-centered forcing, the inverse agrees with the projection
identity in [reference-resolvent.md](gauge-reference-resolvent.md):
\[
 {\cal L}^{-1}
   =\Pi_H A_\nu^{-1}\Pi_\nu C,\qquad
 \Pi_\nu h=h-\nu[h].                                  \tag{14}
\]
For general \(X_b\) forcing, (13) supplies the bounded extension in
\(X_b\). An \(L^2(\nu)\) bound for the right side alone would not prove
that extension: it contains an extra \(C\) on the input.

## 6. A finite inverse plus the all-spin tail can earn a number

Let \(K_R={\cal K}P_R\). Suppose an independently checked finite
calculation bounds
\[
 \|(I-K_R)^{-1}\|_{X_b\to X_b}\le B_R,\qquad
 B_R\tau(R)<1.                                       \tag{15}
\]
Then a Neumann perturbation around this finite-rank approximation gives
\[
 \boxed{\|{\cal L}^{-1}\|\le
       \frac{B_R}{1-B_R\tau(R)}.}                     \tag{16}
\]
The finite number \(B_R\) is an actual premise of this numerical route.
It cannot be replaced by the reference \(L^2\) inverse norm.

One explicit reduction to a finite matrix is as follows. On
\(P_R X_b\), write \(D_R=I-P_R{\cal K}P_R\). If \(D_R\) is invertible,
then
\[
 (I-{\cal K}P_R)^{-1}
       =I+{\cal K}P_RD_R^{-1}P_R.                     \tag{17}
\]
Formula (17) includes the retained-to-omitted output block. For example,
the conservative bound
\(B_R\le1+\|{\cal K}P_R\|\|D_R^{-1}\|\) is sufficient.
A sharper bound may evaluate this finite-rank correction directly in
the anchored norm. Replacing that norm by a coefficient \(\ell^1\) norm
requires explicit comparison constants.

The existence proof also guarantees that sufficiently accurate
finite-rank approximations eventually have bounded inverses, since
\({\cal K}P_R\to{\cal K}\) in norm and \({\cal L}^{-1}\) is bounded.
It supplies no practical cutoff size. A failed finite cutoff or failed
inequality (15) is therefore an inconclusive quantitative attempt, not
evidence that the finite-graph inverse does not exist.

## 7. The adjacent-square graph keeps its physical path weights

Gauge invariance at the bivalent interior vertices of the adjacent-square
graph forces constant representation labels along its three paths.
Their lengths are \(3,3,1\). For half-integer path spins \(j_a,j_b,j_s\),
\[
 E=3j_a(j_a+1)+3j_b(j_b+1)+j_s(j_s+1).                 \tag{18}
\]
The endpoint intertwiner must exist; for SU(2) this is the usual triangle
and integer-sum condition. In doubled-spin notation \(a=2j_a,b'=2j_b,
s=2j_s\), (18) is
\([3a(a+2)+3b'(b'+2)+s(s+2)]/4\).
The symbol \(b'\) here is a representation label, distinct from the
locality weight \(b\).

The original norm (2) has only three distinct anchor sums on this graph,
but it remains their maximum. It is not silently replaced by a single
coefficient norm on three unit-weight links. The energy denominator
remains (18), rather than the three-unit-link Casimir. No orientation
change is used to infer a coefficient nuclear norm.

The universal theta specialization of (8)–(9) is simply
\[
 \|{\cal K}Q_{t^2}\|\le gb^2(32/t+24/t^2).             \tag{19}
\]
A finite theta intertwiner calculation may improve (19) by handling
additional exact shells, but every omitted label must still be covered
by a proved tail. This supplies a concrete finite target: certify the
finite inverse in (17), preserve all three physical path weights, and
check (15).

## 8. What remains unearned

The factor \(P\) in (8) grows with volume. The alternative
\(\sqrt m\) constant also grows. Pointwise compactness at every finite
graph therefore provides neither compactness nor a uniform inverse norm
on an infinite lattice. The source derivative coupling can still be
strong in the low-energy blocks even though its high-energy tail is
small.

Nor does linear invertibility itself make the nonlinear correction
contract. A bound for \({\cal L}^{-1}\), the centered residual after
preconditioning, and the quadratic term in the same norm must all pass
their own inequalities. Finally, finite-graph inverse and vacuum
estimates do not supply uniform physical units, continuum
Osterwalder–Schrader reconstruction, nontriviality, or the Clay mass gap.
