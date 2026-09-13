# An open degree-eight target for Hilbert 16

This companion to [the Hilbert 16 program](HILBERT16.md) specifies one open
algebraic configuration and two constructive proof routes. No realizing
polynomial or general nonrealizability proof has been obtained.

## Exact target and current status

The bounded searches recorded in `artifacts/hilbert16/search-summary.json`
have now evaluated 894,851 patchwork candidates and 100 rational annular layouts.
No target construction was found. An independent classifier matched all 2,367
published degree-eight archive labels; exact lower-hull checks passed for all
41 unmodified seed triangulations. A further 528,384 evaluations used
coordinated changes on 1,532 triangulations, independently checked to be
disjoint from the prior 947 geometries. They found no target and no new
21- or 22-oval candidate. Five changed configurations now have exact integer
height certificates, including two- and three-flip realizations of the known
20-oval scheme retaining the fourteen-oval inner nest. Other changed
triangulations remain topology screens unless supplied with such a certificate.
One annular layout has
an exact Farkas exclusion, confined to its fixed sampled sign prescription.
The other numerical failures are not impossibility certificates. Full reports
and independent replay code are preserved in the corresponding artifact
subdirectories; these finite results do not exclude the target scheme.

Find a real homogeneous polynomial \(F(X,Y,Z)\) of degree eight, nonsingular
over \(\mathbb C\), whose real projective zero set has scheme

\[
\Sigma=\langle4\sqcup1\langle2\sqcup1\langle14\rangle\rangle\rangle.
\]

There are four exterior empty ovals and one oval containing two empty ovals
and another oval; the innermost oval contains fourteen empty ovals. The total
is 22, the Harnack maximum. The depth counts are \((5,3,14)\), and the
even/odd nesting counts are \((p,n)=(19,3)\).

As checked on 13 September 2026, Geiselmann et al.,
[*Limits of combinatorial patchworking*, arXiv:2602.06888v4](https://arxiv.org/pdf/2602.06888v4),
Table 1 on printed page 17, marks this exact scheme as having unknown
algebraic realizability. Section 4.3 also leaves its realization by T-curves
open. Theorem 21 excludes T-curves for the \((3,19)\) column, which does not
include this target. A pseudo-holomorphic realization does not provide the
required algebraic polynomial.

An exact comparison object is the rooted region tree with vertices
\(0,\ldots,22\):

\[
\operatorname{children}(0)=\{1,2,3,4,5\},\qquad
\operatorname{children}(5)=\{6,7,8\},\qquad
\operatorname{children}(8)=\{9,\ldots,22\}.
\]

All other child sets are empty. Vertex zero is the exterior projective
region; each edge corresponds to an oval. Counting components without
checking this nesting tree is insufficient.

## Route 1: a regular patchwork witness

Let
\[
A=\{(i,j)\in\mathbb Z^2:i,j\ge0,\ i+j\le8\}.
\]

A positive certificate consists of a regular unimodular triangulation,
a sign distribution, and the exact projective nesting tree:

1. Triangulate the degree-eight triangle using all 45 lattice points.
   Check disjoint interiors, common-face intersections, and complete
   coverage using exact determinants and segment predicates. Such a
   triangulation has 64 triangles and 108 edges, including 24 boundary
   edges; these counts alone do not certify a triangulation.
2. Assign sign bits \(\sigma_{ij}\). Reflect into the other quadrants using
   \(\sigma(i,-j)=\sigma(i,j)+j\) and
   \(\sigma(-i,j)=\sigma(i,j)+i\), modulo two. Join sign-changing edges
   inside each triangle, identify antipodal boundary points, and verify
   that the resulting region tree is \(\Sigma\).
3. Supply rational heights \(h_p\). For each chosen triangle \(\tau\), let
   \(\ell_\tau\) interpolate its three heights and require
   \[
   h_p-\ell_\tau(p)\ge1\qquad(p\notin\operatorname{vertices}(\tau)).
   \]
   These rational linear inequalities certify a strict lower hull.
   Any feasible strict system can be scaled to this margin.
4. Scale and shift the heights to nonnegative integers and form
   \[
   F_t(X,Y,Z)=\sum_{(i,j)\in A}(-1)^{\sigma_{ij}}t^{h_{ij}}
   X^iY^jZ^{8-i-j}.
   \]
   [Viro's patchworking theorem](https://arxiv.org/abs/math/0611382) supplies
   the algebraic realization for sufficiently small positive \(t\).
   An explicit rational choice of \(t\) additionally needs a certified
   admissible range or a direct topology and nonsingularity certificate.

Bounded integer reconnaissance counted 630 primitive candidate segments
and 1,108 unimodular candidate triangles. It did not enumerate all
triangulations, compatible split collections, or schemes.

A useful search constraint must allow cancellation between nested
Harnack splits. For example, double splits with common bases
\((1,0),(5,0)\) and apices \((2,5)\), \((3,3)\) have nested zones and
noncrossing primitive edges. Each zone has individual effect
\[
m=\#\{\text{interior lattice points}\}
 -2\#\{\text{interior points of parity }(0,0)\}=4.
\]
Applying both twists cancels the inner contribution and gives total effect
zero. Deleting every positive-effect split would therefore be unsound
pruning. This example supplies neither the target tree nor regular heights.

An exhaustive failure of this route would exclude only this patchworking
class. It would not exclude algebraic realizability of \(\Sigma\).

## Route 2: polygonal barriers and exact coefficient inequalities

Choose 22 pairwise disjoint thin polygonal annuli in \(\mathbb{RP}^2\) whose
core curves have tree \(\Sigma\). Use rational homogeneous vertices and
verify their disjointness and nesting exactly. Segment lifts must avoid
zero and remain in valid projective charts; do not assume that one affine
chart contains every possible realization.

Require opposite strict signs of \(F\) on the two boundary polygons of
each annulus, with the signs consistent across the nesting. These
conditions admit linear certificates in all 45 coefficients of \(F\).
For a lifted segment \(P(t)=(1-t)P_0+tP_1\), write
\[
F(P(t))=\sum_{k=0}^8 b_k t^k,\qquad
\beta_j=\sum_{k=0}^j b_k\frac{\binom jk}{\binom8k}.
\]
Each \(\beta_j\) is a rational linear form in the polynomial coefficients.
For required sign \(s\in\{-1,1\}\), impose \(s\beta_j\ge1\) for every
\(j=0,\ldots,8\). The Bernstein basis is nonnegative and sums to one, so
these inequalities certify the sign everywhere on the segment.

This produces an exact rational feasibility problem \(M\alpha\ge\mathbf1\).
One must check the proposed coefficient vector by exact multiplication.
A floating solver's answer alone is not a certificate.

Why would a feasible barrier system suffice? Preserve the strict signs
while choosing a polynomial outside the complex discriminant. Such
polynomials are dense. In every annulus, the opposite boundary signs force
a zero component separating its boundaries. Nonsingularity makes it an
essential oval. The disjoint annuli give at least 22 distinct ovals;
Harnack permits at most 22. Hence there is exactly one per annulus and no
additional real component, establishing the prescribed scheme. This
argument uses maximality and does not apply unchanged to a nonmaximal
target.

For an explicit final polynomial, certify complex nonsingularity
separately, for example by exact elimination of common projective zeros of
\(F_X,F_Y,F_Z\). A real-gradient margin proves only real nonsingularity.

If a smooth realization exists, rational annuli can approximate its ovals
with strict boundary-sign margins. Sufficient segment subdivision then
makes the Bernstein sign certificates succeed. Enumerating these layouts,
subdivisions, and rational feasible coefficients therefore gives a positive
semidecision procedure: it eventually finds a witness if one exists. It
provides no stopping rule when none exists.

An exact Farkas witness
\[
\lambda\ge0,\qquad \lambda^\mathsf TM=0,\qquad
\lambda^\mathsf T\mathbf1>0
\]
excludes only its fixed barrier layout. It does not exclude every
embedding with tree \(\Sigma\).

## What remains unsolved

The difficult variables are the compatible patchwork geometry or the
annulus layout. Neither route has produced a witness for \(\Sigma\).
The recorded counts and cancellation example do not constitute an
exhaustive search or an obstruction to algebraic realization.

Existing [rational polynomial arithmetic](../omnibias-holonomic/src/omnibias/holonomic/_core/poly_n.py),
[convex optimization support](../omnibias-convex/src/omnibias/convex),
and [verified arithmetic](../omnibias-core/src/omnibias/core/verified)
can support candidate certificates. Projective topology checks and exact
witness translation still need to be supplied for this target.

A construction or general obstruction would settle this one open
degree-eight configuration. It would not settle the remaining degree-eight
schemes, arbitrary-degree curves, surfaces, or the dynamical half of
Hilbert's sixteenth problem.
