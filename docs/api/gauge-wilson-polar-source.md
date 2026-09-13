# Volume-independent SU(2) vacuum by gauge-polar cancellation

```python
from fractions import Fraction
from omnibias.geometry.gauge.transfer import (
    su2_wilson_polar_vacuum, replay_su2_wilson_polar_vacuum_certificate,
    su2_wilson_static_family,
)

source = su2_wilson_polar_vacuum(15, correction_radius=Fraction(1, 10))
assert source["status"] == "PASS"
assert source["physical_gap_lower"] == "367/180"
assert source["volume_uniform_actual_vacuum_family_verified"]
assert replay_su2_wilson_polar_vacuum_certificate(source["certificate"])
static = su2_wilson_static_family(source["certificate"])
assert static["witness"]["linear_static_energy_coefficients"] == ["367/180", "45/8"]
assert not source["continuum_claim"]
assert not source["theorem_prover_verified"]
```

This API uses exact rational parameters. Optional arguments are `family="cubic"`
or `"strip"`, `correction_radius=None`, `decay_base=1`, and `exponent_steps=4`.
An omitted radius uses the proved rational witness; a supplied radius is checked
as given. `INCONCLUSIVE` can coexist with an earned reference inverse.


The all-graph [polar cancellation theorem](gauge-graph-bilinear-cancellation.md)
extends the theta bilinear improvement to general gauge-invariant SU(2)
Fourier matrices, including intertwiner multiplicities and arbitrary
complex signs. Its use in the existing complete vacuum construction now
passes at \(\kappa=141/10\) for every specified finite open cubic box.
At \(\kappa=15\) it gives physical gap at least \(367/180>2\).

These are exact fixed-coupling family estimates, not a continuum limit.
The method uses the actual interacting Hamiltonian on each entire graph.
It does not assemble isolated theta spectra or assume that adjacent
two-plaquette vacua agree on their shared links.

## 1. What extends from theta to a general graph

On a finite oriented graph of girth at least four, let
\[
N_{b,i}(f)=\sum_{\mathbf j\ne0}j_i E_{\mathbf j}
  b^{\operatorname{diam}X_{\mathbf j}}\|F_{\mathbf j}\|_1,\qquad
N_b(f)=\max_iN_{b,i}(f),\qquad E_{\mathbf j}=\sum_ej_e(j_e+1).
\]
The coefficient norm is the original-edge matrix trace norm. Require
all-vertex gauge invariance and use the inherited ambient line-graph
distance, with \(b\ge1\). Invariant nonconstant coefficients have
\(E_{\mathbf j}\ge3\); no representation cutoff is imposed.

Gauge invariance makes the one-edge marginals of the polar factors
\(|F|\) and \(|F^\dagger|\) scalar. For a product coefficient \(A=F\otimes G\)
and every complete fusion projection \(P_\ell\),
\[
\|P_\ell A P_\ell\|_1
 \le \tfrac12\operatorname{Tr}P_\ell(|A|+|A^\dagger|).
\]
The right sides form a positive pack of total mass
\(\|F\|_1\|G\|_1\). Its mean coupled Casimir defect is zero.
This replaces a positivity assumption on the unknown matrices by
a proved inequality for their polar factors.

The [full argument](gauge-graph-bilinear-cancellation.md) obtains
\[
N_b(T(f,h))\le\frac89N_b(f)N_b(h),\qquad
T=C_0^{-1}\Pi_0\Gamma.
\]
All derivative directions, output representations and vertex
multiplicities remain present. The bound contains no graph cardinality.
The diameter weight is valid because a nonzero derivative product has
intersecting input supports.

## 2. Exact residual and complete nonlinear correction

Use the actual Hamiltonian and positive-coordinate orientations from
[the Wilson residual theorem](gauge-wilson-residual-source.md):
\[
aH=\frac\kappa2 C+\frac2\kappa\sum_p(2-\chi_p),\qquad
g=\frac4{\kappa^2},\qquad S_*=\frac g3\sum_p\chi_p.
\]
The families are all finite open rectangular three-dimensional cubic
boxes, and separately all finite one-cell-wide square strips. Let
\(d=4,P=42\) for cubes and \(d=2,P=3\) for strips. Here \(d\) bounds the
plaquettes through an edge and \(P\) the adjacent plaquette pairs
touching that edge. The already proved residual includes both the
adjoint single-square and adjacent-pair channels:
\[
A=4dg b^2,\qquad
D=\left(3d b^2+\frac83P b^3\right)g^2.
\]
These bounds do not discard higher-spin tails. They bound the exact
seed and residual; the unknown correction contains every higher order.

With \(B=8/9\), the new linear coefficient is
\[
L=2BA=\frac{64}{9}dg b^2.
\]
It improves the earlier fundamental-fusion \(28dg b^2/3\) bound.
The equation for the complete Haar-centered correction is
\[
u=T(S_*,S_*)+2T(S_*,u)+T(u,u).
\]
A radius \(r>0\) is accepted only if
\[
D+Lr+Br^2\le r,\qquad L+2Br<1.                \tag{1}
\]
A contracting radius exists for this sufficient criterion exactly when
\(L<1\) and \((1-L)^2>4BD\).
The automatic choice \(r=2D/(1-L)\) is rational and passes when those
conditions hold; an explicitly supplied failing radius is not replaced.

Completeness gives a unique correction in the accepted ball on each
finite graph. The original Fourier derivative bounds reconstruct a
\(C^2\) function; elliptic bootstrap gives smoothness. The positive
eigenfunction \(e^{S_*+u}\) and its groundstate Dirichlet identity identify
the true neutral vacuum, as in the preceding source theorem.
Uniqueness in the correction ball is not an assertion about all possible
coordinate descriptions of the vacuum. The Hamiltonian groundstate is
unique by the positive groundstate transform on the connected compact
configuration space.

## 3. Quantitative witnesses

For cubic boxes at \(\kappa=15,b=1,r=1/10\),
\[
D=\frac{1984}{50625},\quad L=\frac{1024}{2025},\quad
r-D-Lr-Br^2=\frac{137}{101250}>0,\quad
L+2Br=\frac{1384}{2025}<1.
\]
The original-edge curvature theorem gives
\[
\rho=\frac12-\frac{16g}{3}-\frac{4r}{3}
     =\frac{367}{1350},
\qquad
\boxed{\operatorname{gap}(aH)\ge\frac\kappa2\rho
       =\frac{367}{180}>2.}                          \tag{2}
\]
The constants are the same for every stated finite cubic box, including
all spins and all generated interactions.

The smaller coupling \(\kappa=141/10\), with \(b=1,r=23/100\), also
passes (1). Its physical gap floor is \(256549/423000>0\).
At \(\kappa=14\), the current discriminant is negative. This is failure
of the sufficient criterion, not a theorem that the physical gap closes.
At \(\kappa=12\), the linear Fourier inverse is bounded by \(81/17\)
while the nonlinear criterion is still inconclusive.

At \(\kappa=15,b=33/32,r=1/8\), the source has exact self-map slack
\(149/144000\) and contraction \(19/25\).
The actual conditional Schur comparison has a positive weighted margin,
so the previously proved covariance bound carries the spatial decay
factor \(b^{-d}=(32/33)^d\) in the inherited microscopic edge distance.
The observable gradient factors and covariance kernel are those of
[the source's conditional Schur comparison](gauge-wilson-residual-source.md);
this is a statement about actual vacuum correlations at fixed coupling,
not continuum Euclidean reconstruction.

## 4. Static sources and genuine boundary conditionals

The original-coordinate curvature in (2) holds after fixing arbitrary
exterior links. The [center-cut argument](gauge-wilson-static-source.md)
therefore gives, at \(\kappa=15\), for every distinct pair \(s,t\)
in every stated finite cubic box,
\[
\boxed{\frac{367}{180}d(s,t)\le E_{s,t}-E_0
       \le\frac{45}{8}d(s,t).}                       \tag{3}
\]
Here the sector has fundamental/antifundamental static charges, no
dynamical matter and zero bare source rest energy. \(E_0\) is the true
neutral ground energy. The upper trial is fundamental transport along
a shortest path. Neither isotropic Euclidean Wilson identification nor
an infinite-volume string tension is inferred.

The [local oscillation theorem](gauge-wilson-conditional-block.md) derives
block bounds for the same actual vacuum, uniformly over exterior link
values. A one-edge conditional diffusion floor must not be presented as
a whole-box Hamiltonian gap or as the spectrum of a newly frozen Wilson
Hamiltonian. Independent block gaps alone do not justify gluing.

## 5. Replay and what remains

The new `su2_wilson_polar_vacuum_v1` type recomputes the exact structural
budgets and nonlinear and spectral gates. Old residual and linear v1
certificates retain their original \(4/3\) constants and digests.
The static and conditional consumers require canonical actual sources.

The finite vector Lean development in the consumer project verifies separate
row-wise radius, weighted contraction and elimination algebra. It does
not formalize the polar-matrix theorem, infinite Fourier completion,
vacuum construction or physical gap.

A volume-independent estimate on every finite member does not itself
construct a compatible thermodynamic state. Weak-coupling control,
continuum scaling, the limiting OS axioms and nontriviality remain
undischarged. Literature-level novelty of these bounds has not been
established.
