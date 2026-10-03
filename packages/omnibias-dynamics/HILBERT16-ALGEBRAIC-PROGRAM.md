# The algebraic part of the full Hilbert 16 program

The dynamics program remains the leading workstream. The algebraic part has
independent obligations: classify real plane curves and spatial surfaces by
degree, including the positions of their components. Neither a local cyclicity
bound nor one new algebraic construction completes these obligations.

## Equivalence and scope

[Hilbert's original statement](https://people.reed.edu/~davidp/341/resources/hilbert.pdf),
printed pages 464--465, discusses relative positions of curve branches, emphasizes
the maximal case, and also asks for the number, form, and position of surface
sheets. This program uses the following precise modern targets:

- A plane curve is defined by a real homogeneous polynomial whose complex
  projective curve is nonsingular. Its **real scheme** is the ambient isotopy
  class of its real locus in the real projective plane. For even degree this
  records the complete forest of nested ovals; for odd degree there is also one
  pseudoline.
- A spatial surface is the real locus of a complex-nonsingular homogeneous
  polynomial in four variables, embedded in real projective three-space.
  Component counts and abstract genera are necessary data but do not replace
  the embedding classification.
- **Rigid isotopy** additionally requires a path through nonsingular
  polynomials of the same degree. Complex orientations and dividing type
  provide refinements. An ambient-isotopy certificate below does not classify
  these refinements. Singular strata and arrangements relative to auxiliary
  curves or the hyperplane at infinity are separate specified problems.

Real nonsingularity alone is weaker than the convention used here. For example,
`(X*X + Y*Y)**2 - Z**4` has a smooth real circle, but has complex singularities
at `[1:i:0]` and `[1:-i:0]`.

## Current octic frontier

As checked on 13 September 2026, the latest version of Geiselmann, Joswig,
Kastner, Mundinger, Pokutta, Spiegel, Wack, and Zimmer,
[*Limits of combinatorial patchworking*](https://arxiv.org/pdf/2602.06888v4),
is v4 dated 31 August 2026. Table 1, printed page 17, and Section 4.3 distinguish
89 pseudoholomorphic maximal octic schemes, 83 algebraically realized and six
algebraically unresolved. Those six are:

| Scheme | Algebraic status | Primitive regular T-curve status |
| --- | --- | --- |
| `4 + 1<2 + 1<14>>` | Open; the selected target | Open |
| `14 + 1<2 + 1<4>>` | Open | Open |
| `1 + 1<1> + 1<18>` | Open | Excluded |
| `1 + 1<4> + 1<15>` | Open | Excluded |
| `1 + 1<7> + 1<12>` | Open | Excluded |
| `1 + 1<9> + 1<10>` | Open | Excluded |

Theorem 21 characterizes the four achievable maximal T-curve columns; Corollary
22 excludes the `(p,n)=(3,19)` column. That exclusion does not touch the two
open `(19,3)` schemes (`4 + 1<2 + 1<14>>` and `14 + 1<2 + 1<4>>`). Its archive
of 2,367 nonempty octic schemes
is not exhaustive. Nonmaximal octics, higher degrees, and rigid-isotopy
refinements therefore remain explicit inventory obligations.

[Orevkov's classification](https://www.math.univ-toulouse.fr/~orevkov/m8.pdf)
concerns pseudoholomorphic maximal octics and distinguishes that category from
algebraic realizability. A general obstruction valid for every
pseudoholomorphic curve cannot exclude a scheme already realized in that
category. An algebraic obstruction must use an additional algebraic ingredient.

The [selected-target companion](HILBERT16-ALGEBRAIC-TARGET.md) records the bounded
patchwork and annular searches. Their failures do not exclude all algebraic
realizations. In particular, an exact Farkas witness excludes only its fixed
sign inequalities, and eliminating all primitive regular triangulations would
exclude only that T-curve class.

## Implemented coefficient-to-certificate route

[`omnibias.geometry.algebraic`](../omnibias-geometry/src/omnibias/geometry/algebraic.py)
uses the existing pure-Python rational `SparsePolynomial`. It accepts actual
homogeneous coefficients and performs the following checks:

1. **Complex projective nonsingularity.** In each chart `X_i=1`, three rational
   polynomial multipliers satisfy the exact identity
   `A_0 F_X + A_1 F_Y + A_2 F_Z = 1` after dehomogenization. The identity holds
   over the complex numbers, so the three homogeneous partials cannot vanish
   together anywhere in that chart. All three charts are required. Euler's
   identity and the chart cover then exclude every projective singularity.
2. **Whole-boundary signs.** For each edge of each rational rectangular
   boundary, substitute its affine parameterization into the polynomial and
   compute rational Bernstein coefficients. Positive signed coefficients
   certify a strict sign on the entire segment. Exact dyadic subdivision
   tightens inconclusive coefficients. Opposite signs are required on the two
   boundaries of each annulus.
3. **Disjointness and nesting.** Every inner rectangle must lie strictly within
   its outer rectangle. Distinct annuli must have strictly separated outer
   rectangles, or one outer rectangle must lie strictly inside the other's
   inner rectangle. The checker derives immediate parents without accepting a
   proposed topology label.
4. **Component conclusion.** Opposite boundary signs force an essential oval in
   each annulus. These are distinct. In odd degree the unique pseudoline is
   additional. Harnack gives the upper bound `(d-1)*(d-2)/2 + 1`. Only if the
   certified lower count attains that bound does the result assert that its
   nesting forest describes the entire curve.

The certificate is bound to a canonical hash of the polynomial coefficients.
`replay_curve_certificate` recomputes the identities, geometry, signs, and all
reported conclusions. A different source polynomial or altered conclusion is
rejected. This is exact Python replay, without a theorem-prover verification
flag. The topological implication uses the usual separation theorem and
Harnack theorem; these analytic/topological results have not been formalized by
the finite rational checker.

`find_smoothness_witness` is a bounded rational linear-algebra producer for the
three identities. `None` is inconclusive at the chosen multiplier degree.
Increasing its unknown budget does not constitute an algebraic classification.
Alternatively, an external symbolic solver can provide identities for exact
replay. No new symbolic-algebra dependency is required by this module.

All annuli in one certificate use the same declared affine chart, selectable
from the three standard projective charts. Rectangular annuli are a conservative
subclass of the polygonal layouts in the selected-target companion. Subdividing
their edges improves sign bounds but does not turn their rectangular shapes
into a complete enumeration of possible oval embeddings.

The following example is a complete maximal conic certificate:

```python
from fractions import Fraction as Q
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.geometry.algebraic import (
    HomogeneousPlaneCurve,
    RationalRectangle,
    RectangularAnnulus,
    certify_curve,
    find_smoothness_witness,
    replay_curve_certificate,
)

X, Y, Z = (P.variable(3, axis) for axis in range(3))
curve = HomogeneousPlaneCurve(X**2 + Y**2 - Z**2)
smoothness = find_smoothness_witness(curve, max_multiplier_degree=0)
assert smoothness is not None
annulus = RectangularAnnulus(
    RationalRectangle(-Q(1, 2), Q(1, 2), -Q(1, 2), Q(1, 2)),
    RationalRectangle(-2, 2, -2, 2),
)
certificate = certify_curve(curve, smoothness, [annulus])
assert certificate.complete_real_scheme
assert certificate.barrier_parents == (-1,)
assert replay_curve_certificate(curve, certificate)
```

## Exact baselines and limits

The focused regression suite reconstructs the following coefficient-level
certificates. These are implementation baselines, not new real schemes.

| Polynomial | Certified conclusion |
| --- | --- |
| `X^2 + Y^2 - Z^2` | Complex nonsingular; one oval; complete maximal scheme |
| `(X^2-Z^2)^2 + (Y^2-Z^2)^2 - Z^4/16` | Complex nonsingular; four separated ovals; complete maximal quartic scheme |
| `(X^2+Y^2-Z^2)(X^2+Y^2-4Z^2) + X^4/100` | Complex nonsingular; at least two nested ovals; full classification not asserted by this certificate |
| `X^8 + Y^8 - Z^8` | Complex nonsingular; at least one oval; upper bound 22 |
| `[(X^2-Z^2)(X^2-9Z^2)]^2 + [(Y^2-Z^2)(Y^2-9Z^2)]^2 - Z^8/16` | Complex nonsingular; at least 16 separated ovals; upper bound 22 |

For the last example the producer finds rational smoothness multipliers of
degree at most 12 in each chart. The 16 annuli have centers `(a,b)` for
`a,b in {-3,-1,1,3}`, inner square half-width `1/1024`, and outer half-width
`1/32`. Whole-boundary signs and all-chart identities replay exactly. This is
not the 22-oval target and does not establish that these are its only ovals.

Tests also reject complex-only singularities, real singularities at infinity,
an incomplete projective chart cover, overlapping annuli, misleading corner
samples, and modified certificate operands/conclusions. The Bernstein
restriction is independently checked by exact evaluation on a deterministic
grid and seeded random rational points.

## Candidate-to-proof workstreams

| Workstream | Positive gate | Negative gate |
| --- | --- | --- |
| Compatible maximal T-curve splits | Exact triangulation, signs, regular heights, and target forest; admissible parameter range for an explicit polynomial | Complete proof-producing enumeration of the declared T class only |
| General rational polynomial and annular search | Rational coefficients, all-chart complex nonsingularity, whole-boundary signs, and maximal count with the target nesting | Fixed-layout Farkas or fixed-family infeasibility only, unless every embedding is covered by a proved reduction |
| Singular smoothing and confluent polynomial coordinates | Exact source coefficient map, controlled smoothing, then the same polynomial certificate | A theorem covering every algebraic degeneration needed before family exclusions become global |
| Algebraic auxiliary pencils and discriminant strata | A surviving exact realization and verified topology | A complete algebraicity-sensitive case reduction plus exact exclusion of every case, including infinity and degenerate fibers |

The existing confluence polynomial pair evaluates terminating finite expansions
in attained collision coordinates. It can stabilize candidate generation.
General sigmoid/tanh approximants are not degree-eight polynomials. Exact
coefficient conversion and a full coefficient-space correction must remain
visible; a restricted architecture cannot silently stand for all octics.

The exact SOS and rational realization modules can replay finite identities and
positivity obligations. Failure of a fixed-degree SOS relaxation is
inconclusive. The finite continuation certificates accept supplied sound
residual/Jacobian enclosures; they do not automatically produce a complete
discriminant atlas or compute ambient topology.

[Deng--Rojas--Telek](https://arxiv.org/html/2403.08497v2) give signed-discriminant
and patchworking-completeness results under specific signed-support and
proper-face conditions, notably for codimension-two supports. These do not
apply automatically to all 45 octic coefficients or all projective charts.
[The 2026 OSCAR curve paper](https://arxiv.org/html/2603.12985v1) is useful for
elimination and critical-fiber candidate analysis, but its singularity-detection
heuristic and floating rendering are not substitutes for exact final replay.

Classical quantifier elimination and semialgebraic triviality give an
in-principle finite coefficient-space route at each fixed degree. The actual
topology encoding, chart coverage, cell decomposition, representatives, and
exclusions still have to be produced. Rational witness enumeration alone is a
positive semidecision procedure, and is not a negative stopping criterion.

## Spatial surfaces remain an independent obligation

The classical degree-four surface classification, with Kharlamov's work and
[Nikulin's final isotopy exclusion](https://www.mathnet.ru/eng/im1677), is a
baseline to reproduce. The complete topology of the real locus cannot be read
from a list of Betti numbers.

An elementary coefficient-level surface baseline is now implemented in
[`omnibias.geometry.algebraic_surfaces`](../omnibias-geometry/src/omnibias/geometry/algebraic_surfaces.py).
For rational `0 < epsilon < 1`, it certifies the normalized homogeneous quartic

`F = (X^2-W^2)^2 + (Y^2-W^2)^2 + (Z^2-W^2)^2 - epsilon*W^4`.

The checker infers `epsilon` from the actual `W^4` coefficient and verifies the
entire polynomial identity. Its nonsingularity and topology proof is specific
to this family:

1. At complex infinity `W=0`, the first three partials are `4X^3`, `4Y^3`, and
   `4Z^3`. Their simultaneous vanishing gives no projective point. In the
   affine chart `W=1`, these partials factor as `4x_i(x_i^2-1)`. Every complex
   critical coordinate therefore belongs to `{0,-1,1}`. The checker enumerates
   all 27 combinations and verifies that the unshifted critical levels are
   exactly `0,1,2,3`, separated from `epsilon` by a positive rational gap.
2. There are no real points at infinity, because a sum of three real fourth
   powers can vanish only when all coordinates vanish. In the affine real
   locus, no `x_i` is zero: that would make the sum at least one.
3. On each of the eight sign orthants, the map `u_i=x_i^2-1` is a
   diffeomorphism onto `(-1,infinity)^3`, with inverse
   `x_i=sign_i*sqrt(1+u_i)`. The equation becomes the sphere
   `u_1^2+u_2^2+u_3^2=epsilon`. Its radius is strictly less than one, so the
   sphere and its closed ball lie entirely in that image. Their inverse images
   give one sphere bounding a ball in each orthant. The eight balls are
   disjoint, and these spheres exhaust the entire real projective locus.

The certificate reports the derived parameter, critical levels, rational gaps,
eight component orthants, and eight zero genera. Replay recomputes every
operand from the supplied source. It rejects modified coefficients or
conclusions, inexact parameters, and the singular boundary parameters zero and
one. The separation and diffeomorphism arguments are not a formal Lean proof.

```python
from omnibias.geometry.algebraic_surfaces import (
    certify_separable_quartic_surface,
    replay_separable_quartic_surface,
    separable_quartic_polynomial,
)

surface_polynomial = separable_quartic_polynomial(Q(1, 16))
surface_certificate = certify_separable_quartic_surface(surface_polynomial)
assert surface_certificate.component_count == 8
assert surface_certificate.component_genera == (0,) * 8
assert replay_separable_quartic_surface(surface_polynomial, surface_certificate)
```

This reproduces one known elementary quartic type. It does not reproduce the
complete quartic classification or the quintic construction below.

[Orevkov's quintic construction](https://www.math.univ-toulouse.fr/~orevkov/q23.pdf)
has 23 connected components: 22 spheres and a real projective plane with two
handles. It is a literature baseline, not a construction replayed by the
plane-curve module. The known component upper bound is 25; the existence of 24
or 25 components remains unresolved in the March 2026 surface account below.

[Akyar--Shkolnikov, *Introduction to non-Abelian Patchworking*](https://arxiv.org/html/2603.08094v1)
propose a spatial-surface construction using curve arrangements on quadrics.
The general realization claim is explicitly conjectural: its complete proof
is missing. Their diagram topology/Euler-characteristic results and checks
through degree three do not establish an all-degree algebraic realization
theorem. Reproducing the quartic list can test this proposal; a rigorously
impossible diagram could refute it. Any new quintic must have its own actual
polynomial realization proof or a proved gluing theorem.

[Gunes Aktas, *Real Structures in the Moduli of Projective Models of K3-Surfaces*](https://arxiv.org/html/2608.05941v1)
settles real representability of real equisingular strata of simple quartics:
exactly three real strata have no real representatives, and every other real
stratum has one. The exceptions have singularity sets
`A7+A6+A3+A2`, `D7+A6+A3+A2`, and `A7+A5+A3+A2+A1`, with one exceptional
stratum for each set. These are proved exceptions, not unresolved cases.
The result concerns real representatives in moduli strata, not a classification
of arbitrary-degree smooth surface embeddings.

[Maletto's 2026 arrangement encoding](https://arxiv.org/html/2606.21449v1)
uses intersection numbers, Dyck words, and rooted trees for transverse curves
relative to a fixed cellular arrangement. It may help organize quadric-surface
input data, but curves on surfaces and spatial surface embeddings remain
distinct problems. The repository replays **one** published
`(n, W, T)` example: the §1.1 quartic

    f = x^4 + x^2 y^2 + 2 x y^3 - y^4 - 2 x^3 z + x y^2 z - y^3 z
        - 3 x^2 z^2 - 2 x y z^2 + 2 y^2 z^2 + y z^3 + z^4

with edge counts `(1, 2, 1, 1, 0, 1)`, four Dyck words, and one floating
tree in region 10. Bézout edge bounds and Dyck validity hold. The existing
plane-curve Nullstellensatz witness certifies complex projective
smoothness of this `f`. A complete real scheme stays `BLOCKED` without
annuli. This is not a classification of 119 cubics, not the octic
target, and not a G1 pass. The Julia library NWT is not ported.

The remaining surface acceptance gates are: exact four-variable polynomial
witnesses, complex projective nonsingularity in all four charts, certified
embedded topology, completeness of the proposed list, and exclusions of every
other configuration for each degree. None is inferred from the plane-curve
certificate implementation. The eight-sphere surface baseline meets the
realization, nonsingularity, and embedded-topology gates for its declared
quartic family only.
