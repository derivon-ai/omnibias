# Hilbert XVI H7: polygonal barriers and the SOS obstruction route

H7 has two parts. The first is already implemented more strongly than
proposed; the second lacks the reduction needed to state a sound global
obstruction.

## Exact polygonal barriers already exist

`omnibias.geometry.algebraic` provides exact `RationalPolygon` and
`PolygonalAnnulus` objects. It checks simplicity, disjointness, strict
containment, nesting, and whole-edge Bernstein signs over \(\mathbb Q\).
`omnibias.geometry.algebraic_layout` constructs and validates 22 annuli with
the wide/deep open \((19,3)\) target tree. This is stronger for certification
than hardening floating oblique gates from `omnibias.partition.arrangement`.

The canonical layout yields 1,584 exact linear inequalities in the 45
homogeneous octic coefficients. Its current normalized full-coefficient LP
has nonpositive proposed margin and no accepted coefficient vector. No exact
Farkas witness was found, so even this fixed layout has not been proved
infeasible.

Thus widening rectangles to polygons removes a representation limitation but
does not produce a 22-oval polynomial.

## What Positivstellensatz can and cannot prove

The SOS engine is functional: the audit certifies emptiness of the toy basic
closed set

\[
x-1\ge0,\qquad -x-1\ge0
\]

by a rational Putinar identity with interval-certified positive Gram blocks.

But "no real octic realizes scheme \(S\)" is not currently presented as a
finite basic closed set in 45 coefficients. It includes:

- quantification over the entire real projective locus;
- ambient-isotopy/nesting conditions;
- complex projective nonsingularity; and
- all possible embeddings, not one chosen annulus layout.

A Positivstellensatz certificate for a fixed polygonal layout or a
symmetry-reduced coefficient ansatz excludes only that finite ansatz. To lift
it to the scheme requires a proved reduction saying every realization can be
moved into the encoded family. No such reduction is known for either open
\((19,3)\) scheme.

Scaling is already severe before those missing constraints are encoded. A
dense SOS basis of half-degree two in 45 variables has 1,081 monomials and
584,821 independent Gram entries. Higher useful degrees grow
combinatorially.

```text
polygonal_layout_valid = true
toy_positivstellensatz_empty_set_certified = true
scheme_basic_closed_encoding_supplied = false
symmetry_reduction_complete_for_all_realizations = false
octic_scheme_obstructed = false
part_a_22_oval_realized = false
full_hilbert16_solved = false
```

Reproduce with:

```bash
python -m pytest \
  packages/omnibias-geometry/tests/test_part_a_obstruction.py -q
python benchmarks/hilbert16_part_a_obstruction.py --full
```
