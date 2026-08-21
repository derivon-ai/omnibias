# Differentiable topology (03-09)

Cell counts and Euler characteristics are integers, so **no
differentiable function equals a Betti number**. Soft cell masses
and 1-D Morse persistence values are surrogates with a stated
gap to the integer truth. Status is **gated**, not shipped.
G1–G6 are CI-gated.

`beta -> inf` is **temperature collapse** (feasibility). The
**founding bias collapse** (`delta -> 0`) appears only in the
underlying field. Do not conflate the two.

Persistence is differentiable **almost everywhere** (pair swaps).
The Euler gap is a **sum over faces** and is loose for large
arrangements. Certified integer counts require a separating
spectral enclosure; otherwise the answer is `Inconclusive`.
Not a continuum topology theorem and not a persistent-homology
library.

Home: `omnibias.shape.topology`. Reuses
`count_eigenvalues_below`. Face masses may come from
`partition_weights`. No new package.

## API

::: omnibias.shape.topology
    options:
      show_root_heading: false
      heading_level: 3
