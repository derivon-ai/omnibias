# Neural quadrature (03-06)

A pack's moments are closed form, so a bank of packs is a quadrature
rule whose nodes and weights are *solved* for exactness on a prescribed
space. Status is **gated**, not shipped. G1–G5 are CI-gated.

Pack functionals come from the **founding bias collapse** (`delta -> 0`).
Temperature collapse (`beta -> inf`, feasibility) does not appear. Do
not conflate the two. The Peano remainder is a sound enclosure when a
bound on `f^(d+1)` is supplied and is **refused** otherwise.
Non-product cubature is out of scope. Gauss–Legendre is hard to beat
on generic smooth integrands; the design win is on a specialized family.

Home: `omnibias.core.cubature` plus `integrate_certified` on
`omnibias.fields._core.quadrature`. No new package.

## API

::: omnibias.core.cubature
    options:
      show_root_heading: false
      heading_level: 3
