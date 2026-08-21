# Jet-Padé singularity tracking (03-10)

A high-order jet is a truncated Taylor series. Domb-Sykes ratios
and Padé poles locate the nearest singularity; a coefficient tail
bound turns `|x_s|` into a sound annulus. Status is **gated**,
not shipped. G1–G6 are CI-gated.

This is a **diagnostic and an estimate, not a proof of blow-up**.
Locating a complex singularity of truncated Taylor data at
sampled times does not control a PDE. No Navier-Stokes
regularity claim.

Jets come from the **founding bias collapse** (`delta -> 0`).
Temperature collapse (`beta -> inf`, feasibility) does not
appear. Do not conflate the two.

Domb-Sykes assumes a single dominant algebraic singularity.
Essential singularities and comparable-distance pairs report
failure. Froissart doublets are filtered at a machine-epsilon
threshold (`1e-8`), not a suite-tuned cutoff. Along a ray only;
not a multivariate singular variety.

Home: `omnibias.difference.singularity` plus
`omnibias.fields.singularity`. Reuses `pade_approximant`,
`pade_certified_remainder`, and `geometric_tail_bound`. No new
package.

## API

::: omnibias.difference.singularity
    options:
      show_root_heading: false
      heading_level: 3
