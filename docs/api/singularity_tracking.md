# Jet-Padé singularity tracking (03-10)

A high-order jet is a truncated Taylor series. Domb-Sykes ratios
and Padé poles estimate the nearest singularity. A proved geometric upper
coefficient tail supplies a lower convergence radius only. The upper endpoint
remains infinity: a finite prefix does not prove that any singularity exists.

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
`pade_certified_remainder`. No new
package.

`convergence_radius_from_geometric_tail(coeffs, tail_bound=M, tail_ratio=q)`
requires the caller to establish `|a_k| <= M.hi*q**k` for every coefficient
beyond the prefix. It returns an extended interval `[1/q, infinity]`, with
outward rounding. An exactly zero tail returns `[infinity, infinity]`.
The legacy `certified_singularity_annulus` name forwards to this API.
Earlier finite upper endpoints were unsound and have been removed; consumers
must not interpret the new result as a bounded annulus or singularity-existence
certificate. A finite upper radius needs a separate infinite lower-tail or
noncancellation argument, which this API does not accept or infer.

## API

::: omnibias.difference.singularity
    options:
      show_root_heading: false
      heading_level: 3
