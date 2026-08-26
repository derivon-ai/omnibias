# Certified scan localization (03-08)

A bias scan's response is a closed-form function of one offset, so
Krawczyk on `r'` produces a **sound enclosure of the peak** and
proves uniqueness in a declared box. Status is **shipped**.
G1–G6 are CI-gated.

The scan template comes from the **founding bias collapse**
(`delta -> 0`). Temperature collapse (`beta -> inf`, feasibility)
does not appear. Do not conflate the two.

`Inconclusive` is a first-class return: flat or inflected peaks
are refused rather than certified. Scope is `local_box`. A
deterministic enclosure of noisy data is **conditional on the
data**. The sealed payload is a sound enclosure, not
`theorem_prover_verified`.

Home: `omnibias.verify.localization`. Reuses
`krawczyk_certificate` and the v1 certificate format. No new
package.

## API

::: omnibias.verify.localization
    options:
      show_root_heading: false
      heading_level: 3
