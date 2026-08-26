# Spectral trial spaces (07-05)

Certified eigenvalue **lower** bounds are limited by the trial
space. Multi-pack bases from the founding bias collapse
(`delta -> 0`) put richness where the ground state lives.
Temperature collapse (`beta -> inf`, feasibility) does not
appear. Do not conflate the two.

Every bound reports `sin(theta)` against a numerical
reference eigenvector. Interval `LDL^T` failure is a safe
refusal. Status is **shipped**. G1–G4 and G6 are
CI-gated here; G5 lives on the SOS adapted-basis page.

Scope is **one fixed finite-dimensional operator**, one
domain, one discretization. This is not a continuum
spectral-gap theorem and not a Yang-Mills mass gap. The
parent problems stay external obligations.

Home: `omnibias.core.verified.trial_spaces`. No new package.
`theorem_prover_verified` is not asserted.

## API

::: omnibias.core.verified.trial_spaces
    options:
      show_root_heading: false
      heading_level: 3
