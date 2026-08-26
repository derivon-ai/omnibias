# Constraint satisfaction by collapse (03-03)

A finite-domain CSP is a product of multilinear relations plus compact
global relaxations. `E = 0` exactly on satisfying one-hot vertices.
Status is **shipped**. G1–G6 are CI-gated.

Simplex sharpness `p = softmax(beta_1 z)` and clause sharpness
`log(m)/beta_2` are two independent **temperature collapse** knobs
(`beta -> inf`, feasibility). Neither is the founding bias collapse
(`delta -> 0` to `sigma^(K-1)`). Do not conflate the two. A
relaxation is not a complete solver and never proves
unsatisfiability. Certified statements are instance-wise gaps, not
P vs NP.

Home: `omnibias.discrete.csp`. No new package.

## API

::: omnibias.discrete.csp
    options:
      show_root_heading: false
      heading_level: 3
