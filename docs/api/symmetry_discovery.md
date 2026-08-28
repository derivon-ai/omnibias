# Lie symmetry discovery (03-11)

A Lie point symmetry's determining equations are linear in the
generator once the prolongation is known. Prolongation is a jet
computation, so the algebra is a nullspace. Status is **shipped**.
G1–G6 are CI-gated.

This recovers **point symmetries inside a declared ansatz**.
Dimensions are in-ansatz, not a classification of every
symmetry of the PDE family. No Navier-Stokes regularity claim.

Jets come from the **founding bias collapse** (`delta -> 0`).
Temperature collapse (`beta -> inf`, feasibility) does not
appear. Do not conflate the two.

The rank threshold and the singular-value separation are required
output from the float SVD proposer. Before a proposed dimension is marked
`verified`, omnibias snaps the finite determining matrix to bounded-denominator
rationals, clears row denominators, and checks its exact integer kernel with
`rank_collapse`. The resulting `ExactSymmetryReport` is `PROVED`,
`DISPROVED`, or `BLOCKED` for that **snapped finite matrix**; it is not an
exact proof about all real samples or the entire PDE.

A symmetry outside the basis is invisible. "No symmetries found" means none in
this basis. Linear PDEs can fill almost the whole ansatz; that case is
reported as `infinite_in_ansatz`.

Home: `omnibias.symbolic.symmetry`. No new package.

## API

::: omnibias.symbolic.symmetry
    options:
      show_root_heading: false
      heading_level: 3
