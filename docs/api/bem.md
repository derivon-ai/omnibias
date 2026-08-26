# BEM-Net (02-06)

Layer potentials from the antiderivative window. The PDE is exact
**off-surface** by construction; the boundary condition is
approximated. Linear constant-coefficient homogeneous equations only.
No 2-D/3-D FMM. The half-plane Dirichlet-to-Neumann map uses the
[conjugate Hilbert](conjugate.md) dictionary.

G1 off-surface residual and G5 DtN (<= 4 ulp) are CI-gated. G4
mollifier order is CI-gated. G2 disc-accuracy (annulus L2 `<= 1e-8`)
is **earned** (leftover #24 closed): `circle_dirichlet_density` is
the single-layer Fourier solve, not a train, and is in CI
`all_passed`. `single_layer` wall vs `n_quad` is **reported**. Cost
is not in CI `all_passed`. G3 exterior win is **leftover-recorded**
unearned (leftover #30): pack-tree 02-07 has a dense crossover, and
no truncated volumetric PINN loop is wired. G3 is not in CI
`all_passed`. Status is **gated**, not shipped. See theory spec 02-06.

## Algebra and twins

::: omnibias.pinn.bem
    options:
      show_root_heading: false
      heading_level: 3
