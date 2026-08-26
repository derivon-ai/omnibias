# BEM-Net (02-06)

Layer potentials from the antiderivative window. The PDE is exact
**off-surface** by construction; the boundary condition is
approximated. Linear constant-coefficient homogeneous equations only.
No 2-D/3-D FMM. The half-plane Dirichlet-to-Neumann map uses the
[conjugate Hilbert](conjugate.md) dictionary.

G1 off-surface residual and G5 DtN (<= 4 ulp) are CI-gated. G4
mollifier order is CI-gated. G2 disc-accuracy (annulus L2 `<= 1e-8`)
is **leftover-recorded** unearned (leftover #24): no Dirichlet
density solve is wired, so the named annulus L2 stays `--full`.
`single_layer` wall vs `n_quad` is **reported**; the previous untimed
`passed=True` stub is withdrawn. Cost is not in CI `all_passed`. G3
exterior win is **reported** unearned: pack-tree 02-07 has no dense
crossover, and no truncated volumetric PINN loop is wired. The
previous spec-status smoke/`--full` line is withdrawn. G3 is not in
CI `all_passed`. Status is **gated**, not shipped. See theory spec
02-06.

## Algebra and twins

::: omnibias.pinn.bem
    options:
      show_root_heading: false
      heading_level: 3
