# BEM-Net (02-06)

Layer potentials from the antiderivative window. The PDE is exact
**off-surface** by construction; the boundary condition is
approximated. Linear constant-coefficient homogeneous equations only.
No 2-D/3-D FMM. The half-plane Dirichlet-to-Neumann map uses the
[conjugate Hilbert](conjugate.md) dictionary.

G1 off-surface residual and G5 DtN (<= 4 ulp) are CI-gated. G4
mollifier order is CI-gated. G2/G3 are smoke/`--full` (small-N if the
pack-tree crossover is high). Status is **gated**, not shipped. See
theory spec 02-06.

## Algebra and twins

::: omnibias.pinn.bem
    options:
      show_root_heading: false
      heading_level: 3
