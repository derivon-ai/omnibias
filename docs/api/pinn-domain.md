# omnibias.pinn.domain (SDF / R-function geometry)

Non-box geometry for PINNs: signed-distance primitives, R-function CSG,
normalized approximate distance functions (ADF), SDF-aware sampling, and the
`DistanceConstrainedField` hard-BC cage `u = g + φ · NN`.

Named `domain` (not `geometry`) to avoid colliding with the
`omnibias-geometry` manifold package.

Maturity: **alpha** submodule of Beta `omnibias-pinn`.

## Guarantee level

| Piece | Level | Acceptance domain |
| --- | --- | --- |
| Dirichlet on `φ = 0` | by construction | smooth primitives / R-compositions with a vanishing factor |
| Neumann / Robin | by construction | smooth primitives with well-defined normals; **raises** at non-smooth CSG junctions |
| Negative-inside R-CSG | algebraic | `intersect` / `union` use `r_*_sdf` (positive-inside Rvachev primitives remain available) |
| ADF / higher jets | autodiff-exact / FD | analytic ω jets for Sphere / Halfspace / Box; normalized ADF order &gt; 1 may use FD |
| Residual accuracy off-boundary | empirical | optimised, not proven |

`solver.Domain.sdf` drives interior / boundary / RAR candidate sampling.

## Core schemas

::: omnibias.pinn.domain
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - Sphere
        - Box
        - Halfspace
        - Cylinder
        - Polygon
        - intersect
        - union
        - complement
        - approximate_distance
        - interior_points_sdf
        - boundary_points_sdf

## Torch drivers

::: omnibias.pinn.domain.torch
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - DistanceConstrainedField
        - build_distance_constrained_field
        - sphere_distance
        - box_distance
        - normalize_distance

## Solver integration

`omnibias.pinn.solver.Domain` accepts an optional `sdf=` carving a non-box
interior out of the bounding box (`Domain.is_sdf`).

## Example

See [`docs/examples/pinn_sdf_geometry.py`](../examples/pinn_sdf_geometry.py).
