# omnibias.pinn.domain (SDF / R-function geometry)

Non-box geometry for PINNs: signed-distance primitives, R-function CSG,
normalized approximate distance functions (ADF), SDF-aware sampling, and the
`DistanceConstrainedField` hard-BC cage `u = g + φ · NN`.

Named `domain` (not `geometry`) to avoid colliding with the
`omnibias-geometry` manifold package.

Maturity: **alpha** submodule of Beta `omnibias-pinn`.

## Honest method labels

- **Dirichlet satisfaction on `φ = 0`** is exact by construction.
- **ADF normalization** of a general SDF uses a numerical gradient unless an
  analytic `grad_fn` / torch autodiff path is supplied (`autodiff-exact` for
  the torch primitives).
- Residual accuracy away from the boundary is **optimised, not proven**.

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
