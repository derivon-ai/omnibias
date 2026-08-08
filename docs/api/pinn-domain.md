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

## Measured geometry gate

[`benchmarks/geometry_sdf.py`](https://github.com/derivon-ai/omnibias/blob/main/benchmarks/geometry_sdf.py) (`--full`,
5 seeds) reports, among other cases:

| Case | Hard skill (median) | Hard boundary max\|u−g\| | Soft boundary max\|u−g\| |
| --- | ---: | ---: | ---: |
| Disk | ≈ 0.91 | ~1×10⁻¹⁶ | ~2×10⁻² |
| Annulus | ≈ 0.66 | ~3×10⁻¹⁷ | ~7×10⁻² |
| CSG box∩cap | ≈ 0.82 | ~6×10⁻¹⁷ | ~3×10⁻² |
| Nonconvex L | interior open (skill can lag soft) | ~1×10⁻¹² (identity retained) | ~0.18 |

Boundary Dirichlet on `φ=0` is by construction; nonconvex interior accuracy
remains an open frontier. Matrix:
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

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
Acceptance artifact + matrix:
[`docs/benchmarks/geometry_sdf.json`](../benchmarks/geometry_sdf.json),
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).
