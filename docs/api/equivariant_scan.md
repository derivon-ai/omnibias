# Equivariant and chart scan (02-08)

Orientation orbits of a `BiasScan` bank. Exact steering is
**gaussian-family only** (`steerable_basis` returns `None` otherwise).
The discrete orbit is `C_L`, not SO(2) or SO(3). Chart-coordinate
scan uses the existing pullback metric `g = J^T h J`.

G1–G4 are CI-gated. G5 anisotropic-interface is `--full`. Status is
**gated**, not shipped. See theory spec 02-08.

## PyTorch module

::: omnibias.torch.scan_equivariant
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.scan_equivariant
    options:
      show_root_heading: false
      heading_level: 3

## Chart scan

::: omnibias.geometry.scan
    options:
      show_root_heading: false
      heading_level: 3
