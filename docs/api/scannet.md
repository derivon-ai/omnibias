# Scan-Net (02-01)

Stacked bias-scan banks with no pixel grid. Equivariance is
**per-layer, per-direction, on-lattice** — not the translation group of
`R^D`. Soft-argmax `gamma` is a readout sharpness, not `delta -> 0`.
Templates reuse the six `OperatorBlock` roles; Scan-Net is not a seventh
role.

G1/G2/G5 are CI-gated. G3 (wall/point vs `N`) and G4 (k-NN may win on
density) are recorded, not in CI `all_passed`. Status is **gated**, not
shipped. See theory spec 02-01.

## PyTorch module

::: omnibias.torch.architectures.scannet
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.architectures.scannet
    options:
      show_root_heading: false
      heading_level: 3
