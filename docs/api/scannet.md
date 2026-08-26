# Scan-Net (02-01)

Stacked bias-scan banks with no pixel grid. Equivariance is
**per-layer, per-direction, on-lattice** — not the translation group of
`R^D`. Soft-argmax `gamma` is a readout sharpness, not `delta -> 0`.
Templates reuse the six `OperatorBlock` roles; Scan-Net is not a seventh
role.

G1/G2/G3/G5 are CI-gated. G3 is wall/point vs `N` over two decades
against named k-NN (Scan-Net stays bounded; k-NN grows). G4 (k-NN may
win on density) is **leftover-recorded** (leftover #17) from a real
Scan-Net lstsq vs calibrated k-NN; Scan-Net wins the constructive
mixture fit. Not in CI `all_passed`. Status is **gated**, not shipped.
See theory spec 02-01.

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
