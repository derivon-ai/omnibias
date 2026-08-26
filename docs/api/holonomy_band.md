# Wilson-line holonomy band (02-14)

The slab as a parallel-transport home. Closed form only in the
**abelian** and **transverse-constant** regimes. Open lines are
gauge-dependent. **No** Yang-Mills / mass-gap / continuum claim. The
gap is held finite (band), the opposite of founding `delta -> 0`.
Wraps existing `parallel_transport` / `wilson_loop`.

G1/G2/G5 are CI-gated. G2 closed-form exactness is **earned** on the
smoke artifact: `band_holonomy` matches PRODUCT at `substeps=4096` to
`<= 1e-12` in the abelian and transverse-constant regimes, at a
fraction of the PRODUCT cost. G3 Magnus soundness is **leftover-recorded**
unearned (leftover #34): the bound contains 0 on a grid and a sample
and refuses `||A|| L >= pi`, but no Magnus-truncated holonomy is wired.
G4 gauge covariance is **earned** (leftover #35 closed):
`random_u1_gauge` plus `g(x_hi) U g(x_lo)^{-1}` matches the gauged
holonomy to `<= 4` ulp, the open-line flag stays, and a forward-back
`band_wilson_loop` identity is within `4` ulp. G3 is not in CI
`all_passed`.
No YM / mass gap / continuum claim. Status is **shipped**.
See theory spec 02-14.

## Algebra and twins

::: omnibias.geometry.gauge.band
    options:
      show_root_heading: false
      heading_level: 3
