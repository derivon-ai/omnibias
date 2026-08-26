# Inverse problems and imaging (05-01)

Most imaging inverse problems ask two questions — **where is the
interface** and **what kind of jump does it carry**. The scan answers
both: the peak location is ``tau*`` and the channel index minus two is
the jump order. Status is **shipped**. G1–G7 are earned
for the locally-seeded logistic scan and the product API in
`omnibias.pinn.inverse`.

This is a submodule of `omnibias-pinn`, not a package. It is distinct
from `omnibias.pinn.solver.torch.inverse` (PDE-coefficient recovery).
Ill-posedness is not removed. The Krawczyk enclosure (spec 03-08) and a
conformal slab (spec 04-02) are different guarantee kinds and must not
be merged. `alpha` is a tempering scale, not founding bias collapse and
not temperature collapse.

Home: `omnibias.pinn.inverse`. Five-layer inversion uses a
known-index or known-thickness regularizer so the transfer map is
identifiable. Sensor design reports the `(1 - 1/e)` submodular
guarantee on the *optimization*, not on recovery quality.

::: omnibias.pinn.inverse
    options:
      show_root_heading: false
      heading_level: 3
      filters: ["!^_"]
