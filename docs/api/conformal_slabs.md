# Uncertainty and conformal slabs (04-02)

A slab `[lo, hi]` is the shared *shape* of three different
guarantees. They are not interchangeable.

| Register | Object | Guarantee | Assumptions |
|---|---|---|---|
| Sound enclosure | `Interval` / `SOUND_ENCLOSURE` | the true value of this function on this box is inside, probability 1 | interval arithmetic |
| Conformal | a prediction set | marginal coverage `>= 1 - alpha` | exchangeability |
| Bayesian / Fisher | a credible or Wald interval | coverage under the model | the model is correct |

The band role uses a **finite** gap. That is the opposite of founding
bias collapse (`delta -> 0`). Temperature collapse (`beta -> inf`,
feasibility) does not appear.

Conformal intervals cannot be sealed into certificate v1. Combined
statements keep the enclosure and the residual quantile separate.
Status is **gated**, not shipped.

Homes: `omnibias.core.uncertainty`, `omnibias.verify.uncertainty`.

::: omnibias.core.uncertainty
    options:
      show_root_heading: false
      heading_level: 3
