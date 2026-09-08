# Anisotropic similarity profile (07-09)

Lemma 4.1 operators `T_b`, `Z_b` on an axis-regular swirl germ, a
radial stress whose leading residual equals `-div_r T` exactly over
`Q`, a `TaylorModel` of `E / sqrt(2X)` at `X = 0` with remainder
`{0}`, and order-`n` source jets via `jet_multiply`.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/anisotropic_profile.py`). This is a fragment of a
constructed forced blowup. It does not re-prove Clay (C)/(D) and does
not touch unforced (A)/(B). See theory spec
[07-09](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/09-similarity-profile-axis-germ.md).

Home: `omnibias.pinn.certified.anisotropic`.

```python
from fractions import Fraction

from omnibias.pinn.certified.anisotropic import (
    apply_T_b,
    axis_germ,
    coefficient_source_jet,
    honesty_payload,
    locked_axis_regular_profile,
    profile_jet_coeffs,
    residual_plus_div,
)

profile = locked_axis_regular_profile()
jet = profile.jet_at(Fraction(1), Fraction(0))
assert apply_T_b(profile.h, Fraction(0), Fraction(1), Fraction(0), jet) == 1
assert residual_plus_div(Fraction(1)) == 0
germ = axis_germ(profile, order=2)
assert germ.remainder.lo == germ.remainder.hi == 0.0
assert coefficient_source_jet(profile_jet_coeffs(order=1), order=1)["matches"]
assert honesty_payload()["navier_stokes_proof_claim"] is False
assert honesty_payload()["forced_blowup_reproof_claim"] is False
```
