# Jet-flat forced concentrating field (07-13)

An explicit axisymmetric polynomial swirl on the locked 07-09 scales
whose regularized leading stress `T_0` vanishes at the axis after one
exact-`Q` correction of the linear slope. Core `||u||_infty ~ tau^{-A}`
and core energy `O(tau^{1/2-3h}) -> 0` are exact monomials. Force is
that residual times a mollifier cutoff; a from-rest ramp is `0` at
`t = 0`.

Status is **shipped**. G1–G6 are CI-gated
(`benchmarks/forced_flat_blowup.py`). This is a **different weaker
object** than Clay (C)/(D): the force need not extend smoothly through
the singular time. It does not touch unforced (A)/(B). See theory spec
[07-13](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/13-jet-flat-forced-blowup.md).

Home: `omnibias.pinn.certified.forced_flat`.

```python
from fractions import Fraction

from omnibias.pinn.certified.forced_flat import (
    axis_T0,
    correct_axis_stress,
    core_energy_scale,
    core_linfty_scale,
    from_rest_ramp,
    honesty_payload,
    locked_jet_flat_profile,
    uncorrected_jet_flat_profile,
)

uncorrected = uncorrected_jet_flat_profile()
assert axis_T0(uncorrected) == (Fraction(599, 400), Fraction(0))
corrected, a_star = correct_axis_stress(uncorrected)
assert a_star == Fraction(201, 800)
assert axis_T0(corrected) == (0, 0)
assert axis_T0(locked_jet_flat_profile()) == (0, 0)
s1 = core_linfty_scale(1, uncorrected)
s2 = core_linfty_scale(Fraction(1, 4), uncorrected)
assert s1.prefactor == s2.prefactor
assert s1.exponent == -uncorrected.scales.A
energy = core_energy_scale(1, uncorrected)
assert energy.prefactor == Fraction(17, 3)
assert energy.exponent == Fraction(97, 200)
assert from_rest_ramp().value() == 0
assert honesty_payload()["forced_blowup_reproof_claim"] is False
```
