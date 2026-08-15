/-
Root module for the omnibias analytic (Mathlib-backed) track.

The `Check.*` modules are the sound, `sorry`-free certificate checkers: rational
enclosed-sign (`Check.EnclosedSign`), sum-of-squares / positivity
(`Check.Positivity`), and the Newton-Kantorovich / Krawczyk finite obligations
(`Check.Kantorovich`). `Tower` is the Riccati / Eulerian / Hermite derivative
tower (polynomial recurrences plus `iteratedDeriv` link theorems). `Generated`
is the bridge-overwritten obligation under test.

Every module in this project is `sorry`-free. The track discharges *finite,
rational* obligations against Mathlib; it makes no analytic or asymptotic claim.
-/

import OmnibiasAnalytic.Check.EnclosedSign
import OmnibiasAnalytic.Check.Positivity
import OmnibiasAnalytic.Check.Kantorovich
import OmnibiasAnalytic.Tower
import OmnibiasAnalytic.Generated
