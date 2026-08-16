/-
Named Weyl-volume prefactor identity (Mathlib-backed).

The unnormalized SU(3) Weyl measure on the maximal torus has
`∫ ρ dθ dφ = 6 (2π)² = 24 π²`. The integer identity here is only
the prefactor `6 * 4 = 24`.

Scope (honest). Finite arithmetic. Not a continuum Haar theorem and
not 4-D SU(3) Yang-Mills. The file contains no `admit`.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- Integer Weyl prefactor `6 * (2)²`. -/
def haarWeylPrefactor : ℚ :=
  6 * 4

theorem haar_weyl_prefactor_24 : haarWeylPrefactor = 24 := by
  unfold haarWeylPrefactor
  norm_num

end OmnibiasAnalytic.Check
