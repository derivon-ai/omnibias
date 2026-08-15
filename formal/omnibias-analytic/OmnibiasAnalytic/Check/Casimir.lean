/-
Named quadratic-Casimir identities (Mathlib-backed).

Locked Freudenthal evaluations matching
`omnibias.geometry.gauge.quadratic_casimir` on explicit partitions:

* SU(2) trivial `(0, 0)` → `0`
* SU(2) fundamental `(1, 0)` → `3/4`
* SU(2) adjoint `(2, 0)` → `2`
* SU(3) fundamental `(1, 0, 0)` → `4/3`

`C2 = 1/2 [ Σ_i λ_i (λ_i + n + 1 - 2(i+1)) - |λ|² / n ]` with `i` 0-based.

Scope (honest). Finite rational identities on named partitions. Not a
continuum gauge claim. The file contains no `admit`.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- SU(2) Freudenthal Casimir on a length-2 partition. -/
def casimirSU2 (lam0 lam1 : ℚ) : ℚ :=
  (1 : ℚ) / 2 *
    (lam0 * (lam0 + 2 + 1 - 2) +
      lam1 * (lam1 + 2 + 1 - 4) -
      (lam0 + lam1) * (lam0 + lam1) / 2)

/-- SU(3) Freudenthal Casimir on a length-3 partition. -/
def casimirSU3 (lam0 lam1 lam2 : ℚ) : ℚ :=
  (1 : ℚ) / 2 *
    (lam0 * (lam0 + 3 + 1 - 2) +
      lam1 * (lam1 + 3 + 1 - 4) +
      lam2 * (lam2 + 3 + 1 - 6) -
      (lam0 + lam1 + lam2) * (lam0 + lam1 + lam2) / 3)

theorem su2_casimir_trivial : casimirSU2 0 0 = 0 := by
  unfold casimirSU2
  norm_num

theorem su2_casimir_fund : casimirSU2 1 0 = 3 / 4 := by
  unfold casimirSU2
  norm_num

theorem su2_casimir_adjoint : casimirSU2 2 0 = 2 := by
  unfold casimirSU2
  norm_num

/-- Heat-kernel gap coefficient of SU(2): `C2(1) - C2(0) = 3/4`.
Finite rational identity, not a continuum gauge claim. -/
theorem su2_casimir_fund_gap : casimirSU2 1 0 - casimirSU2 0 0 = 3 / 4 := by
  unfold casimirSU2
  norm_num

/-- SU(3) fundamental Casimir `C2(1,0) = 4/3`.
Finite rational identity, not a continuum gauge claim. -/
theorem su3_casimir_fund : casimirSU3 1 0 0 = 4 / 3 := by
  unfold casimirSU3
  norm_num

end OmnibiasAnalytic.Check
