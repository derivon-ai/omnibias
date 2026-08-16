/-
Named polymer-coordination identities (Mathlib-backed).

Locked integer evaluations matching
`omnibias.geometry.gauge.transfer.strong_coupling`:

* backtrack `3 * (2*4 - 3) = 15`
* first-step `4 * (2*4 - 3) = 20`
* crude `8 * (4 - 1) = 24`
* `15 < 20` and `15 < 24`

Scope (honest). Finite arithmetic on a named dimension. Not a
continuum gauge claim and not Osterwalder-Seiler. The file contains
no `admit`.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- Backtrack-excluding tree majorant `C = 3(2d - 3)` at `d = 4`. -/
def polymerBacktrack (d : ℚ) : ℚ :=
  3 * (2 * d - 3)

/-- First-attachment majorant `A = 4(2d - 3)` at integer `d`. -/
def polymerFirstStep (d : ℚ) : ℚ :=
  4 * (2 * d - 3)

/-- Crude overcount `C = 8(d - 1)` at integer `d`. -/
def polymerCrude (d : ℚ) : ℚ :=
  8 * (d - 1)

theorem polymer_backtrack_coord_4 : polymerBacktrack 4 = 15 := by
  unfold polymerBacktrack
  norm_num

theorem polymer_first_step_4 : polymerFirstStep 4 = 20 := by
  unfold polymerFirstStep
  norm_num

theorem polymer_crude_coord_4 : polymerCrude 4 = 24 := by
  unfold polymerCrude
  norm_num

/-- Subsequent branching is strictly smaller than the first-step count.
Finite rational comparison, not a continuum gauge claim. -/
theorem polymer_backtrack_lt_first_step : polymerBacktrack 4 < polymerFirstStep 4 := by
  unfold polymerBacktrack polymerFirstStep
  norm_num

/-- The backtrack majorant is strictly smaller than the crude overcount.
Finite rational comparison, not a continuum gauge claim. -/
theorem polymer_backtrack_lt_crude : polymerBacktrack 4 < polymerCrude 4 := by
  unfold polymerBacktrack polymerCrude
  norm_num

end OmnibiasAnalytic.Check
