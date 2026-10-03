/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Post-corridor matching identities: V = -ε x restores a spatial margin
independent of the failing first-root wall a = r₁ - d, and the leading
slow-line cubic factor is (x-r₁)(x-r₂). These theorems do not bound
height-section first-hit, sealed T-h continuation, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16PostCorridor

/-- Slow-line embedding `V = -ε x` at a compact matching section. -/
theorem V_embed_identity (ε x : ℝ) :
    (-ε * x) + ε * x = 0 := by
  ring

/-- Kinetic identity `V² = ε² x²` on that embedding. -/
theorem T_kinetic_identity (ε x : ℝ) :
    (-ε * x) ^ 2 - ε ^ 2 * x ^ 2 = 0 := by
  ring

/-- Leading `q / ε³` on the two-root slow line. -/
theorem q_leading_identity (r1 r2 x : ℝ) :
    (r1 * r2) + (-(r1 + r2)) * x + x ^ 2 - (x - r1) * (x - r2) = 0 := by
  ring

/-- `|V| = ε x_*` is strictly positive for `ε > 0` and `x_* > 0`. -/
theorem restored_V_margin {ε x : ℝ} (hε : 0 < ε) (hx : 0 < x) :
    0 < ε * x := mul_pos hε hx

/-- The saddle wall `r₁ - d` can fail while `x_* - r₁` stays positive. -/
theorem wall_fails_star_holds {r1 d xstar : ℝ}
    (hwall : r1 < d) (hstar : r1 < xstar) :
    r1 - d < 0 ∧ 0 < xstar - r1 := by
  constructor <;> linarith

/-- Leading `T_h` numerator `y0 + (x-r₁)(x-r₂)` is positive once
`-(x-r₁)(x-r₂) < y0`. -/
theorem th_leading_pos {y0 r1 r2 x : ℝ}
    (h : -(x - r1) * (x - r2) < y0) :
    0 < y0 + (x - r1) * (x - r2) := by
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16PostCorridor
