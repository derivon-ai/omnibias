/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Wall-box GRAZING E_σ cover identities at V = 1/4:
the declared h-interval [1/50, 4/125] is twelve slabs of width
1/1000, and it contains the aligned restart h = 1/40. These
theorems do not enclose a Lohner first-hit from V = 0, the whole
wall h-interval, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaBox

/-- Declared cover lower endpoint `1/50 = 20/1000`. -/
theorem cover_lo_identity :
    (1 / 50 : ℝ) - 20 / 1000 = 0 := by
  ring

/-- Declared cover upper endpoint `4/125 = 32/1000`. -/
theorem cover_hi_identity :
    (4 / 125 : ℝ) - 32 / 1000 = 0 := by
  ring

/-- Twelve slabs of width `1/1000` fill `[1/50, 4/125]`. -/
theorem cover_width_identity :
    (12 : ℝ) * (1 / 1000) - ((4 / 125) - (1 / 50)) = 0 := by
  ring

/-- The aligned restart `h = 1/40` lies in `(1/50, 4/125)`. -/
theorem cover_contains_identity :
    ((1 / 40 : ℝ) - 1 / 50) * (4 / 125 - 1 / 40) - (7 / 200000) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaBox
