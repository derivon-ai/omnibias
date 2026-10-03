/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
L = 0 whole-wall GRAZING E_σ span identities at V = 1/4:
the declared h-interval [19/1000, 1/25] is twenty-one slabs of
width 1/1000, and it contains the aligned restart h = 1/40. These
theorems do not enclose a Lohner first-hit from V = 0, the
L ∈ {9/25, 1/16} walls, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpan

/-- Declared span upper endpoint `40/1000 = 1/25`. -/
theorem span_hi_identity :
    (40 / 1000 : ℝ) - 1 / 25 = 0 := by
  ring

/-- Twenty-one slabs of width `1/1000` fill `[19/1000, 1/25]`. -/
theorem span_width_identity :
    (21 : ℝ) * (1 / 1000) - ((1 / 25) - (19 / 1000)) = 0 := by
  ring

/-- The aligned restart `h = 1/40` lies above `19/1000`. -/
theorem span_align_lo_identity :
    ((1 / 40 : ℝ) - 19 / 1000) - (6 / 1000) = 0 := by
  ring

/-- The aligned restart `h = 1/40` lies below `1/25`. -/
theorem span_align_hi_identity :
    ((1 / 25 : ℝ) - 1 / 40) - (15 / 1000) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpan
