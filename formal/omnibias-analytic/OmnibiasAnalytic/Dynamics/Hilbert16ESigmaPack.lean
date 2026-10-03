/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
L ∈ {9/25, 1/16} wall-span GRAZING E_σ identities at V = 1/4:
the declared h-interval [17/1000, 7/200] is eighteen slabs of
width 1/1000, and it contains the aligned restart h = 1/40. These
theorems do not enclose a Lohner first-hit from V = 0, G1, G4, or
Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaPack

/-- Declared pack span lower endpoint `17/1000 = 34/2000`. -/
theorem pack_lo_identity :
    (17 / 1000 : ℝ) - 34 / 2000 = 0 := by
  ring

/-- Declared pack span upper endpoint `35/1000 = 7/200`. -/
theorem pack_hi_identity :
    (35 / 1000 : ℝ) - 7 / 200 = 0 := by
  ring

/-- Eighteen slabs of width `1/1000` fill `[17/1000, 7/200]`. -/
theorem pack_width_identity :
    (18 : ℝ) * (1 / 1000) - ((7 / 200) - (17 / 1000)) = 0 := by
  ring

/-- The aligned restart `h = 1/40` lies above `17/1000`. -/
theorem pack_align_identity :
    ((1 / 40 : ℝ) - 17 / 1000) - (8 / 1000) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaPack
