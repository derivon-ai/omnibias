/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line parametric-sep matching-chart identities: 3/2+1/2=2,
8*(1/16)=1/2, and 16*(1/2)=8. These theorems do not enclose every
r1, complete first-hit on chart O, C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginSpan

/-- Declared cover ``3/2 + 1/2 = 2``. -/
theorem ospan_join_identity :
    (3 : ℝ) / 2 + 1 / 2 - 2 = 0 := by
  ring

/-- Eight equal slabs ``8 * (1/16) = 1/2``. -/
theorem ospan_slabs_identity :
    (8 : ℝ) * (1 / 16) - 1 / 2 = 0 := by
  ring

/-- Cover width in slab units ``16 * (1/2) = 8``. -/
theorem ospan_n16_identity :
    (16 : ℝ) * (1 / 2) - 8 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCOriginSpan
