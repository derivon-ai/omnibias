/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C matching-chart section identities:
(1/4)/(1/16)=4, 120*(1/20)=6, and 80*(1/20)=4. These theorems
do not enclose every eps, chart O, C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshot

/-- Matching-chart image of ``E_out``: ``(1/4) / (1/16) = 4``. -/
theorem x_section_identity :
    (1 : ℝ) / 4 * 16 - 4 = 0 := by
  ring

/-- Compact Lohner horizon ``120 * (1/20) = 6``. -/
theorem hit_T_identity :
    (120 : ℝ) * (1 / 20) - 6 = 0 := by
  ring

/-- Short horizon that does not certify: ``80 * (1/20) = 4``. -/
theorem short_T_identity :
    (80 : ℝ) * (1 / 20) - 4 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshot
