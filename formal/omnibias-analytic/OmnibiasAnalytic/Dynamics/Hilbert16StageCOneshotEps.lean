/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C shrinking-ε matching-chart identities:
(1/4)/(1/20)=5, (1/4)/(1/25)=25/4, and 160*(1/20)=8. These
theorems do not enclose every eps, chart O, C2, G1, G4, or
Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshotEps

/-- Matching-chart image of ``E_out`` at ``n = 20``: ``(1/4) / (1/20) = 5``. -/
theorem ceps_n20_sec_identity :
    (1 : ℝ) / 4 * 20 - 5 = 0 := by
  ring

/-- Matching-chart image of ``E_out`` at ``n = 25``: ``(1/4) / (1/25) = 25/4``. -/
theorem ceps_n25_sec_identity :
    (1 : ℝ) / 4 * 25 - 25 / 4 = 0 := by
  ring

/-- Compact Lohner horizon at ``n = 25``: ``160 * (1/20) = 8``. -/
theorem ceps_T25_identity :
    (160 : ℝ) * (1 / 20) - 8 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCOneshotEps
