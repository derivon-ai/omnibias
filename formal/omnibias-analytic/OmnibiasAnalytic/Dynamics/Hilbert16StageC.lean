/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C a_min identities: written rstar/2 = 1/2, integrating
factor 1/(1/4) = 4, and declared floor (1/2)/2 = 1/4. These theorems
do not enclose an outgoing orbit, first-hit, C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageC

/-- Written Stage-C floor ``1/2 = 1/2``. -/
theorem amin_written_identity :
    (1 : ℝ) / 2 - 1 / 2 = 0 := by
  ring

/-- Integrating factor ``1 / (1/4) = 4``. -/
theorem inv_guess_identity :
    (1 : ℝ) / (1 / 4) - 4 = 0 := by
  ring

/-- Declared floor ``(1/2)/2 = 1/4``. -/
theorem amin_floor_identity :
    (1 / 2 : ℝ) / 2 - 1 / 4 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageC
