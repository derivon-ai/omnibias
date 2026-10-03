/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C comparison first-hit identities: 1/(1/64)=64,
64*3=192, and 1-1/4096=4095/4096. These theorems do not enclose
Lohner, the physical signed-label section, chart O, C2, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCHit

/-- Reciprocal of the ḣ/h floor ``64 * (1/64) = 1``. -/
theorem inv_floor_identity :
    (64 : ℝ) * (1 / 64) - 1 = 0 := by
  ring

/-- Time prefactor ``64 * 3 = 192``. -/
theorem time_pre_identity :
    (64 : ℝ) * 3 - 192 = 0 := by
  ring

/-- Height gap ``1 - 1/4096 = 4095/4096``. -/
theorem hmax_gap_identity :
    (1 : ℝ) - 1 / 4096 - 4095 / 4096 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCHit
