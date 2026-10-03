/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Orbit-aligned GRAZING E_σ identities at the certified V = 1/4 wall:
restart V = 1/4, h = 1/40, E_σ at ν = 0 equals -121/160, and the
sample E_σ value is -619519/819200. These theorems do not enclose a
Lohner first-hit from V = 0, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaWall

/-- Orbit-aligned restart `V = 1/4`. -/
theorem align_v_identity :
    (1 / 4 : ℝ) - 1 / 4 = 0 := by
  ring

/-- Orbit-aligned restart `h = 1/40`. -/
theorem align_h_identity :
    (1 / 40 : ℝ) - 1 / 40 = 0 := by
  ring

/-- `E_σ` at `ν = 0` on the aligned restart is `-121/160`. -/
theorem e_align_nu0_identity :
    ((1 / 4 : ℝ) - 1 + (-1) * (1 / 4) * (1 / 40)) - (-121 / 160) = 0 := by
  ring

/-- Sample `E_σ` on the aligned restart at `ν = 1/16`, `C = 2`. -/
theorem e_align_start_identity :
    ((1 / 4 : ℝ) - 1 + (-1) * (1 / 4) * (1 / 40)
        + (1 / 16) * (1 / 4) * (1 / 4) * (1 / 40) * (1 / 40)
        + (2 : ℝ) * (1 / 16) * (1 / 16) * (-1) * (1 / 4) * (1 / 40) * (1 / 40))
      - (-619519 / 819200) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaWall
