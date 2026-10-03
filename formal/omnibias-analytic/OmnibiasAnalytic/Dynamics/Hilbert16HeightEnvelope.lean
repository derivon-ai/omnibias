/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Alpha-0 T-h envelope identities after the shrinking-root x-corridor.
These theorems do not bound height-section first-hit, the actual-field
T-h = O(ε) bootstrap, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16HeightEnvelope

/-- AM-GM identity behind `|q| ≤ C ε (ε² + T)`. -/
theorem amgm_qbound_identity (ε w : ℝ) :
    ε * (ε ^ 2 + w ^ 2) - 2 * ε ^ 2 * w - ε * (ε - w) ^ 2 = 0 := by
  ring

/-- Exit envelope `T (1 + ε y0) = (ε² + h) x² / 2` after clearing `/2`. -/
theorem envelope_exit_identity (ε x y0 : ℝ) :
    (ε ^ 2 * x ^ 2) * (1 + ε * y0) - (ε ^ 2 + ε ^ 3 * y0) * x ^ 2 = 0 := by
  ring

/-- `C = 0` comparison conserves `T - h`. -/
theorem th_alpha0_identity (Te he h : ℝ) :
    (Te + h - he - h) - (Te - he) = 0 := by
  ring

/-- Leading `|q| / (ε (ε² + T))` identity after clearing `/2`. -/
theorem q_envelope_identity (ε x r1 r2 : ℝ) :
    (ε ^ 3 * (x - r1) * (r2 - x)) * (2 + x ^ 2)
      - ε * (2 * ε ^ 2 + ε ^ 2 * x ^ 2) * (x - r1) * (r2 - x) = 0 := by
  ring

/-- `T_e > h_e` once `2 ε y0 < x²`. -/
theorem te_above_he {ε x y0 : ℝ} (hε : 0 < ε) (hgap : 2 * ε * y0 < x ^ 2) :
    0 < ε ^ 2 * x ^ 2 - 2 * ε ^ 3 * y0 := by
  nlinarith

end OmnibiasAnalytic.Dynamics.Hilbert16HeightEnvelope
