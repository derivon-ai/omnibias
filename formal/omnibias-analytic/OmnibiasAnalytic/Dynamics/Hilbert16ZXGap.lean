/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Unfrozen-Z first-log-derivative identities including Z_x versus the
lifted fold map. These theorems do not bound Z_x, sep > 0, physical
C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ZXGap

/-- Unfrozen first-log-derivative gap equals `ε² (2 x³ Z + x⁴ Z_x)`. -/
theorem unfrozen_log_d1_gap_identity (x r μ ε Z Zx : ℝ) :
    (x * (2 * (x - r) + ε ^ 2 * (3 * x ^ 2 * Z + x ^ 3 * Zx))
        - ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z))
      - (x * (2 * (x - r)) - ((x - r) ^ 2 + μ))
      - ε ^ 2 * (2 * x ^ 3 * Z + x ^ 4 * Zx) = 0 := by
  ring

/-- `Z_x = 0` recovers the frozen-Z first-log-derivative gap `2 ε² x³ Z`. -/
theorem unfrozen_recovers_frozen_identity (x r μ ε Z : ℝ) :
    (x * (2 * (x - r) + ε ^ 2 * (3 * x ^ 2 * Z + x ^ 3 * (0 : ℝ)))
        - ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z))
      - (x * (2 * (x - r)) - ((x - r) ^ 2 + μ))
      - 2 * ε ^ 2 * x ^ 3 * Z = 0 := by
  ring

/-- Unfrozen minus frozen gap equals `ε² x⁴ Z_x`. -/
theorem zx_extra_identity (x r μ ε Z Zx : ℝ) :
    ((x * (2 * (x - r) + ε ^ 2 * (3 * x ^ 2 * Z + x ^ 3 * Zx))
        - ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z))
      - (x * (2 * (x - r)) - ((x - r) ^ 2 + μ)))
      -
    ((x * (2 * (x - r) + 3 * ε ^ 2 * x ^ 2 * Z)
        - ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z))
      - (x * (2 * (x - r)) - ((x - r) ^ 2 + μ)))
      - ε ^ 2 * x ^ 4 * Zx = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ZXGap
