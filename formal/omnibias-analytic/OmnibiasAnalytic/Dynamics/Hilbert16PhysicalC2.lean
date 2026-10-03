/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Frozen-Z remainder identities for log D' versus the lifted fold map.
These theorems do not bound Z_x, sep > 0, physical C2, G1, G4, or
Hilbert XVI.
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16PhysicalC2

/-- Linear-in-ε gap of the lift `μ = β ε r³` versus `β ε x³`. -/
theorem lift_cubic_mismatch_identity (x r β ε : ℝ) :
    ((x - r) ^ 2 + β * ε * x ^ 3)
      - ((x - r) ^ 2 + β * ε * r ^ 3)
      - β * ε * (x ^ 3 - r ^ 3) = 0 := by
  ring

/-- Frozen-Z first log-derivative gap equals `2 ε² x³ Z`. -/
theorem frozen_log_d1_gap_identity (x r μ ε Z : ℝ) :
    (x * (2 * (x - r) + 3 * ε ^ 2 * x ^ 2 * Z)
        - ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z))
      - (x * (2 * (x - r)) - ((x - r) ^ 2 + μ))
      - 2 * ε ^ 2 * x ^ 3 * Z = 0 := by
  ring

/-- Frozen C2 remainder `2 ε² Z B / x` clears after multiplying by `x`. -/
theorem frozen_c2_gap_identity
    {x : ℝ} (hx : x ≠ 0) (r μ ε Z : ℝ) :
    (2 * ε ^ 2 * Z * ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z) / x) * x
      = 2 * ε ^ 2 * Z * ((x - r) ^ 2 + μ + ε ^ 2 * x ^ 3 * Z) := by
  field_simp [hx]

end OmnibiasAnalytic.Dynamics.Hilbert16PhysicalC2
