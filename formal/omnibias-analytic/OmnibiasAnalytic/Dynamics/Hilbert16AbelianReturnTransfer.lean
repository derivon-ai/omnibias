/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact arithmetic for the conditional first-order Abelian-to-return transfer.
The analytic expansion, root cover, remainder bounds, and DRR graphic
membership remain external.  This does not close a DRR case or Hilbert XVI.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16AbelianReturnTransfer

/-- Exact value-margin threshold. -/
theorem valueThresholdIdentity
    (margin bound : ℝ)
    (hbound : bound ≠ 0) :
    bound * (margin / bound) - margin = 0 := by
  field_simp
  ring

/-- Exact derivative-margin threshold. -/
theorem derivativeThresholdIdentity
    (margin bound : ℝ)
    (hbound : bound ≠ 0) :
    bound * (margin / bound) - margin = 0 := by
  field_simp
  ring

/-- Algebraic normalization of a first-order displacement expansion. -/
theorem normalizedDisplacementIdentity
    (epsilon abelian remainder : ℝ) :
    (epsilon * abelian + epsilon ^ 2 * remainder)
      - epsilon * abelian - epsilon ^ 2 * remainder = 0 := by
  ring

/-- A strict epsilon budget preserves a positive value margin. -/
theorem valueMarginSurvives
    (epsilon margin bound : ℝ)
    (hbudget : epsilon * bound < margin) :
    0 < margin - epsilon * bound := by
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16AbelianReturnTransfer
