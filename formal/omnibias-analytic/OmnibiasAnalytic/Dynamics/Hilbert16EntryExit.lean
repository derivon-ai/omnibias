/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact slow-line partial fractions and the tracked sep² × outgoing-factor
product for the coalescing quadratic passage.

These theorems do not prove a physical C2 remainder, orbit continuation,
G1, G4, or Hilbert XVI.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16EntryExit

/-- Numerator form of the two-root partial fraction. -/
theorem partial_fraction_numerators
    {x r1 r2 : ℝ} (hne : r1 ≠ r2) :
    r1 / (r1 - r2) * (x - r2) + r2 / (r2 - r1) * (x - r1) = x := by
  have h12 : r1 - r2 ≠ 0 := sub_ne_zero.mpr hne
  have h21 : r2 - r1 ≠ 0 := sub_ne_zero.mpr hne.symm
  field_simp [h12, h21]
  ring

/-- The slow-line integrand identity away from the two roots. -/
theorem partial_fraction_two_roots
    {x r1 r2 : ℝ} (hne : r1 ≠ r2) (hx1 : x ≠ r1) (hx2 : x ≠ r2) :
    x / ((x - r1) * (x - r2))
      = r1 / ((r1 - r2) * (x - r1)) + r2 / ((r2 - r1) * (x - r2)) := by
  have h12 : r1 - r2 ≠ 0 := sub_ne_zero.mpr hne
  have h21 : r2 - r1 ≠ 0 := sub_ne_zero.mpr hne.symm
  have hx1' : x - r1 ≠ 0 := sub_ne_zero.mpr hx1
  have hx2' : x - r2 ≠ 0 := sub_ne_zero.mpr hx2
  field_simp [h12, h21, hx1', hx2']
  ring

/-- Double-root (saddle-node) integrand identity. -/
theorem double_root_partial_fraction
    {x r : ℝ} (hx : x ≠ r) :
    x / (x - r) ^ 2 = 1 / (x - r) + r / (x - r) ^ 2 := by
  have : x - r ≠ 0 := sub_ne_zero.mpr hx
  field_simp
  ring

/-- Height-dominated ``dx/dy`` is independent of height. -/
theorem dx_dy_y_dominated
    {ε k x y : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) :
    ε * (k * y) / (x * y) = ε * k / x := by
  field_simp [hx, hy]

/-- On the kill sequence, ``log sep = -1/ε²``. -/
theorem kill_sep_log (ε : ℝ) :
    Real.log (Real.exp (-(1 / ε ^ 2))) = -(1 / ε ^ 2) :=
  Real.log_exp _

/-- Sep-power logarithm of the tracked product on the kill sequence. -/
theorem tracked_log_on_kill (ε C : ℝ) :
    (2 - 2 * C * ε) * Real.log (Real.exp (-(1 / ε ^ 2)))
      = (2 - 2 * C * ε) * (-(1 / ε ^ 2)) := by
  rw [kill_sep_log]

/-- For small ``ε`` the tracked sep-power logarithm is strictly negative. -/
theorem tracked_log_neg_on_kill
    {ε C : ℝ} (hε : 0 < ε) (hsmall : C * ε < 1) :
    (2 - 2 * C * ε) * Real.log (Real.exp (-(1 / ε ^ 2))) < 0 := by
  have hpos : 0 < 2 - 2 * C * ε := by linarith
  have hneg : Real.log (Real.exp (-(1 / ε ^ 2))) < 0 := by
    rw [kill_sep_log]
    have hsq : 0 < ε ^ 2 := sq_pos_of_pos hε
    have : 0 < 1 / ε ^ 2 := one_div_pos.mpr hsq
    linarith
  exact mul_neg_of_pos_of_neg hpos hneg

/-- Logarithm of both sides of the tracked outgoing product, in expanded form. -/
theorem tracked_product_log_expanded (sep α log_h log_core : ℝ) :
    2 * sep + α * (log_h - log_core - 2 * sep)
      = (2 - 2 * α) * sep + α * (log_h - log_core) := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16EntryExit
