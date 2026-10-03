/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact two-root slow-line I-map and the r1 → 0 remainder.

These theorems do not prove outgoing first-hit, a uniform physical wall,
G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ShrinkingRoot

/-- Two-root I-map: `dx/dκ = (x-r1)(x-r2)/x`. -/
theorem two_root_dx_identity
    {x r1 r2 : ℝ} (hx : x ≠ 0) :
    ((x - r1) * (x - r2) / x) * x = (x - r1) * (x - r2) := by
  field_simp [hx]

/-- Exact `r1 → 0` remainder: `(x-r1)(x-r2)/x - (x-r2) = -r1 (x-r2)/x`. -/
theorem r1_limit_remainder
    {x r1 r2 : ℝ} (hx : x ≠ 0) :
    (x - r1) * (x - r2) - x * (x - r2) + r1 * (x - r2) = 0 := by
  ring

/-- Partial-fraction coefficient `C = r2/(r2-r1)` tends to 1 as `r1 → 0`. -/
theorem c_limit_identity
    {r1 r2 : ℝ} (hne : r1 ≠ r2) :
    r2 / (r2 - r1) - 1 = r1 / (r2 - r1) := by
  have h : r2 - r1 ≠ 0 := sub_ne_zero.mpr hne.symm
  field_simp [h]
  ring

/-- Radial rescaling `x = r1 ξ` recovers `B_- = r1² (ξ-1)(ξ - r2/r1)`. -/
theorem rescaled_b
    {xi r1 r2 : ℝ} (hr1 : r1 ≠ 0) :
    (r1 * xi - r1) * (r1 * xi - r2)
      = r1 ^ 2 * (xi - 1) * (xi - r2 / r1) := by
  field_simp [hr1]
  ring

/-- Limiting rescaled slow ODE: `ξ' → r2 (1 - ξ)` as `r1 → 0`. -/
theorem xi_limit_ode
    {xi r1 r2 : ℝ} (hr1 : r1 ≠ 0) :
    r1 * (xi - 1) * (xi - r2 / r1) - r2 * (1 - xi)
      = r1 * xi * (xi - 1) := by
  field_simp [hr1]
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ShrinkingRoot
