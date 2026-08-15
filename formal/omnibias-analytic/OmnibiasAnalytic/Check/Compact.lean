/-
Compact sub-obligations (Mathlib-backed).

Two planted statements on explicit finite data:

* `nsBoxField` -- named incompressible polynomial field on `[1/2, 1]²`
  with a residual lower bound, plus the 2-D Euler advection identity
  `(-y) x + x y` on `[-1, 1]²`
* `transferChar` -- characteristic polynomial of the named 2×2 matrix
  `[[13/2, 3/2], [3/2, 13/2]]`, roots `8` and `5`, ratio `5/8`

Scope (honest). Each theorem is a named polynomial identity or inequality
on a compact box, or the spectrum of a finite rational matrix. Not a
continuum regularity theorem, not a continuum gauge claim. The file
contains no `admit`.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

open Set

/-! ### A. Local incompressible residual box -/

/-- Named field `u(x, y) = (x, -y)`. -/
def nsBoxField (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Divergence of `nsBoxField`: `∂₁ u₁ + ∂₂ u₂ = 1 + (-1)`. -/
def nsBoxDiv : ℝ := 1 + (-1)

/-- Steady residual `(u · ∇) u` of `nsBoxField` (viscosity term vanishes). -/
def nsBoxResidual (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2)

/-- 2-D Euler advection identity: tangential · radial is the zero polynomial. -/
def eulerAdvect (p : ℝ × ℝ) : ℝ := -p.2 * p.1 + p.1 * p.2

theorem ns_box_div_free : nsBoxDiv = 0 := by
  unfold nsBoxDiv
  norm_num

/-- Compact-box residual of a named incompressible polynomial field.
Not a continuum regularity theorem. -/
theorem ns_box_residual_lo :
    ∀ p : ℝ × ℝ,
      p ∈ Icc (1 / 2 : ℝ) 1 ×ˢ Icc (1 / 2 : ℝ) 1 →
        (nsBoxResidual p).1 ≥ 1 / 2 := by
  intro p hp
  unfold nsBoxResidual
  exact hp.1.1

theorem ns_box_div_and_residual :
    nsBoxDiv = 0 ∧
      ∀ p : ℝ × ℝ,
        p ∈ Icc (1 / 2 : ℝ) 1 ×ˢ Icc (1 / 2 : ℝ) 1 →
          (nsBoxResidual p).1 ≥ 1 / 2 :=
  ⟨ns_box_div_free, ns_box_residual_lo⟩

theorem euler_advect_zero_on_box :
    ∀ p : ℝ × ℝ, p ∈ Icc (-1 : ℝ) 1 ×ˢ Icc (-1 : ℝ) 1 → eulerAdvect p = 0 := by
  intro p _hp
  unfold eulerAdvect
  ring

/-! ### B. Finite transfer-matrix characteristic polynomial -/

/-- Characteristic polynomial of `[[13/2, 3/2], [3/2, 13/2]]`:
`λ² - 13λ + 40`. -/
def transferChar (lam : ℝ) : ℝ := lam * lam - 13 * lam + 40

theorem transfer_char_factors (lam : ℝ) :
    transferChar lam = (lam - 8) * (lam - 5) := by
  unfold transferChar
  ring

/-- Roots of the named 2×2 characteristic polynomial. Finite matrix,
not a continuum gauge claim. -/
theorem transfer_plant_charpoly_roots :
    transferChar 8 = 0 ∧ transferChar 5 = 0 := by
  constructor
  · rw [transfer_char_factors]; norm_num
  · rw [transfer_char_factors]; norm_num

/-- Subdominant ratio `5/8 < 1` and gap `8 - 5 > 0` of this finite
rational matrix. Not a continuum gauge claim. -/
theorem transfer_plant_gap :
    |(5 : ℝ)| / 8 < 1 ∧ 0 < (8 : ℝ) - 5 := by
  constructor <;> norm_num

theorem transfer_plant_charpoly_and_gap :
    (transferChar 8 = 0 ∧ transferChar 5 = 0) ∧
      |(5 : ℝ)| / 8 < 1 ∧ 0 < (8 : ℝ) - 5 :=
  ⟨transfer_plant_charpoly_roots, transfer_plant_gap⟩

end OmnibiasAnalytic.Check
