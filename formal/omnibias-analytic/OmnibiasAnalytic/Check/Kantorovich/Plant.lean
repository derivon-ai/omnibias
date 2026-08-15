/-
Planted 1-D Newton-Kantorovich / Krawczyk instance: `x² - 2`.

The unique positive square root of 2 lies in the compact interval
`[5/4, 7/4]`. Both routes (radii Lipschitz bound `κ = 1/3`, Krawczyk
derivative bound `κ = 1/6`) conclude `∃! x` in that box with `x² = 2`.

Scope (honest). This is a unique real root of a named quadratic on a
compact interval. It is not a continuum PDE claim.
-/

import OmnibiasAnalytic.Check.Kantorovich

namespace OmnibiasAnalytic.Check

open scoped NNReal
open Set

noncomputable section

/-- Planted map `f(x) = x² - 2`. -/
def quadraticPlant (x : ℝ) : ℝ := x * x - 2

def plantCenter : ℝ := 3 / 2
def plantRadius : ℝ := 1 / 4
def plantA : ℝ := 1 / 3
def plantY0 : ℝ := 1 / 12

theorem plantA_ne_zero : plantA ≠ 0 := by
  unfold plantA; norm_num

theorem plantRadius_nonneg : 0 ≤ plantRadius := by
  unfold plantRadius; norm_num

theorem plant_box :
    Icc (plantCenter - plantRadius) (plantCenter + plantRadius) =
      Icc (5 / 4 : ℝ) (7 / 4) := by
  unfold plantCenter plantRadius
  congr 1 <;> norm_num

theorem hasDerivAt_quadraticPlant (x : ℝ) :
    HasDerivAt quadraticPlant (2 * x) x := by
  change HasDerivAt (fun y => y * y - 2) (2 * x) x
  refine ((hasDerivAt_id' x).mul (hasDerivAt_id' x) |>.sub (hasDerivAt_const x (2 : ℝ))).congr_deriv ?_
  ring

theorem differentiable_quadraticPlant : Differentiable ℝ quadraticPlant :=
  fun x => (hasDerivAt_quadraticPlant x).differentiableAt

theorem deriv_quadraticPlant (x : ℝ) : deriv quadraticPlant x = 2 * x :=
  (hasDerivAt_quadraticPlant x).deriv

theorem newtonOp_plant_center :
    newtonOp plantA quadraticPlant plantCenter - plantCenter = -plantY0 := by
  unfold newtonOp quadraticPlant plantA plantCenter plantY0
  ring

theorem abs_newtonOp_plant_center :
    |newtonOp plantA quadraticPlant plantCenter - plantCenter| ≤ plantY0 := by
  rw [newtonOp_plant_center, abs_neg]
  have hpos : 0 ≤ plantY0 := by unfold plantY0; norm_num
  simp [abs_of_nonneg hpos]

theorem plant_deriv_bound {x : ℝ} (hx : x ∈ Icc (5 / 4 : ℝ) (7 / 4)) :
    |1 - plantA * deriv quadraticPlant x| ≤ (1 / 6 : ℝ) := by
  rw [deriv_quadraticPlant]
  unfold plantA
  have : |1 - (2 * x) / 3| ≤ 1 / 6 := by
    rw [abs_le]
    constructor <;> nlinarith [hx.1, hx.2]
  convert this using 2
  ring

theorem plant_lipschitz_sixth :
    LipschitzOnWith (1 / 6 : ℝ≥0)
      (newtonOp plantA quadraticPlant) (Icc (5 / 4 : ℝ) (7 / 4)) := by
  have h := lipschitzOnWith_newtonOp_of_deriv_le (c := plantCenter) (r := plantRadius)
    (A := plantA) (K := (1 / 6 : ℝ≥0)) differentiable_quadraticPlant ?_
  · rwa [plant_box] at h
  · intro x hx
    rw [plant_box] at hx
    exact plant_deriv_bound hx

theorem one_sixth_le_one_third : (1 / 6 : ℝ≥0) ≤ (1 / 3 : ℝ≥0) := by
  rw [← NNReal.coe_le_coe]
  norm_num

theorem plant_lipschitz_third :
    LipschitzOnWith (1 / 3 : ℝ≥0)
      (newtonOp plantA quadraticPlant) (Icc (5 / 4 : ℝ) (7 / 4)) :=
  plant_lipschitz_sixth.weaken one_sixth_le_one_third

theorem plantK_third_lt_one : (1 / 3 : ℝ≥0) < 1 := by
  rw [← NNReal.coe_lt_coe]
  norm_num

theorem plantK_sixth_lt_one : (1 / 6 : ℝ≥0) < 1 := by
  rw [← NNReal.coe_lt_coe]
  norm_num

theorem plant_radii_bound : plantY0 + ((1 / 3 : ℝ≥0) : ℝ) * plantRadius ≤ plantRadius := by
  unfold plantY0 plantRadius
  norm_num

theorem plant_krawczyk_bound : plantY0 + ((1 / 6 : ℝ≥0) : ℝ) * plantRadius ≤ plantRadius := by
  unfold plantY0 plantRadius
  norm_num

/-- Radii route: center displacement `Y0 = 1/12` and Lipschitz `κ = 1/3`
give a unique root of `x² - 2` in `[5/4, 7/4]`. -/
theorem quadratic_plant_radii_unique_zero :
    ∃! x : ℝ, x ∈ Icc (5 / 4) (7 / 4) ∧ quadraticPlant x = 0 := by
  have h := unique_zero_of_center_lip (f := quadraticPlant) (A := plantA)
    (c := plantCenter) (r := plantRadius) (Y0 := plantY0) (K := (1 / 3 : ℝ≥0))
    plantA_ne_zero plantRadius_nonneg abs_newtonOp_plant_center
    (by
      have hL := plant_lipschitz_third
      rwa [← plant_box] at hL) plantK_third_lt_one plant_radii_bound
  rwa [plant_box] at h

/-- Krawczyk route: `|1 - A f'| ≤ 1/6` on the box gives the same unique root. -/
theorem quadratic_plant_krawczyk_unique_zero :
    ∃! x : ℝ, x ∈ Icc (5 / 4) (7 / 4) ∧ quadraticPlant x = 0 := by
  have h := unique_zero_of_krawczyk_1d (f := quadraticPlant) (A := plantA)
    (c := plantCenter) (r := plantRadius) (Y0 := plantY0) (K := (1 / 6 : ℝ≥0))
    plantA_ne_zero plantRadius_nonneg differentiable_quadraticPlant
    abs_newtonOp_plant_center ?_ plantK_sixth_lt_one plant_krawczyk_bound
  · rwa [plant_box] at h
  · intro x hx
    rw [plant_box] at hx
    exact plant_deriv_bound hx

end

end OmnibiasAnalytic.Check
