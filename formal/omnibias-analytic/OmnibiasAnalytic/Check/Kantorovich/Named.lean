/-
Named Newton-Kantorovich unique-zero instances (Mathlib-backed).

Three planted maps, each a unique root of a named polynomial on an
explicit compact box:

* `circleLine` -- unit circle ∩ line `y = x` in `[5/8, 7/8]²`
* `hopfRadial` -- `r (1 - r²)` on `[3/4, 5/4]`
* `ccfChebyshev` -- Chebyshev `T₃ = 4z³ - 3z` on `[3/4, 1]`

Scope (honest). Each theorem is `∃!` of that named polynomial on that
box. Not a continuum PDE, not a Lohner time-`2π` return map, not a
continuum CCF / Euler blow-up. The file contains no `admit`.
-/

import OmnibiasAnalytic.Check.Kantorovich

namespace OmnibiasAnalytic.Check

open scoped NNReal
open Set

noncomputable section

/-! ### A. Circle ∩ line (structured 2-D, reduced to 1-D NK) -/

/-- Unit circle minus the line `y = x`: `F(x, y) = (x² + y² - 1, x - y)`. -/
def circleLine (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.1 + p.2 * p.2 - 1, p.1 - p.2)

/-- 1-D reduction along `y = x`: `2x² - 1`. -/
def circleLine1d (x : ℝ) : ℝ := 2 * (x * x) - 1

def circleCenter : ℝ := 3 / 4
def circleRadius : ℝ := 1 / 8
def circleA : ℝ := 1 / 3
def circleY0 : ℝ := 1 / 24

theorem circleA_ne_zero : circleA ≠ 0 := by
  unfold circleA; norm_num

theorem circleRadius_nonneg : 0 ≤ circleRadius := by
  unfold circleRadius; norm_num

theorem circle_box :
    Icc (circleCenter - circleRadius) (circleCenter + circleRadius) =
      Icc (5 / 8 : ℝ) (7 / 8) := by
  unfold circleCenter circleRadius
  congr 1 <;> norm_num

theorem hasDerivAt_circleLine1d (x : ℝ) :
    HasDerivAt circleLine1d (4 * x) x := by
  change HasDerivAt (fun y => 2 * (y * y) - 1) (4 * x) x
  refine ((((hasDerivAt_id' x).mul (hasDerivAt_id' x)).const_mul 2).sub
    (hasDerivAt_const x (1 : ℝ))).congr_deriv ?_
  ring

theorem differentiable_circleLine1d : Differentiable ℝ circleLine1d :=
  fun x => (hasDerivAt_circleLine1d x).differentiableAt

theorem deriv_circleLine1d (x : ℝ) : deriv circleLine1d x = 4 * x :=
  (hasDerivAt_circleLine1d x).deriv

theorem newtonOp_circle_center :
    newtonOp circleA circleLine1d circleCenter - circleCenter = -circleY0 := by
  unfold newtonOp circleLine1d circleA circleCenter circleY0
  ring

theorem abs_newtonOp_circle_center :
    |newtonOp circleA circleLine1d circleCenter - circleCenter| ≤ circleY0 := by
  rw [newtonOp_circle_center, abs_neg]
  have hpos : 0 ≤ circleY0 := by unfold circleY0; norm_num
  simp [abs_of_nonneg hpos]

theorem circle_deriv_bound {x : ℝ} (hx : x ∈ Icc (5 / 8 : ℝ) (7 / 8)) :
    |1 - circleA * deriv circleLine1d x| ≤ (1 / 6 : ℝ) := by
  rw [deriv_circleLine1d]
  unfold circleA
  rw [abs_le]
  constructor <;> nlinarith [hx.1, hx.2]

theorem circleK_lt_one : (1 / 6 : ℝ≥0) < 1 := by
  rw [← NNReal.coe_lt_coe]
  norm_num

theorem circle_krawczyk_bound :
    circleY0 + ((1 / 6 : ℝ≥0) : ℝ) * circleRadius ≤ circleRadius := by
  unfold circleY0 circleRadius
  norm_num

/-- Unique root of `2x² - 1` on `[5/8, 7/8]`. -/
theorem circle_line_1d_unique_zero :
    ∃! x : ℝ, x ∈ Icc (5 / 8) (7 / 8) ∧ circleLine1d x = 0 := by
  have h := unique_zero_of_krawczyk_1d (f := circleLine1d) (A := circleA)
    (c := circleCenter) (r := circleRadius) (Y0 := circleY0) (K := (1 / 6 : ℝ≥0))
    circleA_ne_zero circleRadius_nonneg differentiable_circleLine1d
    abs_newtonOp_circle_center ?_ circleK_lt_one circle_krawczyk_bound
  · rwa [circle_box] at h
  · intro x hx
    rw [circle_box] at hx
    exact circle_deriv_bound hx

theorem circleLine_zero_iff (p : ℝ × ℝ) :
    circleLine p = (0, 0) ↔ p.1 = p.2 ∧ circleLine1d p.1 = 0 := by
  constructor
  · intro h
    have h1 : p.1 * p.1 + p.2 * p.2 - 1 = 0 := by
      simpa [circleLine] using congrArg Prod.fst h
    have h2 : p.1 - p.2 = 0 := by
      simpa [circleLine] using congrArg Prod.snd h
    have heq : p.1 = p.2 := by linarith
    refine ⟨heq, ?_⟩
    unfold circleLine1d
    calc
      2 * (p.1 * p.1) - 1 = p.1 * p.1 + p.1 * p.1 - 1 := by ring
      _ = p.1 * p.1 + p.2 * p.2 - 1 := by rw [heq]
      _ = 0 := h1
  · intro ⟨heq, hred⟩
    apply Prod.ext
    · change p.1 * p.1 + p.2 * p.2 - 1 = 0
      unfold circleLine1d at hred
      calc
        p.1 * p.1 + p.2 * p.2 - 1 = p.1 * p.1 + p.1 * p.1 - 1 := by rw [heq]
        _ = 2 * (p.1 * p.1) - 1 := by ring
        _ = 0 := hred
    · change p.1 - p.2 = 0
      linarith

/-- Unique intersection of the unit circle and the line `y = x` in this
square. Not a continuum PDE. -/
theorem circle_line_unique_zero :
    ∃! p : ℝ × ℝ,
      p ∈ Icc (5 / 8) (7 / 8) ×ˢ Icc (5 / 8) (7 / 8) ∧
      circleLine p = (0, 0) := by
  obtain ⟨x, ⟨hx, hf⟩, huniq⟩ := circle_line_1d_unique_zero
  refine ⟨(x, x), ⟨?_, ?_⟩, ?_⟩
  · exact ⟨hx, hx⟩
  · exact (circleLine_zero_iff (x, x)).2 ⟨rfl, hf⟩
  · intro p ⟨hpbox, hpz⟩
    have hp1 : p.1 ∈ Icc (5 / 8 : ℝ) (7 / 8) := hpbox.1
    have ⟨heq, hred⟩ := (circleLine_zero_iff p).1 hpz
    have hxeq : p.1 = x := huniq p.1 ⟨hp1, hred⟩
    exact Prod.ext hxeq (heq.symm.trans hxeq)

/-! ### B. Hopf radial equilibrium (named polynomial) -/

/-- Named polynomial `r (1 - r²)`. Isolated positive root `r = 1` of this
normal form. Not a Lohner time-`2π` fixed-point theorem. -/
def hopfRadial (r : ℝ) : ℝ := r * (1 - r * r)

def hopfCenter : ℝ := 1
def hopfRadius : ℝ := 1 / 4
def hopfA : ℝ := -1 / 2
def hopfY0 : ℝ := 0

theorem hopfA_ne_zero : hopfA ≠ 0 := by
  unfold hopfA; norm_num

theorem hopfRadius_nonneg : 0 ≤ hopfRadius := by
  unfold hopfRadius; norm_num

theorem hopf_box :
    Icc (hopfCenter - hopfRadius) (hopfCenter + hopfRadius) =
      Icc (3 / 4 : ℝ) (5 / 4) := by
  unfold hopfCenter hopfRadius
  congr 1 <;> norm_num

theorem hasDerivAt_hopfRadial (x : ℝ) :
    HasDerivAt hopfRadial (1 - 3 * x * x) x := by
  change HasDerivAt (fun y => y * (1 - y * y)) (1 - 3 * x * x) x
  refine ((hasDerivAt_id' x).mul
    ((hasDerivAt_const x (1 : ℝ)).sub
      ((hasDerivAt_id' x).mul (hasDerivAt_id' x)))).congr_deriv ?_
  simp
  ring

theorem differentiable_hopfRadial : Differentiable ℝ hopfRadial :=
  fun x => (hasDerivAt_hopfRadial x).differentiableAt

theorem deriv_hopfRadial (x : ℝ) : deriv hopfRadial x = 1 - 3 * x * x :=
  (hasDerivAt_hopfRadial x).deriv

theorem newtonOp_hopf_center :
    newtonOp hopfA hopfRadial hopfCenter - hopfCenter = 0 := by
  unfold newtonOp hopfRadial hopfA hopfCenter
  ring

theorem abs_newtonOp_hopf_center :
    |newtonOp hopfA hopfRadial hopfCenter - hopfCenter| ≤ hopfY0 := by
  rw [newtonOp_hopf_center]
  unfold hopfY0
  simp

theorem hopf_deriv_bound {x : ℝ} (hx : x ∈ Icc (3 / 4 : ℝ) (5 / 4)) :
    |1 - hopfA * deriv hopfRadial x| ≤ (27 / 32 : ℝ) := by
  rw [deriv_hopfRadial]
  unfold hopfA
  have hpos : 0 ≤ x := by nlinarith [hx.1]
  have hlo : (3 / 4 : ℝ) * (3 / 4) ≤ x * x :=
    mul_self_le_mul_self (by norm_num) hx.1
  have hhi : x * x ≤ (5 / 4 : ℝ) * (5 / 4) :=
    mul_self_le_mul_self hpos hx.2
  rw [abs_le]
  constructor <;> nlinarith [hlo, hhi]

theorem hopfK_lt_one : (27 / 32 : ℝ≥0) < 1 := by
  rw [← NNReal.coe_lt_coe]
  norm_num

theorem hopf_krawczyk_bound :
    hopfY0 + ((27 / 32 : ℝ≥0) : ℝ) * hopfRadius ≤ hopfRadius := by
  unfold hopfY0 hopfRadius
  norm_num

/-- Unique positive root of the named polynomial `r(1 - r²)` on this
interval. Isolated cycle radius of that polynomial normal form. Not a
theorem that a Lohner time-`2π` map has a fixed point. -/
theorem hopf_radial_unique_zero :
    ∃! r : ℝ, r ∈ Icc (3 / 4) (5 / 4) ∧ hopfRadial r = 0 := by
  have h := unique_zero_of_krawczyk_1d (f := hopfRadial) (A := hopfA)
    (c := hopfCenter) (r := hopfRadius) (Y0 := hopfY0) (K := (27 / 32 : ℝ≥0))
    hopfA_ne_zero hopfRadius_nonneg differentiable_hopfRadial
    abs_newtonOp_hopf_center ?_ hopfK_lt_one hopf_krawczyk_bound
  · rwa [hopf_box] at h
  · intro x hx
    rw [hopf_box] at hx
    exact hopf_deriv_bound hx

/-! ### C. 1-mode CCF Chebyshev amplitude (algebraic CAP leaf) -/

/-- Named Chebyshev `T₃(z) = 4z³ - 3z`. Algebraic CAP leaf. Not a
continuum CCF / Euler blow-up. -/
def ccfChebyshev (z : ℝ) : ℝ := 4 * (z * z * z) - 3 * z

def ccfCenter : ℝ := 7 / 8
def ccfRadius : ℝ := 1 / 8
def ccfA : ℝ := 16 / 99
def ccfY0 : ℝ := 7 / 792

theorem ccfA_ne_zero : ccfA ≠ 0 := by
  unfold ccfA; norm_num

theorem ccfRadius_nonneg : 0 ≤ ccfRadius := by
  unfold ccfRadius; norm_num

theorem ccf_box :
    Icc (ccfCenter - ccfRadius) (ccfCenter + ccfRadius) =
      Icc (3 / 4 : ℝ) 1 := by
  unfold ccfCenter ccfRadius
  congr 1 <;> norm_num

theorem hasDerivAt_ccfChebyshev (x : ℝ) :
    HasDerivAt ccfChebyshev (12 * x * x - 3) x := by
  change HasDerivAt (fun z => 4 * (z * z * z) - 3 * z) (12 * x * x - 3) x
  refine (((((hasDerivAt_id' x).mul (hasDerivAt_id' x)).mul (hasDerivAt_id' x)).const_mul 4).sub
    ((hasDerivAt_id' x).const_mul 3)).congr_deriv ?_
  simp
  ring

theorem differentiable_ccfChebyshev : Differentiable ℝ ccfChebyshev :=
  fun x => (hasDerivAt_ccfChebyshev x).differentiableAt

theorem deriv_ccfChebyshev (x : ℝ) : deriv ccfChebyshev x = 12 * x * x - 3 :=
  (hasDerivAt_ccfChebyshev x).deriv

theorem newtonOp_ccf_center :
    newtonOp ccfA ccfChebyshev ccfCenter - ccfCenter = -ccfY0 := by
  unfold newtonOp ccfChebyshev ccfA ccfCenter ccfY0
  ring

theorem abs_newtonOp_ccf_center :
    |newtonOp ccfA ccfChebyshev ccfCenter - ccfCenter| ≤ ccfY0 := by
  rw [newtonOp_ccf_center, abs_neg]
  have hpos : 0 ≤ ccfY0 := by unfold ccfY0; norm_num
  simp [abs_of_nonneg hpos]

theorem ccf_deriv_bound {x : ℝ} (hx : x ∈ Icc (3 / 4 : ℝ) 1) :
    |1 - ccfA * deriv ccfChebyshev x| ≤ (5 / 11 : ℝ) := by
  rw [deriv_ccfChebyshev]
  unfold ccfA
  have hpos : 0 ≤ x := by nlinarith [hx.1]
  have hlo : (3 / 4 : ℝ) * (3 / 4) ≤ x * x :=
    mul_self_le_mul_self (by norm_num) hx.1
  have hhi : x * x ≤ (1 : ℝ) * 1 :=
    mul_self_le_mul_self hpos hx.2
  rw [abs_le]
  constructor <;> nlinarith [hlo, hhi]

theorem ccfK_lt_one : (5 / 11 : ℝ≥0) < 1 := by
  rw [← NNReal.coe_lt_coe]
  norm_num

theorem ccf_krawczyk_bound :
    ccfY0 + ((5 / 11 : ℝ≥0) : ℝ) * ccfRadius ≤ ccfRadius := by
  unfold ccfY0 ccfRadius
  norm_num

/-- Unique root of the named Chebyshev `T₃` on this interval. Algebraic
CAP leaf. Not a continuum CCF / Euler blow-up. -/
theorem ccf_chebyshev_unique_zero :
    ∃! z : ℝ, z ∈ Icc (3 / 4) 1 ∧ ccfChebyshev z = 0 := by
  have h := unique_zero_of_krawczyk_1d (f := ccfChebyshev) (A := ccfA)
    (c := ccfCenter) (r := ccfRadius) (Y0 := ccfY0) (K := (5 / 11 : ℝ≥0))
    ccfA_ne_zero ccfRadius_nonneg differentiable_ccfChebyshev
    abs_newtonOp_ccf_center ?_ ccfK_lt_one ccf_krawczyk_bound
  · rwa [ccf_box] at h
  · intro x hx
    rw [ccf_box] at hx
    exact ccf_deriv_bound hx

end

end OmnibiasAnalytic.Check
