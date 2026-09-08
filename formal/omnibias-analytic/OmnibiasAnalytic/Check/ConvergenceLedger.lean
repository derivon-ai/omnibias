/-
Finite rational convergence-ledger lemmas (Mathlib-backed).

A staged iterative construction closes when a finite list of affine
margin inequalities holds. This file proves that algebra over `ℚ`
(`linarith` / `norm_num` / `min`). It does not define analytic
classes, construct a correction, or prove a PDE.

The Navier–Stokes instance transcribes OpenAI's
`NavierStokes/ExponentLedger` arithmetic (`σ ≥ 1/5`, `κ ≤ 10⁻⁵`).
The polymer instance restates the locked Kotecký–Preiss majorants
already in `Check/Polymer.lean`.

Scope (honest). Finite rational inequalities. Not Clay (A)/(B), not
a continuum mass gap, not Osterwalder–Schrader. The file contains
no `admit`.
-/

import Mathlib.Tactic
import OmnibiasAnalytic.Check.Polymer

namespace OmnibiasAnalytic.Check

/-! ## Generic affine / min lemmas -/

/-- Affine increment: shifting the argument of `c + a * s` by `h`
changes the value by `a * h`. -/
theorem affine_increment (c a s h : ℚ) :
    (c + a * (s + h)) - (c + a * s) = a * h := by
  ring

/-- A number strictly below every listed part is strictly below the min. -/
theorem lt_min₄ {x a b c d : ℚ}
    (ha : x < a) (hb : x < b) (hc : x < c) (hd : x < d) :
    x < min (min (min a b) c) d := by
  exact lt_min (lt_min (lt_min ha hb) hc) hd

theorem lt_min₅ {x a b c d e : ℚ}
    (ha : x < a) (hb : x < b) (hc : x < c) (hd : x < d) (he : x < e) :
    x < min (min (min (min a b) c) d) e := by
  exact lt_min (lt_min₄ ha hb hc hd) he

theorem lt_min₂ {x a b : ℚ} (ha : x < a) (hb : x < b) :
    x < min a b :=
  lt_min ha hb

/-! ## Navier–Stokes exponent ledger (OpenAI ExponentLedger) -/

def waveExponent (σ : ℚ) : ℚ := 1 / 2 + σ
def meanExponent (σ : ℚ) : ℚ := 1 + σ
def meanUpdateExponent (σ κ : ℚ) : ℚ := meanExponent σ - 2 * κ

def particularGain (σ κ : ℚ) : ℚ :=
  min (min (min (1 / 2 - 3 * κ) (1 / 2 - κ)) (waveExponent σ - κ)) (2 / 5)

def signedGain (σ κ : ℚ) : ℚ :=
  min (min (min (1 / 2 - 4 * κ) (2 / 5 - κ)) (1 / 2 - 2 * κ))
    (waveExponent σ - 3 * κ)

def signedBarGain (σ κ : ℚ) : ℚ :=
  min (min (min (min (9 / 50 - 2 * κ) (1 / 2 - 3 * κ)) (σ - 3 * κ))
    (1 - κ)) (1 - 2 * κ)

theorem particular_gain_eq {σ κ : ℚ} (hσ : 1 / 5 ≤ σ) (hκ : κ ≤ 1 / 100000) :
    particularGain σ κ = 2 / 5 := by
  unfold particularGain waveExponent
  apply le_antisymm (min_le_right _ _)
  simp only [le_min_iff]
  exact ⟨⟨⟨by linarith, by linarith⟩, by linarith⟩, le_refl _⟩

theorem signed_gain_eq {σ κ : ℚ} (hσ : 1 / 5 ≤ σ) (hκ : κ ≤ 1 / 100000) :
    signedGain σ κ = 2 / 5 - κ := by
  unfold signedGain waveExponent
  apply le_antisymm
  · exact le_trans (min_le_left _ _) (le_trans (min_le_left _ _) (min_le_right _ _))
  · simp only [le_min_iff]
    exact ⟨⟨⟨by linarith, le_refl _⟩, by linarith⟩, by linarith⟩

theorem particular_gain_exceeds_tenth {σ κ : ℚ}
    (hσ : 1 / 5 ≤ σ) (hκ : κ ≤ 1 / 100000) :
    waveExponent (σ + 1 / 10) < waveExponent σ + particularGain σ κ := by
  rw [particular_gain_eq hσ hκ]
  unfold waveExponent
  linarith

theorem signed_gain_exceeds_tenth {σ κ : ℚ}
    (hσ : 1 / 5 ≤ σ) (hκ : κ ≤ 1 / 100000) :
    waveExponent (σ + 1 / 10) < waveExponent σ + signedGain σ κ := by
  rw [signed_gain_eq hσ hκ]
  unfold waveExponent
  linarith

theorem mean_update_wave_margin {σ κ : ℚ} (hκ : κ ≤ 1 / 100000) :
    waveExponent (σ + 1 / 10) < meanUpdateExponent σ κ := by
  unfold waveExponent meanUpdateExponent meanExponent
  linarith

theorem completed_mean_gain_eq {κ : ℚ} (hκ : κ ≤ 1 / 100000) :
    min (17 / 100) (1 - 4 * κ) = 17 / 100 := by
  apply min_eq_left
  linarith

theorem completed_mean_margin {σ κ : ℚ} (hκ : κ ≤ 1 / 100000) :
    meanExponent (σ + 1 / 10) <
      meanExponent σ + min (17 / 100) (1 - 4 * κ) := by
  rw [completed_mean_gain_eq hκ]
  unfold meanExponent
  linarith

theorem completed_defect_margin {σ κ : ℚ} (hκ : κ ≤ 1 / 100000) :
    meanExponent (σ + 1 / 10) < meanExponent σ + 9 / 10 - 4 * κ := by
  unfold meanExponent
  linarith

theorem signed_bar_gain_exceeds_seventeen_hundredths {σ κ : ℚ}
    (hσ : 1 / 5 ≤ σ) (hκ : κ ≤ 1 / 100000) :
    17 / 100 < signedBarGain σ κ := by
  unfold signedBarGain
  simp only [lt_min_iff]
  exact ⟨⟨⟨⟨by linarith, by linarith⟩, by linarith⟩, by linarith⟩, by linarith⟩

theorem old_difference_bar_margin_iff (κ : ℚ) :
    17 / 100 < 9 / 50 - 2 * κ ↔ κ < 1 / 200 := by
  constructor <;> intro h <;> linarith

/-- The five `all_stage_arithmetic` margins plus the `.17` bar residual,
at a generic admissible stage. Finite rational algebra only. -/
theorem ns_all_stage_margins {σ κ : ℚ}
    (hσ : 1 / 5 ≤ σ) (hκ : κ ≤ 1 / 100000) :
    waveExponent (σ + 1 / 10) < waveExponent σ + particularGain σ κ ∧
    waveExponent (σ + 1 / 10) < waveExponent σ + signedGain σ κ ∧
    waveExponent (σ + 1 / 10) < meanUpdateExponent σ κ ∧
    meanExponent (σ + 1 / 10) <
      meanExponent σ + min (17 / 100) (1 - 4 * κ) ∧
    meanExponent (σ + 1 / 10) < meanExponent σ + 9 / 10 - 4 * κ ∧
    17 / 100 < signedBarGain σ κ :=
  ⟨particular_gain_exceeds_tenth hσ hκ,
    signed_gain_exceeds_tenth hσ hκ,
    mean_update_wave_margin hκ,
    completed_mean_margin hκ,
    completed_defect_margin hκ,
    signed_bar_gain_exceeds_seventeen_hundredths hσ hκ⟩

/-- Manuscript parameters `σ = 1/5`, `κ = 10⁻⁵`. -/
theorem ns_manuscript_margins :
    waveExponent (1 / 5 + 1 / 10) <
        waveExponent (1 / 5) + particularGain (1 / 5) (1 / 100000) ∧
    waveExponent (1 / 5 + 1 / 10) <
        waveExponent (1 / 5) + signedGain (1 / 5) (1 / 100000) ∧
    waveExponent (1 / 5 + 1 / 10) < meanUpdateExponent (1 / 5) (1 / 100000) ∧
    meanExponent (1 / 5 + 1 / 10) <
      meanExponent (1 / 5) + min (17 / 100) (1 - 4 * (1 / 100000)) ∧
    meanExponent (1 / 5 + 1 / 10) <
      meanExponent (1 / 5) + 9 / 10 - 4 * (1 / 100000) ∧
    17 / 100 < signedBarGain (1 / 5) (1 / 100000) :=
  ns_all_stage_margins (by norm_num) (by norm_num)

/-! ## Strong-coupling polymer ledger -/

/-- Locked majorants `15 < 20` and `15 < 24` at `d = 4`. Finite
arithmetic, not a continuum gauge claim. -/
theorem polymer_ledger_margins :
    polymerBacktrack 4 < polymerFirstStep 4 ∧
      polymerBacktrack 4 < polymerCrude 4 :=
  ⟨polymer_backtrack_lt_first_step, polymer_backtrack_lt_crude⟩

/-! ## Navier–Stokes scale ledger (geometry / energy) -/

/-- Manuscript ``h = 1/200`` is below both the geometry bound ``1/100``
and the energy bound ``1/6``. Finite rational algebra only. -/
theorem ns_scale_margins :
    (1 / 200 : ℚ) < 1 / 100 ∧ (1 / 200 : ℚ) < 1 / 6 := by
  constructor <;> norm_num

end OmnibiasAnalytic.Check
