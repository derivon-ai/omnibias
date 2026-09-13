/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
Exact finite algebra for the invariant-parabola reference in the sharp
regular-passage calculation. The field, parameter surface, coordinate
substitution, and denominators are explicit. These theorems do not establish
ODE existence, passage capture, uniform asymptotics, or a Hilbert XVI bound.
-/

namespace OmnibiasAnalytic.Dynamics.Hilbert16Parabola

noncomputable section

def fieldX (A mu1 p r x y : ℝ) : ℝ :=
  A * x - y + x ^ 2 + (p + r) * x * y + mu1 * y ^ 2

def fieldY (C r x y : ℝ) : ℝ :=
  C * x + x ^ 2 + x * y + r * y ^ 2

def parabolaR (k p : ℝ) : ℝ := k + 5 * k ^ 2 - 2 * p

def parabolaMu1 (k p : ℝ) : ℝ := k * p - k ^ 3

def parabolaM (k p : ℝ) : ℝ := 2 * k + 8 * k ^ 2 - 2 * p

def beta (k p : ℝ) : ℝ := p - 3 * k ^ 2

def reducedSpeed (A C k p s y : ℝ) : ℝ :=
  (1 + k) * s ^ 2 + (A + k * C) * s
    + (-beta k p * s - 1 - k * (A + k * C)) * y

def reducedCofactor (k p s y : ℝ) : ℝ :=
  2 * (1 + k) * s - 2 * beta k p * y

/-- The actual quadratic field has no quadratic-height term after this shear. -/
theorem transformed_speed (A C k p x y : ℝ) :
    fieldX A (parabolaMu1 k p) p (parabolaR k p) x y
        + k * fieldY C (parabolaR k p) x y =
      reducedSpeed A C k p (x + k * y) y := by
  unfold fieldX fieldY parabolaMu1 parabolaR reducedSpeed beta
  ring

/-- The physical linear cofactor in the same shear coordinates. -/
theorem transformed_cofactor (k p x y : ℝ) :
    2 * (1 + k) * x + parabolaM k p * y =
      reducedCofactor k p (x + k * y) y := by
  unfold parabolaM reducedCofactor beta
  ring

def conic (a f m k x y : ℝ) : ℝ :=
  a * (x + k * y) ^ 2 - f * m * x + y + f

def jacobian (f k p : ℝ) : ℝ := 1 + k * f * parabolaM k p

def conicHeight (a f k p s : ℝ) : ℝ :=
  (-a * s ^ 2 + f * parabolaM k p * s - f) / jacobian f k p

/-- The affine conic equation becomes a quadratic height graph when its
coordinate Jacobian is nonzero. -/
theorem conic_height_substitution (a f k p s : ℝ)
    (hJ : jacobian f k p ≠ 0) :
    conic a f (parabolaM k p) k
      (s - k * conicHeight a f k p s) (conicHeight a f k p s) = 0 := by
  unfold conic conicHeight
  field_simp [hJ]
  unfold jacobian
  ring

def speedCubic (a f k p : ℝ) : ℝ := a * beta k p / jacobian f k p

def speedQuadratic (A C a f k p : ℝ) : ℝ :=
  1 + k + ((1 + k * (A + k * C)) * a - beta k p * f * parabolaM k p)
    / jacobian f k p

def speedLinear (A C f k p : ℝ) : ℝ :=
  A + k * C + (beta k p * f - (1 + k * (A + k * C)) * f * parabolaM k p)
    / jacobian f k p

def speedConstant (A C f k p : ℝ) : ℝ :=
  (1 + k * (A + k * C)) * f / jacobian f k p

/-- Substitution of the quadratic height leaves a cubic speed, with every
coefficient explicit; in particular there is no quartic speed term. -/
theorem reduced_speed_on_conic (A C a f k p s : ℝ)
    (hJ : jacobian f k p ≠ 0) :
    reducedSpeed A C k p s (conicHeight a f k p s) =
      speedCubic a f k p * s ^ 3 + speedQuadratic A C a f k p * s ^ 2
        + speedLinear A C f k p * s + speedConstant A C f k p := by
  unfold reducedSpeed conicHeight speedCubic speedQuadratic speedLinear speedConstant
  field_simp [hJ]
  ring

/-- The cubic polynomial above is the restriction of the original vector
field, rather than a separately postulated scalar model. -/
theorem actual_speed_on_conic (A C a f k p s : ℝ)
    (hJ : jacobian f k p ≠ 0) :
    fieldX A (parabolaMu1 k p) p (parabolaR k p)
        (s - k * conicHeight a f k p s) (conicHeight a f k p s)
      + k * fieldY C (parabolaR k p)
        (s - k * conicHeight a f k p s) (conicHeight a f k p s) =
      speedCubic a f k p * s ^ 3 + speedQuadratic A C a f k p * s ^ 2
        + speedLinear A C f k p * s + speedConstant A C f k p := by
  rw [transformed_speed]
  have hs : s - k * conicHeight a f k p s + k * conicHeight a f k p s = s := by ring
  rw [hs]
  exact reduced_speed_on_conic A C a f k p s hJ

def cofactorQuadratic (a f k p : ℝ) : ℝ :=
  2 * a * beta k p / jacobian f k p

def cofactorLinear (f k p : ℝ) : ℝ :=
  2 * (1 + k) - 2 * beta k p * f * parabolaM k p / jacobian f k p

def cofactorConstant (f k p : ℝ) : ℝ :=
  2 * beta k p * f / jacobian f k p

theorem reduced_cofactor_on_conic (a f k p s : ℝ)
    (hJ : jacobian f k p ≠ 0) :
    reducedCofactor k p s (conicHeight a f k p s) =
      cofactorQuadratic a f k p * s ^ 2 + cofactorLinear f k p * s
        + cofactorConstant f k p := by
  unfold reducedCofactor conicHeight cofactorQuadratic cofactorLinear cofactorConstant
  field_simp [hJ]
  ring

/-- The exact coefficient cancellation used before integrating the multiplier. -/
theorem cofactor_quadratic_eq_twice_speed_cubic (a f k p : ℝ) :
    cofactorQuadratic a f k p = 2 * speedCubic a f k p := by
  unfold cofactorQuadratic speedCubic
  ring

/-- The remaining quadratic coefficient in K - 2 U' is -4 U_3. -/
theorem multiplier_quadratic_remainder (a f k p : ℝ) :
    cofactorQuadratic a f k p - 6 * speedCubic a f k p =
      -4 * speedCubic a f k p := by
  rw [cofactor_quadratic_eq_twice_speed_cubic]
  ring

/-- Cancellation of the endpoint numerator before any limiting estimate. -/
theorem endpoint_numerator_cancellation (nu k a s y : ℝ) :
    ((nu * (s - k * y) + 1) * y - (s - k * y) ^ 2)
        + y * (1 + 2 * a * k * s) =
      2 * y - s ^ 2 + (nu + 2 * k * (1 + a)) * s * y
        - k * (nu + k) * y ^ 2 := by
  ring

/-- The exact signed-label numerator on every nonzero physical scale and height. -/
theorem signed_label_numerator (nu x y : ℝ) (hnu : nu ≠ 0) (hy : y ≠ 0) :
    (x / (nu * y) - 1) ^ 2 - 2 / (nu ^ 2 * y) =
      ((x - nu * y) ^ 2 - 2 * y) / (nu ^ 2 * y ^ 2) := by
  field_simp

def referenceDenominator (k m C : ℝ) : ℝ :=
  C * m ^ 2 - 6 * C * m * k ^ 2 - 2 * C * m * k
    + 2 * m + 12 * k ^ 2 + 16 * k + 4

def referenceReducedDenominator (k m C : ℝ) : ℝ :=
  C * m ^ 2 + 12 * k ^ 2 + 16 * k + 2 * m + 4

def referenceP (k m : ℝ) : ℝ := (-m + 8 * k ^ 2 + 2 * k) / 2

def referenceR (k m : ℝ) : ℝ := m - 3 * k ^ 2 - k

def referenceMu1 (k m : ℝ) : ℝ := k * (-m + 6 * k ^ 2 + 2 * k) / 2

def referenceA (k m C : ℝ) : ℝ :=
  (2 + C * m - 2 * C * k * (3 * k + 1)) / (2 * (1 + 3 * k))

def referenceConicA (k m C : ℝ) : ℝ :=
  -(1 + 3 * k) * (m + 6 * k ^ 2 + 8 * k + 2) / referenceDenominator k m C

def referenceConicF (k m C : ℝ) : ℝ :=
  2 * C * (1 + 3 * k) / referenceDenominator k m C

/-- The rational reference lies on the parameter surface used by the
transformed-field identities, with no additional parameter equations assumed. -/
theorem reference_parameter_surface (k m : ℝ) :
    referenceR k m = parabolaR k (referenceP k m) ∧
      referenceMu1 k m = parabolaMu1 k (referenceP k m) ∧
      m = parabolaM k (referenceP k m) := by
  unfold referenceR referenceMu1 referenceP parabolaR parabolaMu1 parabolaM
  constructor
  · ring
  constructor <;> ring

/-- The exact normalized reference satisfies its Darboux polynomial identity.
The factors multiplying the physical field are the explicit partial
derivatives of the conic polynomial. -/
theorem reference_darboux_identity (k m C x y : ℝ)
    (hD : referenceDenominator k m C ≠ 0) (hk : 1 + 3 * k ≠ 0) :
    (2 * referenceConicA k m C * (x + k * y) - referenceConicF k m C * m)
        * fieldX (referenceA k m C) (referenceMu1 k m)
          (referenceP k m) (referenceR k m) x y
      + (1 + 2 * referenceConicA k m C * k * (x + k * y))
        * fieldY C (referenceR k m) x y =
      (2 * (1 + k) * x + m * y)
        * conic (referenceConicA k m C) (referenceConicF k m C) m k x y := by
  unfold referenceConicA referenceConicF referenceA referenceMu1 referenceP referenceR
  unfold fieldX fieldY conic
  field_simp [hD, hk]
  unfold referenceDenominator
  ring

/-- The affine Jacobian has the reduced denominator needed by the cubic coefficient. -/
theorem reference_jacobian (k m C : ℝ) (hD : referenceDenominator k m C ≠ 0) :
    jacobian (referenceConicF k m C) k (referenceP k m) =
      referenceReducedDenominator k m C / referenceDenominator k m C := by
  unfold jacobian referenceConicF referenceP parabolaM
  field_simp [hD]
  unfold referenceDenominator referenceReducedDenominator
  ring

/-- Exact factorization of the cubic reference speed. Its smallness in a
canonical parameter sector remains an analytic conclusion outside this module. -/
theorem reference_cubic_speed_coefficient (k m C : ℝ)
    (hD : referenceDenominator k m C ≠ 0)
    (hE : referenceReducedDenominator k m C ≠ 0) :
    speedCubic (referenceConicA k m C) (referenceConicF k m C) k (referenceP k m) =
      -(1 + 3 * k) * (referenceP k m - 3 * k ^ 2)
        * (6 * k ^ 2 + 8 * k + m + 2) / referenceReducedDenominator k m C := by
  unfold speedCubic
  rw [reference_jacobian k m C hD]
  unfold referenceConicA beta
  field_simp [hD, hE]
  ring

end

end OmnibiasAnalytic.Dynamics.Hilbert16Parabola
