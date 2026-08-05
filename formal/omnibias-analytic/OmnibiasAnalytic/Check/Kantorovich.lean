/-
Newton-Kantorovich / Krawczyk finite-obligation lemmas (Mathlib-backed).

These are the reusable `ℚ` statements behind the contraction certificates that
`omnibias.core.verified.kantorovich` seals (`radii_polynomial`, `krawczyk`, and
`quadratic_radii`, which reduces to `radii_polynomial`). The Python bridge emits
the certificate's concrete rational constants and closes the numeric inequalities
with `norm_num`; these lemmas name and document the finite facts.

Scope (honest). A green build certifies only the finite, rational inequalities
`p(r) < 0`, `κ(r) < 1`, and the Krawczyk containment. The analytic
existence/uniqueness conclusion of the Newton-Kantorovich / Krawczyk theorem is a
trusted (unverified) input, exactly as in the kernel.
-/

import Mathlib.Tactic

namespace OmnibiasAnalytic.Check

/-- The **radii polynomial** `p(r) = Y0 + (Z0 + Z1) r + Z2 r² - r`. -/
def radiiPoly (Y0 Z0 Z1 Z2 r : ℚ) : ℚ := Y0 + (Z0 + Z1) * r + Z2 * r ^ 2 - r

/-- The **contraction constant** `κ(r) = Z0 + Z1 + 2 Z2 r`. -/
def contractionConst (Z0 Z1 Z2 r : ℚ) : ℚ := Z0 + Z1 + 2 * Z2 * r

/-- **Self-map bound.** The radii polynomial being negative at `r` is exactly the
Newton-Kantorovich self-map inequality `Y0 + (Z0 + Z1) r + Z2 r² < r`. -/
theorem radii_selfmap {Y0 Z0 Z1 Z2 r : ℚ} (hp : radiiPoly Y0 Z0 Z1 Z2 r < 0) :
    Y0 + (Z0 + Z1) * r + Z2 * r ^ 2 < r := by
  unfold radiiPoly at hp; linarith

/-- **Radii-polynomial contraction obligation.** Both finite conditions of the
existence test: the polynomial is negative and the contraction constant is `< 1`. -/
theorem radii_contraction {Y0 Z0 Z1 Z2 r : ℚ}
    (hp : radiiPoly Y0 Z0 Z1 Z2 r < 0) (hk : contractionConst Z0 Z1 Z2 r < 1) :
    radiiPoly Y0 Z0 Z1 Z2 r < 0 ∧ contractionConst Z0 Z1 Z2 r < 1 :=
  ⟨hp, hk⟩

/-- **Krawczyk strict containment (one coordinate).** The image interval
`[lo, hi]` lies strictly inside the box `[c - r, c + r]`. -/
theorem krawczyk_strictly_inside {c r lo hi : ℚ} (h1 : c - r < lo) (h2 : hi < c + r) :
    c - r < lo ∧ hi < c + r :=
  ⟨h1, h2⟩

end OmnibiasAnalytic.Check
