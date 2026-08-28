/-
Domain-subdivision (branch-and-bound leaf coverage) certificate
(phase1-lean-replay).

A finite branch-and-bound / bisection search over a starting domain
`[0, resolution]` (rationals expressed as `Int` numerators over the shared
integer `resolution`, so no irrational or floating-point reasoning is
needed at all) visits a finite list of leaf boxes. This module checks,
purely combinatorially, that the *visited* leaves tile the claimed
starting domain edge-to-edge: no gap, no overlap, first leaf starts at
`0`, last leaf ends at `resolution`.

Scope. This is a **coverage** check only. It says nothing about whether a
leaf's own analytic conclusion (e.g. "the eigenvalue count below this
midpoint is `>= index`") is correct -- that conclusion is a trusted Python
input, exactly the same trust boundary `Omnibias/LDLT.lean` already
documents for the division-bearing `L D Lᵀ` factorisation. Mathlib-free,
`sorry`-free, decidable, and cheap: every step is a finite `Int`
comparison.
-/

import Omnibias.Interval

namespace Omnibias.Subdivision

/-- One visited leaf box `[lo, hi]`, as an integer sub-range of
`[0, resolution]` (a fraction `lo / resolution` .. `hi / resolution` of the
claimed domain). -/
structure Leaf where
  lo : Int
  hi : Int
deriving Repr, DecidableEq

/-- Chain contiguity from a running boundary `prevHi`: each leaf must start
exactly where the previous one ended and be non-empty, and the chain must
finish exactly at `resolution`. -/
def chainOk (resolution : Int) (prevHi : Int) : List Leaf → Bool
  | [] => decide (prevHi = resolution)
  | l :: rest => decide (l.lo = prevHi) && decide (l.lo < l.hi) && chainOk resolution l.hi rest

/-- Do the (caller-sorted) leaves tile `[0, resolution]` with no gap and no
overlap? The first leaf must start at `0`; `chainOk` handles the rest. -/
def coversNoGaps (resolution : Int) (leaves : List Leaf) : Bool :=
  match leaves with
  | [] => false
  | l0 :: rest => decide (l0.lo = 0) && decide (l0.lo < l0.hi) && chainOk resolution l0.hi rest

/-- The `Prop` a `coversNoGaps` pass certifies: a genuine edge-to-edge
partition of `[0, resolution]` by strictly-increasing, contiguous integer
sub-ranges. -/
def ChainProp (resolution : Int) : Int → List Leaf → Prop
  | prevHi, [] => prevHi = resolution
  | prevHi, l :: rest => l.lo = prevHi ∧ l.lo < l.hi ∧ ChainProp resolution l.hi rest

def IsPartition (resolution : Int) (leaves : List Leaf) : Prop :=
  match leaves with
  | [] => False
  | l0 :: rest => l0.lo = 0 ∧ l0.lo < l0.hi ∧ ChainProp resolution l0.hi rest

theorem chainOk_sound (resolution : Int) :
    ∀ (prevHi : Int) (leaves : List Leaf),
      chainOk resolution prevHi leaves = true → ChainProp resolution prevHi leaves
  | prevHi, [], h => by
      simpa only [chainOk, decide_eq_true_eq, ChainProp] using h
  | prevHi, l :: rest, h => by
      simp only [chainOk, Bool.and_eq_true, decide_eq_true_eq] at h
      exact ⟨h.1.1, h.1.2, chainOk_sound resolution l.hi rest h.2⟩

/-- **Soundness of `coversNoGaps`.** A passing coverage check certifies the
combinatorial fact that the visited leaves partition `[0, resolution]`
edge-to-edge -- nothing about the analytic content of any individual
leaf's conclusion, which remains a trusted Python input. -/
theorem coversNoGaps_sound (resolution : Int) (leaves : List Leaf)
    (h : coversNoGaps resolution leaves = true) : IsPartition resolution leaves := by
  cases leaves with
  | nil => simp [coversNoGaps] at h
  | cons l0 rest =>
      simp only [coversNoGaps, Bool.and_eq_true, decide_eq_true_eq] at h
      exact ⟨h.1.1, h.1.2, chainOk_sound resolution l0.hi rest h.2⟩

end Omnibias.Subdivision
