# 01-12 The conjugate Hilbert tower

## 1. Thesis and status

The Hilbert transform commutes with differentiation, so whenever `H[sigma]` is
closed form the **entire conjugate tower** `H[sigma^(n)] = (H[sigma])^(n)` is
closed form too; for the Cauchy-Hardy family this makes a dictionary that is
simultaneously closed under differentiation *and* under `H`, which attacks the
recorded Hilbert-and-dictionary floor at the basis level rather than by
quadrature refinement.

- **Status**: designed
- **Depends on**: 01-01
- **Blocks**: 02-06, 07-02, 07-03

## 2. Where it lands

`packages/omnibias-core/src/omnibias/core/verified/hardy_line.py` (extend in
place) plus a new `omnibias/core/conjugate.py` for the non-verified fast path
and the dictionary assembly, with torch and jax twins under
`omnibias.{torch,jax}.conjugate` for use inside training loops.

No new package. Same domain, same tier, same audience as the existing Hardy
module.

## 3. Prior art in omnibias

- `packages/omnibias-core/src/omnibias/core/verified/hardy_line.py` — the
  generalized Cauchy-Hardy pair. For `a > 0` and real `alpha`, with
  `r = sqrt(a^2 + y^2)` and `phi = atan(y / a)`:

```
P_{a,alpha}(y) = r^-alpha cos(alpha phi)      (even)
Q_{a,alpha}(y) = r^-alpha sin(alpha phi)      (odd)
H[P] = Q,     H[Q] = -P
P' = -alpha Q_{a, alpha+1},    Q' = alpha P_{a, alpha+1}
```

  provided as `hardy_even`, `hardy_odd`, `hardy_pair`, `hardy_even_deriv`,
  `hardy_odd_deriv`, `hilbert_of_hardy_even`, `hilbert_of_hardy_odd`, plus
  interval-argument variants and `hardy_even_dalpha` / `hardy_odd_dalpha`.
- `packages/omnibias-core/src/omnibias/core/verified/line.py` — the classical
  Poisson pair, the `alpha = 1` case: `poisson_kernel`, `conjugate_poisson`,
  `hilbert_of_poisson` (`H[p_a] = q_a`), `hilbert_of_conjugate`
  (`H[q_a] = -p_a`), `poisson_kernel_deriv`, `poisson_primitive`.
- `benchmarks/reproduce_deepmind_ccf.py` — records the blocker verbatim:
  spectral and principal-value Hilbert alone err at `O(1e-1)`, and with high
  projection-defect weight the neural profile is pulled into a Hardy span that
  itself floors near `1e-1`; the artifact carries
  `known_floor_note = "hilbert_dictionary_catch22_near_1e-1"` and the stretch
  gate `CCF_STRETCH_RESIDUAL_GATE = 1e-13` is never weakened.

**Confirmed gap.** First derivatives close in the family, and `H` is exact on
the pair, but **nobody has composed the two**. There is no `n`-th derivative of
the Hardy pair, no statement that the dictionary is closed under `H` at every
order, and no derivative atoms in the CCF dictionary.

## 4. Mathematics

### Commutation

For `f` with enough decay, differentiation and the Hilbert transform commute:

```
H[f'] = (H[f])'
```

(immediate from the convolution form, since `H` is a Fourier multiplier
`-i sgn(xi)` and differentiation is multiplication by `i xi`; the multipliers
commute). By induction,

```
H[f^(n)] = (H[f])^(n)      for every n >= 0.
```

So a closed-form `H[sigma]` upgrades to a closed-form conjugate tower for free.
This is the general statement; the Cauchy-Hardy family is where it becomes
completely explicit.

### The closed tower for the Cauchy-Hardy pair

Work with the analytic branch the module already documents,

```
F_{a,alpha}(y) = (a - i y)^-alpha = P_{a,alpha}(y) + i Q_{a,alpha}(y)
```

analytic and single-valued in the upper half-plane. Differentiating,

```
d/dy F_{a,alpha} = i alpha F_{a, alpha+1}
```

and by induction, with the rising factorial `(alpha)_n = alpha (alpha+1) ... (alpha+n-1)`,

```
F^(n)_{a,alpha} = i^n (alpha)_n F_{a, alpha+n}.
```

Taking real and imaginary parts gives the **entire tower in closed form**, with
only a Pochhammer factor and a shift of the exponent:

| `n mod 4` | `P^(n)_{a,alpha}` | `Q^(n)_{a,alpha}` |
|---|---|---|
| 0 | `+(alpha)_n P_{a,alpha+n}` | `+(alpha)_n Q_{a,alpha+n}` |
| 1 | `-(alpha)_n Q_{a,alpha+n}` | `+(alpha)_n P_{a,alpha+n}` |
| 2 | `-(alpha)_n P_{a,alpha+n}` | `-(alpha)_n Q_{a,alpha+n}` |
| 3 | `+(alpha)_n Q_{a,alpha+n}` | `-(alpha)_n P_{a,alpha+n}` |

`n = 1` reproduces the two rules already in the module, which is the consistency
check that the general formula is right.

### The conjugate tower

Combining with `H[P] = Q` and `H[Q] = -P` and the commutation lemma:

```
H[ P^(n)_{a,alpha} ] = Q^(n)_{a,alpha},        H[ Q^(n)_{a,alpha} ] = -P^(n)_{a,alpha}
```

for **every** `n`. Equivalently, `H[F^(n)] = i F^(n)`: every derivative of a
Hardy atom remains in the Hardy space, so the eigenrelation is inherited by the
whole tower.

### The structural consequence

Let `D_alpha = span { P_{a_j, alpha_j}, Q_{a_j, alpha_j} }` be a finite Hardy
dictionary. Then:

1. `D` is **closed under `H`** exactly (this is already used).
2. Adding derivative atoms `P^(n), Q^(n)` keeps it closed under `H`, because
   each derivative atom is (up to a scalar) another atom at exponent
   `alpha + n`.
3. So the enlarged dictionary
   `D' = span { P_{a_j, alpha_j + n}, Q_{a_j, alpha_j + n} : n = 0..N }`
   is strictly larger, still exactly `H`-closed, and costs **no extra Hilbert
   evaluation** — `H` acts on coefficients by a fixed signed permutation.

That is the whole point. The recorded floor is a *capacity* problem: the neural
profile is pulled into a Hardy span that itself floors near `1e-1`. Enlarging
the span along the exponent axis, at zero Hilbert cost and with `H` still exact,
is a direct attack on that capacity, as opposed to refining a quadrature that
was never the binding constraint.

### Why exponent shifts are the right enlargement

For CCF self-similar profiles the physical far-field exponent is
`alpha = 1/(1 + lambda)`, as the module's own documentation records, and a
finite Hardy sum matches that algebraic decay where a finite Poisson sum
(`|y|^-2`) structurally cannot. Derivative atoms live at `alpha + n`, that is
they decay *faster*. So the enlarged dictionary keeps the correct far-field
exponent in its slowest atoms while gaining near-field resolution from the
faster ones. The enlargement is aligned with the physics rather than arbitrary.

### Interval version

Everything above is products of `Interval` quantities the module already has
(`hardy_radius_iv`, `hardy_angle_iv`, `exp_iv`, `ln_iv`, `sin_iv`, `cos_iv`),
plus an exact rational or float Pochhammer factor. So the tower and the
conjugate tower are **soundly enclosable**, with outward rounding, at every
order. That matters because the CCF pipeline's certified arm needs enclosures,
not only fast evaluations.

## 5. Worked example

Take `a = 1`, `alpha = 0.5`, `y = 1`. Then `r = sqrt(2) = 1.41421356`,
`phi = atan(1) = pi/4 = 0.78539816`.

Base atoms:

```
P_{1,0.5}(1) = 2^-0.25 cos(pi/8) = 0.84089642 * 0.92387953 = 0.77688705
Q_{1,0.5}(1) = 2^-0.25 sin(pi/8) = 0.84089642 * 0.38268343 = 0.32179712
```

First derivative (`n = 1`, `(alpha)_1 = 0.5`, exponent `1.5`):

```
Q_{1,1.5}(1) = 2^-0.75 sin(3 pi/8) = 0.59460356 * 0.92387953 = 0.54934204
P'(1) = -(0.5) Q_{1,1.5}(1) = -0.27467102
```

which is exactly what `hardy_even_deriv` returns, confirming the general table
at `n = 1`.

Second derivative (`n = 2`, `(alpha)_2 = 0.5 * 1.5 = 0.75`, exponent `2.5`,
row `n mod 4 = 2`):

```
P_{1,2.5}(1) = 2^-1.25 cos(5 pi/8) = 0.42044821 * (-0.38268343) = -0.16089857
Q_{1,2.5}(1) = 2^-1.25 sin(5 pi/8) = 0.42044821 *   0.92387953  =  0.38844338
P''(1) = -(0.75) P_{1,2.5}(1) =  0.12067393
Q''(1) = -(0.75) Q_{1,2.5}(1) = -0.29133254
```

Conjugate tower at order 2:

```
H[ P'' ](1) = Q''(1) = -0.29133254
H[ Q'' ](1) = -P''(1) = -0.12067393
```

No quadrature, no principal-value integral, no spectral transform. Two
transcendental evaluations (`r^-beta` and one angle) and a scalar.

Cross-check by second-order central differencing of `hardy_even` at `h = 1e-4`
gives `0.1206739`, agreeing to seven digits, which is all that difference can
deliver — and precisely the accuracy ceiling the closed form removes.

## 6. Proposed API

Extends the existing module; new symbols only.

```python
# omnibias/core/verified/hardy_line.py  (additions)
def pochhammer(alpha: float, n: int) -> float:
    """Rising factorial (alpha)_n; exact for integer alpha, float otherwise."""

def hardy_even_deriv_n(y: float, a: float, alpha: float, n: int) -> Interval:
    """P^(n) via the closed table; n >= 0. n = 1 must agree with
    `hardy_even_deriv` exactly."""

def hardy_odd_deriv_n(y: float, a: float, alpha: float, n: int) -> Interval: ...
def hardy_even_deriv_n_iv(y: Interval, a: float, alpha: float, n: int) -> Interval: ...
def hardy_odd_deriv_n_iv(y: Interval, a: float, alpha: float, n: int) -> Interval: ...

def hilbert_of_hardy_even_deriv_n(y, a, alpha, n) -> Interval:
    """H[P^(n)] = Q^(n). Exact."""
def hilbert_of_hardy_odd_deriv_n(y, a, alpha, n) -> Interval:
    """H[Q^(n)] = -P^(n). Exact."""
```

```python
# omnibias/core/conjugate.py  (fast, non-verified path + dictionary)
@dataclass(frozen=True)
class HardyAtom:
    scale: float        # a
    exponent: float     # alpha
    order: int          # n
    parity: Literal["even", "odd"]

@dataclass(frozen=True)
class HardyDictionary:
    atoms: tuple[HardyAtom, ...]
    def hilbert_permutation(self) -> tuple[tuple[int, float], ...]:
        """H acts on coefficients as a signed permutation: (target index, sign)."""

def evaluate(dictionary: HardyDictionary, y) -> FloatArray: ...
def hilbert(dictionary: HardyDictionary, coeffs) -> FloatArray:
    """Apply the signed permutation. No quadrature."""
```

Backends: `omnibias.torch.conjugate` and `omnibias.jax.conjugate` expose
`hardy_atoms(y, dictionary)` and `hilbert_coeffs(coeffs, dictionary)` as
bit-identical twins for use inside the CCF training loop.

## 7. Practical use cases

1. **CCF dictionary capacity** (spec 07-03). Derivative atoms enlarge the span
   along the physically meaningful exponent axis while keeping `H` exact and
   free. This is the concrete mechanism aimed at the recorded
   `hilbert_dictionary_catch22_near_1e-1` note.
2. **Exact `H` inside a training loop.** Any model expanded in the dictionary
   gets `H[u]` as a signed permutation of its coefficients: differentiable,
   exact, and `O(number of atoms)` rather than `O(N log N)` with a quadrature
   error.
3. **Certified conjugate quantities.** The interval variants make statements
   like "the conjugate function on this cell lies in `[lo, hi]`" sound, which
   the enclosure-tier arms of the fluid work need.
4. **Boundary-integral methods** (spec 02-06). Layer potentials on a line are
   Hilbert-type operators; a dictionary closed under `H` gives exact
   Dirichlet-to-Neumann action.
5. **Signal analysis with exact analytic signals.** The analytic signal of an
   atom is `F` itself; envelope and instantaneous phase are then closed form.

## 8. Acceptance gates

- **G1 consistency at `n = 1`.** `hardy_even_deriv_n(..., 1)` and
  `hardy_odd_deriv_n(..., 1)` reproduce the existing `hardy_even_deriv` /
  `hardy_odd_deriv` **exactly** (interval equality, not approximate).
- **G2 tower correctness.** For `n = 0 .. 8` and randomized `(a, alpha, y)`,
  the closed table matches a high-precision (at least 50-digit) reference to
  `<= 1e-30` relative.
- **G3 enclosure soundness.** The `_iv` variants contain the true value on a
  dense deterministic grid **and** a random sample, with zero violations, as the
  verified register requires.
- **G4 exact commutation.** `H[P^(n)] - Q^(n)` is exactly zero by construction
  in the implementation, and a high-precision numerical Hilbert transform of
  `P^(n)` agrees with `Q^(n)` to the quadrature's own accuracy on a test set.
- **G5 dictionary capacity, the one that matters.** On the CCF profile-fitting
  subproblem, the enlarged dictionary's best achievable projection defect is at
  least `10x` smaller than the current Hardy span at matched atom count, and the
  reduction is reported in the campaign artifact. This gate is about the
  *dictionary*, not about the stretch residual.

Note what G5 is not: clearing `CCF_STRETCH_RESIDUAL_GATE = 1e-13` is spec
07-03's gate, is not weakened, and is not claimed here.

## 9. Benchmark plan

- `benchmarks/conjugate_tower.py`: accuracy versus order against a
  high-precision reference, enclosure width growth versus order, wall time per
  atom.
- Dictionary-capacity arm added to the existing CCF pipeline rather than a
  competing script, so the comparison is against the recorded floor with the
  same scoring.
- Smoke JSON committed as `docs/benchmarks/conjugate_tower_smoke.json`; full
  sweep under `$OMNIBIAS_SCRATCH/conjugate/`.

## 10. Honesty and scope

- The commutation lemma requires decay and smoothness sufficient for the
  principal value to exist and for differentiation under the integral. The
  Cauchy-Hardy family satisfies this for `alpha > 0`; state the hypothesis, do
  not treat `H[f^(n)] = (H[f])^(n)` as unconditional.
- Everything here is on the **line**, with the repo's convention
  `H[f](x) = (1/pi) p.v. integral f(t) / (x - t) dt`. Periodic and finite-interval
  Hilbert transforms are different operators and are not covered.
- Enlarging a dictionary raises capacity; it does not by itself reduce a
  residual. G5 measures projection defect, and any claim beyond that must be
  earned by the campaign's own gate.
- The certificate tier available here is **sound enclosure** via `Interval`. It
  is not `theorem_prover_verified`: the tower identity is an analytic statement
  with a limit and a quantifier, which is explicitly outside the Mathlib-free
  kernel's finite-rational scope. The Pochhammer factors are rational for
  rational `alpha`, but the identity they appear in is not.
- No collapse limit of either kind appears in this spec. It is a statement about
  a function family, not about coalescing biases (`delta -> 0`) or hardening
  gates (`beta -> inf`).

## 11. Open questions and risks

- **Enclosure widening with order.** `(alpha)_n` grows and the exponent shift
  makes atoms sharper, so interval widths may grow quickly with `n`. Measure the
  usable order ceiling rather than asserting one.
- **Conditioning of the enlarged dictionary.** Atoms at `alpha` and `alpha + 1`
  with the same scale are correlated; the Gram matrix may become ill-conditioned
  well before the span is exhausted. Report the condition number alongside the
  capacity gain, and consider orthogonalizing.
- **The floor may not be dictionary-limited.** The recorded note describes a
  two-sided problem (Hilbert accuracy *and* dictionary capacity). If capacity
  turns out not to be binding, G5 will show it, and the honest outcome is to say
  so and look elsewhere.
- **Falsifier.** If the enlarged dictionary does not reduce projection defect at
  matched atom count, this spec is a tidy identity with no operational value,
  and spec 07-03 must find a different lever.

## 12. Implementation checklist

- [ ] Extend `packages/omnibias-core/src/omnibias/core/verified/hardy_line.py`
      with the `_deriv_n` family and their `_iv` variants
- [ ] `packages/omnibias-core/src/omnibias/core/conjugate.py` with the
      dictionary and the signed-permutation Hilbert action
- [ ] `omnibias.torch.conjugate` and `omnibias.jax.conjugate` twins with a
      parity test
- [ ] Exact-agreement test at `n = 1` against the existing derivative rules
- [ ] High-precision tower test for `n = 0 .. 8`
- [ ] Enclosure soundness test: dense grid **and** random sample
- [ ] Numerical-Hilbert cross-check test
- [ ] Dictionary-capacity arm in the CCF pipeline, artifact field recorded
- [ ] `benchmarks/conjugate_tower.py` plus smoke JSON
- [ ] Regenerate `__all__` in `omnibias/core/verified/__init__.py`
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
