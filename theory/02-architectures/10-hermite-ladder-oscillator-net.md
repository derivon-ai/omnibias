# 02-10 Hermite ladder oscillator networks

## 1. Thesis and status

The gaussian base's derivative tower **is** the Hermite function basis, and the
quantum harmonic oscillator's raising and lowering operators are first-order
differential operators the tower supplies exactly, so an omnibias network can
carry an exact eigenbasis and exact ladder algebra for oscillator-like
Schrodinger problems.

- **Status**: designed
- **Depends on**: 01-01, 01-06
- **Blocks**: 07-07

## 2. Where it lands

`packages/omnibias-torch/src/omnibias/torch/architectures/ladder.py` plus the
jax twin, with the pure-Python algebra in
`packages/omnibias-core/src/omnibias/core/ladder.py`. The FermiNet bridge
(`omnibias-ferminet`) is a consumer.

## 3. Prior art in omnibias

- `omnibias.core.polynomials` — `hermite_coeffs`, already shared by every
  backend. The gaussian tower is exactly the Hermite family.
- `omnibias.core.transforms` — closed-form Fourier transform of the gaussian,
  which is the self-duality that makes the oscillator special.
- `omnibias-ferminet` — the FermiNet bridge (stable), which needs high-quality
  single-particle orbitals.
- `omnibias-qpinn` — quantum PINNs (alpha).
- `omnibias.{torch,jax}.jet_mv` — mixed partials, which the many-body kinetic
  operator needs.

**Confirmed gap.** `hermite_coeffs` exists as a coefficient table. There is no
ladder-operator surface, no oscillator eigenbasis architecture, and no
exactness statement connecting the tower to the eigenvalue problem.

## 4. Mathematics

### What the tower actually produces

With `G(x) = exp(-x^2 / 2)`, the derivative tower is

```
G^(n)(x) = (-1)^n He_n(x) G(x)
```

with `He_n` the probabilists' Hermite polynomial. Define the **tower-normalized
Hermite functions**

```
h_n(x) = (-1)^n G^(n)(x) = He_n(x) exp(-x^2 / 2)
```

so that an order-`n` collapsed gaussian pack is exactly `h_n` up to sign.

State the next point carefully, because it is where the standard treatment is
usually garbled. `h_n` is **not** the quantum harmonic oscillator eigenfunction.
The eigenfunctions of `H = -1/2 d^2/dx^2 + x^2/2` with eigenvalues `n + 1/2` are
built from the *physicists'* polynomials,

```
psi_n(x) = H_n(x) exp(-x^2 / 2),      H_n(x) = 2^(n/2) He_n( sqrt(2) x )
```

and `h_2 = (x^2 - 1) G` satisfies `H h_2 = 2.5 h_2 + h_0`, not `2.5 h_2`. The two
families differ by a scaling of the argument, and mixing them is the classic
error in this area.

The exact bridge is Rodrigues' formula:

```
psi_n(x) = (-1)^n exp(x^2 / 2) * (d/dx)^n exp(-x^2)
```

So the oscillator eigenbasis is the **tempered gaussian tower** (base
`exp(-x^2)`, which the `tempered` combinator reaches exactly) times a fixed
gaussian reweighting `exp(x^2/2)`. Both operations are exact, so the eigenbasis
is exactly available; it is one factor away from the raw tower, not identical to
it.

### The ladder, in the tower normalization

The clean algebra lives on `h_n`, and it is remarkably simple:

```
raising:    h_{n+1} = -d/dx h_n              (one step up the tower: free)
lowering:   ( x + d/dx ) h_n = n h_{n-1}
```

The first is immediate from `h_n = (-1)^n G^(n)`. The second follows from
`He_n' = n He_{n-1}`. Writing `R = -d/dx` and `L = x + d/dx`:

```
L R h_n = L h_{n+1} = (n+1) h_n
R L h_n = R ( n h_{n-1} ) = n h_n
[L, R] = 1                     exactly
N = R L,   N h_n = n h_n
```

and the number operator expands to

```
N = -( d^2/dx^2 + x d/dx + 1 )
```

which is the Ornstein-Uhlenbeck generator, shifted. That is a genuine and useful
identification: **the natural Hamiltonian of the gaussian tower is the
Ornstein-Uhlenbeck operator, and the quantum oscillator is its gaussian-reweighted
cousin.** Both are exactly realizable; they are not the same operator.

`R` costs nothing (it is the next tower entry) and `L` costs one tower entry plus
a multiplication. So the entire ladder algebra is closed form, and `[L, R] = 1`
is an exact identity the implementation asserts rather than approximates.

### Why this matters for a network

Three concrete consequences.

1. **Exact basis.** An expansion `psi = sum_n c_n psi_n` in the reweighted
   eigenbasis has an exactly known Hamiltonian action:
   `H psi = sum_n (n + 1/2) c_n psi_n`. Energy is a weighted sum of squared
   coefficients with no quadrature and no autodiff.
2. **Exact kinetic energy.** The usual expensive object in variational quantum
   Monte Carlo is the Laplacian of the wavefunction. In this basis it is a
   coefficient operation.
3. **Ladder-structured layers.** A layer that applies `a` or `a+` moves between
   basis levels exactly, which gives an architecture whose depth corresponds to
   excitation number rather than to arbitrary width.

### Beyond the pure oscillator

Real problems are not harmonic. Two honest routes:

- **Perturbation.** Write `H = H_0 + V` with `H_0` harmonic. Matrix elements
  `<psi_m | V | psi_n>` are computable in closed form for polynomial `V` (using
  the ladder algebra) and by quadrature otherwise. The basis is exact; the
  potential is where the work is.
- **Scaled and shifted bases.** A tempered, shifted gaussian pack is an
  oscillator basis centred elsewhere with a different frequency, which is what a
  molecular basis set needs. The tower's exactness survives, since tempering has
  the exact scaling law.

What must not be claimed: that an oscillator basis solves general Schrodinger
problems. It is a good basis for bound states near a minimum and a poor one for
scattering, long-range tails, or strongly anharmonic wells.

### Many-body

For the FermiNet bridge, single-particle orbitals in this basis give exact
kinetic terms and exact derivatives of the Slater determinant entries. The
antisymmetrization and the correlation factors are unchanged. The claim is
narrow and real: **better orbitals with exact derivatives**, not a new many-body
method.

## 5. Worked example

Take `x = 0.7`, `G(0.7) = exp(-0.245) = 0.7827045`:

```
h_0 = G                  = 0.7827045
h_1 = x G                = 0.5478932
h_2 = (x^2 - 1) G        = -0.3991793
h_3 = (x^3 - 3x) G       = (0.343 - 2.1) * 0.7827045 = -1.3752116
```

**Raising is one tower step.** `h_3 = -d/dx h_2`. Symbolically,

```
d/dx h_2 = 2x G + (x^2 - 1)(-x G) = G (3x - x^3)
-d/dx h_2 = G (x^3 - 3x) = h_3
```

Numerically: `d/dx h_2 = 0.7827045 * (2.1 - 0.343) = 1.3752116`, so
`-d/dx h_2 = -1.3752116 = h_3`. Exact, and it cost nothing: `h_3` is the next
entry of the same tower.

**Lowering.** `(x + d/dx) h_2 = 2 h_1`:

```
x h_2      = 0.7 * (-0.3991793)  = -0.2794255
d/dx h_2   =                       1.3752116
sum        =                       1.0957861
2 h_1      = 2 * 0.5478932       = 1.0957864
```

matching to the seven digits carried. Symbolically the identity is exact:
`x h_2 + h_2' = G(x^3 - x) + G(3x - x^3) = 2 x G = 2 h_1`.

**Commutator.** `L R h_2 = L h_3 = 3 h_2` and `R L h_2 = R (2 h_1) = 2 h_2`, so
`[L, R] h_2 = h_2`: the commutator is exactly the identity, with no numerical
content at all.

**The convention trap, shown explicitly.** Apply the quantum Hamiltonian
`H = -1/2 d^2/dx^2 + x^2/2` to `h_2`:

```
h_2''  = d/dx [ G (3x - x^3) ] = G (x^4 - 6 x^2 + 3)
H h_2  = -1/2 G (x^4 - 6x^2 + 3) + (x^2/2) G (x^2 - 1) = (G/2)(5 x^2 - 3)
2.5 h_2 = 2.5 G (x^2 - 1)                              = (G/2)(5 x^2 - 5)
```

so `H h_2 = 2.5 h_2 + h_0`: the tower-normalized function is **not** an
eigenfunction of the quantum oscillator. It *is* an eigenfunction of the number
operator `N = -(d^2/dx^2 + x d/dx + 1)` with eigenvalue exactly `2`:

```
N h_2 = -( x^4 - 6x^2 + 3 ) G - x G (3x - x^3) - G (x^2 - 1)
      = G ( -x^4 + 6x^2 - 3 - 3x^2 + x^4 - x^2 + 1 ) = G ( 2x^2 - 2 ) = 2 h_2
```

The quantum eigenfunction requires the Rodrigues reweighting
`psi_n = (-1)^n e^{x^2/2} (d/dx)^n e^{-x^2}`. Both operators are exactly
available; they are different operators. Gate G2 asserts precisely this
distinction so it cannot be lost in a refactor.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/ladder.py
class Normalization(StrEnum):
    TOWER = "tower"            # h_n = He_n(x) exp(-x^2/2), the raw pack output
    OSCILLATOR = "oscillator"  # psi_n = H_n(x) exp(-x^2/2), the eigenbasis

def hermite_function(
    n: int, x: float, *, normalization: Normalization,
    scale: float = 1.0, centre: float = 0.0,
) -> float:
    """Explicit normalization is required, not defaulted: the two families are
    different functions and silently picking one is the classic bug."""

def rodrigues_reweight(x: float) -> float:
    """exp(x^2 / 2); converts the tempered tower of exp(-x^2) into psi_n."""

def tower_raise(coeffs): ...        # h_{n+1} = -d/dx h_n : shift the tower
def tower_lower(coeffs): ...        # (x + d/dx) h_n = n h_{n-1}
def number_operator_action(coeffs): ...   # N h_n = n h_n, the OU generator
def oscillator_action(coeffs): ...        # H psi_n = (n + 1/2) psi_n

def matrix_element_polynomial(m, n, poly, *, normalization) -> float:
    """<. | p(x) | .> in closed form via the ladder algebra."""
def commutator_residual(n_max: int) -> float:
    """||[L, R] - 1|| on the truncated basis; must be at rounding level."""
```

```python
# omnibias/torch/architectures/ladder.py  (and jax twin)
class HermiteBasis(nn.Module):
    def __init__(self, n_levels: int, *, normalization: Normalization,
                 centres: int = 1, learnable_scale: bool = True,
                 dtype=None) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...            # (..., n_levels)
    def apply_operator(self, coeffs: Tensor, which: Literal["N", "H"]) -> Tensor:
        """Exact: multiply by n, or by (n + 1/2) in the oscillator basis."""
    def raise_(self, coeffs: Tensor) -> Tensor: ...
    def lower(self, coeffs: Tensor) -> Tensor: ...

class LadderNet(nn.Module):
    """Depth indexed by excitation number rather than arbitrary width."""
```

Making `normalization` a required argument rather than a default is deliberate:
the failure mode this spec guards against is a caller who never thinks about it.

## 7. Practical use cases

1. **Bound-state Schrodinger problems** near a potential minimum, where the
   oscillator basis is the natural one and the exact kinetic term removes the
   dominant numerical error.
2. **FermiNet orbitals.** Exact single-particle derivatives feeding the existing
   bridge, with the antisymmetrization untouched.
3. **Vibrational spectroscopy.** Anharmonic corrections as polynomial matrix
   elements, computed in closed form.
4. **Coherent and squeezed states.** These are ladder-algebra objects; an exact
   ladder makes them exactly representable.
5. **Signal processing.** Hermite functions are the eigenfunctions of the
   Fourier transform, so this basis diagonalizes time-frequency operations.

## 8. Acceptance gates

Baselines: a plain MLP wavefunction with autodiff kinetic energy, and a
finite-difference grid solver at matched cost.

- **G1 ladder exactness.** In the tower normalization, `h_{n+1} = -d/dx h_n` and
  `(x + d/dx) h_n = n h_{n-1}` hold to `<= 4 ulp` for `n = 0 .. 20`.
- **G2 two-operator test, the one that matters.** Both of the following are
  asserted, on the same test set:
  `N h_n = n h_n` to `<= 4 ulp` (tower normalization, Ornstein-Uhlenbeck
  generator), and `H psi_n = (n + 1/2) psi_n` to `<= 4 ulp` (oscillator
  normalization). A third assertion checks the **negative**: `H h_n != (n + 1/2)
  h_n`, so a refactor that silently conflates the families fails loudly.
- **G3 commutator.** `commutator_residual(20)` is at rounding level.
- **G4 many-body win.** On a small FermiNet system, exact-derivative orbitals
  reduce the variational energy variance by at least `2x` at matched sample
  count, over five seeds, without changing the mean beyond error bars.
- **G5 anharmonic honesty.** On a strongly anharmonic well, the basis is
  *allowed* to lose to the grid solver, and the result is reported.
- **G6 parity.** torch and jax bit-identical.

## 9. Benchmark plan

- `benchmarks/hermite_ladder.py`: exactness tables, anharmonic sweep, FermiNet
  variance study in `--full`.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/ladder/`.

## 10. Honesty and scope

- The basis comes from the founding bias collapse applied to a **gaussian**
  base, so it uses `hermite_coeffs`, not the Riccati tables. No temperature
  collapse appears.
- **The raw tower is not the oscillator eigenbasis.** The collapsed gaussian
  pack gives `h_n = He_n(x) exp(-x^2/2)`, an eigenfunction of the
  Ornstein-Uhlenbeck generator, not of `-1/2 d^2/dx^2 + x^2/2`. The oscillator
  basis is one exact Rodrigues reweighting away. Both are exactly available;
  saying they are the same object would be false, and G2 asserts the
  distinction, including the negative case.
- The oscillator basis is excellent for bound states near a minimum and poor for
  scattering, long tails and strong anharmonicity. G5 keeps that visible.
- The FermiNet claim is narrow: better orbitals with exact derivatives. It is
  not a new many-body method and not a claim about accuracy on any particular
  chemistry benchmark until measured.
- No certificate tier is claimed. Eigenvalue *lower bounds* for perturbed
  operators are a different exercise and belong to spec 07-05, which uses
  `omnibias.core.verified.eig_operator`.

## 11. Open questions and risks

- **Truncation.** A finite level count truncates the basis, and the ladder
  operators are exact only away from the top level. The API must handle the
  boundary explicitly rather than silently.
- **Conditioning at high `n`.** Hermite polynomials at large `n` and large `|x|`
  overflow; a scaled or recursive evaluation is mandatory, and the usable range
  must be measured.
- **Basis-set choice.** Real chemistry uses contracted gaussians, not pure
  Hermite functions. The bridge to standard basis sets is extra work not covered
  here.
- **Falsifier.** If the exact kinetic term does not reduce variance in the
  FermiNet bridge, the main practical claim fails and this becomes a clean but
  narrow basis library.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/ladder.py` with a required
      `Normalization` argument, never a default
- [ ] torch and jax twins with a parity test
- [ ] Reuse `hermite_coeffs`; do not fork the polynomials
- [ ] Ladder-exactness test for `n = 0 .. 20`
- [ ] Two-operator test including the negative assertion `H h_n != (n+1/2) h_n`
- [ ] Commutator residual test
- [ ] Truncation-boundary behaviour test
- [ ] FermiNet variance study in the benchmark
- [ ] `benchmarks/hermite_ladder.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
