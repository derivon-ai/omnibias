# 02-09 Soliton networks: the tanh method as an omnibias ansatz

## 1. Thesis and status

The classical tanh method writes exact travelling-wave solutions as
**polynomials in `tanh(k x - omega t + b)`** — which is precisely the algebra
that the omnibias tanh tower already implements exactly — so a network built
from that ansatz can represent known soliton solutions with zero residual and
can search for new ones by solving small algebraic systems.

- **Status**: designed
- **Depends on**: 01-01
- **Blocks**: 02-13, 03-11

## 2. Where it lands

`packages/omnibias-pinn/src/omnibias/pinn/travelling/` with torch and jax twins,
plus the pure-Python algebraic solver in
`packages/omnibias-core/src/omnibias/core/tanh_method.py`.

## 3. Prior art in omnibias

- `omnibias.core.polynomials` — `tanh_polynomial_coeffs`: the closed-form
  `sigma^(n)(z) = P_n(tanh z)` coefficients. The tanh tower is *literally* a
  polynomial algebra in `t = tanh`, which is the tanh method's variable.
- `omnibias.symbolic.field_discovery` — `make_burgers_field_split`, which
  generates Burgers data; data generation, not solution machinery.
- `docs/examples/pinn_burgers_shock.py` — a worked shock example.
- `omnibias.{torch,jax}.jet` — `jet_multiply`, `jet_reciprocal`, `jet_exp`,
  `derivative_jet`, `antiderivative_jet`: exact polynomial and analytic
  operations on jets.

**Confirmed gap.** There is no travelling-wave, soliton, or tanh-method
machinery. The only related material is data generation and a single example.

## 4. Mathematics

### The tanh method

For a PDE `N(u, u_t, u_x, u_xx, ...) = 0`, seek a travelling wave
`u(x, t) = U(xi)` with `xi = k x - omega t + b`. Substitute `T = tanh(xi)` and
posit

```
U = sum_{m=0}^{M} a_m T^m
```

Because `dT/dxi = 1 - T^2`, every derivative of `U` is again a polynomial in
`T`:

```
U'   = (1 - T^2) sum_m m a_m T^{m-1}
U''  = (1 - T^2) d/dT [ (1 - T^2) sum_m m a_m T^{m-1} ]
```

So substituting into `N` yields a polynomial in `T`, and setting each
coefficient to zero gives a **finite algebraic system** in `(a_m, k, omega)`. The
balance between the highest nonlinear term and the highest derivative fixes `M`.

This is a classical, complete, and mechanical procedure. What omnibias adds is
that the polynomial algebra `d/dxi = (1 - T^2) d/dT` is exactly the Riccati
identity the whole library is built on, and `tanh_polynomial_coeffs` already
stores the resulting coefficient tables.

### Why this is the right ansatz class for a network

Three properties:

1. **Exact representability.** KdV, Burgers, sine-Gordon, FKPP, Fisher,
   Boussinesq, and many others have travelling-wave solutions in this class.
   A network restricted to it can hit zero residual, not `1e-6`.
2. **Exact derivatives.** The residual is computed from the polynomial algebra,
   so there is no differentiation error at all — the PDE residual is an exact
   rational function of the parameters.
3. **Small parameter count.** A single soliton is `M + 3` numbers.

### Multi-kink superposition

Single travelling waves are the easy case. The interesting ansatz is a
superposition over several wave variables:

```
u(x, t) = sum_{s=1}^{S} sum_{m} a_{s,m} tanh^m( k_s x - omega_s t + b_s )
```

For genuinely integrable equations the exact `n`-soliton solution is *not* a
plain sum (interaction terms appear), which is why spec 02-13's Backlund and
Darboux machinery exists. But the superposition is an excellent **ansatz for
fitting**, and its residual measures exactly how far the true interaction is
from additive — which is itself a useful integrability diagnostic.

Honest framing: a plain multi-kink sum is an ansatz, not a solution formula.
Where the exact `n`-soliton formula is known, use spec 02-13; where it is not,
the sum plus a learned correction is a strong parameterization.

### Balance number

`M` is determined by balancing the highest-order derivative against the highest
nonlinearity. For `u_t + u u_x + u_xxx = 0` (KdV): `u u_x` has degree
`2M + 1` in `T` while `u_xxx` has degree `M + 3`, so `2M + 1 = M + 3` gives
`M = 2`. That calculation is mechanical and should be automated, because getting
it wrong is the most common way the method is misapplied.

## 5. Worked example

**KdV, single soliton.** Solve `u_t + 6 u u_x + u_xxx = 0`.

Balance: `u u_x` gives `2M + 1`, `u_xxx` gives `M + 3`, so `M = 2`. Ansatz
`U = a_0 + a_1 T + a_2 T^2` with `xi = k x - omega t + b`.

Known exact solution: `u = 2 kappa^2 sech^2(kappa(x - 4 kappa^2 t))`. Using
`sech^2 = 1 - T^2`,

```
u = 2 kappa^2 (1 - T^2),   k = kappa,   omega = 4 kappa^3
```

so `a_0 = 2 kappa^2`, `a_1 = 0`, `a_2 = -2 kappa^2`. Take `kappa = 1`:

```
a = (2, 0, -2),   k = 1,   omega = 4
u(x, t) = 2 (1 - tanh^2(x - 4t)) = 2 sech^2(x - 4t)
```

**Symbolic verification.** Every derivative stays polynomial in `T`, using
`dT/dxi = 1 - T^2` throughout:

```
U        =  2 kappa^2 (1 - T^2)
U'       = -4 kappa^2 T (1 - T^2)
U''      = -4 kappa^2 (1 - T^2)(1 - 3 T^2)
U'''     = 16 kappa^2 T (1 - T^2)(2 - 3 T^2)
```

Chain rule with `xi = k x - omega t`, `k = kappa`, `omega = 4 kappa^3`:

```
u_t      = -omega U'  = 16 kappa^5 T (1 - T^2)
u_x      =  k U'      = -4 kappa^3 T (1 - T^2)
u_xxx    =  k^3 U'''  = 16 kappa^5 T (1 - T^2)(2 - 3 T^2)
6 u u_x  = 6 * 2 kappa^2 (1 - T^2) * (-4 kappa^3 T (1 - T^2))
         = -48 kappa^5 T (1 - T^2)^2
```

Summing and factoring out `16 kappa^5 T (1 - T^2)`:

```
u_t + 6 u u_x + u_xxx = 16 kappa^5 T (1 - T^2) [ 1 - 3(1 - T^2) + (2 - 3 T^2) ]
                      = 16 kappa^5 T (1 - T^2) [ 1 - 3 + 3T^2 + 2 - 3T^2 ]
                      = 16 kappa^5 T (1 - T^2) * 0
                      = 0
```

**identically in `T`**, which is what "exact solution" means for this method: not
a small residual, but a bracket that vanishes as a polynomial identity.

Numerically at `kappa = 1`, `x = 0.5`, `t = 0`, so `T = tanh(0.5) = 0.4621172`,
`1 - T^2 = 0.7864477`, `2 - 3T^2 = 1.3593431`:

```
u_t     =  16 * 0.3634310                =  5.8148960
6 u u_x = -48 * 0.3634310 * 0.7864477    = -13.7193408
u_xxx   =  16 * 0.3634310 * 1.3593431    =  7.9044384
sum                                       = -6.4e-6
```

where `0.3634310 = T (1 - T^2)`, and the residual is pure rounding from the
seven-digit intermediates above. In float64 the implementation returns the
identity at machine precision.

The reason G1 below insists on **exact rational** verification rather than a
numerical residual: soliton identities of this shape are easy to get subtly
wrong by hand (the coefficient in `U'''` is `16`, not `8`, and an error there
produces a residual that looks like a modelling issue rather than an algebra
slip). The bracket must be shown to be identically zero as a polynomial, which
is a finite exact computation.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/core/tanh_method.py   (pure Python, exact rational where possible)
@dataclass(frozen=True)
class TravellingWaveAnsatz:
    degree: int                      # M
    coeffs: tuple[Fraction | float, ...]
    wavenumber: float                # k
    frequency: float                 # omega
    shift: float = 0.0               # b

def balance_degree(pde: PDESpec) -> int:
    """Highest derivative versus highest nonlinearity. Mechanical; automated
    because getting it wrong is the classic failure mode."""

def substitute(pde: PDESpec, ansatz: TravellingWaveAnsatz) -> tuple[Fraction, ...]:
    """Coefficients of the resulting polynomial in T. All zero <=> exact
    solution. Exact rational arithmetic when the PDE has rational coefficients."""

def solve_ansatz(pde: PDESpec, *, degree: int | None = None) -> tuple[TravellingWaveAnsatz, ...]:
    """Solve the algebraic system; returns every branch found."""

def verify_exact(pde: PDESpec, ansatz: TravellingWaveAnsatz) -> bool:
    """True only when every polynomial coefficient is exactly zero."""
```

```python
# omnibias/pinn/travelling/torch.py  (and jax twin)
class SolitonField(nn.Module):
    def __init__(self, ansatz: Sequence[TravellingWaveAnsatz], *,
                 learn_interaction: bool = False, dtype=None) -> None: ...
    def forward(self, x: Tensor, t: Tensor) -> Tensor: ...
    def exact_residual(self, x: Tensor, t: Tensor, pde: PDESpec) -> Tensor:
        """Computed from the polynomial algebra, so it carries no
        differentiation error."""
```

## 7. Practical use cases

1. **Exact initialization for PINNs.** Start from the known travelling wave and
   learn only the deviation; the hard part (the sharp front) is already right.
2. **Soliton interaction studies.** Fit a multi-kink ansatz and measure the
   residual of additivity, which quantifies the interaction directly.
3. **Automated ansatz search** for new equations: run `solve_ansatz` and see
   whether a closed-form travelling wave exists at all.
4. **Benchmarks with zero-error references.** Any PDE method can be scored
   against an exactly representable solution instead of a finely resolved
   numerical one.
5. **Integrability screening** (with spec 03-11): equations admitting rich
   families of tanh solutions are candidates for the linearizing transforms of
   spec 02-13.

## 8. Acceptance gates

- **G1 symbolic exactness.** For a curated list of at least ten classical
  equations with known travelling waves (KdV, mKdV, Burgers, sine-Gordon, FKPP,
  Fisher, Boussinesq, Kuramoto-Sivashinsky travelling fronts, Klein-Gordon,
  Camassa-Holm peakon-adjacent cases), `solve_ansatz` recovers the published
  solution and `verify_exact` returns `True` with **exact rational arithmetic**,
  every coefficient identically zero.
- **G2 balance correctness.** `balance_degree` matches the published `M` for
  every equation in the list.
- **G3 numerical residual.** `exact_residual` evaluated in float64 on a dense
  grid is at rounding level (`<= 1e-14` relative to the term magnitudes).
- **G4 initialization win.** A PINN initialized from the ansatz reaches a given
  accuracy target on a perturbed problem in at least `5x` fewer steps than a
  cold start, over five seeds.
- **G5 negative control.** For an equation with no tanh-class travelling wave,
  `solve_ansatz` returns no solutions rather than a spurious one.

## 9. Benchmark plan

- `benchmarks/soliton_ansatz.py`: the ten-equation verification table (fast,
  always in smoke), plus the initialization study in `--full`.
- Smoke JSON committed; full under `$OMNIBIAS_SCRATCH/soliton/`.

## 10. Honesty and scope

- No collapse limit of either kind appears here. This is the tanh *algebra*, the
  same Riccati identity the tower uses, applied to an ansatz. Do not describe it
  as a collapse result.
- **The ansatz class is narrow by construction.** It covers travelling waves
  polynomial in `tanh`. Most PDE solutions are not in it, and the method's value
  is precisely that it is exact where it applies.
- **A multi-kink sum is an ansatz, not the `n`-soliton formula.** For integrable
  equations the true multi-soliton solution has interaction structure; claiming
  otherwise would be wrong, and spec 02-13 is where the correct construction
  lives.
- The tanh method is classical (Malfliet and successors). The contribution is
  the exact implementation on the existing polynomial tables and the integration
  with a differentiable field, not the method.
- Certificate tier: exact rational verification is a finite algebraic fact and
  is a candidate for the Lean obligation class of spec 01-11. That is an
  opportunity, not a claim, until a kernel pass exists.

## 11. Open questions and risks

- **Algebraic system solving.** The coefficient system is polynomial and may
  need Groebner-basis machinery for higher `M`. Adding a computer-algebra
  dependency is a real cost; check how far a hand-rolled resultant approach gets
  first.
- **Branch multiplicity.** Several solution branches exist; returning all of
  them is right, but choosing among them is a modelling decision.
- **The hand-calculation trap.** The worked example above deliberately shows how
  easy it is to slip; the shipped verification must be symbolic.
- **Falsifier.** If ansatz initialization does not speed up training on
  perturbed problems, the practical value reduces to exact reference generation,
  which is still useful but much smaller.

## 12. Implementation checklist

- [ ] `packages/omnibias-core/src/omnibias/core/tanh_method.py` with exact
      rational substitution
- [ ] `packages/omnibias-pinn/src/omnibias/pinn/travelling/` torch and jax twins
- [ ] Reuse `tanh_polynomial_coeffs`; do not fork the tables
- [ ] Ten-equation symbolic verification test with exact zero coefficients
- [ ] Balance-degree test against published values
- [ ] Negative-control test on an equation with no tanh-class wave
- [ ] torch/jax parity test
- [ ] `benchmarks/soliton_ansatz.py` plus smoke JSON
- [ ] Docs page and nav entry
- [ ] Index row in `theory/README.md`
